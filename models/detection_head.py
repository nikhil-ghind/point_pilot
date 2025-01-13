"""
AnchorHead: per-anchor classification + box regression + direction heads.

For each spatial location in the BEV feature map, predicts:
    - cls_preds:  (B, A*C, H, W) — classification logits per anchor per class
    - box_preds:  (B, A*7, H, W) — box regression residuals [dx, dy, dz, dl, dw, dh, dsin, dcos]
    - dir_preds:  (B, A*2, H, W) — binary direction classification (to resolve yaw ambiguity)

where A = anchors per location, C = num_classes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class AnchorHead(nn.Module):
    """
    Single-shot anchor-based detection head for PointPillars.

    Takes BEV feature maps from backbone+neck and produces
    dense predictions for all anchor locations.

    Box encoding follows KITTI convention:
        dx = (x_gt - x_a) / d_a
        dy = (y_gt - y_a) / d_a
        dz = (z_gt - z_a) / h_a
        dl = log(l_gt / l_a)
        dw = log(w_gt / w_a)
        dh = log(h_gt / h_a)
        dsin = sin(theta_gt - theta_a)
        dcos = cos(theta_gt - theta_a)

    where d_a = sqrt(l_a^2 + w_a^2) is the diagonal of the anchor base.
    """

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 3,
        num_anchors_per_loc: int = 2,
        box_code_size: int = 7,
        use_direction_classifier: bool = True,
    ):
        """
        Args:
            in_channels: Input feature channels from backbone neck
            num_classes: Number of object classes
            num_anchors_per_loc: Number of anchors per spatial location
                                  (typically 2: 0 and π/2 rotation)
            box_code_size: Number of box regression parameters (7: x,y,z,l,w,h,yaw)
            use_direction_classifier: Whether to add direction classification head
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_loc = num_anchors_per_loc
        self.box_code_size = box_code_size
        self.use_direction_classifier = use_direction_classifier

        # Separate heads for each output type
        # Classification: A * C channels
        self.cls_head = nn.Conv2d(
            in_channels,
            num_anchors_per_loc * num_classes,
            kernel_size=1,
        )

        # Box regression: A * 7 channels
        self.box_head = nn.Conv2d(
            in_channels,
            num_anchors_per_loc * box_code_size,
            kernel_size=1,
        )

        # Direction: A * 2 channels (binary: forward/backward)
        if use_direction_classifier:
            self.dir_head = nn.Conv2d(
                in_channels,
                num_anchors_per_loc * 2,
                kernel_size=1,
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize classification head with bias for focal loss stability."""
        import math
        # Prior probability for foreground class (from RetinaNet)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, bias_value)

        nn.init.normal_(self.box_head.weight, std=0.001)
        nn.init.constant_(self.box_head.bias, 0.0)

        if self.use_direction_classifier:
            nn.init.normal_(self.dir_head.weight, std=0.01)
            nn.init.constant_(self.dir_head.bias, 0.0)

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C, H, W) — BEV feature map

        Returns:
            dict with:
                cls_preds: (B, A*num_classes, H, W)
                box_preds: (B, A*7, H, W)
                dir_preds: (B, A*2, H, W)  — if use_direction_classifier
        """
        cls_preds = self.cls_head(features)
        box_preds = self.box_head(features)

        preds = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
        }

        if self.use_direction_classifier:
            preds["dir_preds"] = self.dir_head(features)

        return preds

    def get_anchors(
        self,
        feature_map_size: Tuple[int, int],
        anchor_sizes: List[List[float]],
        anchor_rotations: List[float],
        anchor_center_z: float,
        pc_range: List[float],
        voxel_size: List[float],
    ) -> torch.Tensor:
        """
        Generate anchor boxes for a given feature map size.

        Args:
            feature_map_size: (H, W) of the BEV feature map
            anchor_sizes: List of [l, w, h] for each anchor template
            anchor_rotations: List of yaw angles (e.g. [0, pi/2])
            anchor_center_z: Z center of anchors
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            voxel_size: [vx, vy, vz]

        Returns:
            anchors: (H*W*A, 7) — [x, y, z, l, w, h, yaw]
        """
        import numpy as np

        H, W = feature_map_size

        # BEV stride = voxel_size * feature_map_stride
        # For PointPillars, the backbone applies stride-2 twice → stride 4
        # but feature map H, W already accounts for the stride
        x_stride = (pc_range[3] - pc_range[0]) / W
        y_stride = (pc_range[4] - pc_range[1]) / H

        # Center coordinates for each feature map cell
        x_offset = pc_range[0] + x_stride / 2
        y_offset = pc_range[1] + y_stride / 2

        xs = np.arange(W) * x_stride + x_offset   # (W,)
        ys = np.arange(H) * y_stride + y_offset   # (H,)

        # Grid of centers: (H, W, 2)
        xx, yy = np.meshgrid(xs, ys)

        anchors = []
        for size in anchor_sizes:
            l, w, h = size
            for rot in anchor_rotations:
                # Broadcast to (H, W)
                z = np.full_like(xx, anchor_center_z)
                l_arr = np.full_like(xx, l)
                w_arr = np.full_like(xx, w)
                h_arr = np.full_like(xx, h)
                r_arr = np.full_like(xx, rot)
                # Stack: (H, W, 7)
                anchor = np.stack([xx, yy, z, l_arr, w_arr, h_arr, r_arr], axis=-1)
                anchors.append(anchor.reshape(-1, 7))  # (H*W, 7)

        anchors = np.concatenate(anchors, axis=0)  # (H*W*A, 7)
        return torch.from_numpy(anchors).float()
