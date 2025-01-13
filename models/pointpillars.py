"""
PointPillars: full 3D object detection model.

Assembles PillarFeatureNet + BackboneNeck + AnchorHead.
Implements forward pass and multi-task loss (focal + smooth-L1 + direction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .pillar_feature_net import PillarFeatureNet
from .backbone_2d import BackboneNeck
from .detection_head import AnchorHead


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Sigmoid focal loss for class imbalance in dense detection.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N, C) raw classification logits
        targets: (N, C) binary targets in {0, 1}
        alpha: balancing factor for positive class
        gamma: focusing exponent
        reduction: 'sum', 'mean', or 'none'
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    alpha_t = torch.where(targets == 1, alpha * torch.ones_like(probs), (1 - alpha) * torch.ones_like(probs))
    p_t = torch.where(targets == 1, probs, 1 - probs)

    loss = -alpha_t * (1 - p_t) ** gamma * torch.clamp(p_t, min=1e-6).log()

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0 / 9,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss for bounding box regression.

    L = 0.5 * x^2 / beta       if |x| < beta
        |x| - 0.5 * beta        otherwise

    Args:
        pred: (N, 7) predicted box encodings
        target: (N, 7) target box encodings
        beta: transition point between L1 and L2
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta,
    )

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss


class PointPillars(nn.Module):
    """
    PointPillars 3D object detection model.

    Input:
        voxels:     (V, P, C)  — pillar point features
        coords:     (V, 4)     — [batch_idx, z, y, x] pillar coordinates
        num_points: (V,)       — valid points per pillar
        batch_size: int        — samples in batch

    Output:
        predictions dict with cls_preds, box_preds, dir_preds
    """

    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_points_per_voxel: int = 100,
        max_voxels: int = 12000,
        in_channels: int = 9,
        pillar_feat_channels: List[int] = None,
        backbone_out_channels: List[int] = None,
        fpn_upsample_channels: List[int] = None,
        num_classes: int = 3,
        num_anchors_per_loc: int = 2,
    ):
        super().__init__()

        if pillar_feat_channels is None:
            pillar_feat_channels = [64]
        if backbone_out_channels is None:
            backbone_out_channels = [64, 128, 256]
        if fpn_upsample_channels is None:
            fpn_upsample_channels = [128, 128, 128]

        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range

        # Compute BEV grid dimensions
        import numpy as np
        range_size = np.array(point_cloud_range[3:]) - np.array(point_cloud_range[:3])
        self.grid_size = np.round(range_size / np.array(voxel_size)).astype(int)
        # BEV H, W (after stride-2 backbone)
        self.bev_h = self.grid_size[1]
        self.bev_w = self.grid_size[0]

        # Module 1: Pillar Feature Network
        self.pfn = PillarFeatureNet(
            in_channels=in_channels,
            feat_channels=pillar_feat_channels,
        )

        # Module 2: 2D Backbone + FPN Neck
        self.backbone_neck = BackboneNeck(
            in_channels=pillar_feat_channels[-1],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            out_channels=backbone_out_channels,
            upsample_strides=[1, 2, 4],
            upsample_out_channels=fpn_upsample_channels,
        )

        # Module 3: Detection Head
        self.head = AnchorHead(
            in_channels=self.backbone_neck.out_channels,
            num_classes=num_classes,
            num_anchors_per_loc=num_anchors_per_loc,
        )

        # Loss weights
        self.cls_weight = 1.0
        self.reg_weight = 2.0
        self.dir_weight = 0.2

    def forward(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor,
        num_points: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            voxels:     (V, P, C) — pillar features
            coords:     (V, 4)    — [batch_idx, z_idx, y_idx, x_idx]
            num_points: (V,)      — valid points per pillar
            batch_size: int

        Returns:
            predictions: dict with cls_preds, box_preds, dir_preds
        """
        # 1. Encode pillars → BEV canvas
        spatial_shape = (self.bev_h, self.bev_w)

        # coords: (V, 4) — [batch_idx, iz, iy, ix]
        # PillarFeatureNet expects coords as (V, 3) — [batch_idx, iy, ix]
        coords_batch_yx = torch.stack([
            coords[:, 0],   # batch idx
            coords[:, 2],   # y idx
            coords[:, 3],   # x idx
        ], dim=1)

        canvas = self.pfn(
            voxels=voxels,
            coords=coords_batch_yx,
            num_points=num_points,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )  # (B, C_pfn, H, W)

        # 2. Backbone + FPN
        features = self.backbone_neck(canvas)  # (B, C_neck, H', W')

        # 3. Detection head
        predictions = self.head(features)

        return predictions

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        anchors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task detection loss.

        Args:
            predictions: output of forward()
            gt_boxes: list of (M_i, 7) gt boxes per sample
            gt_labels: list of (M_i,) gt labels per sample
            anchors: (H*W*A, 7) anchor boxes

        Returns:
            dict with cls_loss, reg_loss, dir_loss, total_loss
        """
        cls_preds = predictions["cls_preds"]  # (B, A*C, H, W)
        box_preds = predictions["box_preds"]  # (B, A*7, H, W)
        dir_preds = predictions.get("dir_preds", None)

        B = cls_preds.shape[0]
        device = cls_preds.device

        # Reshape predictions for loss computation
        # (B, A*C, H, W) → (B, H*W*A, C)
        A = self.head.num_anchors_per_loc
        C = self.head.num_classes
        H, W = cls_preds.shape[2:]
        HWA = H * W * A

        cls_preds_flat = cls_preds.permute(0, 2, 3, 1).contiguous().view(B, HWA, C)
        box_preds_flat = box_preds.permute(0, 2, 3, 1).contiguous().view(B, HWA, 7)

        # Assign anchors to GT boxes (simplified: positive if IoU > threshold)
        cls_targets, reg_targets, dir_targets, pos_mask = self._assign_targets(
            anchors, gt_boxes, gt_labels, device
        )

        # Classification loss (focal)
        cls_loss = sigmoid_focal_loss(
            cls_preds_flat.view(-1, C),
            cls_targets.view(-1, C),
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        )
        n_pos = pos_mask.float().sum().clamp(min=1.0)
        cls_loss = cls_loss / n_pos

        # Regression loss (smooth L1, only on positive anchors)
        pos_idx = pos_mask.view(-1)
        if pos_idx.any():
            reg_loss = smooth_l1_loss(
                box_preds_flat.view(-1, 7)[pos_idx],
                reg_targets.view(-1, 7)[pos_idx],
                beta=1.0 / 9,
                reduction="sum",
            ) / n_pos
        else:
            reg_loss = torch.tensor(0.0, device=device)

        # Direction loss
        dir_loss = torch.tensor(0.0, device=device)
        if dir_preds is not None and dir_targets is not None and pos_idx.any():
            dir_preds_flat = dir_preds.permute(0, 2, 3, 1).contiguous().view(B, HWA, 2)
            dir_loss = F.cross_entropy(
                dir_preds_flat.view(-1, 2)[pos_idx],
                dir_targets.view(-1)[pos_idx],
                reduction="sum",
            ) / n_pos

        total_loss = (
            self.cls_weight * cls_loss
            + self.reg_weight * reg_loss
            + self.dir_weight * dir_loss
        )

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "dir_loss": dir_loss,
            "total_loss": total_loss,
        }

    def _assign_targets(
        self,
        anchors: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Simplified anchor assignment using 2D BEV IoU.

        Returns:
            cls_targets: (B, HWA, C)
            reg_targets: (B, HWA, 7)
            dir_targets: (B, HWA) long
            pos_mask: (B, HWA) bool
        """
        B = len(gt_boxes)
        n_anchors = anchors.shape[0]
        C = self.head.num_classes

        cls_targets = torch.zeros(B, n_anchors, C, device=device)
        reg_targets = torch.zeros(B, n_anchors, 7, device=device)
        dir_targets = torch.zeros(B, n_anchors, dtype=torch.long, device=device)
        pos_mask = torch.zeros(B, n_anchors, dtype=torch.bool, device=device)

        anchors_dev = anchors.to(device)

        for b in range(B):
            boxes = gt_boxes[b].to(device)  # (M, 7)
            labels = gt_labels[b].to(device)  # (M,)

            if len(boxes) == 0:
                continue

            # Compute 2D BEV IoU between anchors and GT boxes
            iou = self._compute_bev_iou(anchors_dev, boxes)  # (N_anchors, M)

            max_iou, matched_gt = iou.max(dim=1)  # (N_anchors,)

            # Positive: IoU > 0.6
            pos = max_iou >= 0.6
            # Negative: IoU < 0.45
            neg = max_iou < 0.45
            # Ignore: in between

            pos_mask[b] = pos

            # Classification targets
            matched_labels = labels[matched_gt]
            for i, (is_pos, label) in enumerate(zip(pos, matched_labels)):
                if is_pos:
                    cls_targets[b, i, label.long()] = 1.0

            # Regression targets for positive anchors
            if pos.any():
                gt_matched = boxes[matched_gt[pos]]  # (n_pos, 7)
                anch_matched = anchors_dev[pos]       # (n_pos, 7)

                # Encode box targets
                reg_targets[b, pos] = self._encode_boxes(gt_matched, anch_matched)

                # Direction: 0 if yaw in [0, pi), else 1
                dir_targets[b, pos] = (gt_matched[:, 6] > 0).long()

        return cls_targets, reg_targets, dir_targets, pos_mask

    def _compute_bev_iou(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 2D axis-aligned IoU between anchors and GT boxes.
        Uses simplified rectangle IoU (ignores rotation for efficiency).

        Args:
            anchors: (N, 7)
            gt_boxes: (M, 7)

        Returns:
            iou: (N, M)
        """
        # Convert center format to corner format [x1, y1, x2, y2]
        def to_corners(boxes):
            x, y, l, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
            x1 = x - l / 2
            y1 = y - w / 2
            x2 = x + l / 2
            y2 = y + w / 2
            return x1, y1, x2, y2

        ax1, ay1, ax2, ay2 = to_corners(anchors)   # (N,)
        gx1, gy1, gx2, gy2 = to_corners(gt_boxes)  # (M,)

        # Broadcast to (N, M)
        inter_x1 = torch.max(ax1.unsqueeze(1), gx1.unsqueeze(0))  # (N, M)
        inter_y1 = torch.max(ay1.unsqueeze(1), gy1.unsqueeze(0))
        inter_x2 = torch.min(ax2.unsqueeze(1), gx2.unsqueeze(0))
        inter_y2 = torch.min(ay2.unsqueeze(1), gy2.unsqueeze(0))

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h  # (N, M)

        a_area = (ax2 - ax1) * (ay2 - ay1)  # (N,)
        g_area = (gx2 - gx1) * (gy2 - gy1)  # (M,)

        union = a_area.unsqueeze(1) + g_area.unsqueeze(0) - inter_area
        iou = inter_area / union.clamp(min=1e-6)

        return iou

    def _encode_boxes(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode GT boxes relative to anchors.

        Standard PointPillars encoding:
            dx = (x_gt - x_a) / d_a
            dy = (y_gt - y_a) / d_a
            dz = (z_gt - z_a) / h_a
            dl = log(l_gt / l_a)
            dw = log(w_gt / w_a)
            dh = log(h_gt / h_a)
            dsin = sin(r_gt - r_a)

        Returns:
            encoded: (N, 7)
        """
        d_a = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

        dx = (gt_boxes[:, 0] - anchors[:, 0]) / d_a
        dy = (gt_boxes[:, 1] - anchors[:, 1]) / d_a
        dz = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5].clamp(min=1e-6)
        dl = torch.log(gt_boxes[:, 3] / anchors[:, 3].clamp(min=1e-6))
        dw = torch.log(gt_boxes[:, 4] / anchors[:, 4].clamp(min=1e-6))
        dh = torch.log(gt_boxes[:, 5] / anchors[:, 5].clamp(min=1e-6))
        dr = torch.sin(gt_boxes[:, 6] - anchors[:, 6])

        return torch.stack([dx, dy, dz, dl, dw, dh, dr], dim=-1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"PointPillars saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "PointPillars":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
