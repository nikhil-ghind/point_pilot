"""
PillarFeatureNet: per-pillar PointNet that produces a BEV pseudo-image.

Architecture:
    Input:  (V, P, C) — V voxels, P points per voxel, C input features
    Per-pillar PointNet: Linear → BN → ReLU → MaxPool
    Scatter back to BEV canvas: (C', H, W)

References:
    Lang et al., "PointPillars: Fast Encoders for Object Detection from
    Point Clouds", CVPR 2019.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class PFNLayer(nn.Module):
    """
    A single Pillar Feature Network layer.

    Linear → BN → ReLU applied point-wise.
    Followed by channel-wise MaxPool over all points in each pillar.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last_layer: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (V, P, in_channels)

        Returns:
            out: (V, out_channels) if last_layer else (V, P, out_channels)
        """
        V, P, C = inputs.shape

        # (V*P, C) for linear + BN
        x = inputs.view(V * P, C)
        x = self.linear(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = x.view(V, P, -1)  # (V, P, out_channels)

        if self.last_layer:
            # MaxPool over point dimension
            x = x.max(dim=1).values  # (V, out_channels)

        return x


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network.

    Processes each pillar independently through a PointNet-style encoder,
    then scatters the per-pillar feature vectors back to a 2D BEV canvas.

    Input:
        voxels:     (V, P, C_in) — raw pillar point features
        coords:     (V, 3)       — [batch_idx, y_idx, x_idx]
        num_points: (V,)          — valid point count per pillar

    Output:
        canvas: (B, C_out, H, W) — BEV pseudo-image
    """

    def __init__(
        self,
        in_channels: int = 9,
        feat_channels: List[int] = None,
        with_distance: bool = False,
    ):
        """
        Args:
            in_channels: Input feature dimension per point (default 9)
            feat_channels: List of output channels for each PFN layer
            with_distance: If True, append Euclidean distance as extra feature
        """
        super().__init__()
        if feat_channels is None:
            feat_channels = [64]

        self.in_channels = in_channels
        self.with_distance = with_distance

        if with_distance:
            in_channels += 1

        # Build PFN layers
        pfn_layers = []
        for i, out_ch in enumerate(feat_channels):
            is_last = (i == len(feat_channels) - 1)
            pfn_layers.append(PFNLayer(in_channels, out_ch, last_layer=is_last))
            in_channels = out_ch

        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.out_channels = feat_channels[-1]

    def forward(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor,
        num_points: torch.Tensor,
        spatial_shape: Tuple[int, int],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            voxels:       (V, P, C_in) — pillar point features
            coords:       (V, 3)  — [batch_idx, y_idx, x_idx]
            num_points:   (V,)    — number of valid points per pillar
            spatial_shape: (H, W) — BEV grid dimensions
            batch_size:   int     — batch size

        Returns:
            canvas: (B, C_out, H, W) — BEV pseudo-image
        """
        V, P, C = voxels.shape

        # Optionally add distance feature
        if self.with_distance:
            dist = torch.norm(voxels[:, :, :3], dim=-1, keepdim=True)  # (V, P, 1)
            voxels = torch.cat([voxels, dist], dim=-1)

        # Mask padding positions (points beyond num_points) to zero
        # This prevents padded zeros from influencing the max pool
        voxel_count = voxels.shape[1]
        mask = self._get_paddings_indicator(num_points, voxel_count)  # (V, P)
        mask = mask.unsqueeze(-1).float()  # (V, P, 1)
        voxels = voxels * mask

        # Run PFN layers
        x = voxels
        for pfn in self.pfn_layers:
            x = pfn(x)
        # x: (V, C_out)

        # Scatter pillars back to BEV canvas
        H, W = spatial_shape
        canvas = torch.zeros(
            batch_size, self.out_channels, H, W,
            device=voxels.device,
            dtype=voxels.dtype,
        )

        # coords: (V, 3) — [batch_idx, y_idx, x_idx]
        batch_idx = coords[:, 0].long()
        y_idx = coords[:, 1].long()
        x_idx = coords[:, 2].long()

        # Clip indices to be safe
        y_idx = y_idx.clamp(0, H - 1)
        x_idx = x_idx.clamp(0, W - 1)

        # Scatter: canvas[b, :, y, x] = pillar_features[v]
        canvas[batch_idx, :, y_idx, x_idx] = x

        return canvas

    @staticmethod
    def _get_paddings_indicator(
        actual_num: torch.Tensor,
        max_num: int,
    ) -> torch.Tensor:
        """
        Create a boolean mask for valid points in each pillar.

        Args:
            actual_num: (V,) number of valid points per pillar
            max_num: P — maximum points per pillar

        Returns:
            mask: (V, P) bool — True for valid points
        """
        actual_num = actual_num.unsqueeze(-1)  # (V, 1)
        max_num_range = torch.arange(max_num, device=actual_num.device)  # (P,)
        paddings_indicator = actual_num > max_num_range  # (V, P)
        return paddings_indicator
