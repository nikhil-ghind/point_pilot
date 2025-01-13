"""
Voxelizer: pure Python/NumPy voxelization from scratch.

Converts raw point clouds into pillar (voxel) representation for PointPillars.
No external voxelization library — hand-implemented with NumPy.

The voxelization:
1. Computes which voxel cell each point belongs to (grid index)
2. Groups points by voxel, truncating to max_points_per_voxel
3. Returns voxel feature tensors, coordinates, and point counts
"""

import numpy as np
from typing import Tuple


class Voxelizer:
    """
    Point cloud voxelizer for PointPillars.

    For PointPillars, voxels are "pillars" — infinite columns in the Z direction.
    The Z dimension of voxel_size is set large enough to capture the full scene
    height in a single voxel column, reducing 3D voxelization to a 2D grid.

    Usage:
        voxelizer = Voxelizer(
            voxel_size=[0.16, 0.16, 4.0],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_points_per_voxel=100,
            max_voxels=12000,
        )
        voxels, coords, num_points = voxelizer.voxelize(points)
    """

    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        max_points_per_voxel: int = 100,
        max_voxels: int = 12000,
    ):
        """
        Args:
            voxel_size: [vx, vy, vz] — size of each voxel in meters
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points_per_voxel: P — maximum points stored per voxel
            max_voxels: T — maximum number of non-empty voxels to return
        """
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.pc_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

        # Compute grid dimensions
        range_size = self.pc_range[3:] - self.pc_range[:3]  # [rx, ry, rz]
        self.grid_size = np.round(range_size / self.voxel_size).astype(np.int32)
        # grid_size: [nx, ny, nz]

        # Precompute reciprocals for speed
        self._inv_voxel_size = 1.0 / self.voxel_size

    @property
    def grid_x(self) -> int:
        return int(self.grid_size[0])

    @property
    def grid_y(self) -> int:
        return int(self.grid_size[1])

    @property
    def grid_z(self) -> int:
        return int(self.grid_size[2])

    def voxelize(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelize a point cloud.

        Algorithm:
        1. Shift points so that pc_range min is at origin
        2. Compute integer voxel indices (i, j, k) for each point
        3. Filter out-of-range points
        4. Group points by voxel index using a hash map
        5. Truncate each group to max_points_per_voxel
        6. Return first max_voxels non-empty voxels

        Args:
            points: (N, C) float32 point cloud, C >= 3 (x, y, z, ...)

        Returns:
            voxels:       (T, P, C) — T voxels, up to P points each, C features
            coords:       (T, 3)   — voxel grid indices [z_idx, y_idx, x_idx]
            num_points:   (T,)     — number of valid points in each voxel
        """
        assert points.ndim == 2 and points.shape[1] >= 3
        N, C = points.shape

        # Step 1: Shift and compute voxel indices
        shifted = points[:, :3] - self.pc_range[:3]  # (N, 3)
        voxel_indices = (shifted * self._inv_voxel_size).astype(np.int32)  # (N, 3) [ix, iy, iz]

        # Step 2: Filter points outside the grid
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self.grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self.grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self.grid_size[2])
        )
        voxel_indices = voxel_indices[valid_mask]  # (N', 3)
        valid_points = points[valid_mask]           # (N', C)

        if len(valid_points) == 0:
            return (
                np.zeros((0, self.max_points_per_voxel, C), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )

        # Step 3: Compute flat hash key for each voxel
        # key = ix * (ny * nz) + iy * nz + iz
        ny, nz = self.grid_size[1], self.grid_size[2]
        keys = (
            voxel_indices[:, 0] * (ny * nz)
            + voxel_indices[:, 1] * nz
            + voxel_indices[:, 2]
        )  # (N',)

        # Step 4: Group points by voxel key
        # Use np.unique to find voxel membership
        sorted_order = np.argsort(keys)
        keys_sorted = keys[sorted_order]
        unique_keys, start_indices, counts = np.unique(
            keys_sorted, return_index=True, return_counts=True
        )

        # Limit number of voxels
        n_voxels = min(len(unique_keys), self.max_voxels)
        unique_keys = unique_keys[:n_voxels]
        start_indices = start_indices[:n_voxels]
        counts = counts[:n_voxels]

        # Step 5: Fill voxel arrays
        P = self.max_points_per_voxel
        voxels = np.zeros((n_voxels, P, C), dtype=np.float32)
        num_points = np.zeros(n_voxels, dtype=np.int32)
        coords = np.zeros((n_voxels, 3), dtype=np.int32)

        # Recover voxel grid indices from flat keys
        # key = ix * ny * nz + iy * nz + iz
        coords[:, 0] = unique_keys // (ny * nz)                # ix
        coords[:, 1] = (unique_keys % (ny * nz)) // nz         # iy
        coords[:, 2] = unique_keys % nz                         # iz

        for i in range(n_voxels):
            start = start_indices[i]
            n_pts = min(int(counts[i]), P)

            point_indices_in_voxel = sorted_order[start: start + n_pts]
            voxels[i, :n_pts, :] = valid_points[point_indices_in_voxel]
            num_points[i] = n_pts

        # Return coords as [z, y, x] (batch processing convention)
        coords_zyx = coords[:, [2, 1, 0]]  # [iz, iy, ix]

        return voxels, coords_zyx, num_points

    def voxelize_with_augmented_features(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelize and augment point features with pillar-center offsets.

        For PointPillars, the original feature set is:
            [x, y, z, intensity, x_c, y_c, z_c, x_p, y_p]
        where (x_c, y_c, z_c) is the mean of points in the pillar,
        and (x_p, y_p) is the offset from the pillar center in XY.

        Args:
            points: (N, 4) [x, y, z, intensity]

        Returns:
            voxels:     (T, P, 9)  — augmented features
            coords:     (T, 3)    — [iz, iy, ix]
            num_points: (T,)      — valid points per voxel
        """
        voxels_raw, coords, num_points = self.voxelize(points)

        T, P, C = voxels_raw.shape
        if T == 0:
            return np.zeros((0, P, 9), dtype=np.float32), coords, num_points

        # Compute pillar center (mean of valid points) for each voxel
        # Mask invalid (zero-padded) point positions
        voxels_aug = np.zeros((T, P, 9), dtype=np.float32)
        voxels_aug[:, :, :C] = voxels_raw

        for i in range(T):
            n = num_points[i]
            if n == 0:
                continue
            valid_pts = voxels_raw[i, :n, :3]   # (n, 3): xyz
            center = valid_pts.mean(axis=0)      # (3,): x_c, y_c, z_c

            # x_c, y_c, z_c — same for all points in pillar
            voxels_aug[i, :n, C:C+3] = center

            # x_p, y_p — XY offset from pillar center
            voxels_aug[i, :n, C+3] = valid_pts[:, 0] - center[0]
            voxels_aug[i, :n, C+4] = valid_pts[:, 1] - center[1]

        return voxels_aug, coords, num_points

    def get_output_shape(self) -> Tuple[int, int]:
        """Return (H, W) of the BEV pseudo-image output."""
        return (self.grid_y, self.grid_x)

    def __repr__(self) -> str:
        return (
            f"Voxelizer("
            f"voxel_size={self.voxel_size.tolist()}, "
            f"grid={self.grid_size.tolist()}, "
            f"max_pts={self.max_points_per_voxel}, "
            f"max_voxels={self.max_voxels})"
        )
