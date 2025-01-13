"""
KITTIDataset: loads KITTI velodyne point clouds and label files.

Directory structure expected:
    data/KITTI/
    ├── training/
    │   ├── velodyne/       # .bin point cloud files
    │   ├── label_2/        # .txt label files
    │   └── calib/          # .txt calibration files
    └── ImageSets/
        ├── train.txt       # frame indices for training split
        └── val.txt         # frame indices for validation split
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# KITTI class name to integer label
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}

# Classes to ignore during training
IGNORED_CLASSES = {"DontCare", "Misc", "Person_sitting", "Tram", "Truck", "Van"}


def read_velodyne_bin(path: str) -> np.ndarray:
    """
    Read a KITTI velodyne binary point cloud file.

    Returns:
        points: (N, 4) float32 array with columns [x, y, z, intensity]
    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


def read_label_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a KITTI label .txt file.

    Format per line:
        type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y

    Returns:
        gt_boxes: (M, 7) float32 [x, y, z, l, w, h, yaw] in LiDAR frame
        labels: (M,) int32 class labels
    """
    gt_boxes = []
    labels = []

    if not os.path.exists(path):
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int32)

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj_type = parts[0]

            # Skip ignored classes
            if obj_type in IGNORED_CLASSES:
                continue
            if obj_type not in CLASS_TO_LABEL:
                continue

            # Camera coordinates: h, w, l, x, y, z, rot_y
            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            x_cam = float(parts[11])
            y_cam = float(parts[12])
            z_cam = float(parts[13])
            rot_y = float(parts[14])

            # Convert camera coords to LiDAR coords (approximate without calib)
            # In KITTI: LiDAR z = camera -y, LiDAR x = camera z, LiDAR y = camera -x
            x_lidar = z_cam
            y_lidar = -x_cam
            z_lidar = -y_cam + h / 2.0  # bottom to center

            # Yaw: camera rot_y -> LiDAR yaw
            yaw = -rot_y - np.pi / 2.0

            gt_boxes.append([x_lidar, y_lidar, z_lidar, l, w, h, yaw])
            labels.append(CLASS_TO_LABEL[obj_type])

    if not gt_boxes:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return (
        np.array(gt_boxes, dtype=np.float32),
        np.array(labels, dtype=np.int32),
    )


class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI 3D object detection.

    Returns dicts with:
        points:    (N, 4) float32  — x, y, z, intensity
        gt_boxes:  (M, 7) float32  — x, y, z, l, w, h, yaw (LiDAR frame)
        labels:    (M,)   int32    — class ids (0=Car, 1=Pedestrian, 2=Cyclist)
        frame_id:  str             — frame identifier (e.g. "000000")
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        point_cloud_range: Optional[List[float]] = None,
        max_points: int = 120000,
    ):
        """
        Args:
            root_dir: Path to KITTI data root (contains 'training/', 'ImageSets/')
            split: 'train' or 'val'
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] filter
            max_points: Clip point cloud to this many points (random sample)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_points = max_points

        if point_cloud_range is not None:
            self.pc_range = np.array(point_cloud_range, dtype=np.float32)
        else:
            self.pc_range = np.array([0, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)

        # Paths
        self.velodyne_dir = self.root_dir / "training" / "velodyne"
        self.label_dir = self.root_dir / "training" / "label_2"
        self.split_file = self.root_dir / "ImageSets" / f"{split}.txt"

        # Load frame ids
        self.frame_ids = self._load_split()

    def _load_split(self) -> List[str]:
        """Load list of frame IDs from split file, or auto-generate."""
        if self.split_file.exists():
            with open(self.split_file, "r") as f:
                ids = [line.strip() for line in f if line.strip()]
            return ids

        # If no split file, generate from velodyne directory
        if self.velodyne_dir.exists():
            files = sorted(self.velodyne_dir.glob("*.bin"))
            all_ids = [f.stem for f in files]
            # 80/20 train/val split
            n = len(all_ids)
            if self.split == "train":
                return all_ids[:int(0.8 * n)]
            else:
                return all_ids[int(0.8 * n):]

        return []

    def _filter_points_in_range(self, points: np.ndarray) -> np.ndarray:
        """Remove points outside the point cloud range."""
        mask = (
            (points[:, 0] >= self.pc_range[0]) &
            (points[:, 0] <= self.pc_range[3]) &
            (points[:, 1] >= self.pc_range[1]) &
            (points[:, 1] <= self.pc_range[4]) &
            (points[:, 2] >= self.pc_range[2]) &
            (points[:, 2] <= self.pc_range[5])
        )
        return points[mask]

    def _filter_boxes_in_range(
        self,
        gt_boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove GT boxes whose centers are outside the range."""
        if len(gt_boxes) == 0:
            return gt_boxes, labels
        mask = (
            (gt_boxes[:, 0] >= self.pc_range[0]) &
            (gt_boxes[:, 0] <= self.pc_range[3]) &
            (gt_boxes[:, 1] >= self.pc_range[1]) &
            (gt_boxes[:, 1] <= self.pc_range[4])
        )
        return gt_boxes[mask], labels[mask]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Dict:
        frame_id = self.frame_ids[idx]

        # Load point cloud
        bin_path = self.velodyne_dir / f"{frame_id}.bin"
        if bin_path.exists():
            points = read_velodyne_bin(str(bin_path))
        else:
            # Return empty point cloud if file missing
            points = np.zeros((1, 4), dtype=np.float32)

        # Filter points in range
        points = self._filter_points_in_range(points)

        # Subsample if too many points
        if len(points) > self.max_points:
            idx_sample = np.random.choice(len(points), self.max_points, replace=False)
            points = points[idx_sample]

        # Load labels
        label_path = self.label_dir / f"{frame_id}.txt"
        gt_boxes, labels = read_label_file(str(label_path))
        gt_boxes, labels = self._filter_boxes_in_range(gt_boxes, labels)

        return {
            "points": torch.from_numpy(points),         # (N, 4)
            "gt_boxes": torch.from_numpy(gt_boxes),     # (M, 7)
            "labels": torch.from_numpy(labels),          # (M,)
            "frame_id": frame_id,
        }


def collate_fn(batch):
    """
    Custom collate function for variable-size point clouds and GT boxes.

    Returns batch dict where 'points', 'gt_boxes', 'labels' are lists
    (not stacked tensors) since they have different sizes per sample.
    """
    return {
        "points": [b["points"] for b in batch],
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "frame_ids": [b["frame_id"] for b in batch],
    }


def create_kitti_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    point_cloud_range: Optional[List[float]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create KITTI train and val DataLoaders."""
    train_dataset = KITTIDataset(root_dir, split="train", point_cloud_range=point_cloud_range)
    val_dataset = KITTIDataset(root_dir, split="val", point_cloud_range=point_cloud_range)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"KITTI — train: {len(train_dataset)}, val: {len(val_dataset)}")
    return train_loader, val_loader
