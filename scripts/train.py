"""
Training script for PointPillars on KITTI.

Usage:
    python -m scripts.train --config configs/pointpillars.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.kitti_dataset import create_kitti_dataloaders
from data.voxelization import Voxelizer
from models.pointpillars import PointPillars


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def voxelize_batch(
    points_list: List[torch.Tensor],
    voxelizer: Voxelizer,
    device: str,
) -> tuple:
    """
    Voxelize a batch of point clouds and stack into tensors.

    Returns:
        voxels_all:   (V_total, P, C)
        coords_all:   (V_total, 4) — [batch_idx, iz, iy, ix]
        num_pts_all:  (V_total,)
    """
    all_voxels = []
    all_coords = []
    all_num_pts = []

    for b, pts in enumerate(points_list):
        pts_np = pts.numpy().astype(np.float32)
        voxels, coords, num_pts = voxelizer.voxelize_with_augmented_features(pts_np)

        if len(voxels) == 0:
            continue

        # Add batch index as first column to coords
        batch_col = np.full((len(coords), 1), b, dtype=np.int32)
        coords_with_batch = np.concatenate([batch_col, coords], axis=1)  # (V, 4)

        all_voxels.append(voxels)
        all_coords.append(coords_with_batch)
        all_num_pts.append(num_pts)

    if not all_voxels:
        C = voxelizer.max_points_per_voxel
        return (
            torch.zeros(0, voxelizer.max_points_per_voxel, 9, device=device),
            torch.zeros(0, 4, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    voxels_cat = torch.from_numpy(np.concatenate(all_voxels, axis=0)).float().to(device)
    coords_cat = torch.from_numpy(np.concatenate(all_coords, axis=0)).long().to(device)
    num_pts_cat = torch.from_numpy(np.concatenate(all_num_pts, axis=0)).long().to(device)

    return voxels_cat, coords_cat, num_pts_cat


def train(config_path: str):
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    device = train_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    # Voxelizer
    vox_cfg = cfg["voxelization"]
    voxelizer = Voxelizer(
        voxel_size=vox_cfg["voxel_size"],
        point_cloud_range=vox_cfg["point_cloud_range"],
        max_points_per_voxel=vox_cfg["max_points_per_voxel"],
        max_voxels=vox_cfg["max_voxels"],
    )
    print(f"Voxelizer: {voxelizer}")

    # DataLoaders
    train_loader, val_loader = create_kitti_dataloaders(
        root_dir=cfg["data"]["root_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        point_cloud_range=vox_cfg["point_cloud_range"],
    )

    # Model
    model_cfg = cfg["model"]
    model = PointPillars(
        voxel_size=vox_cfg["voxel_size"],
        point_cloud_range=vox_cfg["point_cloud_range"],
        max_points_per_voxel=vox_cfg["max_points_per_voxel"],
        max_voxels=vox_cfg["max_voxels"],
        in_channels=model_cfg["in_channels"],
        pillar_feat_channels=model_cfg["pillar_feat_channels"],
        backbone_out_channels=model_cfg["backbone_out_channels"],
        fpn_upsample_channels=model_cfg["fpn_upsample_channels"],
        num_classes=model_cfg["num_classes"],
        num_anchors_per_loc=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PointPillars parameters: {n_params:,}")

    # Generate anchors for KITTI (one class: Car, for simplicity)
    feature_map_h = voxelizer.grid_y // 2
    feature_map_w = voxelizer.grid_x // 2
    anchors = model.head.get_anchors(
        feature_map_size=(feature_map_h, feature_map_w),
        anchor_sizes=cfg["anchors"]["Car"]["sizes"],
        anchor_rotations=cfg["anchors"]["Car"]["rotations"],
        anchor_center_z=cfg["anchors"]["Car"]["center_z"],
        pc_range=vox_cfg["point_cloud_range"],
        voxel_size=vox_cfg["voxel_size"],
    ).to(device)
    print(f"Anchors: {anchors.shape}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=train_cfg["lr_decay_epochs"],
        gamma=train_cfg["lr_decay_factor"],
    )

    save_dir = Path(cfg["checkpoints"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    print("\n" + "=" * 60)
    print("Starting PointPillars Training")
    print("=" * 60)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_losses = {"cls_loss": 0.0, "reg_loss": 0.0, "dir_loss": 0.0, "total_loss": 0.0}
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            points_list = batch["points"]
            gt_boxes_list = batch["gt_boxes"]
            gt_labels_list = batch["labels"]
            B = len(points_list)

            # Voxelize batch
            voxels, coords, num_pts = voxelize_batch(points_list, voxelizer, device)

            if len(voxels) == 0:
                continue

            # Forward
            optimizer.zero_grad()
            predictions = model(voxels, coords, num_pts, batch_size=B)

            # Compute loss
            losses = model.loss(predictions, gt_boxes_list, gt_labels_list, anchors)
            total_loss = losses["total_loss"]

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

            if n_batches % train_cfg["log_every"] == 0:
                pbar.set_postfix({
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "cls": f"{losses['cls_loss'].item():.4f}",
                    "reg": f"{losses['reg_loss'].item():.4f}",
                })

        scheduler.step()
        n_batches = max(n_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            + " | ".join(f"{k}={v/n_batches:.4f}" for k, v in epoch_losses.items())
            + f" | time={elapsed:.1f}s"
        )

        # Save best model
        avg_loss = epoch_losses["total_loss"] / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg["checkpoints"]["best_model_path"])
            print(f"  -> New best checkpoint saved (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % train_cfg["checkpoint_every"] == 0:
            ckpt_path = str(save_dir / f"pointpillars_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print("\nTraining complete.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PointPillars on KITTI")
    parser.add_argument("--config", type=str, default="configs/pointpillars.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
