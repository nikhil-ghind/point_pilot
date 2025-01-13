"""
KITTI 3D mAP evaluation.

Implements IoU matching for Car/Pedestrian/Cyclist at thresholds:
    Car:        IoU >= 0.7
    Pedestrian: IoU >= 0.5
    Cyclist:    IoU >= 0.5

Usage:
    python -m eval.kitti_eval \
        --config configs/pointpillars.yaml \
        --checkpoint checkpoints/pointpillars_best.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def box3d_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
) -> np.ndarray:
    """
    Compute 3D axis-aligned IoU between predicted and GT boxes.
    Uses BEV 2D IoU times height overlap ratio as approximation.

    Args:
        pred_boxes: (N, 7) [x, y, z, l, w, h, yaw]
        gt_boxes:   (M, 7)

    Returns:
        iou: (N, M) float32
    """
    N = len(pred_boxes)
    M = len(gt_boxes)

    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # BEV 2D IoU (axis-aligned approximation ignoring yaw)
    def corners(boxes):
        x, y, l, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
        return x - l / 2, y - w / 2, x + l / 2, y + w / 2

    px1, py1, px2, py2 = corners(pred_boxes)  # (N,)
    gx1, gy1, gx2, gy2 = corners(gt_boxes)    # (M,)

    # Broadcast
    ix1 = np.maximum(px1[:, None], gx1[None, :])  # (N, M)
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])

    inter_bev = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)

    pa = (px2 - px1) * (py2 - py1)
    ga = (gx2 - gx1) * (gy2 - gy1)
    union_bev = pa[:, None] + ga[None, :] - inter_bev

    # Height overlap
    pz1 = pred_boxes[:, 2] - pred_boxes[:, 5] / 2
    pz2 = pred_boxes[:, 2] + pred_boxes[:, 5] / 2
    gz1 = gt_boxes[:, 2] - gt_boxes[:, 5] / 2
    gz2 = gt_boxes[:, 2] + gt_boxes[:, 5] / 2

    hz1 = np.maximum(pz1[:, None], gz1[None, :])  # (N, M)
    hz2 = np.minimum(pz2[:, None], gz2[None, :])
    h_inter = np.maximum(0, hz2 - hz1)

    ph = pred_boxes[:, 5]
    gh = gt_boxes[:, 5]
    h_union = ph[:, None] + gh[None, :] - h_inter

    iou_3d = (inter_bev * h_inter) / np.maximum(union_bev * h_union, 1e-6)
    return iou_3d.astype(np.float32)


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    n_points: int = 11,
) -> float:
    """
    Compute Average Precision using n-point interpolation.

    Args:
        recalls: sorted recall values
        precisions: corresponding precision values
        n_points: number of recall thresholds (11 for PASCAL VOC)

    Returns:
        ap: average precision
    """
    if len(recalls) == 0:
        return 0.0

    recall_thresholds = np.linspace(0, 1, n_points)
    ap = 0.0

    for thr in recall_thresholds:
        prec_at_thr = precisions[recalls >= thr]
        if len(prec_at_thr) == 0:
            ap += 0.0
        else:
            ap += prec_at_thr.max()

    return ap / n_points


def evaluate_class(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    iou_threshold: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP for a single class.

    Args:
        all_pred_boxes: list of (N_i, 7) predicted boxes per frame
        all_pred_scores: list of (N_i,) confidence scores per frame
        all_gt_boxes: list of (M_i, 7) GT boxes per frame
        iou_threshold: IoU threshold for matching

    Returns:
        ap: average precision
        recalls: recall curve
        precisions: precision curve
    """
    # Collect all predictions with frame index
    all_preds = []
    for frame_id, (boxes, scores) in enumerate(zip(all_pred_boxes, all_pred_scores)):
        for box, score in zip(boxes, scores):
            all_preds.append((score, frame_id, box))

    # Sort by descending confidence
    all_preds.sort(key=lambda x: -x[0])

    n_gt_total = sum(len(gt) for gt in all_gt_boxes)
    if n_gt_total == 0:
        return 0.0, np.array([]), np.array([])

    # Track matched GT boxes
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in all_gt_boxes]

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, (score, frame_id, pred_box) in enumerate(all_preds):
        gt_boxes = all_gt_boxes[frame_id]

        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        # Compute IoU with all GT boxes in this frame
        pred_box_np = pred_box[np.newaxis, :]  # (1, 7)
        iou = box3d_iou(pred_box_np, gt_boxes)[0]  # (M,)

        # Find best matching unmatched GT
        iou[gt_matched[frame_id]] = -1  # mask already matched

        if iou.max() >= iou_threshold:
            best_match = iou.argmax()
            tp[i] = 1
            gt_matched[frame_id][best_match] = True
        else:
            fp[i] = 1

    # Cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / n_gt_total
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)

    ap = compute_ap(recalls, precisions)
    return ap, recalls, precisions


def decode_predictions(
    cls_preds: torch.Tensor,
    box_preds: torch.Tensor,
    dir_preds: Optional[torch.Tensor],
    anchors: torch.Tensor,
    score_threshold: float = 0.4,
    nms_iou_threshold: float = 0.5,
    num_classes: int = 3,
    num_anchors_per_loc: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode raw model predictions to boxes, scores, and labels.

    Returns:
        boxes: (K, 7) decoded box predictions
        scores: (K,) confidence scores
        class_ids: (K,) class ids
    """
    device = cls_preds.device
    B = cls_preds.shape[0]
    A = num_anchors_per_loc
    C = num_classes

    # Flatten
    H, W = cls_preds.shape[2:]
    HWA = H * W * A

    cls_flat = cls_preds.permute(0, 2, 3, 1).contiguous().view(B, HWA, C)
    box_flat = box_preds.permute(0, 2, 3, 1).contiguous().view(B, HWA, 7)

    # Use single-sample (batch index 0)
    scores_all = torch.sigmoid(cls_flat[0])  # (HWA, C)
    box_enc = box_flat[0]                    # (HWA, 7)

    # Max score and class per anchor
    max_scores, class_ids = scores_all.max(dim=1)  # (HWA,)

    # Filter by score threshold
    keep = max_scores >= score_threshold
    if not keep.any():
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=int)

    filtered_scores = max_scores[keep].cpu().numpy()
    filtered_class_ids = class_ids[keep].cpu().numpy()
    filtered_enc = box_enc[keep]
    filtered_anchors = anchors[:HWA][keep]

    # Decode box encodings → actual boxes
    d_a = torch.sqrt(filtered_anchors[:, 3] ** 2 + filtered_anchors[:, 4] ** 2)
    x = filtered_enc[:, 0] * d_a + filtered_anchors[:, 0]
    y = filtered_enc[:, 1] * d_a + filtered_anchors[:, 1]
    z = filtered_enc[:, 2] * filtered_anchors[:, 5] + filtered_anchors[:, 2]
    l = torch.exp(filtered_enc[:, 3]) * filtered_anchors[:, 3]
    w = torch.exp(filtered_enc[:, 4]) * filtered_anchors[:, 4]
    h = torch.exp(filtered_enc[:, 5]) * filtered_anchors[:, 5]
    r = filtered_enc[:, 6] + filtered_anchors[:, 6]

    boxes = torch.stack([x, y, z, l, w, h, r], dim=1).cpu().numpy()

    # Simple greedy NMS (per class)
    final_boxes = []
    final_scores = []
    final_class_ids = []

    for cls_id in range(C):
        mask = filtered_class_ids == cls_id
        if not mask.any():
            continue

        cls_boxes = boxes[mask]
        cls_scores = filtered_scores[mask]

        # Sort by score
        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        keep_nms = []
        suppressed = np.zeros(len(cls_boxes), dtype=bool)

        for i in range(len(cls_boxes)):
            if suppressed[i]:
                continue
            keep_nms.append(i)
            iou = box3d_iou(cls_boxes[i:i+1], cls_boxes[i+1:])[0]
            suppressed[i+1:][iou > nms_iou_threshold] = True

        final_boxes.extend(cls_boxes[keep_nms].tolist())
        final_scores.extend(cls_scores[keep_nms].tolist())
        final_class_ids.extend([cls_id] * len(keep_nms))

    if not final_boxes:
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=int)

    return (
        np.array(final_boxes, dtype=np.float32),
        np.array(final_scores, dtype=np.float32),
        np.array(final_class_ids, dtype=int),
    )


def evaluate(config_path: str, checkpoint_path: str):
    """Run KITTI evaluation and print per-class AP and mAP."""
    from data.kitti_dataset import create_kitti_dataloaders
    from data.voxelization import Voxelizer
    from models.pointpillars import PointPillars
    from scripts.train import voxelize_batch

    cfg = load_config(config_path)
    vox_cfg = cfg["voxelization"]
    model_cfg = cfg["model"]
    device = "cpu"

    voxelizer = Voxelizer(
        voxel_size=vox_cfg["voxel_size"],
        point_cloud_range=vox_cfg["point_cloud_range"],
        max_points_per_voxel=vox_cfg["max_points_per_voxel"],
        max_voxels=vox_cfg["max_voxels"],
    )

    _, val_loader = create_kitti_dataloaders(
        root_dir=cfg["data"]["root_dir"],
        batch_size=1,
        num_workers=0,
        point_cloud_range=vox_cfg["point_cloud_range"],
    )

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
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

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

    # Accumulate predictions and GT
    all_pred_boxes = {cls: [] for cls in CLASS_NAMES}
    all_pred_scores = {cls: [] for cls in CLASS_NAMES}
    all_gt_boxes = {cls: [] for cls in CLASS_NAMES}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            points_list = batch["points"]
            gt_boxes_list = batch["gt_boxes"]
            gt_labels_list = batch["labels"]

            voxels, coords, num_pts = voxelize_batch(points_list, voxelizer, device)
            if len(voxels) == 0:
                for cls in CLASS_NAMES:
                    all_pred_boxes[cls].append(np.zeros((0, 7)))
                    all_pred_scores[cls].append(np.zeros(0))
                    all_gt_boxes[cls].append(np.zeros((0, 7)))
                continue

            preds = model(voxels, coords, num_pts, batch_size=1)
            pred_boxes, pred_scores, pred_class_ids = decode_predictions(
                preds["cls_preds"], preds["box_preds"],
                preds.get("dir_preds"), anchors,
            )

            # Separate by class
            for cls_id, cls_name in enumerate(CLASS_NAMES):
                cls_mask = pred_class_ids == cls_id
                all_pred_boxes[cls_name].append(pred_boxes[cls_mask] if len(pred_boxes) else np.zeros((0, 7)))
                all_pred_scores[cls_name].append(pred_scores[cls_mask] if len(pred_scores) else np.zeros(0))

            # GT boxes
            gt_boxes = gt_boxes_list[0].numpy()
            gt_labels = gt_labels_list[0].numpy()
            for cls_id, cls_name in enumerate(CLASS_NAMES):
                cls_gt = gt_boxes[gt_labels == cls_id]
                all_gt_boxes[cls_name].append(cls_gt)

    # Compute AP per class
    print("\n" + "=" * 50)
    print("KITTI 3D Evaluation Results")
    print("=" * 50)
    ap_scores = {}
    for cls_name in CLASS_NAMES:
        iou_thr = IOU_THRESHOLDS[cls_name]
        ap, recalls, precisions = evaluate_class(
            all_pred_boxes[cls_name],
            all_pred_scores[cls_name],
            all_gt_boxes[cls_name],
            iou_threshold=iou_thr,
        )
        ap_scores[cls_name] = ap
        print(f"  {cls_name:<12} (IoU={iou_thr:.1f}): AP = {ap*100:.2f}%")

    map_score = np.mean(list(ap_scores.values()))
    print(f"\n  mAP: {map_score*100:.2f}%")
    print("=" * 50)

    return ap_scores, map_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointPillars on KITTI")
    parser.add_argument("--config", type=str, default="configs/pointpillars.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/pointpillars_best.pt")
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
