# Point Pilot

## Project Overview
Fine-tuned a PointPillars-based 3D object detection model on the KITTI dataset for real-time vehicle and pedestrian detection from LiDAR point clouds. Target: 78% mAP with a C++ inference pipeline for deployment.

## Tech Stack
- **Training:** PyTorch 2.x, OpenPCDet framework
- **Dataset:** KITTI 3D Object Detection
- **Model:** PointPillars (voxel-based)
- **Inference:** C++ with LibTorch or ONNX Runtime
- **Visualization:** Open3D, Matplotlib
- **Experiment Tracking:** Weights & Biases or MLflow

## Architecture Overview
```
┌──────────────┐    ┌───────────────┐    ┌───────────────┐
│ KITTI Dataset│ ─► │ PointPillars  │ ─► │ 3D BBox       │
│ (point cloud)│    │ Training      │    │ Predictions   │
└──────────────┘    │ (PyTorch)     │    └───────────────┘
                    └───────┬───────┘
                            │ export
                    ┌───────▼───────┐
                    │ C++ Inference │
                    │ Pipeline      │
                    └───────────────┘
```

## Phase 1: Dataset Setup & Preprocessing
**Goal:** Download KITTI dataset and implement preprocessing pipeline.

### Tasks
1. Project structure:
   ```
   lidar3dObjectDetection/
   ├── data/
   │   ├── download_kitti.sh
   │   └── kitti/              # symlink to dataset
   ├── src/
   │   ├── config/
   │   │   └── pointpillars.yaml  # Model config
   │   ├── data/
   │   │   ├── kitti_dataset.py   # Dataset class
   │   │   ├── preprocessing.py   # Point cloud preprocessing
   │   │   └── augmentation.py    # Data augmentation
   │   ├── models/
   │   │   ├── pointpillars.py    # PointPillars model
   │   │   ├── pillar_encoder.py  # Pillar Feature Net
   │   │   ├── backbone.py        # 2D CNN backbone
   │   │   └── detection_head.py  # SSD-style detection head
   │   ├── training/
   │   │   ├── trainer.py
   │   │   └── loss.py            # Focal loss + smooth L1
   │   ├── evaluation/
   │   │   ├── evaluator.py       # mAP calculation
   │   │   └── visualize.py       # 3D bbox visualization
   │   └── inference/
   │       └── export_onnx.py
   ├── cpp_inference/
   │   ├── CMakeLists.txt
   │   ├── src/
   │   │   ├── main.cpp
   │   │   ├── pointpillars_inference.cpp
   │   │   └── voxelization.cpp
   │   └── include/
   ├── scripts/
   │   ├── train.py
   │   └── evaluate.py
   └── requirements.txt
   ```
2. `data/download_kitti.sh` — download KITTI velodyne, labels, calibration files.
3. `src/data/kitti_dataset.py`:
   - `class KITTIDataset(torch.utils.data.Dataset)`:
     - `__init__(root_dir, split)` — loads file lists for train/val
     - `__getitem__` — loads `.bin` point cloud, parses label file, applies calibration
     - Returns: `{"points": (N, 4), "gt_boxes": (M, 7), "gt_labels": (M,)}`
   - 7-DOF box format: `(x, y, z, w, l, h, heading)`
4. `src/data/preprocessing.py`:
   - `voxelize(points, voxel_size, point_range, max_points_per_voxel, max_voxels)`:
     - Divides point cloud into pillars (x-y grid cells)
     - Returns: `pillars (P, N, D)`, `coords (P, 3)`, `num_points_per_pillar (P,)`
   - `filter_point_range(points, range)` — crop to detection region
5. `src/data/augmentation.py`:
   - `random_flip(points, boxes)` — flip along x-axis
   - `global_rotation(points, boxes, angle_range)` — random yaw rotation
   - `global_scaling(points, boxes, scale_range)` — random uniform scale
   - `gt_sampling(points, boxes, db)` — paste ground-truth boxes from other scenes

## Phase 2: PointPillars Model Implementation
**Goal:** Implement the PointPillars architecture in PyTorch.

### Tasks
1. `src/models/pillar_encoder.py`:
   - `class PillarFeatureNet(nn.Module)`:
     - Input: pillars `(B, P, N, D)` where D = (x, y, z, intensity, xc, yc, zc, xp, yp) = 9
     - Linear layer → BN → ReLU → max pool over points dimension
     - Output: `(B, C, H, W)` pseudo-image (scatter pillar features to 2D grid)
2. `src/models/backbone.py`:
   - `class Backbone2D(nn.Module)`:
     - Multi-scale 2D CNN: 3 blocks of Conv2D → BN → ReLU with stride-2 downsampling
     - Upsample and concatenate multi-scale features
     - Output: `(B, 6C, H/2, W/2)` feature map
3. `src/models/detection_head.py`:
   - `class SSDHead(nn.Module)`:
     - Per-anchor predictions: classification (num_classes), box regression (7), direction classification (2)
     - Anchor generation: predefined sizes for Car, Pedestrian, Cyclist at multiple orientations
     - `forward(features) -> cls_preds, box_preds, dir_preds`
4. `src/models/pointpillars.py`:
   - `class PointPillars(nn.Module)`:
     - Composes: PillarFeatureNet → Backbone2D → SSDHead
     - `forward(pillars, coords, num_points)` → predictions
     - `post_process(preds, score_thresh, nms_thresh)` → final 3D bboxes
5. `src/config/pointpillars.yaml`:
   - Voxel size: [0.16, 0.16, 4.0]
   - Point range: [0, -39.68, -3, 69.12, 39.68, 1]
   - Max pillars: 12000, max points per pillar: 32
   - Anchor sizes: Car [3.9, 1.6, 1.56], Pedestrian [0.8, 0.6, 1.73]

## Phase 3: Training Pipeline
**Goal:** Train the model on KITTI with proper loss functions and logging.

### Tasks
1. `src/training/loss.py`:
   - `class PointPillarsLoss`:
     - `focal_loss(cls_preds, cls_targets)` — handles class imbalance
     - `smooth_l1_loss(box_preds, box_targets, mask)` — regression loss
     - `direction_loss(dir_preds, dir_targets)` — binary cross-entropy for heading
     - `total_loss = cls_weight * focal + reg_weight * smooth_l1 + dir_weight * direction`
2. `src/training/trainer.py`:
   - `class Trainer`:
     - `train(model, train_loader, val_loader, epochs, lr, scheduler)`:
       - Adam optimizer, OneCycleLR scheduler
       - Train loop: forward → loss → backward → step
       - Validation every N epochs: compute mAP
       - Save best checkpoint by mAP
       - Log to W&B: loss curves, mAP, learning rate
3. `scripts/train.py`:
   - CLI: `python scripts/train.py --config src/config/pointpillars.yaml --epochs 80 --batch_size 4 --lr 0.003`
   - Mixed precision training with `torch.cuda.amp`
4. Target training metrics:
   - Car: 78% mAP@0.7 IoU
   - Pedestrian: 55% mAP@0.5 IoU
   - Cyclist: 60% mAP@0.5 IoU

## Phase 4: Evaluation & Visualization
**Goal:** Evaluate model performance and visualize detections.

### Tasks
1. `src/evaluation/evaluator.py`:
   - `evaluate_kitti(model, val_loader) -> dict`:
     - Computes mAP at IoU thresholds [0.5, 0.7] for each class
     - Computes mAP for Easy/Moderate/Hard difficulty levels
     - Returns per-class and overall metrics
   - 3D IoU computation for oriented bounding boxes
2. `src/evaluation/visualize.py`:
   - `visualize_detection(points, gt_boxes, pred_boxes, save_path)`:
     - Open3D point cloud rendering
     - Ground truth boxes in green, predictions in red
     - Bird's eye view projection with Matplotlib
   - `visualize_bev(points, boxes, save_path)` — top-down view
3. `scripts/evaluate.py`:
   - Load best checkpoint, run evaluation on val set
   - Generate visualization for 50 sample scenes
   - Output metrics table and save visualizations

## Phase 5: C++ Inference Pipeline
**Goal:** Export model and build C++ inference for deployment.

### Tasks
1. `src/inference/export_onnx.py`:
   - Export trained model to ONNX with dynamic batch
   - Verify ONNX output matches PyTorch output within tolerance
2. `cpp_inference/src/voxelization.cpp`:
   - C++ implementation of point cloud → pillar voxelization
   - Optimized with OpenMP for parallel point assignment
3. `cpp_inference/src/pointpillars_inference.cpp`:
   - `class PointPillarsInference`:
     - `PointPillarsInference(const std::string& onnx_model_path)` — loads ONNX Runtime session
     - `std::vector<Detection3D> detect(const float* points, size_t num_points)`:
       1. Voxelize point cloud
       2. Run ONNX inference
       3. Post-process: decode boxes, apply NMS
       4. Return detections: `{x, y, z, w, l, h, heading, class_id, score}`
     - Target: < 50ms per frame on desktop GPU
4. `cpp_inference/src/main.cpp`:
   - CLI: `./detect --model model.onnx --input scene.bin --output detections.json`
   - Batch mode: process directory of `.bin` files
5. CMakeLists: link ONNX Runtime, OpenMP. Build with `cmake .. && make`.
