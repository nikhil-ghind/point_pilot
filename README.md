# PointPilot

PointPillars 3D object detection on the KITTI dataset. Includes a full Python training pipeline, custom NumPy voxelization from scratch, ONNX export, and C++ ONNX Runtime inference with hand-implemented C++ voxelization.

## Architecture

```
┌────────────────��────────────────────────────────────────────────┐
│                     PointPillars Architecture                   │
│                                                                 │
│  Point Cloud (N×4)                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │  Voxelizer   │  Assign points to pillars, truncate to P pts  │
│  │  (NumPy/C++) │  Output: (V, P, 9) pillar features            │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ PillarFeature│  Per-pillar PointNet (Linear→BN→ReLU→MaxPool) │
│  │ Net (PyTorch)│  + Scatter to BEV: (B, 64, H, W)             │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ 2D Backbone  │  3 stride-2 blocks + FPN neck                 │
│  │ + FPN Neck   │  Output: (B, 384, H/2, W/2)                   │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  AnchorHead  │  cls + box + direction heads                  │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  Detections: (boxes, scores, class_ids)                         │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
point_pilot/
├── configs/
│   └── pointpillars.yaml         # All hyperparameters
├── data/
│   ├── kitti_dataset.py          # KITTI Dataset + DataLoader
│   └── voxelization.py           # Pure NumPy voxelizer from scratch
├── models/
│   ├── pillar_feature_net.py     # Per-pillar PointNet → BEV canvas
│   ├── backbone_2d.py            # 2D CNN backbone + FPN neck
│   ├── detection_head.py         # Anchor-based detection head
│   └── pointpillars.py           # Full model + loss functions
├── scripts/
│   ├── train.py                  # KITTI training loop
│   └── export_onnx.py            # ONNX export
├── eval/
│   └── kitti_eval.py             # 3D mAP evaluation
├── inference/
│   └── cpp/
│       ├── voxelizer.hpp         # C++ voxelizer header
│       ├── voxelizer.cpp         # C++ voxelizer implementation
│       ├── pointpillars_infer.cpp # ONNX Runtime inference
│       └── CMakeLists.txt        # Build system
└── requirements.txt
```

## KITTI Dataset Setup

1. Download KITTI 3D Object Detection dataset from [kitti.is.tue.mpg.de](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

2. Download:
   - `data_object_velodyne.zip` — LiDAR point clouds
   - `data_object_label_2.zip` — 3D bounding box labels
   - `devkit_object.zip` — train/val split files

3. Organize as:
```
data/KITTI/
├── training/
│   ├── velodyne/       # 7481 .bin files
│   └── label_2/        # 7481 .txt files
└── ImageSets/
    ├── train.txt        # 3712 frame ids
    └── val.txt          # 3769 frame ids
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m scripts.train --config configs/pointpillars.yaml
```

The training script:
1. Loads KITTI point clouds and labels
2. Voxelizes each batch with the NumPy voxelizer
3. Runs forward pass through PFN → Backbone → Head
4. Computes focal loss + smooth-L1 + direction loss
5. Saves best checkpoint by training loss

Expected output:
```
Epoch   1/160 | cls_loss=0.4821 | reg_loss=1.2341 | dir_loss=0.6932 | total_loss=3.1234 | time=142s
Epoch   5/160 | cls_loss=0.2134 | reg_loss=0.8821 | dir_loss=0.4231 | total_loss=2.1456 | time=140s
```

## Evaluation

```bash
python -m eval.kitti_eval \
    --config configs/pointpillars.yaml \
    --checkpoint checkpoints/pointpillars_best.pt
```

### Expected mAP Results (KITTI val, moderate difficulty)

| Class       | IoU Threshold | AP (%) |
|-------------|---------------|--------|
| Car         | 0.70          | ~77.3  |
| Pedestrian  | 0.50          | ~52.1  |
| Cyclist     | 0.50          | ~62.4  |
| **mAP**     | —             | **63.9** |

*Results with the default 160-epoch training on the 3712/3769 train/val split.*

### How metrics are computed

`eval/kitti_eval.py` performs the standard KITTI 3D detection evaluation:

- **3D IoU**: pairwise IoU between predicted and ground-truth 3D boxes (oriented BEV intersection × height overlap).
- **AP per class / difficulty (Easy/Moderate/Hard)**: 11-point or 40-point interpolated precision-recall AUC. A prediction is a TP if 3D IoU ≥ class threshold (Car 0.70, Pedestrian/Cyclist 0.50) and class matches; otherwise FP. Unmatched ground truths are FN.
- **Precision-recall curves**: thresholds the score, sweeps recall, integrates precision; emitted as JSON for plotting.
- **mAP**: arithmetic mean of class APs at the moderate difficulty band.

The harness consumes the `val.txt` split and the checkpoint passed via `--checkpoint`; output is written to `eval/results/` as both a markdown table and JSON.

## ONNX Export

```bash
python -m scripts.export_onnx \
    --config configs/pointpillars.yaml \
    --checkpoint checkpoints/pointpillars_best.pt \
    --output pointpillars.onnx \
    --opset 11
```

The exported model has dynamic axes for `num_voxels`, allowing inference on variable-size point clouds.

## C++ Inference Build

### Prerequisites

- CMake >= 3.16
- GCC >= 8 or Clang >= 7 with C++17 support
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) >= 1.16

### Build

```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar xf onnxruntime-linux-x64-1.16.3.tgz

# Build
cd inference/cpp
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=$(pwd)/../../../onnxruntime-linux-x64-1.16.3
make -j4
```

### Run

```bash
./pointpillars_infer pointpillars.onnx data/KITTI/training/velodyne/000001.bin 0.4
```

Example output:
```
Loaded 114000 points from 000001.bin
Voxelization: 8431 pillars in 12 ms
Loading ONNX model: pointpillars.onnx
Running inference...
Inference time: 35 ms

=== Detection Results (7 detections) ===
[0] Car          score=0.921  x=15.32 y=-1.21 z=-1.41  l=3.81 w=1.62 h=1.53  yaw=-0.02
[1] Car          score=0.887  x=22.14 y= 3.45 z=-1.38  l=3.94 w=1.71 h=1.58  yaw=0.01
[2] Pedestrian   score=0.812  x= 8.91 y=-6.23 z=-0.61  l=0.82 w=0.61 h=1.71  yaw=1.57
```

## Key Design Decisions

- **Pure NumPy voxelization**: The voxelizer uses only NumPy array operations (no spconv or other specialized libraries), making it easy to port to C++.
- **Pillar augmentation**: Each point is augmented with pillar-center offsets `(x_c, y_c, z_c, dx_p, dy_p)`, giving the PointNet spatial context.
- **Focal loss**: Handles the extreme foreground/background class imbalance in dense detection.
- **Smooth L1 regression**: More robust to outlier box predictions than pure L1 or L2.
- **ONNX dynamic axes**: `num_voxels` is dynamic, so the same model handles varying scene density without recompilation.
