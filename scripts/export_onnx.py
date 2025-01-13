"""
Export trained PointPillars to ONNX format for C++ inference.

Usage:
    python -m scripts.export_onnx \
        --config configs/pointpillars.yaml \
        --checkpoint checkpoints/pointpillars_best.pt \
        --output pointpillars.onnx
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pointpillars import PointPillars
from data.voxelization import Voxelizer


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class PointPillarsONNXWrapper(torch.nn.Module):
    """
    Wrapper around PointPillars that accepts numpy-compatible inputs
    for clean ONNX export.

    The ONNX model takes:
        voxels:     (V, P, C) float32
        coords:     (V, 4) int64
        num_points: (V,) int64
        batch_size: scalar (fixed at export time or dynamic)

    And outputs:
        cls_preds: (B, A*C, H, W)
        box_preds: (B, A*7, H, W)
        dir_preds: (B, A*2, H, W)
    """

    def __init__(self, model: PointPillars, batch_size: int = 1):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def forward(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor,
        num_points: torch.Tensor,
    ) -> tuple:
        preds = self.model(voxels, coords, num_points, batch_size=self.batch_size)
        return (
            preds["cls_preds"],
            preds["box_preds"],
            preds.get("dir_preds", torch.zeros(1)),
        )


def export_onnx(
    config_path: str,
    checkpoint_path: str,
    output_path: str = "pointpillars.onnx",
    batch_size: int = 1,
    opset_version: int = 11,
):
    """
    Export PointPillars model to ONNX.

    Args:
        config_path: Path to YAML config
        checkpoint_path: Path to .pt checkpoint
        output_path: Output ONNX file path
        batch_size: Batch size for dummy input
        opset_version: ONNX opset version
    """
    cfg = load_config(config_path)
    vox_cfg = cfg["voxelization"]
    model_cfg = cfg["model"]

    # Build model
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
    )

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy inputs
    V = 5000   # number of pillars (dummy)
    P = vox_cfg["max_points_per_voxel"]
    C = model_cfg["in_channels"]

    dummy_voxels = torch.randn(V, P, C)
    dummy_coords = torch.randint(0, 100, (V, 4)).long()
    dummy_coords[:, 0] = 0  # batch index = 0 for single-batch export
    dummy_num_pts = torch.randint(1, P + 1, (V,)).long()

    # Wrap for export
    wrapper = PointPillarsONNXWrapper(model, batch_size=batch_size)
    wrapper.eval()

    print(f"Exporting to ONNX (opset {opset_version})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_voxels, dummy_coords, dummy_num_pts),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["voxels", "coords", "num_points"],
            output_names=["cls_preds", "box_preds", "dir_preds"],
            dynamic_axes={
                "voxels": {0: "num_voxels"},
                "coords": {0: "num_voxels"},
                "num_points": {0: "num_voxels"},
                "cls_preds": {0: "batch_size"},
                "box_preds": {0: "batch_size"},
                "dir_preds": {0: "batch_size"},
            },
        )

    print(f"ONNX model saved to {output_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")

        # Print input/output shapes
        print("\nModel inputs:")
        for inp in onnx_model.graph.input:
            print(f"  {inp.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}")
        print("Model outputs:")
        for out in onnx_model.graph.output:
            print(f"  {out.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]}")

    except ImportError:
        print("onnx package not installed, skipping verification")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    # Test with onnxruntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        inputs = {
            "voxels": dummy_voxels.numpy(),
            "coords": dummy_coords.numpy().astype(np.int64),
            "num_points": dummy_num_pts.numpy().astype(np.int64),
        }
        outputs = sess.run(None, inputs)
        print(f"\nONNXRuntime test passed.")
        print(f"  cls_preds shape: {outputs[0].shape}")
        print(f"  box_preds shape: {outputs[1].shape}")
        print(f"  dir_preds shape: {outputs[2].shape}")

    except ImportError:
        print("onnxruntime not installed, skipping runtime test")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export PointPillars to ONNX")
    parser.add_argument("--config", type=str, default="configs/pointpillars.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/pointpillars_best.pt")
    parser.add_argument("--output", type=str, default="pointpillars.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    export_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
