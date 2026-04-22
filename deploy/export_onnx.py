from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from datasets.custom_resistance_dataset import FeatureConfig
from models.inertial_student import InertialStudent, ModelConfig, export_student_to_onnx
from preprocessing.window_pipeline import WindowConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained student checkpoint to ONNX.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint (default: <output_dir>/student_best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to exported .onnx (default: <output_dir>/student_model.onnx)",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_cfg = FeatureConfig(**raw.get("feature", {}))
    window_cfg = WindowConfig(**raw.get("window", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))

    output_dir = Path(raw.get("io", {}).get("output_dir", "./artifacts_student"))
    checkpoint = args.checkpoint or (output_dir / "student_best.pt")
    onnx_output = args.output or (output_dir / "student_model.onnx")

    label_map_path = output_dir / "label_map.json"
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        model_cfg.num_classes = len(label_map)

    model_cfg.input_channels = len(feature_cfg.imu_columns)
    model = InertialStudent(model_cfg, window_cfg.window_size)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    export_student_to_onnx(
        model=model,
        output_path=str(onnx_output),
        window_size=window_cfg.window_size,
        input_channels=model_cfg.input_channels,
        opset_version=args.opset,
    )

    print(f"Exported ONNX model: {onnx_output}")


if __name__ == "__main__":
    main()
