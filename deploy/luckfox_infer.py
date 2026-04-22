from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd

from datasets.custom_resistance_dataset import FeatureConfig
from preprocessing.window_pipeline import WindowConfig, ZScoreStats


@dataclass
class OnlineWindowBuffer:
    window_size: int
    stride_size: int
    channels: int

    def __post_init__(self) -> None:
        self.data = np.zeros((0, self.channels), dtype=np.float32)

    def push(self, sample: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        sample = np.asarray(sample, dtype=np.float32).reshape(1, self.channels)
        self.data = np.concatenate([self.data, sample], axis=0)

        if len(self.data) < self.window_size:
            return False, None

        window = self.data[: self.window_size].copy()
        self.data = self.data[self.stride_size :]
        return True, window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX action inference with a Luckfox-style sliding buffer.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to student_model.onnx")
    parser.add_argument("--stats", type=Path, required=True, help="Path to zscore_stats.json")
    parser.add_argument("--label-map", type=Path, required=True, help="Path to label_map.json")
    parser.add_argument("--csv", type=Path, required=True, help="CSV input stream/file with IMU schema")
    parser.add_argument("--sample-rate", type=int, default=50)
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_cfg = FeatureConfig()
    window_cfg = WindowConfig(
        sample_rate_hz=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    required = {
        "sensor_ts",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "action_type",
        "subject_id",
    }

    df = pd.read_csv(args.csv)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    stats = ZScoreStats.load(args.stats)
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    index_to_label = {int(v): k for k, v in label_map.items()}

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    buffer = OnlineWindowBuffer(
        window_size=window_cfg.window_size,
        stride_size=window_cfg.stride_size,
        channels=len(feature_cfg.imu_columns),
    )

    for _, row in df.iterrows():
        sample = np.array([row[c] for c in feature_cfg.imu_columns], dtype=np.float32)
        sample = (sample - stats.mean) / stats.std

        ready, window = buffer.push(sample)
        if not ready or window is None:
            continue

        model_input = window[np.newaxis, :, :].astype(np.float32)
        logits = session.run(None, {input_name: model_input})[0]
        pred_idx = int(np.argmax(logits, axis=1)[0])
        pred_label = index_to_label.get(pred_idx, str(pred_idx))
        print(pred_label)


if __name__ == "__main__":
    main()
