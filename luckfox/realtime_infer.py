from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path
from typing import Sequence

try:
    from .features import IMU_COLUMNS, OTHER_LABEL, extract_features
    from .model import LuckFoxModel
except ImportError:  # Allows `python luckfox/realtime_infer.py ...`
    from features import IMU_COLUMNS, OTHER_LABEL, extract_features
    from model import LuckFoxModel


class RingBuffer:
    def __init__(self, keep_seconds: float) -> None:
        self.keep_seconds = float(keep_seconds)
        self.times: list[float] = []
        self.samples: list[list[float]] = []

    def push(self, timestamp: float, sample: Sequence[float]) -> None:
        self.times.append(float(timestamp))
        self.samples.append([float(value) for value in sample])
        cutoff = float(timestamp) - self.keep_seconds
        drop = 0
        while drop < len(self.times) and self.times[drop] < cutoff:
            drop += 1
        if drop:
            del self.times[:drop]
            del self.samples[:drop]


class RealtimeRecognizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model = LuckFoxModel.load(model_path)
        cfg = self.model.config
        self.scales = [float(value) for value in cfg.get("scales_seconds", [0.75, 2.0, 4.0])]
        self.stride_seconds = float(cfg.get("stride_seconds", 0.5))
        self.active_threshold = float(cfg.get("active_threshold", 0.25))
        self.class_active_thresholds = {
            str(label): float(threshold)
            for label, threshold in dict(cfg.get("class_active_thresholds", {})).items()
        }
        self.active_smooth_windows = int(cfg.get("active_smooth_windows", 3))
        self.action_smooth_windows = int(cfg.get("action_smooth_windows", 1))
        self.action_labels = [label for label in self.model.labels if label != OTHER_LABEL]
        self.confirmation_enabled = bool(cfg.get("confirmation_enabled", False))
        self.confirmation_window_seconds = float(cfg.get("confirmation_window_seconds", 4.0))
        self.confirmation_min_action_windows = int(cfg.get("confirmation_min_action_windows", 8))
        self.confirmation_min_action_ratio = float(cfg.get("confirmation_min_action_ratio", 0.9))
        self.confirmation_streak_windows = int(cfg.get("confirmation_streak_windows", 3))
        self.confirmation_min_peaks = int(cfg.get("confirmation_min_peaks", 2))
        self.confirmation_peak_z = float(cfg.get("confirmation_peak_z", 0.7))
        self.confirmation_min_peak_distance_seconds = float(cfg.get("confirmation_min_peak_distance_seconds", 0.55))
        self.mad_gate_enabled = bool(cfg.get("mad_gate_enabled", False))
        self.mad_gate_window_seconds = float(cfg.get("mad_gate_window_seconds", 2.0))
        self.mad_gate_min_acc = float(cfg.get("mad_gate_min_acc", 0.018))
        self.mad_gate_min_gyro = float(cfg.get("mad_gate_min_gyro", 1.0))
        self.mad_gate_mode = str(cfg.get("mad_gate_mode", "or")).lower()
        self.buffer = RingBuffer(
            max(max(self.scales) + 0.5, self.confirmation_window_seconds + 0.5, self.mad_gate_window_seconds + 0.5)
        )
        self.next_emit_time: float | None = None
        self.active_history: deque[bool] = deque(maxlen=self.active_smooth_windows)
        self.action_history: deque[str] = deque(maxlen=self.action_smooth_windows)
        self.confirmation_history: deque[tuple[float, str]] = deque()
        self.confirmation_streak = 0
        self.confirmed_action: str | None = None

    def majority_action(self) -> str:
        counts: dict[str, int] = {}
        for label in self.action_history:
            counts[label] = counts.get(label, 0) + 1
        return max(self.action_labels, key=lambda label: (counts.get(label, 0), -self.action_labels.index(label)))

    def motion_peak_count(self, end_time: float) -> int:
        start_time = end_time - self.confirmation_window_seconds
        pairs = [
            (timestamp, sample)
            for timestamp, sample in zip(self.buffer.times, self.buffer.samples)
            if start_time <= timestamp <= end_time
        ]
        if len(pairs) < 5:
            return 0
        magnitudes = [
            math.sqrt(sample[3] * sample[3] + sample[4] * sample[4] + sample[5] * sample[5])
            for _, sample in pairs
        ]
        sorted_mag = sorted(magnitudes)
        median = sorted_mag[len(sorted_mag) // 2]
        deviations = sorted(abs(value - median) for value in magnitudes)
        mad = deviations[len(deviations) // 2]
        if mad <= 1e-9:
            mean = sum(magnitudes) / float(len(magnitudes))
            mad = math.sqrt(sum((value - mean) * (value - mean) for value in magnitudes) / float(len(magnitudes))) or 1e-6
        threshold = median + self.confirmation_peak_z * 1.4826 * mad
        count = 0
        last_peak_time = -1.0e9
        times = [timestamp for timestamp, _ in pairs]
        for idx in range(1, len(magnitudes) - 1):
            if (
                magnitudes[idx] >= threshold
                and magnitudes[idx] >= magnitudes[idx - 1]
                and magnitudes[idx] > magnitudes[idx + 1]
                and times[idx] - last_peak_time >= self.confirmation_min_peak_distance_seconds
            ):
                count += 1
                last_peak_time = times[idx]
        return count

    def motion_mad_gate(self, end_time: float) -> tuple[bool, float, float]:
        if not self.mad_gate_enabled:
            return True, 0.0, 0.0
        start_time = end_time - self.mad_gate_window_seconds
        block = [
            sample
            for timestamp, sample in zip(self.buffer.times, self.buffer.samples)
            if start_time <= timestamp <= end_time
        ]
        if len(block) < 5:
            return False, 0.0, 0.0

        acc_mag = [
            math.sqrt(sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2])
            for sample in block
        ]
        gyro_mag = [
            math.sqrt(sample[3] * sample[3] + sample[4] * sample[4] + sample[5] * sample[5])
            for sample in block
        ]
        acc_mean = sum(acc_mag) / float(len(acc_mag))
        gyro_mean = sum(gyro_mag) / float(len(gyro_mag))
        acc_mad = sum(abs(value - acc_mean) for value in acc_mag) / float(len(acc_mag))
        gyro_mad = sum(abs(value - gyro_mean) for value in gyro_mag) / float(len(gyro_mag))

        acc_active = acc_mad >= self.mad_gate_min_acc
        gyro_active = gyro_mad >= self.mad_gate_min_gyro
        if self.mad_gate_mode == "and":
            return acc_active and gyro_active, acc_mad, gyro_mad
        return acc_active or gyro_active, acc_mad, gyro_mad

    def confirmed_output(self, end_time: float, active_smooth: bool, pre_gate_output: str) -> tuple[str, dict[str, object]]:
        if not self.confirmation_enabled:
            return pre_gate_output, {
                "pre_gate_prediction": pre_gate_output,
                "confirmation_ready": pre_gate_output != OTHER_LABEL,
                "confirmed_action": pre_gate_output if pre_gate_output != OTHER_LABEL else "",
                "action_consistency": 1.0 if pre_gate_output != OTHER_LABEL else 0.0,
                "repetition_peak_count": 0,
            }
        if not active_smooth or pre_gate_output == OTHER_LABEL:
            self.confirmation_history.clear()
            self.confirmation_streak = 0
            self.confirmed_action = None
            return OTHER_LABEL, {
                "pre_gate_prediction": pre_gate_output,
                "confirmation_ready": False,
                "confirmed_action": "",
                "action_consistency": 0.0,
                "repetition_peak_count": 0,
            }

        self.confirmation_history.append((end_time, pre_gate_output))
        while self.confirmation_history and self.confirmation_history[0][0] < end_time - self.confirmation_window_seconds:
            self.confirmation_history.popleft()

        labels = [label for _, label in self.confirmation_history]
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        majority = max(self.action_labels, key=lambda label: (counts.get(label, 0), -self.action_labels.index(label)))
        consistency = counts.get(majority, 0) / float(len(labels)) if labels else 0.0
        peak_count = self.motion_peak_count(end_time)
        ready = (
            len(labels) >= self.confirmation_min_action_windows
            and consistency >= self.confirmation_min_action_ratio
            and peak_count >= self.confirmation_min_peaks
        )
        if ready:
            self.confirmation_streak += 1
            if self.confirmation_streak >= self.confirmation_streak_windows:
                self.confirmed_action = majority
        else:
            self.confirmation_streak = 0

        output = self.confirmed_action if self.confirmed_action is not None else OTHER_LABEL
        return output, {
            "pre_gate_prediction": pre_gate_output,
            "confirmation_ready": ready,
            "confirmed_action": self.confirmed_action or "",
            "action_consistency": consistency,
            "repetition_peak_count": peak_count,
        }

    def push_sample(self, timestamp: float, sample: Sequence[float]) -> dict[str, object] | None:
        self.buffer.push(timestamp, sample)
        if self.next_emit_time is None:
            self.next_emit_time = timestamp + max(self.scales)
            return None
        if timestamp + 1e-9 < self.next_emit_time:
            return None

        end_time = self.next_emit_time
        self.next_emit_time += self.stride_seconds
        features = extract_features(self.buffer.times, self.buffer.samples, end_time, self.scales, include_posture=True)
        if features is None:
            return None

        active_prob = self.model.active_probability(features)
        action_label, action_confidence = self.model.action_prediction(features)
        threshold = self.class_active_thresholds.get(action_label, self.active_threshold)
        mad_gate_active, acc_mad, gyro_mad = self.motion_mad_gate(end_time)
        active_now = active_prob >= threshold and mad_gate_active
        self.active_history.append(active_now)
        active_smooth = sum(1 for value in self.active_history if value) >= ((len(self.active_history) + 1) // 2)

        pre_gate_output = action_label if active_smooth else OTHER_LABEL
        if active_smooth:
            self.action_history.append(action_label)
            if self.action_smooth_windows > 1:
                pre_gate_output = self.majority_action()
        else:
            self.action_history.clear()
        output, confirmation = self.confirmed_output(end_time, active_smooth, pre_gate_output)
        return {
            "time_seconds": end_time,
            "prediction": output,
            "active_probability": active_prob,
            "active_smooth": active_smooth,
            "active_threshold": threshold,
            "action_candidate": action_label,
            "action_confidence": action_confidence,
            "mad_gate_active": mad_gate_active,
            "acc_mad": acc_mad,
            "gyro_mad": gyro_mad,
            **confirmation,
        }


def parse_sample_csv_line(line: str) -> tuple[float, list[float]]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 1 + len(IMU_COLUMNS):
        raise ValueError(f"Expected timestamp plus {len(IMU_COLUMNS)} IMU values, got {len(parts)} values.")
    return float(parts[0]), [float(value) for value in parts[1:]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure-Python LuckFox realtime IMU inference.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stdin", action="store_true", help="Read lines: timestamp,ax,ay,az,gx,gy,gz,mx,my,mz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognizer = RealtimeRecognizer(args.model)
    if not args.stdin:
        raise SystemExit("Use --stdin on device, or run simulate_dataset_stream.py for dataset replay.")
    for line in iter(input, ""):
        if not line.strip():
            continue
        timestamp, sample = parse_sample_csv_line(line)
        result = recognizer.push_sample(timestamp, sample)
        if result is not None:
            print(
                f"{result['time_seconds']:.3f},"
                f"{result['prediction']},"
                f"{result['active_probability']:.4f},"
                f"{result['active_threshold']:.4f},"
                f"{result['action_candidate']},"
                f"{result['action_confidence']:.4f},"
                f"{result['pre_gate_prediction']},"
                f"{int(bool(result['confirmation_ready']))},"
                f"{result['confirmed_action']},"
                f"{result['action_consistency']:.4f},"
                f"{result['repetition_peak_count']},"
                f"{int(bool(result['mad_gate_active']))},"
                f"{result['acc_mad']:.6f},"
                f"{result['gyro_mad']:.6f}",
                flush=True,
            )
        time.sleep(0.0)


if __name__ == "__main__":
    main()
