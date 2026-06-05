from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from features import (  # noqa: E402
    IMU_COLUMNS,
    OTHER_LABEL,
    active_labels_from_rows,
    endpoint_label,
    infer_time_seconds,
    safe_float,
)
from realtime_infer import RealtimeRecognizer  # noqa: E402


DISPLAY_LABELS = {
    "db_bench_press": "Bench",
    "db_biceps_curl": "Biceps",
    "db_rdl": "RDL",
    "db_shoulder_press": "Shoulder",
    "db_squat": "Squat",
    "db_triceps_curl": "Triceps",
    "db_weighted_crunch": "Crunch",
    "one_arm_db_row": "Row",
    OTHER_LABEL: "Other",
}


def read_dataset_csv(path: Path) -> tuple[list[float], list[list[float]], list[str], list[str]]:
    raw_times: list[float] = []
    samples: list[list[float]] = []
    actions: list[str] = []
    phases: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_times.append(safe_float(row.get("sensor_ts"), float(len(raw_times)) / 100.0))
            samples.append([safe_float(row.get(col)) for col in IMU_COLUMNS])
            actions.append(str(row.get("action_type") or ""))
            phases.append(str(row.get("phase") or ""))
    return infer_time_seconds(raw_times), samples, actions, phases


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        if true in index and pred in index:
            matrix[index[true]][index[pred]] += 1
    return matrix


def metrics_from_confusion(matrix: list[list[int]], labels: list[str]) -> dict[str, object]:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[idx][idx] for idx in range(len(labels)))
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for idx, label in enumerate(labels):
        tp = matrix[idx][idx]
        row_sum = sum(matrix[idx])
        col_sum = sum(matrix[row][idx] for row in range(len(labels)))
        recall = tp / row_sum if row_sum else 0.0
        precision = tp / col_sum if col_sum else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls[label] = recall
        f1s.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls.values()) / len(labels) if labels else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class_recall": recalls,
    }


def write_confusion_csv(path: Path, matrix: list[list[int]], labels: list[str], normalized: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *[DISPLAY_LABELS.get(label, label) for label in labels]])
        for label, row in zip(labels, matrix):
            if normalized:
                total = sum(row)
                values = [f"{(value / total if total else 0.0):.6f}" for value in row]
            else:
                values = [str(value) for value in row]
            writer.writerow([DISPLAY_LABELS.get(label, label), *values])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a dataset CSV as realtime input into the pure-Python LuckFox recognizer.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("luckfox/artifacts/stream_test"))
    parser.add_argument("--realtime", action="store_true", help="Sleep according to sample timestamps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recognizer = RealtimeRecognizer(args.model)
    model_labels = [label for label in recognizer.model.labels]
    exercise_labels = [label for label in model_labels if label != OTHER_LABEL]
    endpoint_seconds = float(recognizer.model.config.get("endpoint_seconds", 0.5))
    endpoint_min_active_fraction = float(recognizer.model.config.get("endpoint_min_active_fraction", 0.25))

    times, samples, actions, phases = read_dataset_csv(args.csv)
    labels, active = active_labels_from_rows(actions, phases, exercise_labels)
    rows: list[dict[str, object]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    start_wall = time.perf_counter()
    previous_time = times[0] if times else 0.0
    inference_seconds = 0.0

    for timestamp, sample in zip(times, samples):
        if args.realtime:
            delay = max(0.0, timestamp - previous_time)
            if delay > 0:
                time.sleep(delay)
            previous_time = timestamp
        infer_start = time.perf_counter()
        result = recognizer.push_sample(timestamp, sample)
        inference_seconds += time.perf_counter() - infer_start
        if result is None:
            continue
        end_time = float(result["time_seconds"])
        true_label, active_fraction = endpoint_label(
            times,
            labels,
            active,
            end_time,
            endpoint_seconds,
            exercise_labels,
            endpoint_min_active_fraction,
        )
        pred = str(result["prediction"])
        y_true.append(true_label)
        y_pred.append(pred)
        rows.append(
            {
                "time_seconds": f"{end_time:.3f}",
                "true_label": true_label,
                "prediction": pred,
                "active_probability": f"{float(result['active_probability']):.6f}",
                "active_threshold": f"{float(result['active_threshold']):.6f}",
                "action_candidate": result["action_candidate"],
                "action_confidence": f"{float(result['action_confidence']):.6f}",
                "mad_gate_active": int(bool(result.get("mad_gate_active", True))),
                "acc_mad": f"{float(result.get('acc_mad', 0.0)):.6f}",
                "gyro_mad": f"{float(result.get('gyro_mad', 0.0)):.6f}",
                "pre_gate_prediction": result.get("pre_gate_prediction", ""),
                "confirmation_ready": int(bool(result.get("confirmation_ready", False))),
                "confirmed_action": result.get("confirmed_action", ""),
                "action_consistency": f"{float(result.get('action_consistency', 0.0)):.6f}",
                "repetition_peak_count": int(result.get("repetition_peak_count", 0)),
                "endpoint_active_fraction": f"{active_fraction:.6f}",
            }
        )

    prediction_path = args.output_dir / "stream_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["time_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    matrix = confusion_matrix(y_true, y_pred, model_labels)
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", matrix, model_labels, normalized=False)
    write_confusion_csv(args.output_dir / "confusion_matrix_row_proportion.csv", matrix, model_labels, normalized=True)
    summary = {
        "csv": str(args.csv),
        "model": str(args.model),
        "stream_seconds": float(times[-1] - times[0]) if times else 0.0,
        "samples": len(samples),
        "emitted_predictions": len(rows),
        "wall_seconds": float(time.perf_counter() - start_wall),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_sample": float((inference_seconds / max(1, len(samples))) * 1000.0),
        "inference_ms_per_emitted_prediction": float((inference_seconds / max(1, len(rows))) * 1000.0),
        **metrics_from_confusion(matrix, model_labels),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
