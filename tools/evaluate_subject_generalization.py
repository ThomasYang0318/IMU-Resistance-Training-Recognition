from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


LABELS = [
    "db_bench_press",
    "db_biceps_curl",
    "db_rdl",
    "db_shoulder_press",
    "db_squat",
    "db_triceps_curl",
    "db_weighted_crunch",
    "one_arm_db_row",
    "Other",
]

DISPLAY = {
    "db_bench_press": "Bench",
    "db_biceps_curl": "Biceps",
    "db_rdl": "RDL",
    "db_shoulder_press": "Shoulder",
    "db_squat": "Squat",
    "db_triceps_curl": "Triceps",
    "db_weighted_crunch": "Crunch",
    "one_arm_db_row": "Row",
    "Other": "Other",
}


def infer_time_seconds(sensor_ts: pd.Series) -> np.ndarray:
    values = pd.to_numeric(sensor_ts, errors="coerce").to_numpy(dtype=np.float64)
    if len(values) == 0:
        return values
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_delta = float(np.median(diffs)) if len(diffs) else 1.0
    if median_delta > 1000.0:
        scale = 1_000_000.0
    elif median_delta > 10.0:
        scale = 1000.0
    else:
        scale = 1.0
    return np.maximum.accumulate((values - values[0]) / scale)


def raw_gyro_context(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=["sensor_ts", "gx", "gy", "gz"])
    times = infer_time_seconds(df["sensor_ts"])
    gyro = np.sqrt(
        df["gx"].to_numpy(dtype=np.float64) ** 2
        + df["gy"].to_numpy(dtype=np.float64) ** 2
        + df["gz"].to_numpy(dtype=np.float64) ** 2
    )
    return times, gyro


def peak_count(
    times: np.ndarray,
    gyro: np.ndarray,
    end_time: float,
    window_seconds: float,
    peak_z: float,
    min_peak_distance_seconds: float,
) -> int:
    start = int(np.searchsorted(times, end_time - window_seconds, side="left"))
    end = int(np.searchsorted(times, end_time, side="right"))
    values = gyro[start:end]
    local_times = times[start:end]
    if len(values) < 5:
        return 0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-9:
        mad = float(np.std(values)) or 1e-6
    threshold = median + peak_z * 1.4826 * mad
    count = 0
    last_peak = -1.0e9
    for idx in range(1, len(values) - 1):
        if (
            values[idx] >= threshold
            and values[idx] >= values[idx - 1]
            and values[idx] > values[idx + 1]
            and local_times[idx] - last_peak >= min_peak_distance_seconds
        ):
            count += 1
            last_peak = float(local_times[idx])
    return count


def majority_action(values: list[str]) -> str:
    counts = Counter(values)
    return max(LABELS[:-1], key=lambda label: (counts.get(label, 0), -LABELS.index(label)))


def apply_confirmation_gate(
    frame: pd.DataFrame,
    prediction_col: str,
    window_seconds: float,
    min_action_windows: int,
    min_action_ratio: float,
    streak_windows: int,
    min_peaks: int,
    peak_z: float,
    min_peak_distance_seconds: float,
) -> pd.Series:
    output = pd.Series("Other", index=frame.index, dtype=object)
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for file_path, group in frame.groupby("file", sort=False):
        times, gyro = raw_cache.setdefault(file_path, raw_gyro_context(file_path))
        history: deque[tuple[float, str]] = deque()
        streak = 0
        confirmed: str | None = None
        for row in group.itertuples():
            end_time = float(row.end_seconds)
            pred = str(getattr(row, prediction_col))
            if pred == "Other":
                history.clear()
                streak = 0
                confirmed = None
                output.loc[row.Index] = "Other"
                continue
            history.append((end_time, pred))
            while history and history[0][0] < end_time - window_seconds:
                history.popleft()
            labels = [label for _, label in history]
            majority = majority_action(labels)
            consistency = labels.count(majority) / float(len(labels)) if labels else 0.0
            peaks = peak_count(times, gyro, end_time, window_seconds, peak_z, min_peak_distance_seconds)
            ready = len(labels) >= min_action_windows and consistency >= min_action_ratio and peaks >= min_peaks
            if ready:
                streak += 1
                if streak >= streak_windows:
                    confirmed = majority
            else:
                streak = 0
            output.loc[row.Index] = confirmed or "Other"
    return output


def metric_rows(df: pd.DataFrame, prediction_col: str, method: str) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject, group in df.groupby("subject", sort=True):
        y_true = group["label"].astype(str).to_numpy()
        y_pred = group[prediction_col].astype(str).to_numpy()
        present = [label for label in LABELS if np.any(y_true == label)]
        cm = confusion_matrix(y_true, y_pred, labels=present)
        row_sum = cm.sum(axis=1, keepdims=True)
        prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
        recalls = {label: float(prop[present.index(label), present.index(label)]) for label in present}
        row = {
            "method": method,
            "subject": subject,
            "windows": int(len(group)),
            "present_classes": int(len(present)),
            "classes": ",".join(present),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy_present": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1_present": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "min_present_recall": float(min(recalls.values())) if recalls else 0.0,
            "other_recall": float(recalls.get("Other", np.nan)),
        }
        for label in LABELS:
            row[f"recall_{label}"] = recalls.get(label, np.nan)
        rows.append(row)

    table = pd.DataFrame(rows)
    summary = {
        "method": method,
        "subjects": int(table["subject"].nunique()),
        "mean_accuracy": float(table["accuracy"].mean()),
        "mean_balanced_accuracy_present": float(table["balanced_accuracy_present"].mean()),
        "mean_macro_f1_present": float(table["macro_f1_present"].mean()),
        "mean_other_recall": float(table["other_recall"].mean()),
        "mean_min_present_recall": float(table["min_present_recall"].mean()),
    }
    return table, summary


def plot_subject_metrics(table: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1_present", "Macro F1"),
        ("other_recall", "Other recall"),
    ]
    subjects = table["subject"].tolist()
    x = np.arange(len(subjects))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    for idx, (col, label) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, table[col].to_numpy(dtype=float), width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Subject-held-out generalization")
    ax.legend()
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-subject generalization from subject-held-out window predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/realtime_recofit_9class_other_20260519/window_predictions_9class_recofit.csv"))
    parser.add_argument("--prediction-col", default="prediction_threshold0p25_smooth3_window_vote")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/subject_generalization_20260519"))
    parser.add_argument("--gate-window-seconds", type=float, default=4.0)
    parser.add_argument("--gate-min-action-windows", type=int, default=8)
    parser.add_argument("--gate-min-action-ratio", type=float, default=0.9)
    parser.add_argument("--gate-streak-windows", type=int, default=3)
    parser.add_argument("--gate-min-peaks", type=int, default=2)
    parser.add_argument("--gate-peak-z", type=float, default=0.7)
    parser.add_argument("--gate-min-peak-distance-seconds", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    needed = ["file", "subject", "end_seconds", "label", args.prediction_col]
    df = pd.read_csv(args.predictions, usecols=needed).sort_values(["file", "end_seconds"]).reset_index(drop=True)
    df["prediction_pre_gate"] = df[args.prediction_col].astype(str)
    df["prediction_confirmed_gate"] = apply_confirmation_gate(
        df,
        prediction_col="prediction_pre_gate",
        window_seconds=args.gate_window_seconds,
        min_action_windows=args.gate_min_action_windows,
        min_action_ratio=args.gate_min_action_ratio,
        streak_windows=args.gate_streak_windows,
        min_peaks=args.gate_min_peaks,
        peak_z=args.gate_peak_z,
        min_peak_distance_seconds=args.gate_min_peak_distance_seconds,
    )
    df.to_csv(args.output_dir / "window_predictions_with_confirmed_gate.csv", index=False)

    pre_table, pre_summary = metric_rows(df, "prediction_pre_gate", "pre_gate")
    gate_table, gate_summary = metric_rows(df, "prediction_confirmed_gate", "confirmed_gate")
    all_table = pd.concat([pre_table, gate_table], ignore_index=True)
    all_table.to_csv(args.output_dir / "subject_metrics.csv", index=False, float_format="%.6f")
    plot_subject_metrics(pre_table, args.output_dir / "subject_metrics_pre_gate.png")
    plot_subject_metrics(gate_table, args.output_dir / "subject_metrics_confirmed_gate.png")
    summary = {
        "prediction_source": str(args.predictions),
        "prediction_col": args.prediction_col,
        "note": "Each subject's rows come from subject-held-out cross-validation predictions. This summarizes per-subject generalization; it is not a fresh train-on-all-other-subjects LOSO retrain.",
        "gate": {
            "window_seconds": args.gate_window_seconds,
            "min_action_windows": args.gate_min_action_windows,
            "min_action_ratio": args.gate_min_action_ratio,
            "streak_windows": args.gate_streak_windows,
            "min_peaks": args.gate_min_peaks,
            "peak_z": args.gate_peak_z,
            "min_peak_distance_seconds": args.gate_min_peak_distance_seconds,
        },
        "pre_gate": pre_summary,
        "confirmed_gate": gate_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(all_table[["method", "subject", "windows", "present_classes", "accuracy", "balanced_accuracy_present", "macro_f1_present", "other_recall", "min_present_recall"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
