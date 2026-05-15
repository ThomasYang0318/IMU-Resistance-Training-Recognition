from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

from evaluate_active_set_detection import (
    IMU_COLUMNS,
    contiguous_intervals,
    estimate_sample_rate_hz,
    ground_truth_mask,
    intervals_to_mask,
    match_intervals,
    merge_intervals,
    sample_metrics,
    subject_from_path,
    whole_session_files,
)


@dataclass
class FileRecord:
    file_id: int
    path: Path
    subject: str
    n_samples: int
    sample_rate_hz: float
    truth: np.ndarray


@dataclass
class WindowRecord:
    file_id: int
    subject: str
    start: int
    end: int
    label: int


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_minimal_session(path: Path, data_dirs: Sequence[Path]) -> pd.DataFrame:
    needed = set(IMU_COLUMNS) | {"action_type", "phase", "sensor_ts", "host_ts", "pc_time", "subject_id", "set", "rep"}
    df = pd.read_csv(path, usecols=lambda col: col in needed)
    if "subject_id" not in df.columns:
        df["subject_id"] = subject_from_path(path, data_dirs)
    if "action_type" not in df.columns:
        df["action_type"] = "none"
    if "phase" not in df.columns:
        df["phase"] = "none"
    for col in IMU_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df.reset_index(drop=True)


def robust_scale_window(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    std = np.std(values, axis=0)
    scale = np.where(scale < 1e-9, std, scale)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (values - median) / scale


def window_features(window: np.ndarray) -> list[float]:
    z = robust_scale_window(window.astype(np.float64))
    diff = np.diff(z, axis=0, prepend=z[:1])
    acc_norm = np.linalg.norm(z[:, :3], axis=1)
    gyro_norm = np.linalg.norm(z[:, 3:6], axis=1)
    features: list[float] = []
    for values in (z, diff):
        features.extend(np.mean(values, axis=0).tolist())
        features.extend(np.std(values, axis=0).tolist())
        features.extend((np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)).tolist())
        features.extend((np.max(values, axis=0) - np.min(values, axis=0)).tolist())
    for norm in (acc_norm, gyro_norm):
        features.extend(
            [
                float(np.mean(norm)),
                float(np.std(norm)),
                float(np.percentile(norm, 75) - np.percentile(norm, 25)),
                float(np.max(norm) - np.min(norm)),
                float(np.sqrt(np.mean(norm**2))),
            ]
        )
    return features


def extract_windows(
    path: Path,
    file_id: int,
    args: argparse.Namespace,
) -> tuple[FileRecord, list[WindowRecord], list[list[float]]]:
    df = read_minimal_session(path, args.data_dirs)
    subject = str(df["subject_id"].iloc[0]).strip() if len(df) else subject_from_path(path, args.data_dirs)
    truth = ground_truth_mask(df, args.target)
    sample_rate_hz = estimate_sample_rate_hz(df)
    file_record = FileRecord(
        file_id=file_id,
        path=path,
        subject=subject,
        n_samples=len(df),
        sample_rate_hz=sample_rate_hz,
        truth=truth,
    )
    values = df.loc[:, IMU_COLUMNS].to_numpy(dtype=np.float64)
    windows: list[WindowRecord] = []
    features: list[list[float]] = []
    if len(values) < args.window_samples:
        return file_record, windows, features
    starts = list(range(0, len(values) - args.window_samples + 1, args.stride_samples))
    tail_start = len(values) - args.window_samples
    if starts and starts[-1] != tail_start:
        starts.append(tail_start)
    for start in starts:
        end = start + args.window_samples
        label = int(float(np.mean(truth[start:end])) >= args.label_threshold)
        windows.append(WindowRecord(file_id=file_id, subject=subject, start=start, end=end, label=label))
        features.append(window_features(values[start:end]))
    return file_record, windows, features


def postprocess_probability(
    score: np.ndarray,
    threshold: float,
    min_segment_samples: int,
    merge_gap_samples: int,
) -> np.ndarray:
    raw = score >= threshold
    intervals = contiguous_intervals(raw, min_samples=1)
    merged = merge_intervals(intervals, merge_gap_samples)
    filtered = [interval for interval in merged if interval.n_samples >= min_segment_samples]
    return intervals_to_mask(filtered, len(score))


def probability_from_windows(
    file_record: FileRecord,
    window_rows: Sequence[WindowRecord],
    probabilities: np.ndarray,
) -> np.ndarray:
    score_sum = np.zeros(file_record.n_samples, dtype=np.float64)
    score_count = np.zeros(file_record.n_samples, dtype=np.float64)
    for row, probability in zip(window_rows, probabilities):
        score_sum[row.start : row.end] += float(probability)
        score_count[row.start : row.end] += 1.0
    score = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count > 0)
    return score


def evaluate_prediction(
    file_record: FileRecord,
    pred_mask: np.ndarray,
    fold: int,
    method: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    true_intervals = contiguous_intervals(file_record.truth, min_samples=args.min_truth_segment_samples)
    pred_intervals = contiguous_intervals(pred_mask, min_samples=args.min_segment_samples)
    samples = sample_metrics(pred_mask, file_record.truth)
    segment = match_intervals(pred_intervals, true_intervals, args.iou_threshold)
    row = {
        "fold": fold,
        "file": str(file_record.path),
        "subject": file_record.subject,
        "target": args.target,
        "method": method,
        "samples": file_record.n_samples,
        "sample_rate_hz": round(file_record.sample_rate_hz, 4),
        "true_active_samples": int(np.sum(file_record.truth)),
        "predicted_active_samples": int(np.sum(pred_mask)),
        **samples,
        **segment,
    }
    if file_record.sample_rate_hz > 0:
        row["sum_abs_start_error_sec"] = round(float(row["sum_abs_start_error_samples"]) / file_record.sample_rate_hz, 4)
        row["sum_abs_end_error_sec"] = round(float(row["sum_abs_end_error_samples"]) / file_record.sample_rate_hz, 4)
        row["mean_abs_start_error_sec"] = round(float(row["mean_abs_start_error_samples"]) / file_record.sample_rate_hz, 4)
        row["mean_abs_end_error_sec"] = round(float(row["mean_abs_end_error_samples"]) / file_record.sample_rate_hz, 4)
    else:
        row["sum_abs_start_error_sec"] = 0.0
        row["sum_abs_end_error_sec"] = 0.0
        row["mean_abs_start_error_sec"] = 0.0
        row["mean_abs_end_error_sec"] = 0.0
    return row


def aggregate(rows: Sequence[dict[str, object]], group_cols: Sequence[str]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    metric_cols = [
        "sample_tp",
        "sample_fp",
        "sample_fn",
        "sample_tn",
        "false_active_samples",
        "missed_active_samples",
        "true_segments",
        "predicted_segments",
        "matched_segments",
        "false_positives",
        "false_negatives",
        "sum_matched_iou",
        "sum_abs_start_error_samples",
        "sum_abs_end_error_samples",
        "sum_abs_start_error_sec",
        "sum_abs_end_error_sec",
    ]
    grouped = df.groupby(list(group_cols), sort=True)[metric_cols].sum().reset_index()
    out: list[dict[str, object]] = []
    for row in grouped.to_dict("records"):
        tp = int(row["sample_tp"])
        fp = int(row["sample_fp"])
        fn = int(row["sample_fn"])
        tn = int(row["sample_tn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        matched = int(row["matched_segments"])
        predicted = int(row["predicted_segments"])
        truth = int(row["true_segments"])
        segment_precision = matched / predicted if predicted else 0.0
        segment_recall = matched / truth if truth else 0.0
        segment_f1 = 2 * segment_precision * segment_recall / (segment_precision + segment_recall) if segment_precision + segment_recall else 0.0
        out.append(
            {
                **{col: row[col] for col in group_cols},
                **{
                    col: round(float(row[col]), 4)
                    if col == "sum_matched_iou" or col.endswith("_sec")
                    else int(row[col])
                    for col in metric_cols
                },
                "sample_precision": round(precision, 4),
                "sample_recall": round(recall, 4),
                "sample_f1": round(f1, 4),
                "sample_accuracy": round(accuracy, 4),
                "segment_precision": round(segment_precision, 4),
                "segment_recall": round(segment_recall, 4),
                "segment_f1": round(segment_f1, 4),
                "mean_matched_iou": round(float(row["sum_matched_iou"]) / matched, 4) if matched else 0.0,
                "mean_abs_start_error_samples": round(float(row["sum_abs_start_error_samples"]) / matched, 2) if matched else 0.0,
                "mean_abs_end_error_samples": round(float(row["sum_abs_end_error_samples"]) / matched, 2) if matched else 0.0,
                "mean_abs_start_error_sec": round(float(row["sum_abs_start_error_sec"]) / matched, 4) if matched else 0.0,
                "mean_abs_end_error_sec": round(float(row["sum_abs_end_error_sec"]) / matched, 4) if matched else 0.0,
            }
        )
    return out


def plot_timeline(
    file_record: FileRecord,
    score: np.ndarray,
    pred_mask: np.ndarray,
    output_dir: Path,
    index: int,
    args: argparse.Namespace,
) -> Path:
    max_samples = min(args.plot_samples, file_record.n_samples)
    x = np.arange(max_samples)
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(x, score[:max_samples], color="#1f77b4", linewidth=0.9, label="active probability")
    for label, mask, color in (
        ("truth", file_record.truth, "#2ca02c"),
        ("window-rf", pred_mask, "#ff7f0e"),
    ):
        min_samples = args.min_truth_segment_samples if label == "truth" else args.min_segment_samples
        intervals = contiguous_intervals(mask, min_samples=min_samples)
        for interval in intervals:
            if interval.start < max_samples:
                ax.axvline(interval.start, color=color, linewidth=1.0, linestyle="-", alpha=0.85)
            if interval.end < max_samples:
                ax.axvline(interval.end, color=color, linewidth=1.0, linestyle="--", alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(active)")
    ax.set_xlabel("Sample index")
    ax.set_title(f"Window RF Active Detection\n{file_record.subject} | {file_record.path.name}", fontsize=11)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", linewidth=0.9, label="active probability"),
            Line2D([0], [0], color="#2ca02c", linewidth=1.2, label="truth start/end"),
            Line2D([0], [0], color="#ff7f0e", linewidth=1.2, label="window-rf start/end"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in file_record.path.stem)
    path = output_dir / f"{index:03d}_{safe_name}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_overall(rows: Sequence[dict[str, object]], output_dir: Path) -> Path | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.38
    labels = [f"{row.target}\n{row.method}" for row in df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - width / 2, df["sample_f1"].to_numpy(dtype=float), width, label="sample F1", color="#4c78a8")
    ax.bar(x + width / 2, df["segment_f1"].to_numpy(dtype=float), width, label="segment IoU@0.50 F1", color="#f58518")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Window RF Active Detection F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "window_rf_active_detection_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.savetxt(output_dir / "window_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    display = ConfusionMatrixDisplay(cm, display_labels=["rest", "active"])
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Window RF Confusion Matrix")
    fig.tight_layout()
    path = output_dir / "window_confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate subject-wise window classifier for active/set detection.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_active_detection/window_rf_action_5fold"))
    parser.add_argument("--target", choices=["action", "phase"], default="action")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--stride-samples", type=int, default=100)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-segment-samples", type=int, default=8000)
    parser.add_argument("--min-truth-segment-samples", type=int, default=50)
    parser.add_argument("--merge-gap-samples", type=int, default=4500)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max-timeline-plots", type=int, default=8)
    parser.add_argument("--plot-samples", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir = args.output_dir / "timeline_examples"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    file_records: dict[int, FileRecord] = {}
    window_records: list[WindowRecord] = []
    features: list[list[float]] = []
    for file_id, path in enumerate(whole_session_files(args.data_dirs)):
        file_record, file_windows, file_features = extract_windows(path, file_id, args)
        if not file_windows:
            continue
        file_records[file_id] = file_record
        window_records.extend(file_windows)
        features.extend(file_features)

    if not features:
        raise RuntimeError("No windows were extracted.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray([row.label for row in window_records], dtype=np.int64)
    groups = np.asarray([row.subject for row in window_records])
    n_splits = min(args.folds, len(set(groups)))
    splitter = GroupKFold(n_splits=n_splits)

    file_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    all_window_true: list[int] = []
    all_window_pred: list[int] = []
    plotted = 0

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=42 + fold,
            n_jobs=-1,
        )
        model.fit(x[train_idx], y[train_idx])
        class_to_idx = {label: idx for idx, label in enumerate(model.classes_)}
        active_idx = class_to_idx.get(1)
        if active_idx is None:
            probabilities = np.zeros(len(val_idx), dtype=np.float64)
        else:
            probabilities = model.predict_proba(x[val_idx])[:, active_idx]
        window_pred = (probabilities >= args.prob_threshold).astype(int)
        all_window_true.extend(y[val_idx].astype(int).tolist())
        all_window_pred.extend(window_pred.astype(int).tolist())

        val_by_file: dict[int, list[tuple[WindowRecord, float]]] = {}
        for idx, probability in zip(val_idx, probabilities):
            row = window_records[int(idx)]
            val_by_file.setdefault(row.file_id, []).append((row, float(probability)))

        train_subjects = sorted(set(groups[train_idx].tolist()))
        val_subjects = sorted(set(groups[val_idx].tolist()))
        fold_rows.append(
            {
                "fold": fold,
                "train_subjects": ";".join(train_subjects),
                "validation_subjects": ";".join(val_subjects),
                "train_windows": len(train_idx),
                "validation_windows": len(val_idx),
            }
        )

        for file_id, pairs in sorted(val_by_file.items()):
            file_record = file_records[file_id]
            rows_for_file = [row for row, _ in pairs]
            probs_for_file = np.asarray([prob for _, prob in pairs], dtype=np.float64)
            score = probability_from_windows(file_record, rows_for_file, probs_for_file)
            pred_mask = postprocess_probability(score, args.prob_threshold, args.min_segment_samples, args.merge_gap_samples)
            file_rows.append(evaluate_prediction(file_record, pred_mask, fold, "window-rf", args))
            if plotted < args.max_timeline_plots:
                plot_timeline(file_record, score, pred_mask, timeline_dir, plotted + 1, args)
                plotted += 1

    overall = aggregate(file_rows, ["target", "method"])
    by_subject = aggregate(file_rows, ["target", "method", "subject"])
    write_csv(args.output_dir / "active_detection_file_metrics.csv", file_rows)
    write_csv(args.output_dir / "active_detection_metrics.csv", overall)
    write_csv(args.output_dir / "active_detection_metrics_by_subject.csv", by_subject)
    write_csv(args.output_dir / "fold_manifest.csv", fold_rows)
    figure_paths = [
        plot_overall(overall, args.output_dir),
        save_confusion_matrix(np.asarray(all_window_true), np.asarray(all_window_pred), args.output_dir),
    ]
    summary = {
        "data_dirs": [str(path) for path in args.data_dirs],
        "target": args.target,
        "method": "window-rf",
        "folds": n_splits,
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "prob_threshold": args.prob_threshold,
        "min_segment_samples": args.min_segment_samples,
        "merge_gap_samples": args.merge_gap_samples,
        "num_windows": int(len(y)),
        "positive_windows": int(np.sum(y)),
        "figures": [str(path) for path in figure_paths if path is not None],
        "overall": overall,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
