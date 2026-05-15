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


IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")
ACTIVE_PHASES = {"concentric", "eccentric"}
REST_LABELS = {"big_rest", "rest", "none", "nan", ""}


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    @property
    def n_samples(self) -> int:
        return self.end - self.start


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def whole_session_files(data_dirs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for data_dir in data_dirs:
        files.extend(sorted(data_dir.rglob("*whole_session*.csv")))
    return sorted(set(files))


def clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def subject_from_path(path: Path, data_dirs: Sequence[Path]) -> str:
    for data_dir in data_dirs:
        try:
            return path.relative_to(data_dir).parts[0]
        except ValueError:
            continue
    return path.parent.name


def read_session(path: Path, data_dirs: Sequence[Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject_id" not in df.columns:
        df["subject_id"] = subject_from_path(path, data_dirs)
    if "action_type" not in df.columns:
        df["action_type"] = "none"
    if "phase" not in df.columns:
        df["phase"] = "none"
    if "set" not in df.columns:
        df["set"] = 0
    return df.reset_index(drop=True)


def contiguous_intervals(mask: np.ndarray, min_samples: int) -> list[Interval]:
    intervals: list[Interval] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_samples:
                intervals.append(Interval(start, idx))
            start = None
    if start is not None and len(mask) - start >= min_samples:
        intervals.append(Interval(start, len(mask)))
    return intervals


def intervals_to_mask(intervals: Sequence[Interval], n_samples: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for interval in intervals:
        mask[max(0, interval.start) : min(n_samples, interval.end)] = True
    return mask


def interval_iou(a: Interval, b: Interval) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / float(union) if union > 0 else 0.0


def match_intervals(predicted: Sequence[Interval], truth: Sequence[Interval], iou_threshold: float) -> dict[str, object]:
    pairs: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predicted):
        for true_idx, true in enumerate(truth):
            iou = interval_iou(pred, true)
            if iou >= iou_threshold:
                pairs.append((iou, pred_idx, true_idx))
    pairs.sort(reverse=True)

    used_pred: set[int] = set()
    used_true: set[int] = set()
    matched_ious: list[float] = []
    start_errors: list[int] = []
    end_errors: list[int] = []
    for iou, pred_idx, true_idx in pairs:
        if pred_idx in used_pred or true_idx in used_true:
            continue
        used_pred.add(pred_idx)
        used_true.add(true_idx)
        matched_ious.append(iou)
        start_errors.append(predicted[pred_idx].start - truth[true_idx].start)
        end_errors.append(predicted[pred_idx].end - truth[true_idx].end)

    matched = len(matched_ious)
    false_positives = len(predicted) - matched
    false_negatives = len(truth) - matched
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "true_segments": len(truth),
        "predicted_segments": len(predicted),
        "matched_segments": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "segment_precision": round(precision, 4),
        "segment_recall": round(recall, 4),
        "segment_f1": round(f1, 4),
        "sum_matched_iou": round(float(np.sum(matched_ious)), 4) if matched_ious else 0.0,
        "sum_abs_start_error_samples": int(np.sum(np.abs(start_errors))) if start_errors else 0,
        "sum_abs_end_error_samples": int(np.sum(np.abs(end_errors))) if end_errors else 0,
        "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "mean_abs_start_error_samples": round(float(np.mean(np.abs(start_errors))), 2) if start_errors else 0.0,
        "mean_abs_end_error_samples": round(float(np.mean(np.abs(end_errors))), 2) if end_errors else 0.0,
    }


def sample_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, object]:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(true) if len(true) else 0.0
    return {
        "sample_tp": tp,
        "sample_fp": fp,
        "sample_fn": fn,
        "sample_tn": tn,
        "sample_precision": round(precision, 4),
        "sample_recall": round(recall, 4),
        "sample_f1": round(f1, 4),
        "sample_accuracy": round(accuracy, 4),
        "false_active_samples": fp,
        "missed_active_samples": fn,
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(np.float64, copy=True)
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(values.astype(np.float64), pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def normalize_robust(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / scale


def imu_activity_score(df: pd.DataFrame, mode: str, smooth_window: int) -> np.ndarray:
    available = [col for col in IMU_COLUMNS if col in df.columns]
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    if x.size == 0:
        return np.zeros(len(df), dtype=np.float64)
    acc_cols = [col for col in ("ax", "ay", "az") if col in df.columns]
    gyro_cols = [col for col in ("gx", "gy", "gz") if col in df.columns]
    if mode == "imu-energy":
        acc = np.linalg.norm(df.loc[:, acc_cols].to_numpy(dtype=np.float64), axis=1) if acc_cols else np.zeros(len(df))
        gyro = np.linalg.norm(df.loc[:, gyro_cols].to_numpy(dtype=np.float64), axis=1) if gyro_cols else np.zeros(len(df))
        score = np.abs(normalize_robust(acc)) + np.abs(normalize_robust(gyro))
        return moving_average(score, smooth_window)
    if mode == "imu-variance":
        z = np.column_stack([normalize_robust(x[:, idx]) for idx in range(x.shape[1])])
        local_energy = np.mean(np.diff(z, axis=0, prepend=z[:1]) ** 2, axis=1)
        return moving_average(local_energy, smooth_window)
    raise ValueError(f"Unsupported score mode: {mode}")


def merge_intervals(intervals: Sequence[Interval], merge_gap_samples: int) -> list[Interval]:
    merged: list[Interval] = []
    for interval in intervals:
        if merged and interval.start - merged[-1].end <= merge_gap_samples:
            merged[-1] = Interval(merged[-1].start, interval.end)
        else:
            merged.append(interval)
    return merged


def threshold_mask(score: np.ndarray, percentile: float, min_samples: int, merge_gap_samples: int) -> np.ndarray:
    if len(score) == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(np.percentile(score, percentile))
    mask = score >= threshold
    intervals = contiguous_intervals(mask, min_samples=max(1, min_samples))
    if not intervals:
        return np.zeros_like(mask, dtype=bool)
    merged = merge_intervals(intervals, merge_gap_samples)
    return intervals_to_mask(merged, len(mask))


def hysteresis_set_mask(score: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(score) == 0:
        return np.zeros(0, dtype=bool)
    smoothed = moving_average(score, args.hysteresis_smooth_window)
    low_threshold = float(np.percentile(smoothed, args.hysteresis_low_percentile))
    high_threshold = float(np.percentile(smoothed, args.hysteresis_high_percentile))
    low_mask = smoothed >= low_threshold
    high_mask = smoothed >= high_threshold

    kept: list[Interval] = []
    for interval in contiguous_intervals(low_mask, min_samples=1):
        if np.any(high_mask[interval.start : interval.end]):
            kept.append(interval)
    if not kept:
        return np.zeros_like(low_mask, dtype=bool)
    merged = merge_intervals(kept, args.hysteresis_merge_gap_samples)
    filtered = [interval for interval in merged if interval.n_samples >= args.hysteresis_min_segment_samples]
    return intervals_to_mask(filtered, len(low_mask))


def ground_truth_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "action":
        actions = df["action_type"].map(clean_label).str.lower()
        return (~actions.isin(REST_LABELS)).to_numpy()
    if target == "phase":
        phases = df["phase"].map(clean_label).str.lower()
        return phases.isin(ACTIVE_PHASES).to_numpy()
    raise ValueError(f"Unsupported target: {target}")


def oracle_action_mask(df: pd.DataFrame) -> np.ndarray:
    actions = df["action_type"].map(clean_label).str.lower()
    return (~actions.isin(REST_LABELS)).to_numpy()


def estimate_sample_rate_hz(df: pd.DataFrame) -> float:
    for col in ("sensor_ts", "host_ts"):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            continue
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            continue
        median_diff = float(np.median(diffs))
        if median_diff <= 0:
            continue
        if median_diff > 1_000_000:
            rate = 1_000_000_000.0 / median_diff
        elif median_diff > 1_000:
            rate = 1_000_000.0 / median_diff
        elif median_diff > 1:
            rate = 1_000.0 / median_diff
        else:
            rate = 1.0 / median_diff
        if 1.0 <= rate <= 1000.0:
            return rate

    if "pc_time" in df.columns:
        times = pd.to_datetime(df["pc_time"], errors="coerce")
        values = times.dropna().astype("int64").to_numpy(dtype=np.float64)
        if len(values) >= 3:
            diffs = np.diff(values) / 1_000_000_000.0
            diffs = diffs[diffs > 0]
            if len(diffs):
                median_diff = float(np.median(diffs))
                if median_diff > 0:
                    rate = 1.0 / median_diff
                    if 1.0 <= rate <= 1000.0:
                        return rate
    return 0.0


def method_mask(df: pd.DataFrame, method: str, args: argparse.Namespace) -> np.ndarray:
    if method == "oracle-action":
        return oracle_action_mask(df)
    if method == "imu-hysteresis":
        score = imu_activity_score(df, "imu-energy", args.smooth_window)
        return hysteresis_set_mask(score, args)
    score = imu_activity_score(df, method, args.smooth_window)
    return threshold_mask(
        score,
        percentile=args.active_percentile,
        min_samples=args.min_segment_samples,
        merge_gap_samples=args.merge_gap_samples,
    )


def file_meta(df: pd.DataFrame, path: Path, data_dirs: Sequence[Path]) -> dict[str, str]:
    subject = clean_label(df["subject_id"].iloc[0]) if "subject_id" in df.columns and len(df) else subject_from_path(path, data_dirs)
    actions = df["action_type"].map(clean_label).str.lower()
    non_rest = sorted(set(actions[~actions.isin(REST_LABELS)].tolist()))
    exercise = ",".join(non_rest[:3]) if non_rest else "none"
    return {"subject": subject, "exercise_hint": exercise}


def evaluate_file(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    df = read_session(path, args.data_dirs)
    meta = file_meta(df, path, args.data_dirs)
    sample_rate_hz = estimate_sample_rate_hz(df)
    rows: list[dict[str, object]] = []
    masks: dict[str, np.ndarray] = {}
    for target in args.targets:
        true_mask = ground_truth_mask(df, target)
        true_intervals = contiguous_intervals(true_mask, min_samples=args.min_segment_samples)
        if args.exclude_empty_truth and not true_intervals:
            continue
        for method in args.methods:
            pred_mask = method_mask(df, method, args)
            pred_intervals = contiguous_intervals(pred_mask, min_samples=args.min_segment_samples)
            segment = match_intervals(pred_intervals, true_intervals, iou_threshold=args.iou_threshold)
            samples = sample_metrics(pred_mask, true_mask)
            row = {
                "file": str(path),
                "subject": meta["subject"],
                "exercise_hint": meta["exercise_hint"],
                "target": target,
                "method": method,
                "samples": len(df),
                "sample_rate_hz": round(sample_rate_hz, 4),
                "true_active_samples": int(np.sum(true_mask)),
                "predicted_active_samples": int(np.sum(pred_mask)),
                **samples,
                **segment,
            }
            if sample_rate_hz > 0:
                row["sum_abs_start_error_sec"] = round(float(row["sum_abs_start_error_samples"]) / sample_rate_hz, 4)
                row["sum_abs_end_error_sec"] = round(float(row["sum_abs_end_error_samples"]) / sample_rate_hz, 4)
                row["mean_abs_start_error_sec"] = round(float(row["mean_abs_start_error_samples"]) / sample_rate_hz, 4)
                row["mean_abs_end_error_sec"] = round(float(row["mean_abs_end_error_samples"]) / sample_rate_hz, 4)
            else:
                row["sum_abs_start_error_sec"] = 0.0
                row["sum_abs_end_error_sec"] = 0.0
                row["mean_abs_start_error_sec"] = 0.0
                row["mean_abs_end_error_sec"] = 0.0
            rows.append(row)
            masks[f"{target}:{method}"] = pred_mask
        masks[f"{target}:truth"] = true_mask
    return rows, {"df": df, "meta": meta, "masks": masks}


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
    out_rows: list[dict[str, object]] = []
    for row in grouped.to_dict("records"):
        tp = int(row["sample_tp"])
        fp = int(row["sample_fp"])
        fn = int(row["sample_fn"])
        tn = int(row["sample_tn"])
        sample_precision = tp / (tp + fp) if tp + fp else 0.0
        sample_recall = tp / (tp + fn) if tp + fn else 0.0
        sample_f1 = 2.0 * sample_precision * sample_recall / (sample_precision + sample_recall) if sample_precision + sample_recall else 0.0
        sample_accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        matched = int(row["matched_segments"])
        predicted = int(row["predicted_segments"])
        true = int(row["true_segments"])
        segment_precision = matched / predicted if predicted else 0.0
        segment_recall = matched / true if true else 0.0
        segment_f1 = 2.0 * segment_precision * segment_recall / (segment_precision + segment_recall) if segment_precision + segment_recall else 0.0
        mean_matched_iou = float(row["sum_matched_iou"]) / matched if matched else 0.0
        mean_abs_start_error_samples = float(row["sum_abs_start_error_samples"]) / matched if matched else 0.0
        mean_abs_end_error_samples = float(row["sum_abs_end_error_samples"]) / matched if matched else 0.0
        mean_abs_start_error_sec = float(row["sum_abs_start_error_sec"]) / matched if matched else 0.0
        mean_abs_end_error_sec = float(row["sum_abs_end_error_sec"]) / matched if matched else 0.0
        out_rows.append(
            {
                **{col: row[col] for col in group_cols},
                **{
                    col: round(float(row[col]), 4)
                    if col.startswith("sum_") and col.endswith("_sec")
                    else int(row[col])
                    if not col == "sum_matched_iou"
                    else round(float(row[col]), 4)
                    for col in metric_cols
                },
                "sample_precision": round(sample_precision, 4),
                "sample_recall": round(sample_recall, 4),
                "sample_f1": round(sample_f1, 4),
                "sample_accuracy": round(sample_accuracy, 4),
                "segment_precision": round(segment_precision, 4),
                "segment_recall": round(segment_recall, 4),
                "segment_f1": round(segment_f1, 4),
                "mean_matched_iou": round(mean_matched_iou, 4),
                "mean_abs_start_error_samples": round(mean_abs_start_error_samples, 2),
                "mean_abs_end_error_samples": round(mean_abs_end_error_samples, 2),
                "mean_abs_start_error_sec": round(mean_abs_start_error_sec, 4),
                "mean_abs_end_error_sec": round(mean_abs_end_error_sec, 4),
            }
        )
    return out_rows


def plot_overall_comparison(rows: Sequence[dict[str, object]], output_dir: Path) -> Path | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    labels = [f"{row.target}\n{row.method}" for row in df.itertuples(index=False)]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 4.8))
    ax.bar(x - width / 2, df["sample_f1"].to_numpy(dtype=float), width, label="sample F1", color="#4c78a8")
    ax.bar(x + width / 2, df["segment_f1"].to_numpy(dtype=float), width, label="segment IoU@0.50 F1", color="#f58518")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Active Detection Overall F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "active_detection_overall_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_subject_heatmap(
    rows: Sequence[dict[str, object]],
    output_dir: Path,
    metric: str,
) -> Path | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["method_target"] = df["target"].astype(str) + "/" + df["method"].astype(str)
    pivot = df.pivot_table(index="subject", columns="method_target", values=metric, aggfunc="mean").fillna(0.0)
    if pivot.empty:
        return None
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, values.shape[1] * 1.2), max(4.8, values.shape[0] * 0.45)))
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"Active Detection {metric.replace('_', ' ').title()} By Subject")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=40, ha="right")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=7, color="white" if values[y, x] < 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = output_dir / f"active_detection_{metric}_by_subject.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_timeline(
    path: Path,
    payload: dict[str, object],
    args: argparse.Namespace,
    output_dir: Path,
    index: int,
) -> Path:
    df = payload["df"]
    masks: dict[str, np.ndarray] = payload["masks"]
    meta: dict[str, str] = payload["meta"]
    score = imu_activity_score(df, "imu-energy", args.smooth_window)
    if len(score):
        denom = float(np.percentile(score, 95) - np.percentile(score, 5))
        if denom < 1e-9:
            denom = float(np.std(score)) or 1.0
        score = (score - float(np.median(score))) / denom

    n = len(df)
    max_samples = min(args.plot_samples, n)
    x = np.arange(max_samples)
    fig, axes = plt.subplots(len(args.targets), 1, figsize=(14, max(4, 2.2 * len(args.targets))), sharex=True)
    if len(args.targets) == 1:
        axes = [axes]
    colors = {
        "truth": "#2ca02c",
        "oracle-action": "#9467bd",
        "imu-energy": "#ff7f0e",
        "imu-variance": "#d62728",
    }
    for ax, target in zip(axes, args.targets):
        ax.plot(x, score[:max_samples], color="#1f77b4", linewidth=0.8, label="IMU activity score")
        labels = ["truth", *args.methods]
        handles: list[Line2D] = [Line2D([0], [0], color="#1f77b4", linewidth=0.8, label="IMU activity score")]
        for label in labels:
            key = f"{target}:{label}"
            mask = masks.get(key)
            if mask is None:
                continue
            color = colors.get(label, "#7f7f7f")
            intervals = contiguous_intervals(mask, min_samples=args.min_segment_samples)
            for interval in intervals:
                if interval.start < max_samples:
                    ax.axvline(interval.start, color=color, linewidth=1.0, linestyle="-", alpha=0.85)
                if interval.end < max_samples:
                    ax.axvline(interval.end, color=color, linewidth=1.0, linestyle="--", alpha=0.85)
            handles.append(Line2D([0], [0], color=color, linewidth=1.2, label=f"{label} start/end"))
        ax.set_ylim(-1.5, 1.8)
        ax.set_ylabel(target)
        ax.grid(axis="x", alpha=0.15)
        ax.legend(handles=handles, loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"Active Detection Timeline\n{meta['subject']} | {meta['exercise_hint']} | {path.name}", fontsize=12)
    fig.tight_layout()
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)
    output_path = output_dir / f"{index:03d}_{safe_name}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active/set detection from workout whole-session IMU files.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_active_detection"))
    parser.add_argument("--targets", choices=["action", "phase"], nargs="+", default=["action", "phase"])
    parser.add_argument(
        "--methods",
        choices=["oracle-action", "imu-hysteresis", "imu-energy", "imu-variance"],
        nargs="+",
        default=["oracle-action", "imu-hysteresis"],
    )
    parser.add_argument("--active-percentile", type=float, default=70.0)
    parser.add_argument("--smooth-window", type=int, default=101)
    parser.add_argument("--min-segment-samples", type=int, default=50)
    parser.add_argument("--merge-gap-samples", type=int, default=100)
    parser.add_argument("--hysteresis-low-percentile", type=float, default=40.0)
    parser.add_argument("--hysteresis-high-percentile", type=float, default=65.0)
    parser.add_argument("--hysteresis-smooth-window", type=int, default=2001)
    parser.add_argument("--hysteresis-min-segment-samples", type=int, default=8000)
    parser.add_argument("--hysteresis-merge-gap-samples", type=int, default=4500)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--exclude-empty-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-timeline-plots", type=int, default=8)
    parser.add_argument("--plot-samples", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir = args.output_dir / "timeline_examples"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    plotted = 0
    for file_idx, path in enumerate(whole_session_files(args.data_dirs), start=1):
        file_rows, payload = evaluate_file(path, args)
        rows.extend(file_rows)
        if plotted < args.max_timeline_plots and file_rows:
            plot_timeline(path, payload, args, timeline_dir, file_idx)
            plotted += 1

    overall = aggregate(rows, ["target", "method"])
    by_subject = aggregate(rows, ["target", "method", "subject"])
    by_exercise = aggregate(rows, ["target", "method", "exercise_hint"])

    write_csv(args.output_dir / "active_detection_file_metrics.csv", rows)
    write_csv(args.output_dir / "active_detection_metrics.csv", overall)
    write_csv(args.output_dir / "active_detection_metrics_by_subject.csv", by_subject)
    write_csv(args.output_dir / "active_detection_metrics_by_exercise.csv", by_exercise)
    figure_paths = [
        plot_overall_comparison(overall, args.output_dir),
        plot_subject_heatmap(by_subject, args.output_dir, "sample_f1"),
        plot_subject_heatmap(by_subject, args.output_dir, "segment_f1"),
    ]
    summary = {
        "data_dirs": [str(path) for path in args.data_dirs],
        "targets": args.targets,
        "methods": args.methods,
        "active_percentile": args.active_percentile,
        "smooth_window": args.smooth_window,
        "min_segment_samples": args.min_segment_samples,
        "merge_gap_samples": args.merge_gap_samples,
        "iou_threshold": args.iou_threshold,
        "num_file_metric_rows": len(rows),
        "figures": [str(path) for path in figure_paths if path is not None],
        "overall": overall,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
