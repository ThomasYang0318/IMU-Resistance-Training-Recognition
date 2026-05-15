from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from evaluate_rep_segmentation_classification import principal_motion_signal


@dataclass(frozen=True)
class SetMeta:
    key: tuple[str, str, str, str]
    file: str
    subject: str
    exercise: str
    set_id: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot waveform rep boundaries with set-level IoU accuracy.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--set-padding-fraction", type=float, default=0.15)
    parser.add_argument("--min-set-reps", type=int, default=1)
    parser.add_argument("--max-sets", type=int)
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    scale = float(np.percentile(values, 95) - np.percentile(values, 5))
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return values - float(np.mean(values))
    return (values - float(np.median(values))) / scale


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    intersection = max(0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(intersection / union) if union > 0 else 0.0


def match_intervals(
    predicted: list[tuple[int, int]],
    truth: list[tuple[int, int]],
    threshold: float,
) -> tuple[int, list[float]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predicted):
        for true_idx, true in enumerate(truth):
            iou = interval_iou(pred, true)
            if iou >= threshold:
                candidates.append((iou, pred_idx, true_idx))
    candidates.sort(reverse=True)

    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matched_ious: list[float] = []
    for iou, pred_idx, true_idx in candidates:
        if pred_idx in used_pred or true_idx in used_truth:
            continue
        used_pred.add(pred_idx)
        used_truth.add(true_idx)
        matched_ious.append(iou)
    return len(matched_ious), matched_ious


def set_metrics(
    predicted: list[tuple[int, int]],
    truth: list[tuple[int, int]],
    threshold: float,
) -> dict[str, float | int]:
    matched, matched_ious = match_intervals(predicted, truth, threshold)
    false_positives = len(predicted) - matched
    false_negatives = len(truth) - matched
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(truth) if truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "true_reps": len(truth),
        "predicted_reps": len(predicted),
        "matched_reps": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
    }


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def truth_set_metas(truth: pd.DataFrame, min_set_reps: int) -> list[SetMeta]:
    metas: list[SetMeta] = []
    group_cols = ["file", "subject", "exercise", "set_id"]
    for keys, group in truth.groupby(group_cols, sort=True):
        if len(group) < min_set_reps:
            continue
        file, subject, exercise, set_id = (str(value) for value in keys)
        metas.append(
            SetMeta(
                key=(file, subject, exercise, set_id),
                file=file,
                subject=subject,
                exercise=exercise,
                set_id=set_id,
                start=int(group["true_start"].min()),
                end=int(group["true_end"].max()),
            )
        )
    return metas


def overlap_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def assign_predictions_to_sets(predictions: pd.DataFrame, metas: list[SetMeta]) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    by_file: dict[str, list[SetMeta]] = {}
    for meta in metas:
        by_file.setdefault(meta.file, []).append(meta)

    assigned_rows: dict[tuple[str, str, str, str], list[pd.Series]] = {meta.key: [] for meta in metas}
    for _, row in predictions.iterrows():
        file = str(row["file"])
        pred_interval = (int(row["start"]), int(row["end"]))
        best_meta: SetMeta | None = None
        best_overlap = 0
        for meta in by_file.get(file, []):
            overlap = overlap_length(pred_interval, (meta.start, meta.end))
            if overlap > best_overlap:
                best_meta = meta
                best_overlap = overlap
        if best_meta is not None and best_overlap > 0:
            assigned_rows[best_meta.key].append(row)

    return {
        key: pd.DataFrame(rows, columns=predictions.columns)
        for key, rows in assigned_rows.items()
    }


def load_waveform(file_path: str, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if file_path not in cache:
        df = pd.read_csv(file_path)
        signal = principal_motion_signal(df, smooth_window=9)
        if {"ax", "ay", "az"}.issubset(df.columns):
            acc_norm = normalize(np.linalg.norm(df.loc[:, ["ax", "ay", "az"]].to_numpy(dtype=np.float64), axis=1))
        else:
            acc_norm = np.zeros(len(df), dtype=np.float64)
        cache[file_path] = (normalize(signal), acc_norm)
    return cache[file_path]


def draw_boundary_lines(
    ax: plt.Axes,
    intervals: list[tuple[int, int]],
    color: str,
    linewidth: float,
) -> None:
    for start, end in intervals:
        ax.axvline(start, color=color, linewidth=linewidth, linestyle="-", alpha=0.92)
        ax.axvline(end, color=color, linewidth=linewidth, linestyle="--", alpha=0.92)


def plot_waveform_background(ax: plt.Axes, x: np.ndarray, signal: np.ndarray, acc_norm: np.ndarray) -> None:
    ax.plot(x, signal, color="#303030", linewidth=0.9, label="PCA motion")
    ax.plot(x, acc_norm, color="#9a9a9a", linewidth=0.65, alpha=0.55, label="Acc magnitude")
    ax.grid(axis="x", alpha=0.12)
    ax.set_ylabel("Normalized signal")


def plot_set_waveform(
    meta: SetMeta,
    truth_intervals: list[tuple[int, int]],
    predicted_intervals: list[tuple[int, int]],
    metrics: dict[str, float | int],
    output_path: Path,
    waveform_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    padding_fraction: float,
    iou_threshold: float,
) -> None:
    signal, acc_norm = load_waveform(meta.file, waveform_cache)
    padding = int(round((meta.end - meta.start) * padding_fraction))
    window_start = max(0, meta.start - padding)
    window_end = min(len(signal), meta.end + padding)
    x = np.arange(window_start, window_end)
    signal_window = signal[window_start:window_end]
    acc_window = acc_norm[window_start:window_end]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True)
    for ax in axes:
        plot_waveform_background(ax, x, signal_window, acc_window)

    draw_boundary_lines(axes[0], truth_intervals, color="#0066cc", linewidth=1.8)
    draw_boundary_lines(axes[1], predicted_intervals, color="#d62728", linewidth=1.8)
    axes[0].set_title("Ground Truth")
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Sample index")

    waveform_handles = [
        Line2D([0], [0], color="#303030", linewidth=0.9, label="PCA motion"),
        Line2D([0], [0], color="#9a9a9a", linewidth=0.65, alpha=0.55, label="Acc magnitude"),
    ]
    axes[0].legend(
        handles=[
            *waveform_handles,
            Line2D([0], [0], color="#0066cc", linewidth=1.8, linestyle="-", label="GT start"),
            Line2D([0], [0], color="#0066cc", linewidth=1.8, linestyle="--", label="GT end"),
        ],
        loc="upper right",
        ncol=4,
        fontsize=8,
    )
    axes[1].legend(
        handles=[
            *waveform_handles,
            Line2D([0], [0], color="#d62728", linewidth=1.8, linestyle="-", label="Pred start"),
            Line2D([0], [0], color="#d62728", linewidth=1.8, linestyle="--", label="Pred end"),
        ],
        loc="upper right",
        ncol=4,
        fontsize=8,
    )

    fig.suptitle(
        f"{meta.subject} | {meta.exercise} | set {meta.set_id} | "
        f"IoU@{iou_threshold:.2f} F1={float(metrics['f1']):.2f}, "
        f"P={float(metrics['precision']):.2f}, R={float(metrics['recall']):.2f}, "
        f"TP/FP/FN={int(metrics['matched_reps'])}/{int(metrics['false_positives'])}/{int(metrics['false_negatives'])}, "
        f"mean IoU={float(metrics['mean_matched_iou']):.2f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aggregate_metrics(summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = summary.groupby(group_col, sort=True).agg(
        true_reps=("true_reps", "sum"),
        predicted_reps=("predicted_reps", "sum"),
        matched_reps=("matched_reps", "sum"),
        false_positives=("false_positives", "sum"),
        false_negatives=("false_negatives", "sum"),
    )
    grouped["precision"] = grouped["matched_reps"] / grouped["predicted_reps"].clip(lower=1)
    grouped["recall"] = grouped["matched_reps"] / grouped["true_reps"].clip(lower=1)
    denom = grouped["precision"] + grouped["recall"]
    grouped["f1"] = np.where(denom > 0, 2 * grouped["precision"] * grouped["recall"] / denom, 0.0)
    return grouped.reset_index()


def plot_bar(summary: pd.DataFrame, group_col: str, output_path: Path, title: str) -> None:
    agg = aggregate_metrics(summary, group_col).sort_values("f1")
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(agg) * 0.42)))
    bars = ax.barh(agg[group_col], agg["f1"], color="#4c78a8")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Rep segmentation F1 at IoU threshold")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, agg["f1"], strict=True):
        ax.text(min(float(value) + 0.015, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_subject_exercise_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    agg = summary.groupby(["subject", "exercise"], sort=True).agg(
        true_reps=("true_reps", "sum"),
        predicted_reps=("predicted_reps", "sum"),
        matched_reps=("matched_reps", "sum"),
    )
    agg["precision"] = agg["matched_reps"] / agg["predicted_reps"].clip(lower=1)
    agg["recall"] = agg["matched_reps"] / agg["true_reps"].clip(lower=1)
    denom = agg["precision"] + agg["recall"]
    agg["f1"] = np.where(denom > 0, 2 * agg["precision"] * agg["recall"] / denom, 0.0)
    pivot = agg["f1"].reset_index().pivot(index="subject", columns="exercise", values="f1").fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(9, len(pivot.columns) * 1.2), max(5, len(pivot) * 0.48)))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Waveform Rep Segmentation F1 by Subject and Exercise")
    for y in range(len(pivot.index)):
        for x in range(len(pivot.columns)):
            value = float(pivot.iloc[y, x])
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value >= 0.5 else "black", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_set_distribution(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(summary["f1"].to_numpy(dtype=float), bins=np.linspace(0, 1, 21), color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Set-level F1")
    ax.set_ylabel("Set count")
    ax.set_title("Waveform Set-Level Rep Segmentation F1 Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_dir = args.output_dir / "sets_all"
    set_dir.mkdir(parents=True, exist_ok=True)

    truth = read_required_csv(args.run_dir / "rep_segmentation_truth_matches.csv")
    predictions = read_required_csv(args.run_dir / "rep_segmentation_matches.csv")
    metas = truth_set_metas(truth, args.min_set_reps)
    if args.max_sets is not None:
        metas = metas[: args.max_sets]
    assigned_predictions = assign_predictions_to_sets(predictions, metas)

    rows: list[dict[str, object]] = []
    waveform_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, meta in enumerate(metas, start=1):
        truth_group = truth[
            (truth["file"].astype(str) == meta.file)
            & (truth["subject"].astype(str) == meta.subject)
            & (truth["exercise"].astype(str) == meta.exercise)
            & (truth["set_id"].astype(str) == meta.set_id)
        ].sort_values("true_start")
        pred_group = assigned_predictions.get(meta.key, pd.DataFrame(columns=predictions.columns)).sort_values("start")
        truth_intervals = [(int(row.true_start), int(row.true_end)) for row in truth_group.itertuples(index=False)]
        predicted_intervals = [(int(row.start), int(row.end)) for row in pred_group.itertuples(index=False)]
        metrics = set_metrics(predicted_intervals, truth_intervals, args.iou_threshold)
        output_name = f"{idx:03d}_{safe_name(meta.subject)}_{safe_name(meta.exercise)}_set_{safe_name(meta.set_id)}.png"
        plot_path = set_dir / output_name
        plot_set_waveform(
            meta,
            truth_intervals,
            predicted_intervals,
            metrics,
            plot_path,
            waveform_cache,
            args.set_padding_fraction,
            args.iou_threshold,
        )
        rows.append(
            {
                "plot": str(plot_path),
                "file": meta.file,
                "subject": meta.subject,
                "exercise": meta.exercise,
                "set_id": meta.set_id,
                "set_start": meta.start,
                "set_end": meta.end,
                "iou_threshold": args.iou_threshold,
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "waveform_rep_accuracy_set_summary.csv", index=False)
    plot_bar(summary, "subject", args.output_dir / "waveform_rep_accuracy_by_subject.png", "Waveform Rep Segmentation F1 by Subject")
    plot_bar(summary, "exercise", args.output_dir / "waveform_rep_accuracy_by_exercise.png", "Waveform Rep Segmentation F1 by Exercise")
    plot_subject_exercise_heatmap(summary, args.output_dir / "waveform_rep_accuracy_subject_exercise_heatmap.png")
    plot_set_distribution(summary, args.output_dir / "waveform_rep_accuracy_set_f1_distribution.png")

    overall = {
        "run_dir": str(args.run_dir),
        "iou_threshold": args.iou_threshold,
        "set_count": int(len(summary)),
        "true_reps": int(summary["true_reps"].sum()),
        "predicted_reps": int(summary["predicted_reps"].sum()),
        "matched_reps": int(summary["matched_reps"].sum()),
        "false_positives": int(summary["false_positives"].sum()),
        "false_negatives": int(summary["false_negatives"].sum()),
    }
    precision = overall["matched_reps"] / overall["predicted_reps"] if overall["predicted_reps"] else 0.0
    recall = overall["matched_reps"] / overall["true_reps"] if overall["true_reps"] else 0.0
    overall["precision"] = round(float(precision), 4)
    overall["recall"] = round(float(recall), 4)
    overall["f1"] = round(float(2 * precision * recall / (precision + recall)), 4) if (precision + recall) else 0.0
    (args.output_dir / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    print(f"set_plots={len(summary)}")
    print(f"summary={args.output_dir / 'waveform_rep_accuracy_set_summary.csv'}")
    print(f"subject_plot={args.output_dir / 'waveform_rep_accuracy_by_subject.png'}")


if __name__ == "__main__":
    main()
