from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_metrics(run_dir: Path, method: str) -> pd.DataFrame:
    path = run_dir / "rep_segmentation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing segmentation metrics: {path}")
    df = pd.read_csv(path)
    df["method"] = method
    return df


def read_exercise_metrics(run_dir: Path, method: str) -> pd.DataFrame:
    path = run_dir / "rep_segmentation_metrics_by_exercise.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-exercise segmentation metrics: {path}")
    df = pd.read_csv(path)
    df["method"] = method
    return df


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_overall(df: pd.DataFrame, output_dir: Path, filename: str = "rep_segmentation_methods_f1.png") -> None:
    thresholds = sorted(df["iou_threshold"].astype(float).unique())
    methods = df["method"].drop_duplicates().tolist()
    x = np.arange(len(thresholds))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method_idx, method in enumerate(methods):
        values = []
        for threshold in thresholds:
            row = df[(df["method"] == method) & (df["iou_threshold"].astype(float) == threshold)]
            values.append(float(row["f1"].iloc[0]) if not row.empty else 0.0)
        offset = (method_idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU >= {threshold:.2f}" for threshold in thresholds])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Rep Segmentation F1 by Method")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_precision_recall(
    df: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    filename_prefix: str = "rep_segmentation_methods",
) -> None:
    subset = df[np.isclose(df["iou_threshold"].astype(float), threshold)]
    methods = subset["method"].drop_duplicates().tolist()
    x = np.arange(len(methods))
    width = 0.28

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, subset["precision"].astype(float).to_numpy(), width, label="Precision")
    ax.bar(x, subset["recall"].astype(float).to_numpy(), width, label="Recall")
    ax.bar(x + width, subset["f1"].astype(float).to_numpy(), width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Rep Segmentation Metrics at IoU >= {threshold:.2f}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_prefix}_iou_{threshold:.2f}.png", dpi=180)
    plt.close(fig)


def plot_error_breakdown(df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    subset = df[np.isclose(df["iou_threshold"].astype(float), threshold)]
    methods = subset["method"].drop_duplicates().tolist()
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, subset["matched_reps"].astype(float).to_numpy(), width, label="Matched reps")
    ax.bar(x, subset["false_positives"].astype(float).to_numpy(), width, label="False positives")
    ax.bar(x + width, subset["false_negatives"].astype(float).to_numpy(), width, label="False negatives")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Segments")
    ax.set_title(f"Rep Segmentation Error Breakdown at IoU >= {threshold:.2f}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"rep_segmentation_methods_error_breakdown_iou_{threshold:.2f}.png", dpi=180)
    plt.close(fig)


def plot_exercise_delta(df: pd.DataFrame, output_dir: Path, baseline: str, candidate: str, threshold: float) -> None:
    subset = df[np.isclose(df["iou_threshold"].astype(float), threshold)]
    pivot = subset.pivot_table(index="exercise", columns="method", values="f1", aggfunc="first").fillna(0.0)
    if baseline not in pivot.columns or candidate not in pivot.columns:
        return
    delta = (pivot[candidate] - pivot[baseline]).sort_values()

    fig, ax = plt.subplots(figsize=(9, max(5, len(delta) * 0.45)))
    colors = ["#d95f02" if value < 0 else "#1b9e77" for value in delta]
    ax.barh(delta.index.tolist(), delta.to_numpy(), color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("F1 Delta")
    ax.set_title(f"Per-Exercise F1 Change at IoU >= {threshold:.2f}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"rep_segmentation_fft_exercise_delta_iou_{threshold:.2f}.png", dpi=180)
    plt.close(fig)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as method_name=run_dir")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("method_name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rep segmentation IoU results across methods.")
    parser.add_argument("--run", type=parse_run, action="append", help="Method result in the form method_name=run_dir.")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--baseline-name", default="pca-extrema")
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-name", default="pca-extrema-fft")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = args.run or []
    if not runs:
        if args.baseline_dir is None or args.candidate_dir is None:
            raise ValueError("Provide either --run method=dir entries or both --baseline-dir and --candidate-dir.")
        runs = [(args.baseline_name, args.baseline_dir), (args.candidate_name, args.candidate_dir)]

    overall = pd.concat(
        [read_metrics(run_dir, method) for method, run_dir in runs],
        ignore_index=True,
    )
    by_exercise = pd.concat(
        [read_exercise_metrics(run_dir, method) for method, run_dir in runs],
        ignore_index=True,
    )

    write_csv(args.output_dir / "rep_segmentation_methods_comparison.csv", overall.to_dict("records"))
    write_csv(args.output_dir / "rep_segmentation_methods_comparison_by_exercise.csv", by_exercise.to_dict("records"))
    plot_overall(overall, args.output_dir)
    plot_precision_recall(overall, args.output_dir, args.focus_iou)
    plot_error_breakdown(overall, args.output_dir, args.focus_iou)
    if len(runs) == 2:
        plot_exercise_delta(by_exercise, args.output_dir, runs[0][0], runs[1][0], args.focus_iou)


if __name__ == "__main__":
    main()
