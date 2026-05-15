from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_id_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["subject"].astype(str)
        + " | "
        + df["exercise"].astype(str)
        + " | set "
        + df["set_id"].astype(str)
    )


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["set_key"] = set_id_series(out)
    out["matched_rate_at_iou_0.50"] = out["predicted_with_best_iou_ge_0.50"] / out["true_reps"].clip(lower=1)
    out["prediction_ratio"] = out["predicted_reps"] / out["true_reps"].clip(lower=1)
    out["over_segmentation_ratio"] = (out["predicted_reps"] - out["true_reps"]) / out["true_reps"].clip(lower=1)
    return out


def plot_method_average(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        df.groupby("method", sort=False)
        .agg(
            mean_matched_rate=("matched_rate_at_iou_0.50", "mean"),
            mean_best_iou=("mean_best_true_iou", "mean"),
            mean_prediction_ratio=("prediction_ratio", "mean"),
            mean_over_segmentation_ratio=("over_segmentation_ratio", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "set_level_method_average.csv", index=False)

    x = np.arange(len(summary))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, summary["mean_matched_rate"], width, label="Matched rate IoU>=0.50")
    ax.bar(x, summary["mean_best_iou"], width, label="Mean best IoU")
    ax.bar(x + width, summary["mean_prediction_ratio"], width, label="Predicted / true reps")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Score / Ratio")
    ax.set_title("Set-Level Method Result Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "set_level_method_average_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, output_path: Path, vmax: float | None = None) -> Path:
    pivot = df.pivot_table(index="set_key", columns="method", values=value_col, aggfunc="first")
    pivot = pivot.loc[df["set_key"].drop_duplicates().tolist()]
    matrix = pivot.to_numpy(dtype=float)
    height = max(10, min(42, len(pivot) * 0.16))
    fig, ax = plt.subplots(figsize=(9, height))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=5)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_best_method_counts(df: pd.DataFrame, output_dir: Path) -> Path:
    idx = df.groupby("set_key")["matched_rate_at_iou_0.50"].idxmax()
    winners = df.loc[idx, "method"].value_counts().reindex(df["method"].drop_duplicates(), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(winners.index.tolist(), winners.to_numpy())
    ax.set_ylabel("Number of Sets")
    ax.set_title("Best Method per Set by IoU>=0.50 Matched Rate")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "set_level_best_method_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot set-level method comparison from waveform summary CSV.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = add_rates(pd.read_csv(args.summary))
    df.to_csv(args.output_dir / "set_level_method_results.csv", index=False)
    print(plot_method_average(df, args.output_dir))
    print(
        plot_heatmap(
            df,
            "matched_rate_at_iou_0.50",
            "Set-Level Matched Rate at IoU >= 0.50",
            args.output_dir / "set_level_matched_rate_heatmap.png",
            vmax=1.0,
        )
    )
    print(
        plot_heatmap(
            df,
            "prediction_ratio",
            "Set-Level Predicted / True Rep Ratio",
            args.output_dir / "set_level_prediction_ratio_heatmap.png",
            vmax=float(min(12.0, max(1.0, df["prediction_ratio"].quantile(0.95)))),
        )
    )
    print(plot_best_method_counts(df, args.output_dir))


if __name__ == "__main__":
    main()
