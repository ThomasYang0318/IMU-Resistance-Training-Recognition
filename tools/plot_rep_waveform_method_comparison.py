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

from evaluate_rep_segmentation_classification import IMU_COLUMNS, principal_motion_signal


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as method_name=run_dir")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("method_name cannot be empty")
    return name, Path(path)


def read_truth(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "rep_segmentation_truth_matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing truth matches: {path}")
    return pd.read_csv(path)


def read_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "rep_segmentation_matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predicted matches: {path}")
    return pd.read_csv(path)


def choose_file(truth: pd.DataFrame, min_reps: int, max_reps: int) -> str:
    counts = truth.groupby("file").size().sort_values()
    candidates = counts[(counts >= min_reps) & (counts <= max_reps)]
    if candidates.empty:
        candidates = counts
    return str(candidates.index[len(candidates) // 2])


def choose_window(truth_file: pd.DataFrame, window_reps: int, padding_fraction: float = 0.3) -> tuple[int, int]:
    truth_file = truth_file.sort_values("true_start")
    if len(truth_file) <= window_reps:
        start = int(truth_file["true_start"].min())
        end = int(truth_file["true_end"].max())
    else:
        start_idx = max(0, (len(truth_file) - window_reps) // 2)
        selected = truth_file.iloc[start_idx : start_idx + window_reps]
        start = int(selected["true_start"].min())
        end = int(selected["true_end"].max())
    padding = int(round((end - start) * padding_fraction))
    return max(0, start - padding), end + padding


def overlapping_intervals(df: pd.DataFrame, start_col: str, end_col: str, start: int, end: int) -> pd.DataFrame:
    return df[(df[start_col].astype(int) < end) & (df[end_col].astype(int) > start)].copy()


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    scale = float(np.percentile(values, 95) - np.percentile(values, 5))
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return values - float(np.mean(values))
    return (values - float(np.median(values))) / scale


def add_intervals(ax, intervals: pd.DataFrame, start_col: str, end_col: str, color: str, alpha: float, label: str) -> None:
    first = True
    for row in intervals.itertuples(index=False):
        start = int(getattr(row, start_col))
        end = int(getattr(row, end_col))
        ax.axvspan(start, end, color=color, alpha=alpha, label=label if first else None)
        first = False


def file_method_summary(
    file_path: str,
    truth: pd.DataFrame,
    runs: list[tuple[str, Path]],
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    truth_file = truth[truth["file"] == file_path]
    true_count = int(len(overlapping_intervals(truth_file, "true_start", "true_end", window_start, window_end)))
    for method, run_dir in runs:
        pred = read_predictions(run_dir)
        pred_file = overlapping_intervals(pred[pred["file"] == file_path], "start", "end", window_start, window_end)
        good = pred_file[pred_file["best_true_iou"].astype(float) >= 0.5]
        rows.append(
            {
                "method": method,
                "file": file_path,
                "window_start": window_start,
                "window_end": window_end,
                "true_reps": true_count,
                "predicted_reps": int(len(pred_file)),
                "predicted_with_best_iou_ge_0.50": int(len(good)),
                "mean_best_true_iou": round(float(pred_file["best_true_iou"].astype(float).mean()), 4) if len(pred_file) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_waveform(
    file_path: str,
    truth: pd.DataFrame,
    runs: list[tuple[str, Path]],
    output_dir: Path,
    window_start: int,
    window_end: int,
) -> Path:
    df = pd.read_csv(file_path)
    signal = principal_motion_signal(df, smooth_window=9)
    if "ax" in df.columns and "ay" in df.columns and "az" in df.columns:
        acc_norm = normalize(np.linalg.norm(df.loc[:, ["ax", "ay", "az"]].to_numpy(dtype=np.float64), axis=1))
    else:
        acc_norm = np.zeros(len(df), dtype=np.float64)
    signal = normalize(signal)
    window_end = min(window_end, len(signal))
    x = np.arange(window_start, window_end)
    signal = signal[window_start:window_end]
    acc_norm = acc_norm[window_start:window_end]

    truth_file = overlapping_intervals(truth[truth["file"] == file_path], "true_start", "true_end", window_start, window_end)
    n_rows = 1 + len(runs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, max(6, n_rows * 1.8)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(x, signal, color="#1f77b4", linewidth=0.9, label="PCA motion")
    axes[0].plot(x, acc_norm, color="#7f7f7f", linewidth=0.65, alpha=0.65, label="Acc magnitude")
    add_intervals(axes[0], truth_file, "true_start", "true_end", "#2ca02c", 0.24, "Ground truth rep")
    axes[0].set_ylabel("truth")
    axes[0].legend(loc="upper right", fontsize=8)

    for ax, (method, run_dir) in zip(axes[1:], runs):
        pred = read_predictions(run_dir)
        pred_file = overlapping_intervals(pred[pred["file"] == file_path], "start", "end", window_start, window_end)
        ax.plot(x, signal, color="#1f77b4", linewidth=0.9)
        add_intervals(ax, truth_file, "true_start", "true_end", "#2ca02c", 0.16, "Ground truth")
        add_intervals(ax, pred_file, "start", "end", "#ff7f0e", 0.28, "Predicted")
        ax.set_ylabel(method)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(
        f"Rep Boundary Comparison on Waveform\n{Path(file_path).name} | samples {window_start}-{window_end}",
        fontsize=13,
    )
    fig.tight_layout()
    output_path = output_dir / f"waveform_method_comparison_{Path(file_path).stem}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_file_counts(summary: pd.DataFrame, output_dir: Path) -> Path:
    x = np.arange(len(summary))
    width = 0.28
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, summary["true_reps"].to_numpy(dtype=float), width, label="True reps")
    ax.bar(x, summary["predicted_reps"].to_numpy(dtype=float), width, label="Predicted reps")
    ax.bar(
        x + width,
        summary["predicted_with_best_iou_ge_0.50"].to_numpy(dtype=float),
        width,
        label="Predicted IoU >= 0.50",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Per-File Rep Count Difference by Method")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path = output_dir / "waveform_method_count_difference.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot waveform-level rep boundary differences across methods.")
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file", type=str)
    parser.add_argument("--min-reps", type=int, default=6)
    parser.add_argument("--max-reps", type=int, default=14)
    parser.add_argument("--window-reps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = args.run
    truth = read_truth(runs[0][1])
    file_path = args.file or choose_file(truth, args.min_reps, args.max_reps)
    truth_file = truth[truth["file"] == file_path]
    window_start, window_end = choose_window(truth_file, args.window_reps)
    summary = file_method_summary(file_path, truth, runs, window_start, window_end)
    summary.to_csv(args.output_dir / "waveform_method_file_summary.csv", index=False)
    waveform_path = plot_waveform(file_path, truth, runs, args.output_dir, window_start, window_end)
    count_path = plot_file_counts(summary, args.output_dir)
    print(f"waveform_plot={waveform_path}")
    print(f"count_plot={count_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
