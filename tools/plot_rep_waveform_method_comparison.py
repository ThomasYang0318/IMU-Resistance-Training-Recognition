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


def load_predictions(runs: list[tuple[str, Path]]) -> dict[str, pd.DataFrame]:
    return {method: read_predictions(run_dir) for method, run_dir in runs}


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


def add_boundary_lines(ax, intervals: pd.DataFrame, start_col: str, end_col: str, color: str, label: str) -> None:
    first = True
    for row in intervals.itertuples(index=False):
        start = int(getattr(row, start_col))
        end = int(getattr(row, end_col))
        ax.axvline(start, color=color, linewidth=0.9, alpha=0.9, label=label if first else None)
        ax.axvline(end, color=color, linewidth=0.9, alpha=0.9, linestyle="--")
        first = False


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def file_method_summary(
    file_path: str,
    truth: pd.DataFrame,
    runs: list[tuple[str, Path]],
    predictions: dict[str, pd.DataFrame],
    window_start: int,
    window_end: int,
    group: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    truth_file = truth[truth["file"] == file_path]
    true_count = int(len(overlapping_intervals(truth_file, "true_start", "true_end", window_start, window_end)))
    for method, run_dir in runs:
        pred = predictions[method]
        pred_file = overlapping_intervals(pred[pred["file"] == file_path], "start", "end", window_start, window_end)
        good = pred_file[pred_file["best_true_iou"].astype(float) >= 0.5]
        row = {
            "method": method,
            "file": file_path,
            "window_start": window_start,
            "window_end": window_end,
            "true_reps": true_count,
            "predicted_reps": int(len(pred_file)),
            "predicted_with_best_iou_ge_0.50": int(len(good)),
            "mean_best_true_iou": round(float(pred_file["best_true_iou"].astype(float).mean()), 4) if len(pred_file) else 0.0,
        }
        if group:
            row = {**group, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def plot_waveform(
    file_path: str,
    truth: pd.DataFrame,
    runs: list[tuple[str, Path]],
    predictions: dict[str, pd.DataFrame],
    waveform_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    window_start: int,
    window_end: int,
    output_name: str | None = None,
    title_suffix: str | None = None,
) -> Path:
    if file_path not in waveform_cache:
        df = pd.read_csv(file_path)
        signal = principal_motion_signal(df, smooth_window=9)
        if "ax" in df.columns and "ay" in df.columns and "az" in df.columns:
            acc_norm = normalize(np.linalg.norm(df.loc[:, ["ax", "ay", "az"]].to_numpy(dtype=np.float64), axis=1))
        else:
            acc_norm = np.zeros(len(df), dtype=np.float64)
        waveform_cache[file_path] = (normalize(signal), acc_norm)
    signal, acc_norm = waveform_cache[file_path]
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
    add_boundary_lines(axes[0], truth_file, "true_start", "true_end", "#2ca02c", "Ground truth rep")
    axes[0].set_ylabel("truth")
    axes[0].legend(loc="upper right", fontsize=8)

    for ax, (method, run_dir) in zip(axes[1:], runs):
        pred = predictions[method]
        pred_file = overlapping_intervals(pred[pred["file"] == file_path], "start", "end", window_start, window_end)
        ax.plot(x, signal, color="#1f77b4", linewidth=0.9)
        add_boundary_lines(ax, truth_file, "true_start", "true_end", "#2ca02c", "Ground truth")
        add_boundary_lines(ax, pred_file, "start", "end", "#ff7f0e", "Predicted")
        ax.set_ylabel(method)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(
        f"Rep Boundary Comparison on Waveform\n{Path(file_path).name} | samples {window_start}-{window_end}"
        + (f" | {title_suffix}" if title_suffix else ""),
        fontsize=13,
    )
    fig.tight_layout()
    output_path = output_dir / (output_name or f"waveform_method_comparison_{Path(file_path).stem}.png")
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


def set_groups(truth: pd.DataFrame, min_reps: int) -> list[tuple[dict[str, object], pd.DataFrame]]:
    group_cols = ["file", "subject", "exercise", "set_id"]
    groups: list[tuple[dict[str, object], pd.DataFrame]] = []
    for keys, group_df in truth.groupby(group_cols, sort=True):
        if len(group_df) < min_reps:
            continue
        meta = dict(zip(group_cols, keys, strict=True))
        groups.append((meta, group_df.sort_values("true_start").copy()))
    return groups


def plot_all_sets(
    truth: pd.DataFrame,
    runs: list[tuple[str, Path]],
    predictions: dict[str, pd.DataFrame],
    output_dir: Path,
    min_set_reps: int,
    padding_fraction: float,
    max_sets: int | None,
) -> pd.DataFrame:
    set_output_dir = output_dir / "sets_all"
    set_output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    groups = set_groups(truth, min_set_reps)
    if max_sets is not None:
        groups = groups[:max_sets]
    waveform_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for idx, (meta, group_df) in enumerate(groups, start=1):
        start = int(group_df["true_start"].min())
        end = int(group_df["true_end"].max())
        padding = int(round((end - start) * padding_fraction))
        window_start = max(0, start - padding)
        window_end = end + padding
        output_name = (
            f"{idx:03d}_{safe_name(meta['subject'])}_{safe_name(meta['exercise'])}_"
            f"set_{safe_name(meta['set_id'])}.png"
        )
        plot_path = plot_waveform(
            str(meta["file"]),
            truth,
            runs,
            predictions,
            waveform_cache,
            set_output_dir,
            window_start,
            window_end,
            output_name=output_name,
            title_suffix=f"{meta['subject']} / {meta['exercise']} / set {meta['set_id']}",
        )
        summary = file_method_summary(
            str(meta["file"]),
            truth,
            runs,
            predictions,
            window_start,
            window_end,
            group={
                "plot": str(plot_path),
                "subject": meta["subject"],
                "exercise": meta["exercise"],
                "set_id": meta["set_id"],
            },
        )
        rows.append(summary)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot waveform-level rep boundary differences across methods.")
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file", type=str)
    parser.add_argument("--min-reps", type=int, default=6)
    parser.add_argument("--max-reps", type=int, default=14)
    parser.add_argument("--window-reps", type=int, default=10)
    parser.add_argument("--plot-all-sets", action="store_true")
    parser.add_argument("--min-set-reps", type=int, default=1)
    parser.add_argument("--set-padding-fraction", type=float, default=0.15)
    parser.add_argument("--max-sets", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = args.run
    truth = read_truth(runs[0][1])
    predictions = load_predictions(runs)
    if args.plot_all_sets:
        summary = plot_all_sets(
            truth,
            runs,
            predictions,
            args.output_dir,
            min_set_reps=args.min_set_reps,
            padding_fraction=args.set_padding_fraction,
            max_sets=args.max_sets,
        )
        summary.to_csv(args.output_dir / "waveform_method_all_sets_summary.csv", index=False)
        print(f"set_plots={len(summary['plot'].unique()) if not summary.empty else 0}")
        print(f"summary={args.output_dir / 'waveform_method_all_sets_summary.csv'}")
        return

    file_path = args.file or choose_file(truth, args.min_reps, args.max_reps)
    truth_file = truth[truth["file"] == file_path]
    window_start, window_end = choose_window(truth_file, args.window_reps)
    summary = file_method_summary(file_path, truth, runs, predictions, window_start, window_end)
    summary.to_csv(args.output_dir / "waveform_method_file_summary.csv", index=False)
    waveform_path = plot_waveform(file_path, truth, runs, predictions, {}, args.output_dir, window_start, window_end)
    count_path = plot_file_counts(summary, args.output_dir)
    print(f"waveform_plot={waveform_path}")
    print(f"count_plot={count_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
