from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_literature_inspired_rep_methods import (  # noqa: E402
    IMU9_COLUMNS,
    infer_sensor_period_seconds,
    principal_signal,
    read_session_9axis,
    safe_name,
)
from tools.evaluate_rep_segmentation_classification import (  # noqa: E402
    PhaseSegment,
    RepSegment,
    active_phase_contiguous_blocks_from_truth,
    phase_iou,
    phase_metric_rows,
    phase_metric_rows_by_phase,
    phase_order_by_exercise,
    plot_phase_metrics,
    plot_phase_metrics_by_phase,
    predict_phase_segments,
    robust_zscore,
    true_phase_segments,
    true_rep_segments,
    whole_session_files,
    write_csv,
)


PHASE_COLORS = {
    "concentric": "#d95f02",
    "eccentric": "#1b9e77",
}


def read_predicted_reps(path: Path) -> list[RepSegment]:
    df = pd.read_csv(path)
    segments: list[RepSegment] = []
    for row in df.itertuples(index=False):
        segments.append(
            RepSegment(
                Path(str(row.file)),
                str(row.subject),
                str(row.exercise),
                str(row.set_id),
                str(row.rep_id),
                int(row.start),
                int(row.end),
                str(getattr(row, "source", "prediction")),
            )
        )
    return segments


def same_block_phase_segments(block: RepSegment, phases: Sequence[PhaseSegment]) -> list[PhaseSegment]:
    base_set_id = block.set_id.split(":active", 1)[0]
    return [
        segment
        for segment in phases
        if segment.file_path == block.file_path
        and segment.subject == block.subject
        and segment.exercise == block.exercise
        and str(segment.set_id).split(":active", 1)[0] == base_set_id
        and min(segment.end, block.end) > max(segment.start, block.start)
    ]


def metric_rows_by_key(
    predicted: Sequence[PhaseSegment],
    truth: Sequence[PhaseSegment],
    thresholds: Sequence[float],
    key_name: str,
) -> list[dict[str, object]]:
    if key_name == "exercise":
        keys = sorted({segment.exercise for segment in truth})
        key_fn = lambda segment: segment.exercise
    elif key_name == "subject":
        keys = sorted({segment.subject for segment in truth})
        key_fn = lambda segment: segment.subject
    else:
        raise ValueError(f"Unsupported key: {key_name}")

    rows: list[dict[str, object]] = []
    for key in keys:
        truth_subset = [segment for segment in truth if key_fn(segment) == key]
        pred_subset = [segment for segment in predicted if key_fn(segment) == key]
        for row in phase_metric_rows(pred_subset, truth_subset, thresholds):
            rows.append({key_name: key, **row})
    return rows


def best_phase_match_rows(predicted: Sequence[PhaseSegment], truth: Sequence[PhaseSegment]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    truth_by_file_phase: dict[tuple[Path, str], list[PhaseSegment]] = {}
    for segment in truth:
        truth_by_file_phase.setdefault((segment.file_path, segment.phase), []).append(segment)

    for segment in predicted:
        candidates = truth_by_file_phase.get((segment.file_path, segment.phase), [])
        best = max(candidates, key=lambda item: phase_iou(segment, item), default=None)
        best_iou = phase_iou(segment, best) if best is not None else 0.0
        rows.append(
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "exercise": segment.exercise,
                "set_id": segment.set_id,
                "rep_id": segment.rep_id,
                "phase": segment.phase,
                "pred_start": segment.start,
                "pred_end": segment.end,
                "best_true_start": best.start if best is not None else "",
                "best_true_end": best.end if best is not None else "",
                "best_iou": round(best_iou, 4),
            }
        )
    return rows


def phase_duration_rows(
    predicted: Sequence[PhaseSegment],
    truth: Sequence[PhaseSegment],
    periods: dict[Path, float],
) -> list[dict[str, object]]:
    grouped: dict[tuple[Path, str, str, str, str], dict[str, float]] = {}
    for source, segments in (("true", truth), ("pred", predicted)):
        for segment in segments:
            key = (segment.file_path, segment.subject, segment.exercise, str(segment.set_id).split(":active", 1)[0], segment.phase)
            row = grouped.setdefault(
                key,
                {
                    "true_samples": 0.0,
                    "pred_samples": 0.0,
                },
            )
            row[f"{source}_samples"] += float(segment.n_samples)

    rows: list[dict[str, object]] = []
    for (file_path, subject, exercise, set_id, phase), values in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2], item[0][3], item[0][4])
    ):
        period = periods.get(file_path, 0.01)
        true_sec = values["true_samples"] * period
        pred_sec = values["pred_samples"] * period
        rows.append(
            {
                "file": str(file_path),
                "subject": subject,
                "exercise": exercise,
                "set_id": set_id,
                "phase": phase,
                "true_phase_tut_sec": round(true_sec, 4),
                "pred_phase_tut_sec": round(pred_sec, 4),
                "phase_tut_error_sec": round(pred_sec - true_sec, 4),
                "phase_tut_abs_error_sec": round(abs(pred_sec - true_sec), 4),
            }
        )
    return rows


def summarize_phase_duration(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    output: list[dict[str, object]] = []
    for phase, group in df.groupby("phase", sort=True):
        output.append(
            {
                "phase": phase,
                "sets": int(len(group)),
                "phase_tut_mae_sec": round(float(group["phase_tut_abs_error_sec"].mean()), 4),
                "phase_tut_bias_sec": round(float(group["phase_tut_error_sec"].mean()), 4),
                "phase_tut_median_abs_error_sec": round(float(group["phase_tut_abs_error_sec"].median()), 4),
            }
        )
    output.append(
        {
            "phase": "overall",
            "sets": int(len(df)),
            "phase_tut_mae_sec": round(float(df["phase_tut_abs_error_sec"].mean()), 4),
            "phase_tut_bias_sec": round(float(df["phase_tut_error_sec"].mean()), 4),
            "phase_tut_median_abs_error_sec": round(float(df["phase_tut_abs_error_sec"].median()), 4),
        }
    )
    return output


def plot_phase_waveform(
    block: RepSegment,
    truth: Sequence[PhaseSegment],
    predicted: Sequence[PhaseSegment],
    df: pd.DataFrame,
    output_path: Path,
    smooth_window: int,
) -> None:
    local_df = df.iloc[block.start : block.end]
    signal = robust_zscore(principal_signal(local_df, smooth_window, IMU9_COLUMNS))
    x = np.arange(block.n_samples)
    rows = [
        ("Ground truth", truth),
        ("Prediction", predicted),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True)
    for ax, (label, segments) in zip(axes, rows):
        ax.plot(x, signal, color="#39424e", linewidth=0.85)
        for segment in sorted(segments, key=lambda item: (item.start, item.end)):
            left = max(0, segment.start - block.start)
            right = min(block.n_samples, segment.end - block.start)
            if right <= left:
                continue
            color = PHASE_COLORS.get(segment.phase, "#7570b3")
            ax.axvline(left, color=color, linewidth=0.9, alpha=0.95)
            ax.axvline(right, color=color, linewidth=0.9, alpha=0.55)
            mid = (left + right) / 2.0
            tag = "C" if segment.phase == "concentric" else "E" if segment.phase == "eccentric" else segment.phase[:1].upper()
            y = 0.92 if segment.phase == "concentric" else 0.74
            ax.text(mid, y, tag, color=color, fontsize=7, ha="center", va="center", transform=ax.get_xaxis_transform())
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=74, fontsize=8)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.16)

    handles = [
        plt.Line2D([0], [0], color=PHASE_COLORS["concentric"], lw=1.4, label="concentric"),
        plt.Line2D([0], [0], color=PHASE_COLORS["eccentric"], lw=1.4, label="eccentric"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Sample in active set")
    axes[0].set_title(f"{block.subject} | {block.exercise} | set {block.set_id}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_outputs(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_cache: dict[Path, pd.DataFrame] = {}
    true_reps: list[RepSegment] = []
    true_phases: list[PhaseSegment] = []
    periods: dict[Path, float] = {}

    for path in whole_session_files(args.data_dirs):
        df = read_session_9axis(path, args.data_dirs)
        session_cache[path] = df
        periods[path] = infer_sensor_period_seconds(df)
        true_reps.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))
        true_phases.extend(true_phase_segments(df, path, min_samples=args.min_phase_segment_samples))

    predicted_reps = read_predicted_reps(args.predicted_reps_csv)
    phase_orders = phase_order_by_exercise(true_phases)
    predicted_phases = predict_phase_segments(
        predicted_reps,
        session_cache,
        phase_orders,
        method=args.phase_split_method,
        smooth_window=args.smooth_window,
        min_phase_samples=args.min_phase_segment_samples,
    )

    phase_rows = phase_metric_rows(predicted_phases, true_phases, args.phase_iou_thresholds)
    by_phase = phase_metric_rows_by_phase(predicted_phases, true_phases, args.phase_iou_thresholds)
    by_exercise = metric_rows_by_key(predicted_phases, true_phases, args.phase_iou_thresholds, "exercise")
    by_subject = metric_rows_by_key(predicted_phases, true_phases, args.phase_iou_thresholds, "subject")
    duration_rows = phase_duration_rows(predicted_phases, true_phases, periods)
    duration_summary = summarize_phase_duration(duration_rows)

    write_csv(args.output_dir / "phase_split_metrics.csv", phase_rows)
    write_csv(args.output_dir / "phase_split_metrics_by_phase.csv", by_phase)
    write_csv(args.output_dir / "phase_split_metrics_by_exercise.csv", by_exercise)
    write_csv(args.output_dir / "phase_split_metrics_by_subject.csv", by_subject)
    write_csv(args.output_dir / "phase_split_best_matches.csv", best_phase_match_rows(predicted_phases, true_phases))
    write_csv(args.output_dir / "phase_tut_error_by_set_phase.csv", duration_rows)
    write_csv(args.output_dir / "phase_tut_error_summary.csv", duration_summary)
    plot_phase_metrics(phase_rows, args.output_dir)
    plot_phase_metrics_by_phase(by_phase, args.output_dir)

    blocks = active_phase_contiguous_blocks_from_truth(true_reps, min_samples=args.min_segment_samples)
    waveform_dir = args.output_dir / "phase_waveforms"
    for index, block in enumerate(blocks):
        if args.max_plots is not None and index >= args.max_plots:
            break
        df = session_cache[block.file_path]
        truth_for_block = same_block_phase_segments(block, true_phases)
        pred_for_block = same_block_phase_segments(block, predicted_phases)
        filename = safe_name(f"{index + 1:03d}_{block.subject}_{block.exercise}_{block.set_id}_{block.file_path.stem}") + ".png"
        plot_phase_waveform(block, truth_for_block, pred_for_block, df, waveform_dir / filename, args.smooth_window)

    summary = {
        "output_dir": str(args.output_dir),
        "predicted_reps_csv": str(args.predicted_reps_csv),
        "phase_split_method": args.phase_split_method,
        "true_phase_segments": len(true_phases),
        "predicted_phase_segments": len(predicted_phases),
        "phase_orders_by_exercise": phase_orders,
        "waveform_plots": str(waveform_dir),
        "max_plots": args.max_plots,
        "metrics_csv": str(args.output_dir / "phase_split_metrics.csv"),
        "duration_summary_csv": str(args.output_dir / "phase_tut_error_summary.csv"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(phase_rows).to_string(index=False))
    print()
    print(pd.DataFrame(duration_summary).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot concentric/eccentric phase split diagnostics from predicted rep segments.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument(
        "--predicted-reps-csv",
        type=Path,
        default=Path("artifacts_rep_classification/016_dense_candidate_dp_decoder/methods/dcp_dp_fs/rep_segments_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/017_phase_split_dcp_dp_fs"))
    parser.add_argument("--phase-split-method", choices=["midpoint", "pca-reversal"], default="pca-reversal")
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--min-phase-segment-samples", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--phase-iou-thresholds", type=float, nargs="+", default=[0.50, 0.75, 0.90])
    parser.add_argument("--max-plots", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    build_outputs(parse_args())


if __name__ == "__main__":
    main()
