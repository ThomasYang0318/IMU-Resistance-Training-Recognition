from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_literature_inspired_rep_methods import (  # noqa: E402
    METHOD_SPECS,
    block_features,
    boundaries_from_centers,
    multi_axis_count_vote,
    period_count,
    read_session_9axis,
    refine_boundaries_by_score,
    select_top_peaks,
    uniform_boundaries,
    valley_cost_signal,
)
from tools.evaluate_rep_segmentation_classification import (  # noqa: E402
    RepSegment,
    active_phase_contiguous_blocks_from_truth,
    true_rep_segments,
    truth_segments_for_block,
    whole_session_files,
    write_csv,
)


@dataclass(frozen=True)
class FinalRun:
    source_id: str
    source_name: str
    manifest_path: Path


DEFAULT_FINAL_RUNS = (
    FinalRun(
        "final_010_ugv",
        "Final 010 UGV",
        Path("artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/rep_segmentation_matches.csv"),
    ),
    FinalRun(
        "final_011_mfbs",
        "Final 011 MFBS",
        Path("artifacts_rep_classification/011_multifeature_boundary_score_high_iou/rep_segmentation_matches.csv"),
    ),
    FinalRun(
        "final_014_stayfit_ba",
        "Final 014 STAYFIT-BA",
        Path("artifacts_rep_classification/014_literature_inspired_rep_methods/methods/stayfit_ba/rep_segments_manifest.csv"),
    ),
    FinalRun(
        "final_014_maxxyt_map",
        "Final 014 MAXXYT-MAP",
        Path("artifacts_rep_classification/014_literature_inspired_rep_methods/methods/maxxyt_map/rep_segments_manifest.csv"),
    ),
    FinalRun(
        "final_014_mfitness_fste",
        "Final 014 MFIT-FSTE",
        Path("artifacts_rep_classification/014_literature_inspired_rep_methods/methods/mfitness_fste/rep_segments_manifest.csv"),
    ),
    FinalRun(
        "final_014_cara_dtw_fs",
        "Final 014 CARA-DTW-FS",
        Path("artifacts_rep_classification/014_literature_inspired_rep_methods/methods/cara_dtw_fs/rep_segments_manifest.csv"),
    ),
    FinalRun(
        "final_014_lift_fusion",
        "Final 014 LIFT-Fusion",
        Path("artifacts_rep_classification/014_literature_inspired_rep_methods/methods/lift_fusion/rep_segments_manifest.csv"),
    ),
)


def base_set_id(set_id: str) -> str:
    return str(set_id).split(":active", 1)[0]


def local_internal_boundaries(boundaries: Sequence[int], length: int) -> list[int]:
    return sorted({int(boundary) for boundary in boundaries if 0 < int(boundary) < length})


def true_internal_boundaries(block: RepSegment, true_segments: Sequence[RepSegment]) -> list[dict[str, object]]:
    truth = sorted(truth_segments_for_block(block, true_segments), key=lambda item: (item.start, item.end))
    rows: list[dict[str, object]] = []
    if len(truth) < 2:
        return rows
    for boundary_idx, (left, right) in enumerate(zip(truth[:-1], truth[1:]), start=1):
        boundary = int(round((left.end + right.start) / 2.0))
        if boundary <= block.start or boundary >= block.end:
            continue
        rows.append(
            {
                "file": str(block.file_path),
                "subject": block.subject,
                "exercise": block.exercise,
                "set_id": base_set_id(block.set_id),
                "active_block_id": block.set_id,
                "block_start": block.start,
                "block_end": block.end,
                "block_samples": block.n_samples,
                "boundary_index": boundary_idx,
                "true_boundary": boundary,
                "local_true_boundary": boundary - block.start,
                "left_rep_id": left.rep_id,
                "right_rep_id": right.rep_id,
            }
        )
    return rows


def source_row(
    source_id: str,
    source_name: str,
    source_kind: str,
    candidates: Sequence[int],
    block: RepSegment,
    truth_row: dict[str, object],
    thresholds: Sequence[int],
) -> dict[str, object]:
    local_candidates = local_internal_boundaries(candidates, block.n_samples)
    local_true = int(truth_row["local_true_boundary"])
    if local_candidates:
        distances = np.abs(np.asarray(local_candidates, dtype=int) - local_true)
        best_idx = int(np.argmin(distances))
        nearest = int(local_candidates[best_idx])
        abs_error = int(distances[best_idx])
        signed_error = int(nearest - local_true)
        nearest_global: int | str = block.start + nearest
    else:
        nearest = ""
        nearest_global = ""
        abs_error = ""
        signed_error = ""
    row: dict[str, object] = {
        **truth_row,
        "source_id": source_id,
        "source_name": source_name,
        "source_kind": source_kind,
        "candidate_count": len(local_candidates),
        "nearest_candidate": nearest_global,
        "local_nearest_candidate": nearest,
        "signed_error_samples": signed_error,
        "abs_error_samples": abs_error,
        "candidate_points": " ".join(str(value) for value in local_candidates),
    }
    for threshold in thresholds:
        row[f"within_{threshold}_samples"] = bool(local_candidates and abs_error <= threshold)
    return row


def candidate_sources_for_block(block: RepSegment, df: pd.DataFrame, args: argparse.Namespace) -> dict[str, tuple[str, str, list[int]]]:
    features = block_features(block, df, args)
    signals: dict[str, np.ndarray] = features["signals"]
    period_scores: dict[str, tuple[float | None, float]] = features["periods"]
    pca = features["pca"]
    period, _score = period_scores.get("imu9_pca", (None, 0.0))
    count = period_count(block.n_samples, period, args.min_segment_samples, args.max_reps)
    period_guess = period if period is not None else block.n_samples / max(count, 1)
    search_radius = max(args.min_segment_samples, int(round(period_guess * args.boundary_search_fraction)))

    uniform = local_internal_boundaries(uniform_boundaries(block.n_samples, count), block.n_samples)

    pca_peaks, _ = select_top_peaks(
        pca,
        count,
        args.min_segment_samples,
        period,
        args.peak_prominence_scale,
        distance_scale=0.55,
    )
    pca_midpoints = local_internal_boundaries(boundaries_from_centers(block.n_samples, pca_peaks, count), block.n_samples)

    all_axis_midpoints: set[int] = set()
    axis_peak_points: set[int] = set()
    for name, signal in signals.items():
        signal_period, _signal_score = period_scores.get(name, (None, 0.0))
        signal_count = period_count(block.n_samples, signal_period, args.min_segment_samples, args.max_reps)
        peaks, _strength = select_top_peaks(
            signal,
            signal_count,
            args.min_segment_samples,
            signal_period,
            args.peak_prominence_scale,
            distance_scale=0.55,
        )
        axis_peak_points.update(int(value) for value in peaks)
        all_axis_midpoints.update(local_internal_boundaries(boundaries_from_centers(block.n_samples, peaks, signal_count), block.n_samples))

    consensus_count, _chosen_name, chosen_peaks, _vote_score = multi_axis_count_vote(
        signals,
        block.n_samples,
        args.min_segment_samples,
        args.max_reps,
        args.peak_prominence_scale,
        period_scores=period_scores,
    )
    if len(chosen_peaks):
        multi_axis_consensus = local_internal_boundaries(boundaries_from_centers(block.n_samples, chosen_peaks, consensus_count), block.n_samples)
    else:
        multi_axis_consensus = local_internal_boundaries(uniform_boundaries(block.n_samples, consensus_count), block.n_samples)

    gyro_valley = local_internal_boundaries(
        refine_boundaries_by_score(
            uniform_boundaries(block.n_samples, count),
            features["gyro_score"],
            args.min_segment_samples,
            search_radius,
        ),
        block.n_samples,
    )
    energy_valley = local_internal_boundaries(
        refine_boundaries_by_score(
            uniform_boundaries(block.n_samples, count),
            valley_cost_signal(np.abs(features["energy"])),
            args.min_segment_samples,
            search_radius,
        ),
        block.n_samples,
    )
    fusion_refined = local_internal_boundaries(
        refine_boundaries_by_score(
            uniform_boundaries(block.n_samples, count),
            features["boundary_score"],
            args.min_segment_samples,
            search_radius,
        ),
        block.n_samples,
    )

    gyro_raw, _ = find_peaks(-features["gyro_score"], distance=max(3, args.min_segment_samples // 2))
    energy_raw, _ = find_peaks(-valley_cost_signal(np.abs(features["energy"])), distance=max(3, args.min_segment_samples // 2))
    raw_valleys = local_internal_boundaries(list(gyro_raw) + list(energy_raw), block.n_samples)

    fusion_pool = local_internal_boundaries(
        list(uniform)
        + list(pca_midpoints)
        + list(all_axis_midpoints)
        + list(multi_axis_consensus)
        + list(gyro_valley)
        + list(energy_valley)
        + list(fusion_refined),
        block.n_samples,
    )

    return {
        "uniform_autocorr_priors": ("Uniform Autocorr Priors", "candidate_pool", uniform),
        "pca_peak_midpoints": ("PCA Peak Midpoints", "candidate_pool", pca_midpoints),
        "multi_axis_all_midpoints": ("Multi-Axis All Midpoints", "candidate_pool", sorted(all_axis_midpoints)),
        "multi_axis_consensus": ("Multi-Axis Consensus", "candidate_pool", multi_axis_consensus),
        "gyro_valley_at_priors": ("Gyro Valley at Priors", "candidate_pool", gyro_valley),
        "energy_valley_at_priors": ("Energy Valley at Priors", "candidate_pool", energy_valley),
        "fusion_refined_score": ("Fusion Refined Score", "candidate_pool", fusion_refined),
        "raw_gyro_energy_valleys": ("Raw Gyro + Energy Valleys", "dense_candidate_pool", raw_valleys),
        "fusion_candidate_pool": ("Fusion Candidate Pool", "dense_candidate_pool", fusion_pool),
    }


def read_final_boundaries(manifest_path: Path) -> dict[tuple[str, str, str, str], list[tuple[int, int]]]:
    if not manifest_path.exists():
        return {}
    try:
        df = pd.read_csv(manifest_path)
    except pd.errors.EmptyDataError:
        return {}
    exercise_col = "exercise" if "exercise" in df.columns else "exercise_hint" if "exercise_hint" in df.columns else None
    if exercise_col is None:
        return {}
    required = {"file", "subject", exercise_col, "start", "end"}
    if not required.issubset(df.columns):
        return {}
    if "set_id" not in df.columns:
        df["set_id"] = "*"
    if exercise_col != "exercise":
        df["exercise"] = df[exercise_col]
    grouped: dict[tuple[str, str, str, str], list[tuple[int, int]]] = {}
    for key, group in df.groupby(["file", "subject", "exercise", "set_id"], sort=True):
        segments = sorted((int(row.start), int(row.end)) for row in group.itertuples(index=False))
        grouped[(str(key[0]), str(key[1]), str(key[2]), str(key[3]))] = segments
    return grouped


def final_candidates_for_block(block: RepSegment, grouped: dict[tuple[str, str, str, str], list[tuple[int, int]]]) -> list[int]:
    key = (str(block.file_path), block.subject, block.exercise, base_set_id(block.set_id))
    wildcard_key = (str(block.file_path), block.subject, block.exercise, "*")
    local: list[int] = []
    for start, end in grouped.get(key, []) + grouped.get(wildcard_key, []):
        if min(end, block.end) <= max(start, block.start):
            continue
        if block.start < end < block.end:
            local.append(end - block.start)
    return local_internal_boundaries(local, block.n_samples)


def summarize_recall(rows: pd.DataFrame, group_cols: Sequence[str], thresholds: Sequence[int]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    data = rows.copy()
    data["abs_error_numeric"] = pd.to_numeric(data["abs_error_samples"], errors="coerce")
    grouped = data.groupby(list(group_cols), sort=True)
    summary = grouped.agg(
        true_boundaries=("local_true_boundary", "count"),
        mean_candidates=("candidate_count", "mean"),
        median_candidates=("candidate_count", "median"),
        median_abs_error_samples=("abs_error_numeric", "median"),
        p80_abs_error_samples=("abs_error_numeric", lambda x: float(np.nanpercentile(x, 80)) if x.notna().any() else np.nan),
        p90_abs_error_samples=("abs_error_numeric", lambda x: float(np.nanpercentile(x, 90)) if x.notna().any() else np.nan),
    ).reset_index()
    for threshold in thresholds:
        col = f"within_{threshold}_samples"
        summary[f"recall_within_{threshold}"] = grouped[col].mean().to_numpy(dtype=float)
    for col in summary.columns:
        if col.startswith("recall_") or col in {"mean_candidates", "median_candidates", "median_abs_error_samples", "p80_abs_error_samples", "p90_abs_error_samples"}:
            summary[col] = summary[col].astype(float).round(4)
    return summary


def plot_recall_bars(summary: pd.DataFrame, output_dir: Path, thresholds: Sequence[int]) -> None:
    if summary.empty:
        return
    selected = summary.sort_values([f"recall_within_{thresholds[1]}", "median_abs_error_samples"], ascending=[False, True])
    sources = selected["source_name"].astype(str).tolist()
    x = np.arange(len(sources))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(12, len(sources) * 0.8), 5.8))
    for idx, threshold in enumerate(thresholds):
        col = f"recall_within_{threshold}"
        ax.bar(x + (idx - (len(thresholds) - 1) / 2) * width, selected[col].astype(float).to_numpy(), width, label=f"+/-{threshold} samples")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Candidate recall")
    ax.set_title("Boundary Candidate Recall by Source")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "015_candidate_recall_by_source.png", dpi=180)
    plt.close(fig)


def plot_error_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    selected = summary.sort_values("median_abs_error_samples", ascending=True)
    fig, ax = plt.subplots(figsize=(max(12, len(selected) * 0.8), 5.6))
    ax.bar(selected["source_name"].astype(str), selected["median_abs_error_samples"].astype(float), label="Median")
    ax.scatter(selected["source_name"].astype(str), selected["p90_abs_error_samples"].astype(float), color="#d62728", label="P90", zorder=3)
    ax.set_ylabel("Nearest candidate error (samples)")
    ax.set_title("Nearest Boundary Candidate Error")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "015_candidate_nearest_error_by_source.png", dpi=180)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, output_dir: Path, threshold: int, group_col: str) -> None:
    if summary.empty:
        return
    value_col = f"recall_within_{threshold}"
    pivot = summary.pivot(index=group_col, columns="source_name", values=value_col).fillna(0.0)
    columns = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[:, columns]
    fig, ax = plt.subplots(figsize=(max(12, len(columns) * 0.85), max(5, len(pivot.index) * 0.45)))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Candidate Recall +/-{threshold} Samples by {group_col.title()}")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value >= 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=value_col)
    fig.tight_layout()
    fig.savefig(output_dir / f"015_candidate_recall_{group_col}_within_{threshold}.png", dpi=180)
    plt.close(fig)


def plot_error_hist(rows: pd.DataFrame, output_dir: Path, source_ids: Sequence[str]) -> None:
    if rows.empty:
        return
    data = rows[rows["source_id"].isin(source_ids)].copy()
    data["abs_error_numeric"] = pd.to_numeric(data["abs_error_samples"], errors="coerce")
    data = data.dropna(subset=["abs_error_numeric"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bins = np.arange(0, min(251, int(data["abs_error_numeric"].max()) + 10), 10)
    if len(bins) < 3:
        bins = np.arange(0, 101, 10)
    for source_name, group in data.groupby("source_name", sort=True):
        ax.hist(group["abs_error_numeric"].to_numpy(dtype=float), bins=bins, alpha=0.42, label=source_name)
    ax.set_xlabel("Nearest candidate absolute error (samples)")
    ax.set_ylabel("Boundary count")
    ax.set_title("Boundary Error Distribution for Key Sources")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "015_key_source_error_distribution.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_cache: dict[Path, pd.DataFrame] = {}
    truth: list[RepSegment] = []
    for path in whole_session_files(args.data_dirs):
        df = read_session_9axis(path, args.data_dirs)
        if not {"ax", "ay", "az", "gx", "gy", "gz"}.issubset(df.columns):
            continue
        session_cache[path] = df
        truth.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))
    blocks = active_phase_contiguous_blocks_from_truth(truth, min_samples=args.min_segment_samples)
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]

    final_runs = [run for run in DEFAULT_FINAL_RUNS if run.manifest_path.exists()]
    final_boundaries = {run.source_id: read_final_boundaries(run.manifest_path) for run in final_runs}

    rows: list[dict[str, object]] = []
    true_boundary_count = 0
    for block in blocks:
        df = session_cache[block.file_path]
        truth_rows = true_internal_boundaries(block, truth)
        if not truth_rows:
            continue
        true_boundary_count += len(truth_rows)
        feature_sources = candidate_sources_for_block(block, df, args)
        for truth_row in truth_rows:
            for source_id, (source_name, source_kind, candidates) in feature_sources.items():
                rows.append(source_row(source_id, source_name, source_kind, candidates, block, truth_row, args.thresholds))
            for run in final_runs:
                candidates = final_candidates_for_block(block, final_boundaries.get(run.source_id, {}))
                rows.append(source_row(run.source_id, run.source_name, "final_boundary", candidates, block, truth_row, args.thresholds))

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "015_boundary_candidate_recall_details.csv", index=False)
    overall = summarize_recall(detail, ["source_id", "source_name", "source_kind"], args.thresholds)
    by_exercise = summarize_recall(detail, ["exercise", "source_id", "source_name", "source_kind"], args.thresholds)
    by_subject = summarize_recall(detail, ["subject", "source_id", "source_name", "source_kind"], args.thresholds)
    overall.to_csv(args.output_dir / "015_boundary_candidate_recall_summary.csv", index=False)
    by_exercise.to_csv(args.output_dir / "015_boundary_candidate_recall_by_exercise.csv", index=False)
    by_subject.to_csv(args.output_dir / "015_boundary_candidate_recall_by_subject.csv", index=False)

    plot_recall_bars(overall, args.output_dir, args.thresholds)
    plot_error_bars(overall, args.output_dir)
    plot_heatmap(by_exercise, args.output_dir, args.primary_threshold, "exercise")
    plot_heatmap(by_subject, args.output_dir, args.primary_threshold, "subject")
    plot_error_hist(
        detail,
        args.output_dir,
        source_ids=[
            "gyro_valley_at_priors",
            "multi_axis_consensus",
            "fusion_candidate_pool",
            "final_014_maxxyt_map",
            "final_014_lift_fusion",
        ],
    )

    best_sources = overall.sort_values([f"recall_within_{args.primary_threshold}", "median_abs_error_samples"], ascending=[False, True]).head(8)
    summary = {
        "output_dir": str(args.output_dir),
        "data_dirs": [str(path) for path in args.data_dirs],
        "active_blocks": len(blocks),
        "true_internal_boundaries": int(true_boundary_count),
        "thresholds_samples": args.thresholds,
        "primary_threshold": args.primary_threshold,
        "best_sources": best_sources.to_dict(orient="records"),
        "interpretation": {
            "candidate_pool": "Feature-derived candidate cut points before final decoding.",
            "final_boundary": "Actual selected internal boundaries emitted by an evaluated method.",
            "high_candidate_recall_low_final_recall": "The feature pool contains useful points, but scoring/decoding is choosing the wrong point.",
            "low_candidate_recall": "The waveform features do not put candidate points near the label; new features or labels are needed.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(overall.sort_values([f"recall_within_{args.primary_threshold}", "median_abs_error_samples"], ascending=[False, True]).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate recall around true internal rep boundaries.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/015_boundary_candidate_recall_analysis"))
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--energy-window", type=int, default=21)
    parser.add_argument("--peak-prominence-scale", type=float, default=0.28)
    parser.add_argument("--boundary-search-fraction", type=float, default=0.38)
    parser.add_argument("--max-period-fraction", type=float, default=0.80)
    parser.add_argument("--max-reps", type=int, default=40)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--primary-threshold", type=int, default=20)
    parser.add_argument("--max-blocks", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
