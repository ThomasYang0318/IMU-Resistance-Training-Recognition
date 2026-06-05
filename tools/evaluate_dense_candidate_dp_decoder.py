from __future__ import annotations

import argparse
import json
import math
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
    ACC_COLUMNS,
    GYRO_COLUMNS,
    IMU9_COLUMNS,
    MethodSpec,
    block_features,
    boundaries_from_centers,
    build_templates,
    infer_sensor_period_seconds,
    magnitude_signal,
    multi_axis_count_vote,
    period_count,
    plot_method_bars,
    plot_method_exercise_heatmap,
    plot_method_subject_heatmap,
    plot_method_table,
    plot_waveform_comparison,
    predictions_for_block,
    read_session_9axis,
    safe_name,
    select_top_peaks,
    template_for_block,
    uniform_boundaries,
    valley_cost_signal,
    weighted_integer_median,
    write_method_outputs,
)
from tools.evaluate_rep_segmentation_classification import (  # noqa: E402
    PhaseSegment,
    RepSegment,
    active_phase_contiguous_blocks_from_truth,
    phase_order_by_exercise,
    segmentation_metric_rows_by_exercise,
    segmentation_metric_rows_by_subject,
    true_phase_segments,
    true_rep_segments,
    truth_segments_for_block,
    whole_session_files,
    write_csv,
)


METHOD_SPECS_016 = (
    MethodSpec(
        "dcp_dp",
        "DCP-DP",
        "Dense Candidate Pool Dynamic Programming",
        "Use dense gyro/energy valleys plus multi-axis midpoint candidates, then decode a whole set with dynamic programming.",
        "Does not personalize duration or boundary offset for a new subject.",
        "Replaces uniform-prior local valley picking with sequence-level candidate selection.",
        False,
    ),
    MethodSpec(
        "dcp_dp_fs",
        "DCP-DP-FS",
        "Few-shot Dense Candidate Pool Dynamic Programming",
        "DCP-DP with per-subject/exercise few-shot duration calibration from the first labeled reps.",
        "Still assumes active-only exercise spans and a small calibration label set.",
        "Uses few-shot duration scale to improve count and DP duration constraints.",
        True,
    ),
)


@dataclass(frozen=True)
class CandidatePool:
    points: np.ndarray
    base_cost: np.ndarray
    votes: np.ndarray
    labels: list[str]


def local_internal(boundaries: Sequence[int], length: int) -> list[int]:
    return sorted({int(value) for value in boundaries if 0 < int(value) < length})


def robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    scale = high - low
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / scale, 0.0, 1.0)


def merge_labeled_points(
    labeled_points: list[tuple[int, str]],
    base_signal: np.ndarray,
    length: int,
    cluster_radius: int,
) -> CandidatePool:
    valid = sorted((int(point), label) for point, label in labeled_points if 0 < int(point) < length)
    if not valid:
        return CandidatePool(np.array([], dtype=int), np.array([], dtype=np.float64), np.array([], dtype=np.float64), [])

    clusters: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = [valid[0]]
    for point, label in valid[1:]:
        if point - current[-1][0] <= cluster_radius:
            current.append((point, label))
        else:
            clusters.append(current)
            current = [(point, label)]
    clusters.append(current)

    points: list[int] = []
    costs: list[float] = []
    votes: list[float] = []
    labels: list[str] = []
    for cluster in clusters:
        unique_labels = sorted({label for _point, label in cluster})
        cluster_points = [point for point, _label in cluster]
        best_point = min(cluster_points, key=lambda point: float(base_signal[min(max(point, 0), length - 1)]))
        points.append(best_point)
        costs.append(float(base_signal[best_point]))
        votes.append(float(len(unique_labels)))
        labels.append("|".join(unique_labels))
    order = np.argsort(points)
    return CandidatePool(
        np.asarray(points, dtype=int)[order],
        np.asarray(costs, dtype=np.float64)[order],
        np.asarray(votes, dtype=np.float64)[order],
        [labels[int(idx)] for idx in order],
    )


def dense_candidate_pool(block: RepSegment, df: pd.DataFrame, args: argparse.Namespace) -> tuple[CandidatePool, dict[str, object]]:
    features = block_features(block, df, args)
    signals: dict[str, np.ndarray] = features["signals"]
    period_scores: dict[str, tuple[float | None, float]] = features["periods"]
    length = block.n_samples

    pca = features["pca"]
    pca_period, _pca_score = period_scores.get("imu9_pca", (None, 0.0))
    pca_count = period_count(length, pca_period, args.min_segment_samples, args.max_reps)
    consensus_count, _chosen_name, chosen_peaks, _vote_score = multi_axis_count_vote(
        signals,
        length,
        args.min_segment_samples,
        args.max_reps,
        args.peak_prominence_scale,
        period_scores=period_scores,
    )

    energy_low = valley_cost_signal(np.abs(features["energy"]))
    gyro_low = features["gyro_score"]
    acc_low = valley_cost_signal(magnitude_signal(features["local_df"], ACC_COLUMNS, args.smooth_window))
    base_signal = robust_unit(0.46 * gyro_low + 0.32 * features["boundary_score"] + 0.16 * energy_low + 0.06 * acc_low)

    labeled: list[tuple[int, str]] = []
    for point in local_internal(uniform_boundaries(length, pca_count), length):
        labeled.append((point, "uniform"))
    pca_peaks, _ = select_top_peaks(
        pca,
        pca_count,
        args.min_segment_samples,
        pca_period,
        args.peak_prominence_scale,
        distance_scale=0.55,
    )
    for point in local_internal(boundaries_from_centers(length, pca_peaks, pca_count), length):
        labeled.append((point, "pca_midpoint"))
    if len(chosen_peaks):
        for point in local_internal(boundaries_from_centers(length, chosen_peaks, consensus_count), length):
            labeled.append((point, "multi_axis_consensus"))
    else:
        for point in local_internal(uniform_boundaries(length, consensus_count), length):
            labeled.append((point, "multi_axis_consensus"))

    for name, signal in signals.items():
        signal_period, _score = period_scores.get(name, (None, 0.0))
        signal_count = period_count(length, signal_period, args.min_segment_samples, args.max_reps)
        peaks, _ = select_top_peaks(
            signal,
            signal_count,
            args.min_segment_samples,
            signal_period,
            args.peak_prominence_scale,
            distance_scale=0.55,
        )
        for point in local_internal(boundaries_from_centers(length, peaks, signal_count), length):
            labeled.append((point, f"axis_mid_{name}"))

    raw_distance = max(3, args.raw_candidate_min_distance)
    for label, score in (
        ("raw_fused_valley", base_signal),
        ("raw_gyro_valley", gyro_low),
        ("raw_energy_valley", energy_low),
    ):
        valleys, _ = find_peaks(-score, distance=raw_distance)
        for point in valleys:
            labeled.append((int(point), label))

    pool = merge_labeled_points(labeled, base_signal, length, args.cluster_radius)
    meta = {
        "features": features,
        "pca_count": pca_count,
        "consensus_count": consensus_count,
        "pca_period": pca_period,
        "base_signal": base_signal,
    }
    return pool, meta


def estimate_count(
    block: RepSegment,
    pool_meta: dict[str, object],
    args: argparse.Namespace,
    use_few_shot: bool,
    template_duration: float | None,
) -> tuple[int, float | None]:
    length = block.n_samples
    values = [int(pool_meta["pca_count"]), int(pool_meta["consensus_count"])]
    weights = [1.0, 1.7]
    period = pool_meta.get("pca_period")
    if period is not None and isinstance(period, float) and period > 0:
        values.append(period_count(length, period, args.min_segment_samples, args.max_reps))
        weights.append(1.2)
    target_duration = None
    if use_few_shot and template_duration is not None and template_duration > 0:
        t_count = int(np.clip(round(length / template_duration), 1, max(1, min(args.max_reps, length // max(args.min_segment_samples, 1)))))
        values.extend([t_count, t_count])
        weights.extend([1.8, 1.2])
        target_duration = float(template_duration)
    count = weighted_integer_median(values, weights)
    count = int(np.clip(count, 1, max(1, min(args.max_reps, length // max(args.min_segment_samples, 1)))))
    return count, target_duration


def slot_indices(
    pool: CandidatePool,
    boundary_idx: int,
    count: int,
    length: int,
    target_duration: float,
    args: argparse.Namespace,
) -> np.ndarray:
    target = boundary_idx * length / float(count)
    radius = max(args.slot_search_min_samples, int(round(target_duration * args.slot_search_fraction)))
    lo = max(args.min_segment_samples * boundary_idx, int(round(target - radius)))
    hi = min(length - args.min_segment_samples * (count - boundary_idx), int(round(target + radius)))
    if hi <= lo:
        return np.array([], dtype=int)
    candidates = np.flatnonzero((pool.points >= lo) & (pool.points <= hi))
    if len(candidates) == 0:
        nearest = int(np.argmin(np.abs(pool.points - target))) if len(pool.points) else -1
        return np.asarray([nearest], dtype=int) if nearest >= 0 else np.array([], dtype=int)
    source_bonus = np.minimum(pool.votes[candidates], args.vote_cap) / args.vote_cap
    target_cost = ((pool.points[candidates].astype(np.float64) - target) / max(radius, 1)) ** 2
    rank_cost = pool.base_cost[candidates] + args.target_cost_weight * target_cost - args.vote_bonus_weight * source_bonus
    keep = min(args.max_slot_candidates, len(candidates))
    return candidates[np.argsort(rank_cost)[:keep]]


def decode_boundaries_dp(
    pool: CandidatePool,
    length: int,
    count: int,
    target_duration: float,
    args: argparse.Namespace,
) -> tuple[list[int], float]:
    if count <= 1 or len(pool.points) == 0:
        return [0, length], 0.0

    slot_lists = [slot_indices(pool, slot, count, length, target_duration, args) for slot in range(1, count)]
    if any(len(slot) == 0 for slot in slot_lists):
        return uniform_boundaries(length, count), float("inf")

    min_duration = max(args.min_segment_samples, int(round(target_duration * args.min_duration_scale)))
    max_duration = max(min_duration, int(round(target_duration * args.max_duration_scale)))

    prev_positions = np.asarray([0], dtype=int)
    prev_costs = np.asarray([0.0], dtype=np.float64)
    prev_paths: list[list[int]] = [[]]

    for slot in slot_lists:
        positions = pool.points[slot]
        boundary_cost = pool.base_cost[slot] - args.vote_bonus_weight * np.minimum(pool.votes[slot], args.vote_cap) / args.vote_cap
        costs = np.full(len(slot), np.inf, dtype=np.float64)
        paths: list[list[int]] = [[] for _ in range(len(slot))]
        for cand_idx, position in enumerate(positions):
            durations = position - prev_positions
            feasible = (durations >= min_duration) & (durations <= max_duration)
            if not np.any(feasible):
                continue
            duration_costs = args.duration_cost_weight * ((durations.astype(np.float64) - target_duration) / max(target_duration, 1.0)) ** 2
            candidate_costs = prev_costs + duration_costs
            candidate_costs[~feasible] = np.inf
            best_prev = int(np.argmin(candidate_costs))
            costs[cand_idx] = float(candidate_costs[best_prev] + boundary_cost[cand_idx])
            paths[cand_idx] = prev_paths[best_prev] + [int(position)]
        finite = np.isfinite(costs)
        if not np.any(finite):
            return uniform_boundaries(length, count), float("inf")
        order = np.argsort(costs[finite])
        finite_indices = np.flatnonzero(finite)[order[: args.max_states_per_stage]]
        prev_positions = positions[finite_indices]
        prev_costs = costs[finite_indices]
        prev_paths = [paths[int(idx)] for idx in finite_indices]

    final_durations = length - prev_positions
    feasible_final = (final_durations >= min_duration) & (final_durations <= max_duration)
    final_costs = prev_costs + args.duration_cost_weight * ((final_durations.astype(np.float64) - target_duration) / max(target_duration, 1.0)) ** 2
    final_costs[~feasible_final] = np.inf
    if not np.any(np.isfinite(final_costs)):
        best_idx = int(np.argmin(prev_costs))
        return [0, *prev_paths[best_idx], length], float(prev_costs[best_idx] / max(count, 1))
    best_idx = int(np.argmin(final_costs))
    return [0, *prev_paths[best_idx], length], float(final_costs[best_idx] / max(count, 1))


def predict_dcp_dp(
    block: RepSegment,
    df: pd.DataFrame,
    args: argparse.Namespace,
    use_few_shot: bool,
    subject_templates,
    exercise_templates,
) -> list[RepSegment]:
    pool, meta = dense_candidate_pool(block, df, args)
    template = template_for_block(block, subject_templates, exercise_templates) if use_few_shot else None
    template_duration = template.duration_samples if template is not None else None
    center_count, calibrated_duration = estimate_count(block, meta, args, use_few_shot, template_duration)

    max_count = max(1, min(args.max_reps, block.n_samples // max(args.min_segment_samples, 1)))
    count_options = sorted({int(np.clip(center_count + delta, 1, max_count)) for delta in range(-args.count_search_radius, args.count_search_radius + 1)})
    best_boundaries: list[int] | None = None
    best_score = float("inf")
    for count in count_options:
        if calibrated_duration is not None and calibrated_duration > 0:
            target_duration = float((0.65 * calibrated_duration) + (0.35 * block.n_samples / max(count, 1)))
        else:
            target_duration = block.n_samples / float(max(count, 1))
        boundaries, score = decode_boundaries_dp(pool, block.n_samples, count, target_duration, args)
        count_cost = args.count_cost_weight * abs(count - center_count) / max(center_count, 1)
        total_score = score + count_cost
        if total_score < best_score:
            best_score = total_score
            best_boundaries = boundaries

    if best_boundaries is None:
        best_boundaries = uniform_boundaries(block.n_samples, center_count)
    source = "dcp_dp_fs" if use_few_shot else "dcp_dp"
    return segments_from_boundaries(block, best_boundaries, args.min_segment_samples, source)


def segments_from_boundaries(
    block: RepSegment,
    boundaries: Sequence[int],
    min_samples: int,
    source: str,
) -> list[RepSegment]:
    clean = [0]
    for boundary in sorted(set(int(value) for value in boundaries[1:-1])):
        if boundary - clean[-1] >= min_samples and block.n_samples - boundary >= min_samples:
            clean.append(boundary)
    clean.append(block.n_samples)
    base_set_id = str(block.set_id).split(":active", 1)[0]
    segments: list[RepSegment] = []
    for rep_idx, (start, end) in enumerate(zip(clean[:-1], clean[1:])):
        if end - start < min_samples:
            continue
        segments.append(
            RepSegment(
                block.file_path,
                block.subject,
                block.exercise,
                base_set_id,
                str(rep_idx),
                block.start + int(start),
                block.start + int(end),
                source,
            )
        )
    return segments


def method_summary_table(rows: list[dict[str, object]], prior_csv: Path) -> pd.DataFrame:
    current = pd.DataFrame(rows)
    if not prior_csv.exists():
        return current
    prior = pd.read_csv(prior_csv)
    keep = {
        "010_universal_gyro_valley",
        "011_multifeature_boundary_score",
        "012_exercise_only_ds_ms_tcn",
        "stayfit_ba",
        "maxxyt_map",
        "cara_dtw_fs",
        "lift_fusion",
    }
    prior = prior[prior["method_id"].astype(str).isin(keep)].copy()
    if prior.empty:
        return current
    prior["paper_anchor"] = prior.get("paper_anchor", "Existing project baseline")
    prior["uses_few_shot_labels"] = prior.get("uses_few_shot_labels", False)
    common = sorted(set(prior.columns) | set(current.columns))
    return pd.concat([prior.reindex(columns=common), current.reindex(columns=common)], ignore_index=True)


def rename_inherited_plot_names(output_dir: Path, thresholds: Sequence[float]) -> None:
    renames = {
        "014_literature_method_comparison_table.png": "016_dense_candidate_dp_comparison_table.png",
        "014_literature_method_score_bars.png": "016_dense_candidate_dp_score_bars.png",
    }
    for threshold in thresholds:
        renames[f"014_method_exercise_f1_iou_{threshold:.2f}.png"] = f"016_method_exercise_f1_iou_{threshold:.2f}.png"
        renames[f"014_method_subject_f1_iou_{threshold:.2f}.png"] = f"016_method_subject_f1_iou_{threshold:.2f}.png"
    for old_name, new_name in renames.items():
        old_path = output_dir / old_name
        if old_path.exists():
            new_path = output_dir / new_name
            if new_path.exists():
                new_path.unlink()
            old_path.rename(new_path)


def evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods_dir = args.output_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)

    session_cache: dict[Path, pd.DataFrame] = {}
    periods: dict[Path, float] = {}
    truth: list[RepSegment] = []
    phase_truth: list[PhaseSegment] = []
    for path in whole_session_files(args.data_dirs):
        df = read_session_9axis(path, args.data_dirs)
        if not {"ax", "ay", "az", "gx", "gy", "gz"}.issubset(df.columns):
            continue
        session_cache[path] = df
        periods[path] = infer_sensor_period_seconds(df)
        truth.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))
        phase_truth.extend(true_phase_segments(df, path, min_samples=args.min_phase_segment_samples))

    blocks = active_phase_contiguous_blocks_from_truth(truth, min_samples=args.min_segment_samples)
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
        truth = [segment for block in blocks for segment in truth_segments_for_block(block, truth)]
        phase_truth = [
            segment
            for segment in phase_truth
            if any(segment.file_path == block.file_path and min(segment.end, block.end) > max(segment.start, block.start) for block in blocks)
        ]

    subject_templates, exercise_templates = build_templates(
        truth,
        session_cache,
        calibration_reps=args.calibration_reps,
        smooth_window=args.smooth_window,
        template_points=args.template_points,
    )
    phase_orders = phase_order_by_exercise(phase_truth) if phase_truth else {}

    predictions: dict[str, list[RepSegment]] = {}
    for spec in METHOD_SPECS_016:
        use_few_shot = spec.method_id == "dcp_dp_fs"
        method_predictions: list[RepSegment] = []
        for block in blocks:
            method_predictions.extend(
                predict_dcp_dp(
                    block,
                    session_cache[block.file_path],
                    args,
                    use_few_shot=use_few_shot,
                    subject_templates=subject_templates,
                    exercise_templates=exercise_templates,
                )
            )
        predictions[spec.method_id] = method_predictions

    summary_rows: list[dict[str, object]] = []
    exercise_rows_all: list[dict[str, object]] = []
    subject_rows_all: list[dict[str, object]] = []
    spec_by_id = {spec.method_id: spec for spec in METHOD_SPECS_016}
    for method_id, predicted in predictions.items():
        spec = spec_by_id[method_id]
        method_dir = methods_dir / method_id
        summary_rows.append(
            write_method_outputs(
                spec,
                predicted,
                truth,
                phase_truth,
                session_cache,
                phase_orders,
                periods,
                args,
                method_dir,
            )
        )
        by_exercise = segmentation_metric_rows_by_exercise(predicted, truth, args.segmentation_iou_thresholds)
        by_subject = segmentation_metric_rows_by_subject(predicted, truth, args.segmentation_iou_thresholds)
        exercise_rows_all.extend({"method_id": method_id, "method_name": spec.method_name, **row} for row in by_exercise)
        subject_rows_all.extend({"method_id": method_id, "method_name": spec.method_name, **row} for row in by_subject)

    current = pd.DataFrame(summary_rows)
    current.to_csv(args.output_dir / "016_dense_candidate_dp_comparison.csv", index=False)
    combined = method_summary_table(summary_rows, args.prior_comparison_csv)
    combined.to_csv(args.output_dir / "016_dense_candidate_dp_comparison_with_prior.csv", index=False)
    plot_method_table(combined, args.output_dir)
    plot_method_bars(combined, args.output_dir)

    exercise_df = pd.DataFrame(exercise_rows_all)
    subject_df = pd.DataFrame(subject_rows_all)
    exercise_df.to_csv(args.output_dir / "016_dense_candidate_dp_by_exercise.csv", index=False)
    subject_df.to_csv(args.output_dir / "016_dense_candidate_dp_by_subject.csv", index=False)
    for threshold in args.segmentation_iou_thresholds:
        plot_method_exercise_heatmap(exercise_df, args.output_dir, float(threshold))
        plot_method_subject_heatmap(subject_df, args.output_dir, float(threshold))
    rename_inherited_plot_names(args.output_dir, args.segmentation_iou_thresholds)

    if args.max_waveform_plots != 0:
        waveform_dir = args.output_dir / "waveform_all_sets"
        for plot_idx, block in enumerate(blocks):
            if args.max_waveform_plots is not None and plot_idx >= args.max_waveform_plots:
                break
            block_truth = truth_segments_for_block(block, truth)
            block_predictions = {
                spec.method_name: predictions_for_block(block, predictions[spec.method_id])
                for spec in METHOD_SPECS_016
            }
            filename = safe_name(f"{plot_idx + 1:03d}_{block.subject}_{block.exercise}_{block.set_id}_{block.file_path.stem}") + ".png"
            plot_waveform_comparison(
                block,
                block_truth,
                block_predictions,
                session_cache[block.file_path],
                waveform_dir / filename,
                args.smooth_window,
            )

    summary = {
        "output_dir": str(args.output_dir),
        "active_blocks": len(blocks),
        "true_reps": len(truth),
        "phase_segments": len(phase_truth),
        "methods": [spec.method_name for spec in METHOD_SPECS_016],
        "comparison_csv": str(args.output_dir / "016_dense_candidate_dp_comparison.csv"),
        "comparison_with_prior_csv": str(args.output_dir / "016_dense_candidate_dp_comparison_with_prior.csv"),
        "assumptions": {
            "domain": "active-only exercise spans; upstream active/rest detection is not evaluated.",
            "candidate_pool": "Dense gyro/energy valleys, PCA/multi-axis midpoints, and autocorr priors are clustered before DP.",
            "few_shot": f"DCP-DP-FS uses up to {args.calibration_reps} labeled reps per subject/exercise for duration calibration.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(current.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense candidate pool dynamic-programming rep boundary decoder.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/016_dense_candidate_dp_decoder"))
    parser.add_argument("--prior-comparison-csv", type=Path, default=Path("artifacts_rep_classification/014_literature_inspired_rep_methods/014_literature_method_comparison_with_prior.csv"))
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--min-phase-segment-samples", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--energy-window", type=int, default=21)
    parser.add_argument("--peak-prominence-scale", type=float, default=0.28)
    parser.add_argument("--boundary-search-fraction", type=float, default=0.38)
    parser.add_argument("--max-period-fraction", type=float, default=0.80)
    parser.add_argument("--max-reps", type=int, default=40)
    parser.add_argument("--calibration-reps", type=int, default=3)
    parser.add_argument("--template-points", type=int, default=32)
    parser.add_argument("--cluster-radius", type=int, default=5)
    parser.add_argument("--raw-candidate-min-distance", type=int, default=6)
    parser.add_argument("--slot-search-fraction", type=float, default=0.95)
    parser.add_argument("--slot-search-min-samples", type=int, default=90)
    parser.add_argument("--max-slot-candidates", type=int, default=42)
    parser.add_argument("--max-states-per-stage", type=int, default=42)
    parser.add_argument("--min-duration-scale", type=float, default=0.52)
    parser.add_argument("--max-duration-scale", type=float, default=1.75)
    parser.add_argument("--duration-cost-weight", type=float, default=0.42)
    parser.add_argument("--target-cost-weight", type=float, default=0.10)
    parser.add_argument("--vote-bonus-weight", type=float, default=0.18)
    parser.add_argument("--vote-cap", type=float, default=6.0)
    parser.add_argument("--count-search-radius", type=int, default=1)
    parser.add_argument("--count-cost-weight", type=float, default=0.20)
    parser.add_argument("--segmentation-iou-thresholds", type=float, nargs="+", default=[0.50, 0.75, 0.90])
    parser.add_argument("--phase-iou-thresholds", type=float, nargs="+", default=[0.50, 0.75, 0.90])
    parser.add_argument("--phase-split-method", choices=["midpoint", "pca-reversal"], default="pca-reversal")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-waveform-plots", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
