from __future__ import annotations

import argparse
import json
import math
import os
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_rep_segmentation_classification import (
    IMU_COLUMNS,
    RepSegment,
    candidate_blocks,
    estimate_autocorrelation_period,
    estimate_fft_period,
    moving_average,
    phase_metric_rows,
    phase_metric_rows_by_phase,
    phase_order_by_exercise,
    plot_exercise_accuracy_table,
    plot_phase_metrics,
    plot_phase_metrics_by_phase,
    plot_segmentation_metrics,
    plot_segmentation_metrics_by_exercise,
    plot_segmentation_metrics_by_subject,
    predict_phase_segments,
    principal_motion_signal,
    read_session,
    robust_zscore,
    segment_iou,
    segmentation_metric_rows,
    segmentation_metric_rows_by_exercise,
    segmentation_metric_rows_by_subject,
    segmentation_summary,
    segments_from_block_boundaries,
    true_phase_segments,
    true_rep_segments,
    truth_segments_for_block,
    vector_magnitude_signal,
    whole_session_files,
    write_csv,
    best_truth_match_rows,
)


@dataclass(frozen=True)
class BoundaryBlock:
    key: tuple[str, str, str, str, int, int]
    file_path: Path
    subject: str
    exercise: str
    set_id: str
    block: RepSegment
    period: float
    expected_count: int
    count_options: tuple[int, ...]
    train_candidates: pd.DataFrame
    eval_candidates: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-wise multifeature boundary scoring for high-IoU rep segmentation.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-source", choices=["active-phase-span", "active-phase-contiguous", "action-label"], default="active-phase-contiguous")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--feature-window-samples", type=int, default=15)
    parser.add_argument("--local-window-samples", type=int, default=25)
    parser.add_argument("--candidate-top-k", type=int, default=3)
    parser.add_argument("--candidate-search-fraction", type=float, default=0.35)
    parser.add_argument("--count-search-radius", type=int, default=2)
    parser.add_argument("--max-reps", type=int, default=30)
    parser.add_argument("--autocorr-min-period-samples", type=int, default=25)
    parser.add_argument("--autocorr-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--positive-radius-samples", type=int, default=10)
    parser.add_argument("--negative-radius-samples", type=int, default=25)
    parser.add_argument("--top-candidates-per-slot", type=int, default=16)
    parser.add_argument("--duration-weight", type=float, default=0.35)
    parser.add_argument("--prior-distance-weight", type=float, default=0.20)
    parser.add_argument("--count-weight", type=float, default=0.25)
    parser.add_argument("--min-duration-scale", type=float, default=0.55)
    parser.add_argument("--max-duration-scale", type=float, default=1.65)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--model", choices=["logistic", "random-forest"], default="logistic")
    parser.add_argument("--negative-sample-ratio", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--segmentation-iou-thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.85, 0.9, 0.95])
    parser.add_argument("--evaluate-phase-split", action="store_true")
    parser.add_argument("--phase-split-method", choices=["midpoint", "pca-reversal"], default="pca-reversal")
    parser.add_argument("--min-phase-segment-samples", type=int, default=10)
    parser.add_argument("--phase-iou-thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    return parser.parse_args()


def safe_get(values: np.ndarray, index: int) -> float:
    if len(values) == 0:
        return 0.0
    index = int(np.clip(index, 0, len(values) - 1))
    return float(values[index])


def z(values: np.ndarray) -> np.ndarray:
    return robust_zscore(values.astype(np.float64))


def build_boundary_signals(segment_df: pd.DataFrame, smooth_window: int, local_window: int) -> dict[str, np.ndarray]:
    pca = z(principal_motion_signal(segment_df, smooth_window))
    pca_velocity = z(moving_average(np.abs(np.diff(pca, prepend=pca[:1])), max(3, smooth_window)))
    pca_acceleration = z(moving_average(np.abs(np.diff(pca_velocity, prepend=pca_velocity[:1])), max(3, smooth_window)))
    acc_mag = z(vector_magnitude_signal(segment_df, ("ax", "ay", "az"), smooth_window))
    gyro_mag = z(vector_magnitude_signal(segment_df, ("gx", "gy", "gz"), smooth_window))
    acc_jerk = z(moving_average(np.abs(np.diff(acc_mag, prepend=acc_mag[:1])), max(3, smooth_window)))
    gyro_jerk = z(moving_average(np.abs(np.diff(gyro_mag, prepend=gyro_mag[:1])), max(3, smooth_window)))
    transition_energy = z(pca_velocity + acc_jerk + gyro_jerk)
    motion_energy = z(moving_average(pca_velocity**2, max(3, local_window)))

    x = segment_df.loc[:, [col for col in IMU_COLUMNS if col in segment_df.columns]].to_numpy(dtype=np.float64)
    if x.size:
        x = np.apply_along_axis(robust_zscore, 0, x)
        axis_idx = int(np.argmax(np.var(x, axis=0)))
        dominant = z(moving_average(x[:, axis_idx], smooth_window))
    else:
        dominant = np.zeros(len(segment_df), dtype=np.float64)
    dominant_velocity = z(moving_average(np.abs(np.diff(dominant, prepend=dominant[:1])), max(3, smooth_window)))

    return {
        "pca": pca,
        "abs_pca": z(np.abs(pca)),
        "pca_velocity": pca_velocity,
        "pca_acceleration": pca_acceleration,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
        "acc_jerk": acc_jerk,
        "gyro_jerk": gyro_jerk,
        "transition_energy": transition_energy,
        "motion_energy": motion_energy,
        "dominant": dominant,
        "dominant_velocity": dominant_velocity,
    }


def top_local_points(signal: np.ndarray, lo: int, hi: int, objective: str, top_k: int, min_distance: int) -> list[int]:
    if hi <= lo:
        return []
    local = signal[lo : hi + 1]
    if len(local) == 0:
        return []
    points: set[int] = set()
    if objective == "min":
        points.add(int(lo + np.argmin(local)))
        peaks, _ = find_peaks(-local, distance=max(1, min_distance))
        if len(peaks):
            ranked = peaks[np.argsort(local[peaks])[:top_k]]
            points.update(int(lo + idx) for idx in ranked)
    elif objective == "max":
        points.add(int(lo + np.argmax(local)))
        peaks, _ = find_peaks(local, distance=max(1, min_distance))
        if len(peaks):
            ranked = peaks[np.argsort(-local[peaks])[:top_k]]
            points.update(int(lo + idx) for idx in ranked)
    else:
        raise ValueError(objective)
    return sorted(points)


def count_options_for_length(length: int, period: float, count_radius: int, max_reps: int) -> tuple[int, ...]:
    expected = int(round(length / max(period, 1.0)))
    max_count = min(max(1, length // 20), max(1, max_reps))
    expected = int(np.clip(expected, 1, max_count))
    lo = max(1, expected - count_radius)
    hi = min(max_count, expected + count_radius)
    return tuple(range(lo, hi + 1))


def estimate_period(signals: dict[str, np.ndarray], length: int, min_period: int, max_period_fraction: float) -> float:
    max_period = max(min_period, int(round(length * max_period_fraction)))
    period = estimate_autocorrelation_period(signals["pca"], min_period=min_period, max_period=max_period)
    if period is None:
        period = estimate_fft_period(signals["pca"], min_period=min_period, max_period=max_period)
    if period is None:
        period = max(float(min_period), length / 8.0)
    return float(period)


def generate_candidate_points(
    signals: dict[str, np.ndarray],
    length: int,
    period: float,
    count_options: Sequence[int],
    min_samples: int,
    search_fraction: float,
    top_k: int,
    include_truth: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(min_samples, int(round(period * search_fraction)))
    radius = min(radius, max(min_samples, length // 2))
    min_distance = max(3, min_samples // 3)
    candidate_points: set[int] = set()
    prior_points: set[int] = set()

    signal_objectives = [
        ("gyro_mag", "min"),
        ("gyro_mag", "max"),
        ("acc_mag", "min"),
        ("acc_mag", "max"),
        ("pca", "min"),
        ("pca", "max"),
        ("abs_pca", "max"),
        ("pca_velocity", "max"),
        ("pca_velocity", "min"),
        ("pca_acceleration", "max"),
        ("acc_jerk", "max"),
        ("gyro_jerk", "max"),
        ("transition_energy", "max"),
        ("transition_energy", "min"),
        ("motion_energy", "min"),
        ("motion_energy", "max"),
        ("dominant", "min"),
        ("dominant", "max"),
        ("dominant_velocity", "max"),
    ]

    for count in count_options:
        if count <= 1:
            continue
        for boundary_idx in range(1, count):
            prior = int(round(boundary_idx * length / float(count)))
            lo = max(min_samples, prior - radius)
            hi = min(length - min_samples, prior + radius)
            if hi <= lo:
                continue
            prior_points.add(prior)
            candidate_points.add(prior)
            for name, objective in signal_objectives:
                candidate_points.update(top_local_points(signals[name], lo, hi, objective, top_k=top_k, min_distance=min_distance))

    if include_truth:
        for boundary in include_truth:
            if min_samples <= int(boundary) <= length - min_samples:
                candidate_points.add(int(boundary))

    candidates = np.array(sorted(point for point in candidate_points if min_samples <= point <= length - min_samples), dtype=int)
    priors = np.array(sorted(point for point in prior_points if min_samples <= point <= length - min_samples), dtype=int)
    if len(priors) == 0:
        priors = candidates.copy()
    return candidates, priors


def build_signal_stats(signals: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, values in signals.items():
        values = values.astype(np.float64)
        stats[name] = (
            values,
            np.concatenate([[0.0], np.cumsum(values)]),
            np.concatenate([[0.0], np.cumsum(values**2)]),
        )
    return stats


def range_mean(cumsum: np.ndarray, start: int, end: int) -> float:
    start = max(0, int(start))
    end = min(len(cumsum) - 1, int(end))
    if end <= start:
        return 0.0
    return float((cumsum[end] - cumsum[start]) / float(end - start))


def fast_local_stats(
    values: np.ndarray,
    cumsum: np.ndarray,
    cumsum_sq: np.ndarray,
    center: int,
    radius: int,
) -> tuple[float, float, float, float, float, float, float]:
    lo = max(0, center - radius)
    hi = min(len(values), center + radius + 1)
    if hi <= lo:
        value = safe_get(values, center)
        return (value, 0.0, value, value, value, value, 0.0)
    count = float(hi - lo)
    mean = float((cumsum[hi] - cumsum[lo]) / count)
    mean_sq = float((cumsum_sq[hi] - cumsum_sq[lo]) / count)
    std = math.sqrt(max(0.0, mean_sq - mean * mean))
    left_mean = range_mean(cumsum, max(0, center - radius), center)
    right_mean = range_mean(cumsum, center + 1, min(len(values), center + radius + 1))
    if center <= lo:
        left_mean = safe_get(values, center)
    if center + 1 >= hi:
        right_mean = safe_get(values, center)
    return (
        mean,
        std,
        mean - std,
        mean + std,
        left_mean,
        right_mean,
        right_mean - left_mean,
    )


def candidate_features(
    signal_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    candidate: int,
    length: int,
    period: float,
    expected_count: int,
    priors: np.ndarray,
    feature_window: int,
) -> dict[str, float]:
    if len(priors):
        prior_dist = float(np.min(np.abs(priors - candidate)))
    else:
        prior_dist = 0.0
    features: dict[str, float] = {
        "feat_position_frac": candidate / max(float(length), 1.0),
        "feat_period_frac": period / max(float(length), 1.0),
        "feat_expected_count": float(expected_count),
        "feat_prior_abs_distance_frac": prior_dist / max(period, 1.0),
    }
    for name, (values, cumsum, cumsum_sq) in signal_stats.items():
        mean, std, min_value, max_value, left_mean, right_mean, diff = fast_local_stats(values, cumsum, cumsum_sq, candidate, feature_window)
        features[f"feat_{name}_value"] = safe_get(values, candidate)
        features[f"feat_{name}_local_mean"] = mean
        features[f"feat_{name}_local_std"] = std
        features[f"feat_{name}_local_min"] = min_value
        features[f"feat_{name}_local_max"] = max_value
        features[f"feat_{name}_left_mean"] = left_mean
        features[f"feat_{name}_right_mean"] = right_mean
        features[f"feat_{name}_right_minus_left"] = diff
    return features


def build_candidate_table(
    df: pd.DataFrame,
    block: RepSegment,
    truth: Sequence[RepSegment],
    args: argparse.Namespace,
    include_truth_candidates: bool,
) -> tuple[pd.DataFrame, float, int, tuple[int, ...]]:
    segment_df = df.iloc[block.start : block.end].reset_index(drop=True)
    length = len(segment_df)
    signals = build_boundary_signals(segment_df, smooth_window=args.smooth_window, local_window=args.local_window_samples)
    signal_stats = build_signal_stats(signals)
    period = estimate_period(
        signals,
        length=length,
        min_period=max(args.min_segment_samples, args.autocorr_min_period_samples),
        max_period_fraction=args.autocorr_max_period_fraction,
    )
    count_options = count_options_for_length(length, period, args.count_search_radius, args.max_reps)
    expected_count = int(round(length / max(period, 1.0)))
    expected_count = int(np.clip(expected_count, 1, max(count_options) if count_options else args.max_reps))
    true_boundaries = sorted(int(segment.start - block.start) for segment in truth[1:])
    candidate_points, priors = generate_candidate_points(
        signals,
        length=length,
        period=period,
        count_options=count_options,
        min_samples=args.min_segment_samples,
        search_fraction=args.candidate_search_fraction,
        top_k=args.candidate_top_k,
        include_truth=true_boundaries if include_truth_candidates else None,
    )

    rows: list[dict[str, object]] = []
    for candidate in candidate_points:
        if true_boundaries:
            nearest_error = int(min(abs(int(candidate) - boundary) for boundary in true_boundaries))
        else:
            nearest_error = length
        if nearest_error <= args.positive_radius_samples:
            label = 1
            use_for_training = True
        elif nearest_error >= args.negative_radius_samples:
            label = 0
            use_for_training = True
        else:
            label = -1
            use_for_training = False
        row: dict[str, object] = {
            "file": str(block.file_path),
            "subject": block.subject,
            "exercise": block.exercise,
            "set_id": block.set_id.split(":active", 1)[0],
            "block_start": block.start,
            "block_end": block.end,
            "candidate": int(candidate),
            "candidate_global": int(block.start + int(candidate)),
            "nearest_gt_error_samples": nearest_error,
            "label": label,
            "use_for_training": use_for_training,
        }
        row.update(candidate_features(signal_stats, int(candidate), length, period, expected_count, priors, args.feature_window_samples))
        rows.append(row)
    return pd.DataFrame(rows), period, expected_count, count_options


def select_boundaries(
    candidate_table: pd.DataFrame,
    probabilities: np.ndarray,
    length: int,
    period: float,
    count_options: Sequence[int],
    min_samples: int,
    search_fraction: float,
    top_per_slot: int,
    duration_weight: float,
    prior_distance_weight: float,
    count_weight: float,
    min_duration_scale: float,
    max_duration_scale: float,
) -> list[int]:
    if candidate_table.empty:
        return [0, length]
    candidates = candidate_table.copy()
    candidates["probability"] = probabilities.astype(float)
    expected_count = int(round(length / max(period, 1.0)))
    radius = max(min_samples, int(round(period * search_fraction)))
    radius = min(radius, max(min_samples, length // 2))
    best_score = -math.inf
    best_boundaries: list[int] | None = None

    for count in count_options:
        target_duration = length / float(count)
        min_duration = max(min_samples, int(round(target_duration * min_duration_scale)))
        max_duration = max(min_duration, int(round(target_duration * max_duration_scale)))
        if count <= 1:
            boundaries = [0, length]
            durations = np.diff(np.asarray(boundaries, dtype=np.float64))
            score = -float(np.mean(((durations - target_duration) / max(target_duration, 1.0)) ** 2))
        else:
            slot_options: list[list[tuple[int, float]]] = []
            for boundary_idx in range(1, count):
                prior = int(round(boundary_idx * target_duration))
                lo = max(min_samples, prior - radius)
                hi = min(length - min_samples, prior + radius)
                local = candidates[(candidates["candidate"] >= lo) & (candidates["candidate"] <= hi)].copy()
                if local.empty:
                    slot_options = []
                    break
                local["slot_score"] = local["probability"] - prior_distance_weight * (np.abs(local["candidate"] - prior) / max(radius, 1))
                local = local.sort_values("slot_score", ascending=False).head(top_per_slot)
                slot_options.append([(int(row.candidate), float(row.slot_score)) for row in local.itertuples(index=False)])
            if not slot_options:
                continue

            dp: list[dict[int, tuple[float, int | None]]] = []
            first: dict[int, tuple[float, int | None]] = {}
            for candidate, slot_score in slot_options[0]:
                duration = candidate
                if min_duration <= duration <= max_duration:
                    duration_cost = duration_weight * ((duration - target_duration) / max(target_duration, 1.0)) ** 2
                    first[candidate] = (slot_score - duration_cost, None)
            if not first:
                continue
            dp.append(first)

            for slot_idx in range(1, len(slot_options)):
                current: dict[int, tuple[float, int | None]] = {}
                for candidate, slot_score in slot_options[slot_idx]:
                    best_prev_score = -math.inf
                    best_prev: int | None = None
                    for prev_candidate, (prev_score, _) in dp[-1].items():
                        duration = candidate - prev_candidate
                        if min_duration <= duration <= max_duration:
                            duration_cost = duration_weight * ((duration - target_duration) / max(target_duration, 1.0)) ** 2
                            score = prev_score + slot_score - duration_cost
                            if score > best_prev_score:
                                best_prev_score = score
                                best_prev = prev_candidate
                    if best_prev is not None:
                        current[candidate] = (best_prev_score, best_prev)
                if not current:
                    dp = []
                    break
                dp.append(current)
            if not dp:
                continue

            best_last_score = -math.inf
            best_last: int | None = None
            for candidate, (score_so_far, _) in dp[-1].items():
                duration = length - candidate
                if min_duration <= duration <= max_duration:
                    duration_cost = duration_weight * ((duration - target_duration) / max(target_duration, 1.0)) ** 2
                    score = score_so_far - duration_cost
                    if score > best_last_score:
                        best_last_score = score
                        best_last = candidate
            if best_last is None:
                continue

            internal = [best_last]
            current_candidate = best_last
            for slot_idx in range(len(dp) - 1, 0, -1):
                prev_candidate = dp[slot_idx][current_candidate][1]
                if prev_candidate is None:
                    break
                internal.append(prev_candidate)
                current_candidate = prev_candidate
            boundaries = [0, *reversed(internal), length]
            durations = np.diff(np.asarray(boundaries, dtype=np.float64))
            if np.any(durations < min_duration) or np.any(durations > max_duration):
                continue
            score = best_last_score / max(1, count - 1)

        count_penalty = count_weight * abs(count - expected_count) / max(expected_count, 1)
        score -= count_penalty
        if score > best_score:
            best_score = score
            best_boundaries = [int(boundary) for boundary in boundaries]

    if best_boundaries is None:
        return [0, length]
    return sorted(set(best_boundaries))


def block_key(block: RepSegment) -> tuple[str, str, str, str, int, int]:
    return (str(block.file_path), block.subject, block.exercise, block.set_id.split(":active", 1)[0], block.start, block.end)


def make_predicted_segments(file_path: Path, block: RepSegment, boundaries: Sequence[int], min_samples: int) -> list[RepSegment]:
    return segments_from_block_boundaries(file_path, block, boundaries, min_samples, "multifeature_boundary_score")


def boundary_error_rows(predicted: Sequence[RepSegment], truth: Sequence[RepSegment], sample_rate_hz: float) -> list[dict[str, object]]:
    by_key_truth: dict[tuple[str, str, str, str], list[RepSegment]] = {}
    by_key_pred: dict[tuple[str, str, str, str], list[RepSegment]] = {}
    for segment in truth:
        key = (str(segment.file_path), segment.subject, segment.exercise, segment.set_id)
        by_key_truth.setdefault(key, []).append(segment)
    for segment in predicted:
        key = (str(segment.file_path), segment.subject, segment.exercise, segment.set_id)
        by_key_pred.setdefault(key, []).append(segment)

    rows: list[dict[str, object]] = []
    for key, true_segments in sorted(by_key_truth.items()):
        file_path, subject, exercise, set_id = key
        true_segments = sorted(true_segments, key=lambda item: item.start)
        pred_segments = sorted(by_key_pred.get(key, []), key=lambda item: item.start)
        true_boundaries = [segment.start for segment in true_segments[1:]]
        pred_boundaries = [segment.start for segment in pred_segments[1:]]
        for idx, boundary in enumerate(true_boundaries, start=1):
            if pred_boundaries:
                nearest = min(pred_boundaries, key=lambda value: abs(value - boundary))
                error = int(nearest - boundary)
                abs_error = abs(error)
            else:
                nearest = ""
                error = ""
                abs_error = math.inf
            rows.append(
                {
                    "file": file_path,
                    "subject": subject,
                    "exercise": exercise,
                    "set_id": set_id,
                    "boundary_index": idx,
                    "true_boundary": int(boundary),
                    "nearest_pred_boundary": nearest,
                    "error_samples": error,
                    "abs_error_samples": abs_error if math.isfinite(float(abs_error)) else "",
                    "abs_error_ms": round(float(abs_error) * 1000.0 / sample_rate_hz, 2) if math.isfinite(float(abs_error)) else "",
                }
            )
    return rows


def summarize_boundary_errors(rows: Sequence[dict[str, object]], group_cols: Sequence[str], sample_rate_hz: float) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    valid = df[df["abs_error_samples"] != ""].copy()
    if valid.empty:
        return []
    valid["abs_error_samples"] = valid["abs_error_samples"].astype(float)
    grouped = valid.groupby(list(group_cols), sort=True) if group_cols else [((), valid)]
    out: list[dict[str, object]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        values = group["abs_error_samples"].to_numpy(dtype=float)
        row.update(
            {
                "boundaries": int(len(values)),
                "mean_abs_error_samples": round(float(np.mean(values)), 3),
                "median_abs_error_samples": round(float(np.median(values)), 3),
                "p80_abs_error_samples": round(float(np.percentile(values, 80)), 3),
                "within_5_samples": round(float(np.mean(values <= 5)), 4),
                "within_10_samples": round(float(np.mean(values <= 10)), 4),
                "within_20_samples": round(float(np.mean(values <= 20)), 4),
                "median_abs_error_ms": round(float(np.median(values)) * 1000.0 / sample_rate_hz, 2),
            }
        )
        out.append(row)
    return out


def plot_boundary_error_by_exercise(summary_rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows).sort_values("median_abs_error_samples", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(df))))
    bars = ax.barh(df["exercise"], df["median_abs_error_samples"], color="#4c78a8")
    ax.set_xlabel("Median internal boundary error (samples)")
    ax.set_title("Internal Boundary Error by Exercise")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, df["median_abs_error_samples"], strict=True):
        ax.text(float(value) + 0.4, bar.get_y() + bar.get_height() / 2, f"{float(value):.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_error_by_exercise.png", dpi=180)
    plt.close(fig)


def plot_high_iou_exercise_table(rows: Sequence[dict[str, object]], output_dir: Path, target_iou: float = 0.9) -> None:
    if not rows:
        return
    target_rows = [row for row in rows if abs(float(row["iou_threshold"]) - target_iou) < 1e-9]
    if not target_rows:
        return
    df = pd.DataFrame(target_rows).sort_values("f1", ascending=False)
    cols = ["exercise", "precision", "recall", "f1", "matched_reps", "true_reps", "predicted_reps", "mean_matched_iou"]
    display = df.loc[:, cols].copy()
    for col in ["precision", "recall", "f1", "mean_matched_iou"]:
        display[col] = display[col].astype(float).map(lambda value: f"{value:.3f}")
    for col in ["matched_reps", "true_reps", "predicted_reps"]:
        display[col] = display[col].astype(int).astype(str)
    write_csv(output_dir / "rep_segmentation_iou_0.90_by_exercise_table.csv", df.to_dict("records"))

    fig, ax = plt.subplots(figsize=(11, max(3.8, 0.42 * len(display) + 1.4)))
    ax.axis("off")
    table = ax.table(
        cellText=display.to_numpy(),
        colLabels=["Exercise", "P@0.90", "R@0.90", "F1@0.90", "Matched", "GT reps", "Pred reps", "Mean IoU"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#dbe8f6")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f5f7fa")
        if col_idx == 0 and row_idx > 0:
            cell.set_text_props(ha="left")
    ax.set_title("Rep Segmentation Accuracy by Exercise at IoU 0.90", pad=18, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "rep_segmentation_iou_0.90_by_exercise_table.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session_cache: dict[Path, pd.DataFrame] = {}
    truth: list[RepSegment] = []
    phase_truth = []
    for path in whole_session_files(args.data_dirs):
        df = read_session(path, args.data_dirs)
        if not all(col in df.columns for col in IMU_COLUMNS):
            continue
        session_cache[path] = df
        truth.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))
        if args.evaluate_phase_split:
            phase_truth.extend(true_phase_segments(df, path, min_samples=args.min_phase_segment_samples))
    if not truth:
        raise RuntimeError("No truth rep segments found.")

    truth_by_file: dict[Path, list[RepSegment]] = {}
    for segment in truth:
        truth_by_file.setdefault(segment.file_path, []).append(segment)

    blocks: list[BoundaryBlock] = []
    all_train_candidates: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    for path, df in session_cache.items():
        for block in candidate_blocks(df, path, truth_by_file.get(path, []), min_samples=args.min_segment_samples, block_source=args.block_source):
            block_truth = sorted(truth_segments_for_block(block, truth_by_file.get(path, [])), key=lambda item: item.start)
            if len(block_truth) < 2:
                continue
            train_table, period, expected_count, count_options = build_candidate_table(df, block, block_truth, args, include_truth_candidates=True)
            eval_table, _, _, _ = build_candidate_table(df, block, block_truth, args, include_truth_candidates=False)
            if train_table.empty or eval_table.empty:
                continue
            if not feature_cols:
                feature_cols = [col for col in train_table.columns if col.startswith("feat_")]
            blocks.append(
                BoundaryBlock(
                    key=block_key(block),
                    file_path=path,
                    subject=block.subject,
                    exercise=block.exercise,
                    set_id=block.set_id.split(":active", 1)[0],
                    block=block,
                    period=period,
                    expected_count=expected_count,
                    count_options=count_options,
                    train_candidates=train_table,
                    eval_candidates=eval_table,
                )
            )
            all_train_candidates.append(train_table)

    if not blocks:
        raise RuntimeError("No boundary blocks could be built.")

    subjects = np.asarray([block.subject for block in blocks], dtype=object)
    unique_subjects = sorted(set(subjects.tolist()))
    n_splits = min(args.folds, len(unique_subjects))
    if n_splits < 2:
        raise RuntimeError("Need at least two subjects for subject-wise folds.")

    predicted: list[RepSegment] = []
    fold_rows: list[dict[str, object]] = []
    candidate_diagnostics: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=n_splits)
    block_indices = np.arange(len(blocks))
    for fold_idx, (train_block_idx, val_block_idx) in enumerate(splitter.split(block_indices, groups=subjects), start=1):
        train_subjects = sorted(set(subjects[train_block_idx].tolist()))
        val_subjects = sorted(set(subjects[val_block_idx].tolist()))
        train_df = pd.concat([blocks[idx].train_candidates for idx in train_block_idx], ignore_index=True)
        train_df = train_df[train_df["use_for_training"].astype(bool)].copy()
        positives = train_df[train_df["label"].astype(int) == 1]
        negatives = train_df[train_df["label"].astype(int) == 0]
        max_negatives = int(round(len(positives) * args.negative_sample_ratio))
        if len(negatives) > max_negatives > 0:
            negatives = negatives.sample(n=max_negatives, random_state=args.seed + fold_idx)
            train_df = pd.concat([positives, negatives], ignore_index=True)
        x_train = train_df.loc[:, feature_cols].fillna(0.0)
        y_train = train_df["label"].astype(int).to_numpy()
        if args.model == "random-forest":
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=args.seed + fold_idx,
                n_jobs=-1,
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-3,
                    alpha=1e-4,
                    random_state=args.seed + fold_idx,
                ),
            )
        model.fit(x_train, y_train)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": ",".join(train_subjects),
                "val_subjects": ",".join(val_subjects),
                "train_candidates": int(len(train_df)),
                "train_positive_candidates": int(np.sum(y_train == 1)),
                "train_negative_candidates": int(np.sum(y_train == 0)),
            }
        )
        for idx in val_block_idx:
            block = blocks[idx]
            eval_df = block.eval_candidates.copy()
            probabilities = model.predict_proba(eval_df.loc[:, feature_cols].fillna(0.0))[:, 1]
            boundaries = select_boundaries(
                eval_df,
                probabilities,
                length=block.block.end - block.block.start,
                period=block.period,
                count_options=block.count_options,
                min_samples=args.min_segment_samples,
                search_fraction=args.candidate_search_fraction,
                top_per_slot=args.top_candidates_per_slot,
                duration_weight=args.duration_weight,
                prior_distance_weight=args.prior_distance_weight,
                count_weight=args.count_weight,
                min_duration_scale=args.min_duration_scale,
                max_duration_scale=args.max_duration_scale,
            )
            predicted.extend(make_predicted_segments(block.file_path, block.block, boundaries, args.min_segment_samples))
            candidate_diagnostics.append(
                {
                    "fold": fold_idx,
                    "file": str(block.file_path),
                    "subject": block.subject,
                    "exercise": block.exercise,
                    "set_id": block.set_id,
                    "period": round(float(block.period), 4),
                    "expected_count": block.expected_count,
                    "count_options": ",".join(str(value) for value in block.count_options),
                    "candidate_count": int(len(eval_df)),
                    "predicted_reps": max(0, len(boundaries) - 1),
                    "boundaries": ",".join(str(value + block.block.start) for value in boundaries),
                }
            )

    segmentation_rows = segmentation_metric_rows(predicted, truth, args.segmentation_iou_thresholds)
    segmentation_by_exercise_rows = segmentation_metric_rows_by_exercise(predicted, truth, args.segmentation_iou_thresholds)
    segmentation_by_subject_rows = segmentation_metric_rows_by_subject(predicted, truth, args.segmentation_iou_thresholds)
    boundary_rows = boundary_error_rows(predicted, truth, sample_rate_hz=args.sample_rate_hz)
    boundary_overall = summarize_boundary_errors(boundary_rows, [], args.sample_rate_hz)
    boundary_by_exercise = summarize_boundary_errors(boundary_rows, ["exercise"], args.sample_rate_hz)
    boundary_by_subject = summarize_boundary_errors(boundary_rows, ["subject"], args.sample_rate_hz)

    write_csv(args.output_dir / "fold_manifest.csv", fold_rows)
    write_csv(args.output_dir / "candidate_diagnostics.csv", candidate_diagnostics)
    write_csv(args.output_dir / "rep_segmentation_matches.csv", segmentation_summary(predicted, truth))
    write_csv(args.output_dir / "rep_segmentation_truth_matches.csv", best_truth_match_rows(predicted, truth))
    write_csv(args.output_dir / "rep_segmentation_metrics.csv", segmentation_rows)
    write_csv(args.output_dir / "rep_segmentation_metrics_by_exercise.csv", segmentation_by_exercise_rows)
    write_csv(args.output_dir / "rep_segmentation_metrics_by_subject.csv", segmentation_by_subject_rows)
    write_csv(args.output_dir / "boundary_error_samples.csv", boundary_rows)
    write_csv(args.output_dir / "boundary_error_overall.csv", boundary_overall)
    write_csv(args.output_dir / "boundary_error_by_exercise.csv", boundary_by_exercise)
    write_csv(args.output_dir / "boundary_error_by_subject.csv", boundary_by_subject)

    plot_segmentation_metrics(segmentation_rows, args.output_dir)
    plot_segmentation_metrics_by_exercise(segmentation_by_exercise_rows, args.output_dir)
    plot_segmentation_metrics_by_subject(segmentation_by_subject_rows, args.output_dir)
    plot_exercise_accuracy_table(segmentation_by_exercise_rows, args.output_dir, args.segmentation_iou_thresholds)
    plot_high_iou_exercise_table(segmentation_by_exercise_rows, args.output_dir, target_iou=0.9)
    plot_boundary_error_by_exercise(boundary_by_exercise, args.output_dir)

    phase_rows: list[dict[str, object]] = []
    phase_by_phase_rows: list[dict[str, object]] = []
    predicted_phases = []
    if args.evaluate_phase_split and phase_truth:
        phase_orders = phase_order_by_exercise(phase_truth)
        predicted_phases = predict_phase_segments(
            predicted,
            session_cache,
            phase_orders=phase_orders,
            method=args.phase_split_method,
            smooth_window=args.smooth_window,
            min_phase_samples=args.min_phase_segment_samples,
        )
        phase_rows = phase_metric_rows(predicted_phases, phase_truth, args.phase_iou_thresholds)
        phase_by_phase_rows = phase_metric_rows_by_phase(predicted_phases, phase_truth, args.phase_iou_thresholds)
        write_csv(args.output_dir / "phase_split_metrics.csv", phase_rows)
        write_csv(args.output_dir / "phase_split_metrics_by_phase.csv", phase_by_phase_rows)
        plot_phase_metrics(phase_rows, args.output_dir)
        plot_phase_metrics_by_phase(phase_by_phase_rows, args.output_dir)

    summary = {
        "data_dirs": [str(path) for path in args.data_dirs],
        "method": "multifeature_boundary_score",
        "block_source": args.block_source,
        "folds": n_splits,
        "subjects": unique_subjects,
        "num_truth_reps": len(truth),
        "num_predicted_reps": len(predicted),
        "num_blocks": len(blocks),
        "feature_count": len(feature_cols),
        "segmentation_metrics": segmentation_rows,
        "boundary_error_overall": boundary_overall,
        "phase_split_method": args.phase_split_method if args.evaluate_phase_split else None,
        "phase_split_metrics": phase_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
