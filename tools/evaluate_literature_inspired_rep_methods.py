from __future__ import annotations

import argparse
import json
import math
import os
import re
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
from scipy.signal import find_peaks, peak_prominences

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_rep_segmentation_classification import (  # noqa: E402
    ACTIVE_PHASES,
    PhaseSegment,
    RepSegment,
    active_phase_contiguous_blocks_from_truth,
    best_truth_match_rows,
    clean_label_series,
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
    robust_zscore,
    segment_iou,
    segmentation_metric_rows,
    segmentation_metric_rows_by_exercise,
    segmentation_metric_rows_by_subject,
    segmentation_summary,
    subject_from_path,
    true_phase_segments,
    true_rep_segments,
    truth_segments_for_block,
    whole_session_files,
    write_csv,
)


IMU9_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")
ACC_COLUMNS = ("ax", "ay", "az")
GYRO_COLUMNS = ("gx", "gy", "gz")
MAG_COLUMNS = ("mx", "my", "mz")
LABEL_COLUMNS = {"subject_id", "action_type", "set", "rep", "phase", "sensor_ts"}
READ_COLUMNS = set(IMU9_COLUMNS) | LABEL_COLUMNS


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    method_name: str
    paper_anchor: str
    description: str
    weakness: str
    improvement_used: str
    uses_few_shot_labels: bool


@dataclass(frozen=True)
class Template:
    values: np.ndarray
    duration_samples: float
    source_reps: int


FEATURE_CACHE: dict[tuple[str, int, int, int, int], dict[str, object]] = {}


METHOD_SPECS = (
    MethodSpec(
        "stayfit_ba",
        "STAYFIT-BA",
        "StayFit-style best-axis peak counting",
        "Choose the most periodic IMU axis or magnitude per active set, then cut reps from peak midpoints.",
        "Single-axis peak counting is cheap, but axis choice changes across people and can over-count phase wiggles.",
        "Use autocorrelation score to choose the axis automatically instead of hard-coding exercise axes.",
        False,
    ),
    MethodSpec(
        "maxxyt_map",
        "MAXXYT-MAP",
        "Maxxyt-style multi-axis adaptive peak aggregation",
        "Count candidate peaks over multiple accelerometer, gyroscope, magnetometer, magnitude, and PCA signals, then use a consensus count.",
        "Multi-axis voting is robust for counting, but it does not by itself place precise boundaries.",
        "After count voting, refine internal cuts to low gyro-motion valleys.",
        False,
    ),
    MethodSpec(
        "mfitness_fste",
        "MFIT-FSTE",
        "M-Fitness-style frequency-weighted short-time energy",
        "Build a short-time energy curve from 9-axis PCA, acceleration magnitude, and gyroscope magnitude; boundaries come from low-energy valleys.",
        "Energy methods can merge slow reps or split noisy reps when energy peaks are not one-to-one with repetitions.",
        "Use autocorrelation as a period prior before searching energy peaks and valleys.",
        False,
    ),
    MethodSpec(
        "cara_dtw_fs",
        "CARA-DTW-FS",
        "CaRaCount / DTW-style few-shot template alignment",
        "Use a few labeled reps from the same subject and exercise as a template, then refine cuts by template-shape matching.",
        "Template matching adapts to the user, but it needs small labeled calibration and is heavier than pure peak counting.",
        "Use the template as a boundary refinement prior after period-based count estimation.",
        True,
    ),
    MethodSpec(
        "lift_fusion",
        "LIFT-Fusion",
        "New literature-inspired fusion method",
        "Fuse PCA/autocorrelation period, multi-axis count consensus, short-time energy valley score, and optional few-shot template refinement.",
        "Fusion still depends on active-only exercise spans and does not solve upstream active/rest detection.",
        "Use complementary weak signals: count consensus for number of reps, energy/gyro valleys for boundaries, and template shape for personalization.",
        True,
    ),
)


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


def read_session_9axis(path: Path, data_dirs: Sequence[Path]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda column: column in READ_COLUMNS)
    if "subject_id" not in df.columns:
        df["subject_id"] = subject_from_path(path, data_dirs)
    if "action_type" not in df.columns:
        df["action_type"] = path.parent.name
    if "set" not in df.columns:
        df["set"] = 0
    if "rep" not in df.columns:
        df["rep"] = np.arange(len(df))
    if "phase" not in df.columns:
        df["phase"] = "none"
    return df.reset_index(drop=True)


def available_imu_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in IMU9_COLUMNS if column in df.columns]


def zscore_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    available = [column for column in columns if column in df.columns]
    if not available:
        return np.zeros((len(df), 0), dtype=np.float64)
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    return np.apply_along_axis(robust_zscore, 0, x)


def principal_signal(df: pd.DataFrame, smooth_window: int, columns: Sequence[str] = IMU9_COLUMNS) -> np.ndarray:
    x = zscore_matrix(df, columns)
    if x.shape[1] == 0:
        return np.zeros(len(df), dtype=np.float64)
    variances = np.var(x, axis=0)
    x = x[:, variances > 1e-9]
    if x.shape[1] == 0:
        return np.zeros(len(df), dtype=np.float64)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return moving_average(x @ vt[0], smooth_window)


def magnitude_signal(df: pd.DataFrame, columns: Sequence[str], smooth_window: int) -> np.ndarray:
    x = zscore_matrix(df, columns)
    if x.shape[1] == 0:
        return np.zeros(len(df), dtype=np.float64)
    return moving_average(np.linalg.norm(x, axis=1), smooth_window)


def candidate_signals(df: pd.DataFrame, smooth_window: int) -> dict[str, np.ndarray]:
    signals: dict[str, np.ndarray] = {}
    for column in available_imu_columns(df):
        signals[column] = moving_average(robust_zscore(df[column].to_numpy(dtype=np.float64)), smooth_window)
    signals["acc_mag"] = magnitude_signal(df, ACC_COLUMNS, smooth_window)
    signals["gyro_mag"] = magnitude_signal(df, GYRO_COLUMNS, smooth_window)
    if any(column in df.columns for column in MAG_COLUMNS):
        signals["mag_mag"] = magnitude_signal(df, MAG_COLUMNS, smooth_window)
    signals["imu9_pca"] = principal_signal(df, smooth_window, IMU9_COLUMNS)
    signals["acc_pca"] = principal_signal(df, smooth_window, ACC_COLUMNS)
    signals["gyro_pca"] = principal_signal(df, smooth_window, GYRO_COLUMNS)
    return {name: values for name, values in signals.items() if len(values) == len(df)}


def block_features(block: RepSegment, df: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    key = (str(block.file_path), block.start, block.end, args.smooth_window, args.energy_window)
    cached = FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    local_df = df.iloc[block.start : block.end]
    signals = candidate_signals(local_df, args.smooth_window)
    pca = signals.get("imu9_pca", principal_signal(local_df, args.smooth_window, IMU9_COLUMNS))
    max_period = max(args.min_segment_samples, int(round(block.n_samples * args.max_period_fraction)))
    periods = {
        name: autocorr_period_and_score(signal, args.min_segment_samples, max_period)
        for name, signal in signals.items()
    }
    features: dict[str, object] = {
        "local_df": local_df,
        "signals": signals,
        "pca": pca,
        "periods": periods,
        "boundary_score": fused_boundary_score(local_df, args.smooth_window, args.energy_window),
        "energy": frequency_weighted_energy(local_df, args.smooth_window, args.energy_window),
        "gyro_score": valley_cost_signal(magnitude_signal(local_df, GYRO_COLUMNS, args.smooth_window)),
    }
    FEATURE_CACHE[key] = features
    return features


def autocorr_period_and_score(signal: np.ndarray, min_period: int, max_period: int) -> tuple[float | None, float]:
    if len(signal) < max(min_period * 2, 4):
        return None, 0.0
    max_period = min(max_period, len(signal) - 1)
    if max_period < min_period:
        return None, 0.0

    values = robust_zscore(signal)
    values = values - float(np.mean(values))
    if float(np.std(values)) < 1e-9:
        return None, 0.0

    n_fft = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(values, n=n_fft)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft)[: len(values)]
    if float(autocorr[0]) <= 1e-9:
        return None, 0.0
    autocorr = autocorr / float(autocorr[0])
    lags = np.arange(len(autocorr))
    valid = (lags >= min_period) & (lags <= max_period)
    if not np.any(valid):
        return None, 0.0

    valid_lags = lags[valid]
    valid_autocorr = autocorr[valid]
    peaks, props = find_peaks(valid_autocorr, prominence=0.02)
    if len(peaks):
        prominences = props.get("prominences", peak_prominences(valid_autocorr, peaks)[0])
        positive = valid_autocorr[peaks] > 0
        if np.any(positive):
            peaks = peaks[positive]
            prominences = prominences[positive]
        if len(peaks):
            best_idx = peaks[int(np.argmax(prominences))]
            score = float(max(valid_autocorr[best_idx], 0.0) + 0.25 * np.max(prominences))
            return float(valid_lags[best_idx]), score

    best_idx = int(np.argmax(valid_autocorr))
    score = float(max(valid_autocorr[best_idx], 0.0))
    return float(valid_lags[best_idx]), score


def period_count(length: int, period: float | None, min_samples: int, max_reps: int) -> int:
    max_count = max(1, min(max_reps, length // max(min_samples, 1)))
    if period is None or not math.isfinite(period) or period <= 0:
        return 1
    return int(np.clip(round(length / max(period, 1.0)), 1, max_count))


def select_top_peaks(
    signal: np.ndarray,
    expected_count: int,
    min_samples: int,
    period: float | None,
    prominence_scale: float,
    distance_scale: float,
) -> tuple[np.ndarray, float]:
    if len(signal) < 3:
        return np.array([], dtype=int), 0.0
    distance_base = period if period is not None and math.isfinite(period) and period > 0 else min_samples
    distance = max(3, int(round(distance_base * distance_scale)), min_samples // 2)
    prominence = max(float(np.std(signal)) * prominence_scale, 1e-6)

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for orientation, values in ((1.0, signal), (-1.0, -signal)):
        peaks, props = find_peaks(values, distance=distance, prominence=prominence)
        if len(peaks) == 0:
            continue
        prominences = props.get("prominences", peak_prominences(values, peaks)[0])
        count_error = abs(len(peaks) - expected_count)
        strength = float(np.mean(prominences)) if len(prominences) else 0.0
        score = -float(count_error) + 0.05 * strength
        candidates.append((score, peaks, prominences))
    if not candidates:
        return np.array([], dtype=int), 0.0

    _score, peaks, prominences = max(candidates, key=lambda item: item[0])
    if expected_count > 0 and len(peaks) > expected_count:
        top = np.argsort(prominences)[-expected_count:]
        peaks = peaks[np.sort(top)]
        prominences = prominences[np.sort(top)]
    return np.sort(peaks.astype(int)), float(np.mean(prominences)) if len(prominences) else 0.0


def uniform_boundaries(length: int, count: int) -> list[int]:
    count = max(1, count)
    return [int(round(i * length / float(count))) for i in range(count + 1)]


def boundaries_from_centers(length: int, centers: np.ndarray, fallback_count: int) -> list[int]:
    if len(centers) == 0:
        return uniform_boundaries(length, fallback_count)
    centers = np.sort(np.clip(centers.astype(int), 0, max(0, length - 1)))
    boundaries = [0]
    if len(centers) > 1:
        boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(len(centers) - 1))
    boundaries.append(length)
    return sorted(set(boundaries))


def valley_cost_signal(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    scale = high - low
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / scale, 0.0, 1.0)


def fused_boundary_score(df: pd.DataFrame, smooth_window: int, energy_window: int) -> np.ndarray:
    pca = principal_signal(df, smooth_window, IMU9_COLUMNS)
    acc_mag = magnitude_signal(df, ACC_COLUMNS, smooth_window)
    gyro_mag = magnitude_signal(df, GYRO_COLUMNS, smooth_window)
    pca_vel = moving_average(np.abs(np.diff(pca, prepend=pca[:1])), max(3, energy_window))
    acc_jerk = moving_average(np.abs(np.diff(acc_mag, prepend=acc_mag[:1])), max(3, energy_window))
    gyro_jerk = moving_average(np.abs(np.diff(gyro_mag, prepend=gyro_mag[:1])), max(3, energy_window))
    transition = robust_zscore(pca_vel) + robust_zscore(acc_jerk) + robust_zscore(gyro_jerk)
    transition = moving_average(np.abs(transition), max(3, energy_window))
    return (
        0.45 * valley_cost_signal(gyro_mag)
        + 0.35 * valley_cost_signal(transition)
        + 0.20 * valley_cost_signal(pca_vel)
    )


def frequency_weighted_energy(df: pd.DataFrame, smooth_window: int, energy_window: int) -> np.ndarray:
    pca = principal_signal(df, smooth_window, IMU9_COLUMNS)
    acc_mag = magnitude_signal(df, ACC_COLUMNS, smooth_window)
    gyro_mag = magnitude_signal(df, GYRO_COLUMNS, smooth_window)
    components = []
    for values in (pca, acc_mag, gyro_mag):
        derivative = np.abs(np.diff(values, prepend=values[:1]))
        envelope = moving_average(np.abs(robust_zscore(values)), max(3, energy_window))
        local_energy = moving_average(derivative**2, max(3, energy_window))
        components.append(robust_zscore(local_energy * (1.0 + 0.2 * envelope)))
    return moving_average(np.sum(np.vstack(components), axis=0), max(3, energy_window))


def enforce_min_duration(boundaries: Sequence[int], length: int, min_samples: int) -> list[int]:
    if not boundaries:
        return [0, length]
    cleaned = [0]
    for boundary in sorted(set(int(value) for value in boundaries[1:-1])):
        if boundary - cleaned[-1] >= min_samples and length - boundary >= min_samples:
            cleaned.append(boundary)
    cleaned.append(length)
    return cleaned if len(cleaned) >= 2 else [0, length]


def refine_boundaries_by_score(
    boundaries: Sequence[int],
    score: np.ndarray,
    min_samples: int,
    search_radius: int,
) -> list[int]:
    length = len(score)
    if len(boundaries) <= 2 or length < 3:
        return list(boundaries)

    refined = [0]
    count = len(boundaries) - 1
    for boundary_idx, boundary in enumerate(boundaries[1:-1], start=1):
        left_prior = boundaries[boundary_idx - 1]
        right_prior = boundaries[boundary_idx + 1]
        lo = max(refined[-1] + min_samples, int(boundary) - search_radius)
        hi = min(length - (count - boundary_idx) * min_samples, int(boundary) + search_radius)
        hi = min(hi, right_prior - min_samples // 2 if right_prior > boundary else hi)
        lo = max(lo, left_prior + min_samples // 2)
        if hi <= lo:
            refined.append(int(boundary))
            continue
        local = score[lo : hi + 1]
        prior = float(boundary)
        duration_cost = ((np.arange(lo, hi + 1, dtype=np.float64) - prior) / max(search_radius, 1)) ** 2
        chosen = int(lo + np.argmin(local + 0.12 * duration_cost))
        refined.append(chosen)
    refined.append(length)
    return enforce_min_duration(refined, length, min_samples)


def segments_from_boundaries(
    block: RepSegment,
    boundaries: Sequence[int],
    min_samples: int,
    source: str,
) -> list[RepSegment]:
    boundaries = enforce_min_duration(boundaries, block.n_samples, min_samples)
    base_set_id = block.set_id.split(":active", 1)[0]
    rows: list[RepSegment] = []
    for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end - start < min_samples:
            continue
        rows.append(
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
    return rows


def normalize_count_votes(counts: Sequence[int]) -> list[int]:
    positive = [int(count) for count in counts if int(count) > 0]
    if not positive:
        return []
    median = float(np.median(positive))
    normalized: list[int] = []
    for count in positive:
        adjusted = count
        if median >= 1:
            if count >= 2 * median - 1 and abs(count / 2.0 - median) <= max(1.0, 0.35 * median):
                adjusted = max(1, int(round(count / 2.0)))
            elif count >= 3 * median - 1 and abs(count / 3.0 - median) <= max(1.0, 0.35 * median):
                adjusted = max(1, int(round(count / 3.0)))
        normalized.append(adjusted)
    return normalized


def weighted_integer_median(values: Sequence[int], weights: Sequence[float]) -> int:
    if not values:
        return 1
    pairs = sorted((int(value), float(weight)) for value, weight in zip(values, weights))
    total = sum(weight for _value, weight in pairs)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= total / 2.0:
            return value
    return pairs[-1][0]


def multi_axis_count_vote(
    signals: dict[str, np.ndarray],
    length: int,
    min_samples: int,
    max_reps: int,
    prominence_scale: float,
    period_scores: dict[str, tuple[float | None, float]] | None = None,
) -> tuple[int, str, np.ndarray, float]:
    estimates: list[int] = []
    weighted_values: list[int] = []
    weights: list[float] = []
    signal_details: list[tuple[float, str, np.ndarray, int]] = []
    max_period = max(min_samples, int(round(length * 0.8)))
    for name, signal in signals.items():
        if period_scores is not None and name in period_scores:
            period, score = period_scores[name]
        else:
            period, score = autocorr_period_and_score(signal, min_period=min_samples, max_period=max_period)
        p_count = period_count(length, period, min_samples, max_reps)
        peaks, strength = select_top_peaks(
            signal,
            expected_count=p_count,
            min_samples=min_samples,
            period=period,
            prominence_scale=prominence_scale,
            distance_scale=0.55,
        )
        count = len(peaks) if len(peaks) else p_count
        count = int(np.clip(count, 1, max(1, min(max_reps, length // max(min_samples, 1)))))
        estimates.append(count)
        weighted_values.append(count)
        weights.append(max(0.2, score + 0.05 * strength))
        signal_details.append((score + 0.05 * strength, name, peaks, count))

    normalized = normalize_count_votes(estimates)
    if normalized:
        for count in normalized:
            weighted_values.append(count)
            weights.append(0.8)
    consensus = int(np.clip(weighted_integer_median(weighted_values, weights), 1, max(1, min(max_reps, length // max(min_samples, 1)))))

    if signal_details:
        signal_details.sort(key=lambda item: (abs(item[3] - consensus), -item[0]))
        _score, name, peaks, _count = signal_details[0]
        if len(peaks) == 0 or abs(len(peaks) - consensus) > 1:
            peaks = np.array([], dtype=int)
        return consensus, name, peaks, float(signal_details[0][0])
    return consensus, "none", np.array([], dtype=int), 0.0


def resample(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float64)
    if len(values) == 1:
        return np.full(target_length, float(values[0]), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, target_length)
    return np.interp(x_new, x_old, values.astype(np.float64))


def dtw_distance(a: np.ndarray, b: np.ndarray, band_fraction: float = 0.25) -> float:
    a = robust_zscore(a)
    b = robust_zscore(b)
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return float("inf")
    band = max(abs(n - m), int(round(max(n, m) * band_fraction)))
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr.fill(np.inf)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = abs(float(a[i - 1] - b[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / max(n + m, 1))


def template_shape_cost(signal: np.ndarray, start: int, end: int, template: Template | None, use_dtw: bool) -> float:
    if template is None or end - start < 5:
        return 0.0
    values = resample(signal[start:end], len(template.values))
    values = robust_zscore(values)
    template_values = robust_zscore(template.values)
    if use_dtw:
        return min(dtw_distance(values, template_values), dtw_distance(-values, template_values))
    return min(float(np.mean((values - template_values) ** 2)), float(np.mean((-values - template_values) ** 2)))


def refine_boundaries_template_aware(
    boundaries: Sequence[int],
    score: np.ndarray,
    signal: np.ndarray,
    template: Template | None,
    min_samples: int,
    search_radius: int,
    shape_weight: float,
    use_dtw: bool,
) -> list[int]:
    length = len(score)
    if len(boundaries) <= 2 or length < 3:
        return list(boundaries)
    count = len(boundaries) - 1
    refined = [0]
    for boundary_idx, boundary in enumerate(boundaries[1:-1], start=1):
        next_boundary = boundaries[boundary_idx + 1]
        lo = max(refined[-1] + min_samples, int(boundary) - search_radius)
        hi = min(length - (count - boundary_idx) * min_samples, int(boundary) + search_radius)
        if hi <= lo:
            refined.append(int(boundary))
            continue
        best_boundary = int(boundary)
        best_cost = float("inf")
        stride = max(4, int(round((hi - lo) / 10.0)))
        for candidate in range(lo, hi + 1, stride):
            duration_prior = ((candidate - float(boundary)) / max(search_radius, 1)) ** 2
            shape_cost = 0.0
            if template is not None and shape_weight > 0:
                left_cost = template_shape_cost(signal, refined[-1], candidate, template, use_dtw=use_dtw)
                right_cost = template_shape_cost(signal, candidate, min(next_boundary, length), template, use_dtw=use_dtw)
                shape_cost = min(3.0, 0.5 * (left_cost + right_cost))
            cost = float(score[candidate]) + 0.12 * duration_prior + shape_weight * shape_cost
            if cost < best_cost:
                best_cost = cost
                best_boundary = candidate
        refined.append(best_boundary)
    refined.append(length)
    return enforce_min_duration(refined, length, min_samples)


def build_templates(
    true_segments: Sequence[RepSegment],
    session_cache: dict[Path, pd.DataFrame],
    calibration_reps: int,
    smooth_window: int,
    template_points: int,
) -> tuple[dict[tuple[str, str], Template], dict[str, Template]]:
    by_subject_exercise: dict[tuple[str, str], list[RepSegment]] = {}
    by_exercise: dict[str, list[RepSegment]] = {}
    for segment in sorted(true_segments, key=lambda item: (item.subject, item.exercise, str(item.file_path), item.start)):
        by_subject_exercise.setdefault((segment.subject, segment.exercise), []).append(segment)
        by_exercise.setdefault(segment.exercise, []).append(segment)

    def build_template(segments: Sequence[RepSegment], limit: int) -> Template | None:
        waves: list[np.ndarray] = []
        durations: list[int] = []
        reference: np.ndarray | None = None
        for segment in segments[:limit]:
            df = session_cache.get(segment.file_path)
            if df is None:
                continue
            local_df = df.iloc[segment.start : segment.end]
            signal = principal_signal(local_df, smooth_window, IMU9_COLUMNS)
            values = robust_zscore(resample(signal, template_points))
            if reference is None:
                reference = values
            elif float(np.dot(values, reference)) < 0:
                values = -values
            waves.append(values)
            durations.append(segment.n_samples)
        if not waves:
            return None
        return Template(np.median(np.vstack(waves), axis=0), float(np.median(durations)), len(waves))

    subject_templates: dict[tuple[str, str], Template] = {}
    for key, segments in by_subject_exercise.items():
        template = build_template(segments, calibration_reps)
        if template is not None:
            subject_templates[key] = template

    exercise_templates: dict[str, Template] = {}
    for exercise, segments in by_exercise.items():
        template = build_template(segments, min(max(calibration_reps * 2, 4), len(segments)))
        if template is not None:
            exercise_templates[exercise] = template
    return subject_templates, exercise_templates


def template_for_block(
    block: RepSegment,
    subject_templates: dict[tuple[str, str], Template],
    exercise_templates: dict[str, Template],
) -> Template | None:
    return subject_templates.get((block.subject, block.exercise)) or exercise_templates.get(block.exercise)


def predict_stayfit_ba(block: RepSegment, df: pd.DataFrame, args: argparse.Namespace) -> list[RepSegment]:
    features = block_features(block, df, args)
    signals = features["signals"]
    if not signals:
        return segments_from_boundaries(block, [0, block.n_samples], args.min_segment_samples, "stayfit_ba")
    period_scores = features["periods"]
    ranked: list[tuple[float, str, np.ndarray, float | None]] = []
    for name, signal in signals.items():
        period, score = period_scores.get(name, (None, 0.0))
        ranked.append((score, name, signal, period))
    score, _name, signal, period = max(ranked, key=lambda item: item[0])
    count = period_count(block.n_samples, period, args.min_segment_samples, args.max_reps)
    peaks, _strength = select_top_peaks(
        signal,
        count,
        args.min_segment_samples,
        period,
        args.peak_prominence_scale,
        0.55,
    )
    if len(peaks) and abs(len(peaks) - count) <= max(1, int(round(0.35 * count))):
        boundaries = boundaries_from_centers(block.n_samples, peaks, count)
    else:
        boundaries = uniform_boundaries(block.n_samples, count)
    search_radius = max(args.min_segment_samples, int(round((period or block.n_samples / max(count, 1)) * args.boundary_search_fraction)))
    boundary_score = features["boundary_score"]
    boundaries = refine_boundaries_by_score(boundaries, boundary_score, args.min_segment_samples, search_radius)
    return segments_from_boundaries(block, boundaries, args.min_segment_samples, "stayfit_ba")


def predict_maxxyt_map(block: RepSegment, df: pd.DataFrame, args: argparse.Namespace) -> list[RepSegment]:
    features = block_features(block, df, args)
    signals = features["signals"]
    period_scores = features["periods"]
    count, chosen_name, peaks, _score = multi_axis_count_vote(
        signals,
        block.n_samples,
        args.min_segment_samples,
        args.max_reps,
        args.peak_prominence_scale,
        period_scores=period_scores,
    )
    if len(peaks) and abs(len(peaks) - count) <= 1:
        boundaries = boundaries_from_centers(block.n_samples, peaks, count)
    else:
        boundaries = uniform_boundaries(block.n_samples, count)
    chosen = signals.get(chosen_name, signals.get("imu9_pca", np.zeros(block.n_samples)))
    period, _ = period_scores.get(chosen_name, (None, 0.0))
    if period is None:
        period, _ = autocorr_period_and_score(chosen, args.min_segment_samples, max(args.min_segment_samples, int(round(block.n_samples * args.max_period_fraction))))
    search_radius = max(args.min_segment_samples, int(round((period or block.n_samples / max(count, 1)) * args.boundary_search_fraction)))
    gyro_score = features["gyro_score"]
    boundaries = refine_boundaries_by_score(boundaries, gyro_score, args.min_segment_samples, search_radius)
    return segments_from_boundaries(block, boundaries, args.min_segment_samples, "maxxyt_map")


def predict_mfitness_fste(block: RepSegment, df: pd.DataFrame, args: argparse.Namespace) -> list[RepSegment]:
    features = block_features(block, df, args)
    energy = features["energy"]
    period, _score = features["periods"].get("imu9_pca", (None, 0.0))
    count = period_count(block.n_samples, period, args.min_segment_samples, args.max_reps)
    peaks, _strength = select_top_peaks(
        energy,
        count,
        args.min_segment_samples,
        period,
        prominence_scale=max(0.15, args.peak_prominence_scale * 0.7),
        distance_scale=0.55,
    )
    if len(peaks) and len(peaks) >= 2 and abs(len(peaks) - count) <= max(1, int(round(0.35 * count))):
        boundaries = [0]
        for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
            if right_peak <= left_peak:
                continue
            local = energy[left_peak : right_peak + 1]
            boundaries.append(int(left_peak + np.argmin(local)))
        boundaries.append(block.n_samples)
    elif len(peaks):
        boundaries = boundaries_from_centers(block.n_samples, peaks, count)
    else:
        boundaries = uniform_boundaries(block.n_samples, count)
    search_radius = max(args.min_segment_samples, int(round((period or block.n_samples / max(count, 1)) * args.boundary_search_fraction)))
    boundaries = refine_boundaries_by_score(boundaries, valley_cost_signal(np.abs(energy)), args.min_segment_samples, search_radius)
    return segments_from_boundaries(block, boundaries, args.min_segment_samples, "mfitness_fste")


def predict_cara_dtw_fs(
    block: RepSegment,
    df: pd.DataFrame,
    args: argparse.Namespace,
    subject_templates: dict[tuple[str, str], Template],
    exercise_templates: dict[str, Template],
) -> list[RepSegment]:
    features = block_features(block, df, args)
    signal = features["pca"]
    template = template_for_block(block, subject_templates, exercise_templates)
    if template is not None and template.duration_samples > 0:
        count = int(round(block.n_samples / template.duration_samples))
        count = int(np.clip(count, 1, max(1, min(args.max_reps, block.n_samples // max(args.min_segment_samples, 1)))))
    else:
        period, _ = autocorr_period_and_score(signal, args.min_segment_samples, max(args.min_segment_samples, int(round(block.n_samples * args.max_period_fraction))))
        count = period_count(block.n_samples, period, args.min_segment_samples, args.max_reps)
    boundaries = uniform_boundaries(block.n_samples, count)
    period_guess = template.duration_samples if template is not None else block.n_samples / max(count, 1)
    search_radius = max(args.min_segment_samples, int(round(period_guess * args.boundary_search_fraction)))
    score = features["boundary_score"]
    boundaries = refine_boundaries_template_aware(
        boundaries,
        score,
        signal,
        template,
        args.min_segment_samples,
        search_radius,
        shape_weight=0.16,
        use_dtw=args.use_dtw_shape_cost,
    )
    return segments_from_boundaries(block, boundaries, args.min_segment_samples, "cara_dtw_fs")


def predict_lift_fusion(
    block: RepSegment,
    df: pd.DataFrame,
    args: argparse.Namespace,
    subject_templates: dict[tuple[str, str], Template],
    exercise_templates: dict[str, Template],
) -> list[RepSegment]:
    features = block_features(block, df, args)
    signals = features["signals"]
    period_scores = features["periods"]
    pca = features["pca"]
    period, period_score = period_scores.get("imu9_pca", (None, 0.0))
    p_count = period_count(block.n_samples, period, args.min_segment_samples, args.max_reps)
    max_count, _chosen_name, _peaks, _score = multi_axis_count_vote(
        signals,
        block.n_samples,
        args.min_segment_samples,
        args.max_reps,
        args.peak_prominence_scale,
        period_scores=period_scores,
    )
    energy = features["energy"]
    energy_peaks, _ = select_top_peaks(
        energy,
        p_count,
        args.min_segment_samples,
        period,
        prominence_scale=max(0.15, args.peak_prominence_scale * 0.7),
        distance_scale=0.55,
    )
    energy_count = len(energy_peaks) if len(energy_peaks) else p_count
    template = template_for_block(block, subject_templates, exercise_templates)

    values = [p_count, max_count, energy_count]
    weights = [1.4 + period_score, 1.5, 0.9]
    if template is not None and template.duration_samples > 0:
        t_count = int(round(block.n_samples / template.duration_samples))
        t_count = int(np.clip(t_count, 1, max(1, min(args.max_reps, block.n_samples // max(args.min_segment_samples, 1)))))
        values.extend([t_count, t_count])
        weights.extend([1.2, 1.0])
    count = weighted_integer_median(values, weights)
    count = int(np.clip(count, 1, max(1, min(args.max_reps, block.n_samples // max(args.min_segment_samples, 1)))))

    boundaries = uniform_boundaries(block.n_samples, count)
    period_guess = period if period is not None else (template.duration_samples if template is not None else block.n_samples / max(count, 1))
    search_radius = max(args.min_segment_samples, int(round(period_guess * args.boundary_search_fraction)))
    score = features["boundary_score"]
    boundaries = refine_boundaries_template_aware(
        boundaries,
        score,
        pca,
        template,
        args.min_segment_samples,
        search_radius,
        shape_weight=0.10 if template is not None else 0.0,
        use_dtw=args.use_dtw_shape_cost,
    )
    return segments_from_boundaries(block, boundaries, args.min_segment_samples, "lift_fusion")


def infer_sensor_period_seconds(df: pd.DataFrame) -> float:
    if "sensor_ts" not in df.columns:
        return 0.01
    values = pd.to_numeric(df["sensor_ts"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(values) < 2:
        return 0.01
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.01
    median_delta = float(np.median(diffs))
    if median_delta > 1000.0:
        return median_delta / 1_000_000.0
    if median_delta > 10.0:
        return median_delta / 1000.0
    return median_delta


def set_table_from_truth(truth: Sequence[RepSegment], periods: dict[Path, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[Path, str, str, str], list[RepSegment]] = {}
    for segment in truth:
        grouped.setdefault((segment.file_path, segment.subject, segment.exercise, segment.set_id), []).append(segment)
    for (file_path, subject, exercise, set_id), segments in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2], item[0][3])):
        period = periods.get(file_path, 0.01)
        rows.append(
            {
                "file": str(file_path),
                "subject": subject,
                "exercise": exercise,
                "set_id": set_id,
                "set_start": min(segment.start for segment in segments),
                "set_end": max(segment.end for segment in segments),
                "true_count": len(segments),
                "true_tut_samples": sum(segment.n_samples for segment in segments),
                "true_tut_sec": sum(segment.n_samples for segment in segments) * period,
            }
        )
    return pd.DataFrame(rows)


def assign_predictions_to_sets(predicted: Sequence[RepSegment], sets: pd.DataFrame, periods: dict[Path, float]) -> tuple[pd.DataFrame, int]:
    rows = sets.copy()
    rows["pred_count"] = 0
    rows["pred_tut_samples"] = 0
    rows["pred_tut_sec"] = 0.0
    by_file_exercise: dict[tuple[str, str], list[int]] = {}
    for idx, row in rows.iterrows():
        by_file_exercise.setdefault((str(row["file"]), str(row["exercise"])), []).append(int(idx))

    unassigned = 0
    for segment in predicted:
        key = (str(segment.file_path), segment.exercise)
        best_idx: int | None = None
        best_overlap = 0
        for idx in by_file_exercise.get(key, []):
            row = rows.loc[idx]
            overlap = max(0, min(segment.end, int(row["set_end"])) - max(segment.start, int(row["set_start"])))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx is None or best_overlap <= 0:
            unassigned += 1
            continue
        period = periods.get(segment.file_path, 0.01)
        rows.loc[best_idx, "pred_count"] += 1
        rows.loc[best_idx, "pred_tut_samples"] += segment.n_samples
        rows.loc[best_idx, "pred_tut_sec"] += segment.n_samples * period
    return rows, unassigned


def count_metric_row(
    method: MethodSpec,
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    periods: dict[Path, float],
    output_dir: Path,
) -> dict[str, object]:
    sets = set_table_from_truth(truth, periods)
    set_counts, unassigned = assign_predictions_to_sets(predicted, sets, periods)
    set_counts["count_error"] = set_counts["pred_count"] - set_counts["true_count"]
    set_counts["abs_count_error"] = set_counts["count_error"].abs()
    set_counts["tut_error_sec"] = set_counts["pred_tut_sec"] - set_counts["true_tut_sec"]
    set_counts["abs_tut_error_sec"] = set_counts["tut_error_sec"].abs()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_counts.to_csv(output_dir / f"{method.method_id}_set_count_details.csv", index=False)
    return {
        "method_id": method.method_id,
        "method_name": method.method_name,
        "paper_anchor": method.paper_anchor,
        "uses_few_shot_labels": method.uses_few_shot_labels,
        "true_sets": int(len(set_counts)),
        "true_reps": int(len(truth)),
        "predicted_reps": int(len(predicted)),
        "unassigned_pred_reps": int(unassigned),
        "count_exact_acc": round(float((set_counts["abs_count_error"] == 0).mean()), 4),
        "count_pm1_acc": round(float((set_counts["abs_count_error"] <= 1).mean()), 4),
        "count_mae_reps": round(float(set_counts["abs_count_error"].mean()), 4),
        "count_bias_reps": round(float(set_counts["count_error"].mean()), 4),
        "tut_mae_sec": round(float(set_counts["abs_tut_error_sec"].mean()), 4),
    }


def metric_lookup(rows: Sequence[dict[str, object]]) -> dict[float, dict[str, object]]:
    return {float(row["iou_threshold"]): row for row in rows}


def write_method_outputs(
    method: MethodSpec,
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    phase_truth: Sequence[PhaseSegment],
    session_cache: dict[Path, pd.DataFrame],
    phase_orders: dict[str, tuple[str, str]],
    periods: dict[Path, float],
    args: argparse.Namespace,
    method_dir: Path,
) -> dict[str, object]:
    method_dir.mkdir(parents=True, exist_ok=True)
    write_csv(method_dir / "rep_segmentation_pred_segments.csv", segmentation_summary(predicted, truth))
    write_csv(method_dir / "rep_segmentation_truth_segments.csv", best_truth_match_rows(predicted, truth))
    write_csv(method_dir / "rep_segments_manifest.csv", rep_segment_manifest(predicted))

    metric_rows = segmentation_metric_rows(predicted, truth, args.segmentation_iou_thresholds)
    by_exercise_rows = segmentation_metric_rows_by_exercise(predicted, truth, args.segmentation_iou_thresholds)
    by_subject_rows = segmentation_metric_rows_by_subject(predicted, truth, args.segmentation_iou_thresholds)
    write_csv(method_dir / "rep_segmentation_metrics.csv", metric_rows)
    write_csv(method_dir / "rep_segmentation_metrics_by_exercise.csv", by_exercise_rows)
    write_csv(method_dir / "rep_segmentation_metrics_by_subject.csv", by_subject_rows)
    plot_segmentation_metrics(metric_rows, method_dir)
    plot_segmentation_metrics_by_exercise(by_exercise_rows, method_dir)
    plot_exercise_accuracy_table(by_exercise_rows, method_dir, args.segmentation_iou_thresholds)
    plot_segmentation_metrics_by_subject(by_subject_rows, method_dir)

    phase_rows: list[dict[str, object]] = []
    if phase_truth:
        predicted_phases = predict_phase_segments(
            predicted,
            session_cache,
            phase_orders,
            method=args.phase_split_method,
            smooth_window=args.smooth_window,
            min_phase_samples=args.min_phase_segment_samples,
        )
        phase_rows = phase_metric_rows(predicted_phases, phase_truth, args.phase_iou_thresholds)
        phase_by_phase_rows = phase_metric_rows_by_phase(predicted_phases, phase_truth, args.phase_iou_thresholds)
        write_csv(method_dir / "phase_split_metrics.csv", phase_rows)
        write_csv(method_dir / "phase_split_metrics_by_phase.csv", phase_by_phase_rows)
        plot_phase_metrics(phase_rows, method_dir)
        plot_phase_metrics_by_phase(phase_by_phase_rows, method_dir)

    count_row = count_metric_row(method, predicted, truth, periods, method_dir / "set_details")
    rep_lookup = metric_lookup(metric_rows)
    phase_lookup = metric_lookup(phase_rows)
    for threshold in args.segmentation_iou_thresholds:
        row = rep_lookup.get(float(threshold), {})
        count_row[f"rep_precision_iou_{threshold:.2f}"] = row.get("precision", np.nan)
        count_row[f"rep_recall_iou_{threshold:.2f}"] = row.get("recall", np.nan)
        count_row[f"rep_f1_iou_{threshold:.2f}"] = row.get("f1", np.nan)
        count_row[f"rep_mean_matched_iou_{threshold:.2f}"] = row.get("mean_matched_iou", np.nan)
    for threshold in args.phase_iou_thresholds:
        row = phase_lookup.get(float(threshold), {})
        count_row[f"phase_f1_iou_{threshold:.2f}"] = row.get("f1", np.nan)
    return count_row


def rep_segment_manifest(segments: Sequence[RepSegment]) -> list[dict[str, object]]:
    return [
        {
            "file": str(segment.file_path),
            "subject": segment.subject,
            "exercise": segment.exercise,
            "set_id": segment.set_id,
            "rep_id": segment.rep_id,
            "start": segment.start,
            "end": segment.end,
            "samples": segment.n_samples,
            "source": segment.source,
        }
        for segment in segments
    ]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:180] or "item"


def plot_method_table(df: pd.DataFrame, output_dir: Path) -> None:
    display_cols = [
        "method_name",
        "count_exact_acc",
        "count_pm1_acc",
        "count_mae_reps",
        "rep_f1_iou_0.50",
        "rep_f1_iou_0.75",
        "rep_f1_iou_0.90",
        "phase_f1_iou_0.50",
        "tut_mae_sec",
    ]
    labels = [
        "Method",
        "Count exact",
        "Count +/-1",
        "Count MAE",
        "Rep F1@0.50",
        "Rep F1@0.75",
        "Rep F1@0.90",
        "Phase F1@0.50",
        "TUT MAE (s)",
    ]
    available = [column for column in display_cols if column in df.columns]
    table = df.loc[:, available].copy()
    for column in table.columns:
        if column == "method_name":
            continue
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    fig, ax = plt.subplots(figsize=(16, max(3.6, 0.48 * len(table) + 1.8)))
    ax.axis("off")
    artists = ax.table(
        cellText=table.to_numpy(),
        colLabels=[labels[display_cols.index(column)] for column in available],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    artists.auto_set_font_size(False)
    artists.set_fontsize(8)
    artists.scale(1, 1.35)
    for (row_idx, _col_idx), cell in artists.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#dbe8f6")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    ax.set_title("014 Literature-Inspired Rep Segmentation Method Comparison", pad=18)
    fig.tight_layout()
    fig.savefig(output_dir / "014_literature_method_comparison_table.png", dpi=180)
    plt.close(fig)


def plot_method_bars(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("count_pm1_acc", "Count +/-1"),
        ("rep_f1_iou_0.50", "Rep F1@0.50"),
        ("rep_f1_iou_0.75", "Rep F1@0.75"),
        ("rep_f1_iou_0.90", "Rep F1@0.90"),
    ]
    x = np.arange(len(df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 1.4), 5.8))
    for idx, (column, label) in enumerate(metrics):
        if column not in df.columns:
            continue
        ax.bar(x + (idx - 1.5) * width, df[column].fillna(0.0).astype(float).to_numpy(), width, label=label)
    ax.axhline(0.90, color="#d62728", linestyle="--", linewidth=1.1, label="0.90 target")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method_name"].tolist(), rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Count Accuracy and Rep Boundary IoU")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "014_literature_method_score_bars.png", dpi=180)
    plt.close(fig)


def plot_method_exercise_heatmap(rows: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    if rows.empty:
        return
    subset = rows[np.isclose(rows["iou_threshold"].astype(float), threshold)]
    if subset.empty:
        return
    methods = sorted(subset["method_name"].astype(str).unique().tolist())
    exercises = sorted(subset["exercise"].astype(str).unique().tolist())
    matrix = np.zeros((len(methods), len(exercises)), dtype=np.float64)
    lookup = {
        (str(row.method_name), str(row.exercise)): float(row.f1)
        for row in subset.itertuples(index=False)
    }
    for i, method in enumerate(methods):
        for j, exercise in enumerate(exercises):
            matrix[i, j] = lookup.get((method, exercise), 0.0)
    fig, ax = plt.subplots(figsize=(max(10, len(exercises) * 1.0), max(5, len(methods) * 0.55)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(exercises)))
    ax.set_xticklabels(exercises, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(f"Method x Exercise Rep F1@IoU {threshold:.2f}")
    for i in range(len(methods)):
        for j in range(len(exercises)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value >= 0.5 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(output_dir / f"014_method_exercise_f1_iou_{threshold:.2f}.png", dpi=180)
    plt.close(fig)


def plot_method_subject_heatmap(rows: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    if rows.empty:
        return
    subset = rows[np.isclose(rows["iou_threshold"].astype(float), threshold)]
    if subset.empty:
        return
    methods = sorted(subset["method_name"].astype(str).unique().tolist())
    subjects = sorted(subset["subject"].astype(str).unique().tolist())
    matrix = np.zeros((len(methods), len(subjects)), dtype=np.float64)
    lookup = {
        (str(row.method_name), str(row.subject)): float(row.f1)
        for row in subset.itertuples(index=False)
    }
    for i, method in enumerate(methods):
        for j, subject in enumerate(subjects):
            matrix[i, j] = lookup.get((method, subject), 0.0)
    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.9), max(5, len(methods) * 0.55)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(subjects)))
    ax.set_xticklabels(subjects, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(f"Method x Subject Rep F1@IoU {threshold:.2f}")
    for i in range(len(methods)):
        for j in range(len(subjects)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value >= 0.5 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(output_dir / f"014_method_subject_f1_iou_{threshold:.2f}.png", dpi=180)
    plt.close(fig)


def plot_waveform_comparison(
    block: RepSegment,
    truth: Sequence[RepSegment],
    predictions_by_method: dict[str, Sequence[RepSegment]],
    df: pd.DataFrame,
    output_path: Path,
    smooth_window: int,
) -> None:
    local_df = df.iloc[block.start : block.end]
    signal = principal_signal(local_df, smooth_window, IMU9_COLUMNS)
    signal = robust_zscore(signal)
    x = np.arange(block.n_samples)
    rows = [("Ground truth", truth, "#2ca02c")]
    rows.extend((method, segments, "#d62728") for method, segments in predictions_by_method.items())
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, max(4.0, 1.45 * len(rows))), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, (label, segments, color) in zip(axes, rows):
        ax.plot(x, signal, color="#3b4856", linewidth=0.9)
        boundaries: set[int] = set()
        for segment in segments:
            if segment.file_path != block.file_path:
                continue
            if min(segment.end, block.end) <= max(segment.start, block.start):
                continue
            boundaries.add(max(0, segment.start - block.start))
            boundaries.add(min(block.n_samples, segment.end - block.start))
        for boundary in sorted(boundaries):
            ax.axvline(boundary, color=color, linewidth=1.0, alpha=0.9)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=82, fontsize=8)
        ax.grid(axis="x", alpha=0.16)
        ax.set_yticks([])
    axes[-1].set_xlabel("Sample in active set")
    title = f"{block.subject} | {block.exercise} | set {block.set_id}"
    axes[0].set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def predictions_for_block(block: RepSegment, predictions: Sequence[RepSegment]) -> list[RepSegment]:
    base_set_id = block.set_id.split(":active", 1)[0]
    return [
        segment
        for segment in predictions
        if segment.file_path == block.file_path
        and segment.exercise == block.exercise
        and segment.set_id == base_set_id
        and min(segment.end, block.end) > max(segment.start, block.start)
    ]


def write_literature_notes(output_dir: Path) -> None:
    rows = [
        {
            "method_name": spec.method_name,
            "method_id": spec.method_id,
            "paper_anchor": spec.paper_anchor,
            "implemented_role": spec.description,
            "known_weakness": spec.weakness,
            "new_method_uses": spec.improvement_used,
            "uses_few_shot_labels": spec.uses_few_shot_labels,
        }
        for spec in METHOD_SPECS
    ]
    write_csv(output_dir / "014_literature_method_notes.csv", rows)


def append_prior_rows(df: pd.DataFrame, prior_csv: Path) -> pd.DataFrame:
    if not prior_csv.exists():
        return df
    prior = pd.read_csv(prior_csv)
    keep_ids = {"010_universal_gyro_valley", "011_multifeature_boundary_score", "012_exercise_only_ds_ms_tcn"}
    prior = prior[prior["method_id"].astype(str).isin(keep_ids)].copy()
    if prior.empty:
        return df
    prior["paper_anchor"] = "Existing project baseline"
    prior["uses_few_shot_labels"] = False
    common_cols = sorted(set(df.columns) | set(prior.columns))
    return pd.concat([prior.reindex(columns=common_cols), df.reindex(columns=common_cols)], ignore_index=True)


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
        if not all(column in df.columns for column in GYRO_COLUMNS):
            continue
        session_cache[path] = df
        periods[path] = infer_sensor_period_seconds(df)
        truth.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))
        phase_truth.extend(true_phase_segments(df, path, min_samples=args.min_phase_segment_samples))

    if not truth:
        raise RuntimeError("No labeled active rep segments found.")

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
    for spec in METHOD_SPECS:
        method_predictions: list[RepSegment] = []
        for block in blocks:
            df = session_cache[block.file_path]
            if spec.method_id == "stayfit_ba":
                method_predictions.extend(predict_stayfit_ba(block, df, args))
            elif spec.method_id == "maxxyt_map":
                method_predictions.extend(predict_maxxyt_map(block, df, args))
            elif spec.method_id == "mfitness_fste":
                method_predictions.extend(predict_mfitness_fste(block, df, args))
            elif spec.method_id == "cara_dtw_fs":
                method_predictions.extend(predict_cara_dtw_fs(block, df, args, subject_templates, exercise_templates))
            elif spec.method_id == "lift_fusion":
                method_predictions.extend(predict_lift_fusion(block, df, args, subject_templates, exercise_templates))
            else:
                raise ValueError(f"Unsupported method: {spec.method_id}")
        predictions[spec.method_id] = method_predictions

    summary_rows: list[dict[str, object]] = []
    exercise_rows_all: list[dict[str, object]] = []
    subject_rows_all: list[dict[str, object]] = []
    spec_by_id = {spec.method_id: spec for spec in METHOD_SPECS}
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

    summary_df = pd.DataFrame(summary_rows)
    ordered_cols = [
        "method_name",
        "method_id",
        "paper_anchor",
        "uses_few_shot_labels",
        "true_sets",
        "true_reps",
        "predicted_reps",
        "unassigned_pred_reps",
        "count_exact_acc",
        "count_pm1_acc",
        "count_mae_reps",
        "count_bias_reps",
        "tut_mae_sec",
        "rep_precision_iou_0.50",
        "rep_recall_iou_0.50",
        "rep_f1_iou_0.50",
        "rep_f1_iou_0.75",
        "rep_f1_iou_0.90",
        "phase_f1_iou_0.50",
    ]
    ordered_cols = [column for column in ordered_cols if column in summary_df.columns]
    summary_df = summary_df.loc[:, ordered_cols]
    summary_df.to_csv(args.output_dir / "014_literature_method_comparison.csv", index=False)

    combined_df = append_prior_rows(summary_df, args.prior_comparison_csv)
    combined_df.to_csv(args.output_dir / "014_literature_method_comparison_with_prior.csv", index=False)
    plot_method_table(combined_df, args.output_dir)
    plot_method_bars(combined_df, args.output_dir)

    exercise_df = pd.DataFrame(exercise_rows_all)
    subject_df = pd.DataFrame(subject_rows_all)
    exercise_df.to_csv(args.output_dir / "014_literature_method_by_exercise.csv", index=False)
    subject_df.to_csv(args.output_dir / "014_literature_method_by_subject.csv", index=False)
    for threshold in args.segmentation_iou_thresholds:
        plot_method_exercise_heatmap(exercise_df, args.output_dir, float(threshold))
        plot_method_subject_heatmap(subject_df, args.output_dir, float(threshold))

    write_literature_notes(args.output_dir)

    waveform_dir = args.output_dir / "waveform_all_sets"
    for plot_idx, block in enumerate(blocks):
        if args.max_waveform_plots is not None and plot_idx >= args.max_waveform_plots:
            break
        df = session_cache[block.file_path]
        block_truth = truth_segments_for_block(block, truth)
        block_predictions = {
            spec.method_name: predictions_for_block(block, predictions[spec.method_id])
            for spec in METHOD_SPECS
        }
        filename = safe_name(f"{plot_idx + 1:03d}_{block.subject}_{block.exercise}_{block.set_id}_{block.file_path.stem}") + ".png"
        plot_waveform_comparison(block, block_truth, block_predictions, df, waveform_dir / filename, args.smooth_window)

    summary = {
        "output_dir": str(args.output_dir),
        "data_dirs": [str(path) for path in args.data_dirs],
        "active_blocks": len(blocks),
        "true_reps": len(truth),
        "phase_segments": len(phase_truth),
        "methods": [spec.method_name for spec in METHOD_SPECS],
        "comparison_csv": str(args.output_dir / "014_literature_method_comparison.csv"),
        "comparison_with_prior_csv": str(args.output_dir / "014_literature_method_comparison_with_prior.csv"),
        "waveform_plots": str(waveform_dir),
        "assumptions": {
            "domain": "active-only exercise spans from labels; upstream active/rest detection is not evaluated here",
            "exercise_hint": "Each active block keeps its labeled exercise hint, matching the current rep-boundary refinement setup.",
            "few_shot": f"CARA-DTW-FS and LIFT-Fusion use up to {args.calibration_reps} labeled reps per subject/exercise as personalization templates.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate literature-inspired active-only rep segmentation methods.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/014_literature_inspired_rep_methods"))
    parser.add_argument("--prior-comparison-csv", type=Path, default=Path("artifacts_rep_classification/013_count_iou_tut_method_table/count_iou_tut_method_comparison.csv"))
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
    parser.add_argument("--use-dtw-shape-cost", action="store_true", help="Use true DTW for template cost. Default uses faster sign-invariant resampled shape distance.")
    parser.add_argument("--segmentation-iou-thresholds", type=float, nargs="+", default=[0.50, 0.75, 0.90])
    parser.add_argument("--phase-iou-thresholds", type=float, nargs="+", default=[0.50, 0.75, 0.90])
    parser.add_argument("--phase-split-method", choices=["midpoint", "pca-reversal"], default="pca-reversal")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-waveform-plots", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
