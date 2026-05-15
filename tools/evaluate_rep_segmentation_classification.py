from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")
ACTIVE_PHASES = {"concentric", "eccentric"}
REST_LABELS = {"big_rest", "rest", "none", "nan", ""}


@dataclass(frozen=True)
class RepSegment:
    file_path: Path
    subject: str
    exercise: str
    set_id: str
    rep_id: str
    start: int
    end: int
    source: str

    @property
    def n_samples(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PhaseSegment:
    file_path: Path
    subject: str
    exercise: str
    set_id: str
    rep_id: str
    phase: str
    start: int
    end: int
    source: str

    @property
    def n_samples(self) -> int:
        return self.end - self.start


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def whole_session_files(data_dirs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for data_dir in data_dirs:
        files.extend(sorted(data_dir.rglob("*whole_session*.csv")))
    return sorted(set(files))


def subject_from_path(path: Path, data_dirs: Sequence[Path]) -> str:
    for data_dir in data_dirs:
        try:
            return path.relative_to(data_dir).parts[0]
        except ValueError:
            continue
    return path.parent.name


def clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_session(path: Path, data_dirs: Sequence[Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / scale


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


def principal_motion_signal(df: pd.DataFrame, smooth_window: int) -> np.ndarray:
    available = [col for col in IMU_COLUMNS if col in df.columns]
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    x = np.apply_along_axis(robust_zscore, 0, x)
    variances = np.var(x, axis=0)
    x = x[:, variances > 1e-9]
    if x.shape[1] == 0:
        return np.zeros(len(df), dtype=np.float64)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return moving_average(x @ vt[0], smooth_window)


def dominant_axis_signal(df: pd.DataFrame, smooth_window: int) -> np.ndarray:
    available = [col for col in IMU_COLUMNS if col in df.columns]
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    if x.size == 0:
        return np.zeros(len(df), dtype=np.float64)
    x = np.apply_along_axis(robust_zscore, 0, x)
    axis_idx = int(np.argmax(np.var(x, axis=0)))
    return moving_average(x[:, axis_idx], smooth_window)


def acceleration_magnitude_signal(df: pd.DataFrame, smooth_window: int) -> np.ndarray:
    available = [col for col in ("ax", "ay", "az") if col in df.columns]
    if not available:
        return np.zeros(len(df), dtype=np.float64)
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    magnitude = np.linalg.norm(x, axis=1)
    magnitude = robust_zscore(magnitude)
    return moving_average(np.abs(magnitude), smooth_window)


def short_time_energy(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, min(window, len(signal)))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return signal**2
    pad = window // 2
    padded = np.pad(signal.astype(np.float64), pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64)
    return np.convolve(padded**2, kernel, mode="valid")


def true_rep_segments(df: pd.DataFrame, path: Path, min_samples: int) -> list[RepSegment]:
    phases = df["phase"].map(clean_label).str.lower()
    active = phases.isin(ACTIVE_PHASES).to_numpy()
    if not active.any():
        return []

    subject_values = df["subject_id"].map(clean_label)
    exercise_values = df["action_type"].map(clean_label)
    set_values = df["set"].map(clean_label)
    rep_values = df["rep"].map(clean_label)

    segments: list[RepSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str, str] | None = None

    for idx, is_active in enumerate(active):
        key = (
            subject_values.iloc[idx],
            exercise_values.iloc[idx],
            set_values.iloc[idx],
            rep_values.iloc[idx],
        )
        if is_active and start is None:
            start = idx
            last_key = key
        elif is_active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                segments.append(RepSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], start, idx, "label"))
            start = idx
            last_key = key
        elif (not is_active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                segments.append(RepSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], start, idx, "label"))
            start = None
            last_key = None

    if start is not None and last_key is not None and len(df) - start >= min_samples:
        segments.append(RepSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], start, len(df), "label"))
    return segments


def set_blocks_from_labels(df: pd.DataFrame, path: Path, min_samples: int) -> list[RepSegment]:
    actions = df["action_type"].map(clean_label)
    sets = df["set"].map(clean_label)
    subjects = df["subject_id"].map(clean_label)
    non_rest = ~actions.str.lower().isin(REST_LABELS)

    blocks: list[RepSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str] | None = None
    for idx, active in enumerate(non_rest.to_numpy()):
        key = (subjects.iloc[idx], actions.iloc[idx], sets.iloc[idx])
        if active and start is None:
            start = idx
            last_key = key
        elif active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                blocks.append(RepSegment(path, last_key[0], last_key[1], last_key[2], "set", start, idx, "set_block"))
            start = idx
            last_key = key
        elif (not active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                blocks.append(RepSegment(path, last_key[0], last_key[1], last_key[2], "set", start, idx, "set_block"))
            start = None
            last_key = None
    if start is not None and last_key is not None and len(df) - start >= min_samples:
        blocks.append(RepSegment(path, last_key[0], last_key[1], last_key[2], "set", start, len(df), "set_block"))
    return blocks


def active_phase_blocks_from_truth(true_segments: Sequence[RepSegment], min_samples: int) -> list[RepSegment]:
    grouped: dict[tuple[Path, str, str, str], list[RepSegment]] = {}
    for segment in true_segments:
        grouped.setdefault((segment.file_path, segment.subject, segment.exercise, segment.set_id), []).append(segment)

    blocks: list[RepSegment] = []
    for (file_path, subject, exercise, set_id), segments in grouped.items():
        start = min(segment.start for segment in segments)
        end = max(segment.end for segment in segments)
        if end - start >= min_samples:
            blocks.append(RepSegment(file_path, subject, exercise, set_id, "set", start, end, "active_phase_span"))
    return sorted(blocks, key=lambda segment: (str(segment.file_path), segment.subject, segment.exercise, segment.set_id, segment.start))


def candidate_blocks(
    df: pd.DataFrame,
    path: Path,
    true_segments: Sequence[RepSegment],
    min_samples: int,
    block_source: str,
) -> list[RepSegment]:
    if block_source == "action-label":
        return set_blocks_from_labels(df, path, min_samples=min_samples)
    if block_source == "active-phase-span":
        return active_phase_blocks_from_truth(true_segments, min_samples=min_samples)
    raise ValueError(f"Unsupported block source: {block_source}")


def pca_extrema_segments(
    df: pd.DataFrame,
    path: Path,
    true_segments: Sequence[RepSegment],
    smooth_window: int,
    min_samples: int,
    peak_prominence_scale: float,
    peak_distance_scale: float,
    block_source: str,
) -> list[RepSegment]:
    by_block: dict[tuple[str, str, str], list[RepSegment]] = {}
    for segment in true_segments:
        by_block.setdefault((segment.subject, segment.exercise, segment.set_id), []).append(segment)

    predicted: list[RepSegment] = []
    for block in candidate_blocks(df, path, true_segments, min_samples=min_samples, block_source=block_source):
        truth = sorted(by_block.get((block.subject, block.exercise, block.set_id), []), key=lambda s: s.start)
        if not truth:
            continue
        expected = len(truth)
        segment_df = df.iloc[block.start : block.end]
        signal = principal_motion_signal(segment_df, smooth_window)
        prominence = max(float(np.std(signal)) * peak_prominence_scale, 1e-6)
        distance = max(int(round(min_samples * peak_distance_scale)), 1)

        candidates: list[np.ndarray] = []
        for candidate_signal in (signal, -signal):
            peaks, _ = find_peaks(candidate_signal, distance=distance, prominence=prominence)
            if len(peaks) >= 2:
                candidates.append(peaks)

        if not candidates:
            continue
        peaks = min(candidates, key=lambda p: abs(len(p) - expected))
        if len(peaks) == 0:
            continue

        centers = np.sort(peaks)
        boundaries = [0]
        if len(centers) > 1:
            boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(len(centers) - 1))
        boundaries.append(len(segment_df))

        for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < min_samples:
                continue
            predicted.append(
                RepSegment(
                    path,
                    block.subject,
                    block.exercise,
                    block.set_id,
                    str(rep_idx),
                    block.start + start,
                    block.start + end,
                    "pca_extrema",
                )
            )
    return predicted


def extrema_segments_from_signal(
    df: pd.DataFrame,
    path: Path,
    signal_fn,
    source: str,
    smooth_window: int,
    min_samples: int,
    peak_prominence_scale: float,
    peak_distance_scale: float,
    true_segments: Sequence[RepSegment],
    block_source: str,
) -> list[RepSegment]:
    predicted: list[RepSegment] = []
    for block in candidate_blocks(df, path, true_segments, min_samples=min_samples, block_source=block_source):
        segment_df = df.iloc[block.start : block.end]
        signal = signal_fn(segment_df, smooth_window)
        if len(signal) < min_samples * 2:
            continue
        prominence = max(float(np.std(signal)) * peak_prominence_scale, 1e-6)
        distance = max(int(round(min_samples * peak_distance_scale)), 1)

        candidates: list[np.ndarray] = []
        for candidate_signal in (signal, -signal):
            peaks, _ = find_peaks(candidate_signal, distance=distance, prominence=prominence)
            if len(peaks) >= 1:
                candidates.append(peaks)
        if not candidates:
            continue
        peaks = max(candidates, key=len)
        centers = np.sort(peaks)
        boundaries = [0]
        if len(centers) > 1:
            boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(len(centers) - 1))
        boundaries.append(len(segment_df))
        boundaries = sorted(set(boundaries))
        for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < min_samples:
                continue
            predicted.append(
                RepSegment(
                    path,
                    block.subject,
                    block.exercise,
                    block.set_id,
                    str(rep_idx),
                    block.start + start,
                    block.start + end,
                    source,
                )
            )
    return predicted


def short_time_energy_segments(
    df: pd.DataFrame,
    path: Path,
    smooth_window: int,
    min_samples: int,
    peak_prominence_scale: float,
    peak_distance_scale: float,
    true_segments: Sequence[RepSegment],
    block_source: str,
) -> list[RepSegment]:
    predicted: list[RepSegment] = []
    for block in candidate_blocks(df, path, true_segments, min_samples=min_samples, block_source=block_source):
        segment_df = df.iloc[block.start : block.end]
        motion = acceleration_magnitude_signal(segment_df, smooth_window)
        energy = short_time_energy(motion, window=max(min_samples, smooth_window))
        if len(energy) < min_samples * 2:
            continue
        prominence = max(float(np.std(energy)) * peak_prominence_scale, 1e-6)
        distance = max(int(round(min_samples * peak_distance_scale)), 1)
        peaks, _ = find_peaks(energy, distance=distance, prominence=prominence)
        if len(peaks) == 0:
            continue

        boundaries = [0]
        for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
            if right_peak <= left_peak:
                continue
            valley = int(left_peak + np.argmin(energy[left_peak:right_peak + 1]))
            boundaries.append(valley)
        boundaries.append(len(segment_df))
        boundaries = sorted(set(boundaries))

        for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < min_samples:
                continue
            predicted.append(
                RepSegment(
                    path,
                    block.subject,
                    block.exercise,
                    block.set_id,
                    str(rep_idx),
                    block.start + start,
                    block.start + end,
                    "short_time_energy",
                )
            )
    return predicted


def estimate_fft_period(signal: np.ndarray, min_period: int, max_period: int) -> float | None:
    if len(signal) < max(min_period * 2, 4):
        return None
    values = robust_zscore(signal)
    values = values - float(np.mean(values))
    if float(np.std(values)) < 1e-9:
        return None

    windowed = values * np.hanning(len(values))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0)
    power = np.abs(spectrum) ** 2
    valid = freqs > 0
    periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    valid &= periods >= min_period
    valid &= periods <= max_period
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    best_idx = valid_indices[int(np.argmax(power[valid]))]
    period = float(periods[best_idx])
    return period if math.isfinite(period) and period > 0 else None


def estimate_autocorrelation_period(signal: np.ndarray, min_period: int, max_period: int) -> float | None:
    if len(signal) < max(min_period * 2, 4):
        return None
    max_period = min(max_period, len(signal) - 1)
    if max_period < min_period:
        return None

    values = robust_zscore(signal)
    values = values - float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-9:
        return None

    n_fft = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(values, n=n_fft)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft)[: len(values)]
    if float(autocorr[0]) <= 1e-9:
        return None
    autocorr = autocorr / float(autocorr[0])
    lags = np.arange(len(autocorr))
    valid = (lags >= min_period) & (lags <= max_period)
    if not np.any(valid):
        return None

    valid_lags = lags[valid]
    valid_autocorr = autocorr[valid]
    peaks, props = find_peaks(valid_autocorr, prominence=0.03)
    if len(peaks):
        prominences = props.get("prominences", peak_prominences(valid_autocorr, peaks)[0])
        positive = valid_autocorr[peaks] > 0
        if np.any(positive):
            peaks = peaks[positive]
            prominences = prominences[positive]
        best_idx = peaks[int(np.argmax(prominences))]
        return float(valid_lags[best_idx])

    best_idx = int(np.argmax(valid_autocorr))
    period = float(valid_lags[best_idx])
    return period if math.isfinite(period) and period > 0 else None


def select_period_guided_peaks(
    signal: np.ndarray,
    period: float,
    min_samples: int,
    peak_prominence_scale: float,
    peak_distance_scale: float,
) -> np.ndarray:
    expected_reps = max(1, int(round(len(signal) / max(period, 1.0))))
    distance = max(int(round(period * peak_distance_scale)), min_samples, 1)
    prominence = max(float(np.std(signal)) * peak_prominence_scale, 1e-6)

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for candidate_signal in (signal, -signal):
        peaks, props = find_peaks(candidate_signal, distance=distance, prominence=prominence)
        if len(peaks):
            candidates.append((peaks, props.get("prominences", peak_prominences(candidate_signal, peaks)[0])))
    if not candidates:
        centers = (np.arange(expected_reps, dtype=np.float64) + 0.5) * len(signal) / float(expected_reps)
        return np.clip(np.round(centers).astype(int), 0, max(0, len(signal) - 1))

    peaks, prominences = min(candidates, key=lambda item: abs(len(item[0]) - expected_reps))
    if len(peaks) > expected_reps:
        top_indices = np.argsort(prominences)[-expected_reps:]
        peaks = peaks[np.sort(top_indices)]
    return np.sort(peaks)


def select_fft_guided_peaks(
    signal: np.ndarray,
    period: float,
    min_samples: int,
    peak_prominence_scale: float,
    fft_peak_distance_scale: float,
) -> np.ndarray:
    return select_period_guided_peaks(
        signal,
        period=period,
        min_samples=min_samples,
        peak_prominence_scale=peak_prominence_scale,
        peak_distance_scale=fft_peak_distance_scale,
    )


def autocorr_guided_pca_extrema_segments(
    df: pd.DataFrame,
    path: Path,
    smooth_window: int,
    min_samples: int,
    peak_prominence_scale: float,
    autocorr_min_period_samples: int,
    autocorr_max_period_fraction: float,
    autocorr_peak_distance_scale: float,
    true_segments: Sequence[RepSegment],
    block_source: str,
) -> list[RepSegment]:
    predicted: list[RepSegment] = []
    for block in candidate_blocks(df, path, true_segments, min_samples=min_samples, block_source=block_source):
        segment_df = df.iloc[block.start : block.end]
        signal = principal_motion_signal(segment_df, smooth_window)
        max_period = max(min_samples, int(round(len(signal) * autocorr_max_period_fraction)))
        period = estimate_autocorrelation_period(
            signal,
            min_period=max(min_samples, autocorr_min_period_samples),
            max_period=max_period,
        )
        if period is None:
            continue

        centers = select_period_guided_peaks(
            signal,
            period=period,
            min_samples=min_samples,
            peak_prominence_scale=peak_prominence_scale,
            peak_distance_scale=autocorr_peak_distance_scale,
        )
        if len(centers) == 0:
            continue

        boundaries = [0]
        if len(centers) > 1:
            boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(len(centers) - 1))
        boundaries.append(len(segment_df))
        boundaries = sorted(set(boundaries))

        for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < min_samples:
                continue
            predicted.append(
                RepSegment(
                    path,
                    block.subject,
                    block.exercise,
                    block.set_id,
                    str(rep_idx),
                    block.start + start,
                    block.start + end,
                    "pca_autocorr",
                )
            )
    return predicted


def fft_guided_pca_extrema_segments(
    df: pd.DataFrame,
    path: Path,
    smooth_window: int,
    min_samples: int,
    peak_prominence_scale: float,
    fft_min_period_samples: int,
    fft_max_period_fraction: float,
    fft_peak_distance_scale: float,
    true_segments: Sequence[RepSegment],
    block_source: str,
) -> list[RepSegment]:
    predicted: list[RepSegment] = []
    for block in candidate_blocks(df, path, true_segments, min_samples=min_samples, block_source=block_source):
        segment_df = df.iloc[block.start : block.end]
        signal = principal_motion_signal(segment_df, smooth_window)
        max_period = max(min_samples, int(round(len(signal) * fft_max_period_fraction)))
        period = estimate_fft_period(
            signal,
            min_period=max(min_samples, fft_min_period_samples),
            max_period=max_period,
        )
        if period is None:
            continue

        centers = select_fft_guided_peaks(
            signal,
            period=period,
            min_samples=min_samples,
            peak_prominence_scale=peak_prominence_scale,
            fft_peak_distance_scale=fft_peak_distance_scale,
        )
        if len(centers) == 0:
            continue

        boundaries = [0]
        if len(centers) > 1:
            boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(len(centers) - 1))
        boundaries.append(len(segment_df))
        boundaries = sorted(set(boundaries))

        for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < min_samples:
                continue
            predicted.append(
                RepSegment(
                    path,
                    block.subject,
                    block.exercise,
                    block.set_id,
                    str(rep_idx),
                    block.start + start,
                    block.start + end,
                    "pca_extrema_fft",
                )
            )
    return predicted


def segment_iou(a: RepSegment, b: RepSegment) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / float(union) if union > 0 else 0.0


def phase_iou(a: PhaseSegment, b: PhaseSegment) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / float(union) if union > 0 else 0.0


def true_phase_segments(df: pd.DataFrame, path: Path, min_samples: int) -> list[PhaseSegment]:
    phases = df["phase"].map(clean_label).str.lower()
    active = phases.isin(ACTIVE_PHASES).to_numpy()
    if not active.any():
        return []

    subject_values = df["subject_id"].map(clean_label)
    exercise_values = df["action_type"].map(clean_label)
    set_values = df["set"].map(clean_label)
    rep_values = df["rep"].map(clean_label)

    segments: list[PhaseSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str, str, str] | None = None
    for idx, is_active in enumerate(active):
        key = (
            subject_values.iloc[idx],
            exercise_values.iloc[idx],
            set_values.iloc[idx],
            rep_values.iloc[idx],
            phases.iloc[idx],
        )
        if is_active and start is None:
            start = idx
            last_key = key
        elif is_active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                segments.append(PhaseSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], last_key[4], start, idx, "label"))
            start = idx
            last_key = key
        elif (not is_active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                segments.append(PhaseSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], last_key[4], start, idx, "label"))
            start = None
            last_key = None
    if start is not None and last_key is not None and len(df) - start >= min_samples:
        segments.append(PhaseSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], last_key[4], start, len(df), "label"))
    return segments


def phase_order_by_exercise(true_phases: Sequence[PhaseSegment]) -> dict[str, tuple[str, str]]:
    by_rep: dict[tuple[Path, str, str, str, str], list[PhaseSegment]] = {}
    for segment in true_phases:
        by_rep.setdefault((segment.file_path, segment.subject, segment.exercise, segment.set_id, segment.rep_id), []).append(segment)

    votes: dict[str, dict[tuple[str, str], int]] = {}
    for (_, _, exercise, _, _), segments in by_rep.items():
        ordered = sorted(segments, key=lambda item: item.start)
        unique_order: list[str] = []
        for segment in ordered:
            if segment.phase not in unique_order:
                unique_order.append(segment.phase)
        if len(unique_order) >= 2:
            pair = (unique_order[0], unique_order[1])
        elif len(unique_order) == 1:
            other = "concentric" if unique_order[0] == "eccentric" else "eccentric"
            pair = (unique_order[0], other)
        else:
            continue
        votes.setdefault(exercise, {})
        votes[exercise][pair] = votes[exercise].get(pair, 0) + 1

    orders: dict[str, tuple[str, str]] = {}
    for exercise, counts in votes.items():
        orders[exercise] = max(counts.items(), key=lambda item: item[1])[0]
    return orders


def infer_phase_split_point(
    df: pd.DataFrame,
    segment: RepSegment,
    method: str,
    smooth_window: int,
    min_phase_samples: int,
) -> int | None:
    length = segment.n_samples
    if length < max(2 * min_phase_samples, 2):
        return None
    if method == "midpoint":
        return segment.start + length // 2
    if method != "pca-reversal":
        raise ValueError(f"Unsupported phase split method: {method}")

    segment_df = df.iloc[segment.start : segment.end]
    signal = principal_motion_signal(segment_df, smooth_window=smooth_window)
    if len(signal) < max(2 * min_phase_samples, 2):
        return segment.start + length // 2
    lower = max(min_phase_samples, int(round(length * 0.25)))
    upper = min(length - min_phase_samples, int(round(length * 0.75)))
    if upper <= lower:
        return segment.start + length // 2

    trend = np.linspace(float(signal[0]), float(signal[-1]), len(signal))
    residual = signal - trend
    local = residual[lower:upper]
    if len(local) == 0 or float(np.std(local)) < 1e-9:
        return segment.start + length // 2
    split = lower + int(np.argmax(np.abs(local)))
    return int(segment.start + split)


def predict_phase_segments(
    predicted_reps: Sequence[RepSegment],
    session_cache: dict[Path, pd.DataFrame],
    phase_orders: dict[str, tuple[str, str]],
    method: str,
    smooth_window: int,
    min_phase_samples: int,
) -> list[PhaseSegment]:
    predicted: list[PhaseSegment] = []
    for segment in predicted_reps:
        df = session_cache.get(segment.file_path)
        if df is None:
            continue
        split = infer_phase_split_point(df, segment, method=method, smooth_window=smooth_window, min_phase_samples=min_phase_samples)
        if split is None:
            continue
        first_phase, second_phase = phase_orders.get(segment.exercise, ("eccentric", "concentric"))
        parts = (
            (first_phase, segment.start, split),
            (second_phase, split, segment.end),
        )
        for phase, start, end in parts:
            if end - start < min_phase_samples:
                continue
            predicted.append(
                PhaseSegment(
                    segment.file_path,
                    segment.subject,
                    segment.exercise,
                    segment.set_id,
                    segment.rep_id,
                    phase,
                    start,
                    end,
                    method,
                )
            )
    return predicted


def label_predicted_segments(
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    class_names: Sequence[str],
    include_other: bool,
    min_iou: float,
) -> tuple[list[RepSegment], list[str], list[dict[str, object]]]:
    by_file: dict[Path, list[RepSegment]] = {}
    for segment in truth:
        by_file.setdefault(segment.file_path, []).append(segment)

    labeled_segments: list[RepSegment] = []
    labels: list[str] = []
    rows: list[dict[str, object]] = []
    class_set = set(class_names)

    for segment in predicted:
        candidates = by_file.get(segment.file_path, [])
        best = max(candidates, key=lambda true_segment: segment_iou(segment, true_segment), default=None)
        best_iou = segment_iou(segment, best) if best is not None else 0.0
        label = best.exercise if best is not None and best_iou >= min_iou else "other"
        if label not in class_set:
            if include_other:
                label = "other"
            else:
                continue
        labeled_segments.append(segment)
        labels.append(label)
        rows.append(
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "pred_start": segment.start,
                "pred_end": segment.end,
                "matched_exercise": label,
                "matched_iou": round(best_iou, 4),
                "source": segment.source,
            }
        )
    return labeled_segments, labels, rows


def segment_features(df: pd.DataFrame, segment: RepSegment) -> dict[str, float]:
    x = df.iloc[segment.start : segment.end].loc[:, IMU_COLUMNS].to_numpy(dtype=np.float64)
    features: dict[str, float] = {
        "duration_samples": float(len(x)),
    }
    if len(x) == 0:
        return features
    diff = np.diff(x, axis=0) if len(x) > 1 else np.zeros_like(x)
    for col_idx, col in enumerate(IMU_COLUMNS):
        values = x[:, col_idx]
        d_values = diff[:, col_idx] if len(diff) else np.zeros(1)
        features[f"{col}_mean"] = float(np.mean(values))
        features[f"{col}_std"] = float(np.std(values))
        features[f"{col}_min"] = float(np.min(values))
        features[f"{col}_max"] = float(np.max(values))
        features[f"{col}_range"] = float(np.ptp(values))
        features[f"{col}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{col}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        features[f"{col}_diff_abs_mean"] = float(np.mean(np.abs(d_values)))
        features[f"{col}_diff_std"] = float(np.std(d_values))

    acc_norm = np.linalg.norm(x[:, :3], axis=1)
    gyro_norm = np.linalg.norm(x[:, 3:6], axis=1)
    features["acc_norm_mean"] = float(np.mean(acc_norm))
    features["acc_norm_std"] = float(np.std(acc_norm))
    features["gyro_norm_mean"] = float(np.mean(gyro_norm))
    features["gyro_norm_std"] = float(np.std(gyro_norm))
    try:
        signal = principal_motion_signal(pd.DataFrame(x, columns=IMU_COLUMNS), smooth_window=7)
        features["principal_range"] = float(np.ptp(signal))
        features["principal_turning_points"] = float(len(find_peaks(signal)[0]) + len(find_peaks(-signal)[0]))
    except Exception:
        features["principal_range"] = 0.0
        features["principal_turning_points"] = 0.0
    return features


def build_feature_table(segments: Sequence[RepSegment], labels: Sequence[str], session_cache: dict[Path, pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, float]] = []
    subjects: list[str] = []
    for segment in segments:
        rows.append(segment_features(session_cache[segment.file_path], segment))
        subjects.append(segment.subject)
    return pd.DataFrame(rows).fillna(0.0), np.asarray(labels, dtype=object), np.asarray(subjects, dtype=object)


def select_classes(labels: Sequence[str], num_classes: int, include_other: bool) -> list[str]:
    counts = pd.Series(labels).value_counts()
    counts = counts.drop(labels=["big_rest", "rest", "other"], errors="ignore")
    classes = counts.head(num_classes).index.astype(str).tolist()
    if include_other:
        classes.append("other")
    return classes


def run_group_kfold(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Sequence[str],
    folds: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    unique_groups = sorted(set(groups.tolist()))
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two subjects for subject-wise K-fold validation.")

    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
        train_subjects = sorted(set(groups[train_idx].tolist()))
        val_subjects = sorted(set(groups[val_idx].tolist()))
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed + fold_idx,
                n_jobs=-1,
            ),
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        pred = model.predict(x.iloc[val_idx])
        y_true_all.extend(y[val_idx].tolist())
        y_pred_all.extend(pred.tolist())
        cm = confusion_matrix(y[val_idx], pred, labels=class_names)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": ",".join(train_subjects),
                "val_subjects": ",".join(val_subjects),
                "val_samples": len(val_idx),
                "accuracy": round(float(np.trace(cm) / max(1, cm.sum())), 4),
            }
        )

    labels = list(class_names)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    report = classification_report(y_true_all, y_pred_all, labels=labels, output_dict=True, zero_division=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_manifest.csv", fold_rows)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true_label": true_label, "pred_label": pred_label, "count": int(cm[i, j])}
            for i, true_label in enumerate(labels)
            for j, pred_label in enumerate(labels)
        ],
    )
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(7, len(labels) * 1.1)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True, xticks_rotation=45)
    ax.set_title("Subject-wise K-fold Exercise Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(7, len(labels) * 1.1)))
    display = ConfusionMatrixDisplay(confusion_matrix=norm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True, xticks_rotation=45)
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)

    return {
        "folds": n_splits,
        "subjects": unique_groups,
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def segmentation_summary(predicted: Sequence[RepSegment], truth: Sequence[RepSegment]) -> list[dict[str, object]]:
    by_file_truth: dict[Path, list[RepSegment]] = {}
    for segment in truth:
        by_file_truth.setdefault(segment.file_path, []).append(segment)
    rows: list[dict[str, object]] = []
    for segment in predicted:
        candidates = by_file_truth.get(segment.file_path, [])
        best_iou = max((segment_iou(segment, candidate) for candidate in candidates), default=0.0)
        rows.append(
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "exercise_hint": segment.exercise,
                "start": segment.start,
                "end": segment.end,
                "samples": segment.n_samples,
                "best_true_iou": round(best_iou, 4),
            }
        )
    return rows


def group_segments_by_file(segments: Sequence[RepSegment]) -> dict[Path, list[RepSegment]]:
    grouped: dict[Path, list[RepSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.file_path, []).append(segment)
    return grouped


def best_truth_match_rows(predicted: Sequence[RepSegment], truth: Sequence[RepSegment]) -> list[dict[str, object]]:
    by_file_predicted = group_segments_by_file(predicted)

    rows: list[dict[str, object]] = []
    for true_segment in truth:
        candidates = by_file_predicted.get(true_segment.file_path, [])
        best = max(candidates, key=lambda pred_segment: segment_iou(pred_segment, true_segment), default=None)
        best_iou = segment_iou(best, true_segment) if best is not None else 0.0
        rows.append(
            {
                "file": str(true_segment.file_path),
                "subject": true_segment.subject,
                "exercise": true_segment.exercise,
                "set_id": true_segment.set_id,
                "rep_id": true_segment.rep_id,
                "true_start": true_segment.start,
                "true_end": true_segment.end,
                "best_pred_start": best.start if best is not None else "",
                "best_pred_end": best.end if best is not None else "",
                "best_iou": round(best_iou, 4),
            }
        )
    return rows


def greedy_match_count(predicted: Sequence[RepSegment], truth: Sequence[RepSegment], threshold: float) -> tuple[int, list[float]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    predicted_by_file = group_segments_by_file(predicted)
    truth_by_file = group_segments_by_file(truth)
    pred_offset = 0
    truth_offset = 0
    for file_path in sorted(set(predicted_by_file) | set(truth_by_file)):
        file_predicted = predicted_by_file.get(file_path, [])
        file_truth = truth_by_file.get(file_path, [])
        for pred_idx, pred_segment in enumerate(file_predicted):
            for true_idx, true_segment in enumerate(file_truth):
                iou = segment_iou(pred_segment, true_segment)
                if iou >= threshold:
                    candidate_pairs.append((iou, pred_offset + pred_idx, truth_offset + true_idx))
        pred_offset += len(file_predicted)
        truth_offset += len(file_truth)
    candidate_pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_truth: set[int] = set()
    matched_ious: list[float] = []
    for iou, pred_idx, true_idx in candidate_pairs:
        if pred_idx in matched_pred or true_idx in matched_truth:
            continue
        matched_pred.add(pred_idx)
        matched_truth.add(true_idx)
        matched_ious.append(iou)
    return len(matched_ious), matched_ious


def segmentation_metric_rows(
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        tp, matched_ious = greedy_match_count(predicted, truth, threshold)
        fp = len(predicted) - tp
        fn = len(truth) - tp
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "iou_threshold": threshold,
                "true_reps": len(truth),
                "predicted_reps": len(predicted),
                "matched_reps": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
            }
        )
    return rows


def segmentation_metric_rows_by_exercise(
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    exercises = sorted({segment.exercise for segment in truth})
    for exercise in exercises:
        truth_subset = [segment for segment in truth if segment.exercise == exercise]
        predicted_subset = [segment for segment in predicted if segment.exercise == exercise]
        for row in segmentation_metric_rows(predicted_subset, truth_subset, thresholds):
            rows.append({"exercise": exercise, **row})
    return rows


def group_phase_segments_by_file(segments: Sequence[PhaseSegment]) -> dict[Path, list[PhaseSegment]]:
    grouped: dict[Path, list[PhaseSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.file_path, []).append(segment)
    return grouped


def greedy_phase_match_count(
    predicted: Sequence[PhaseSegment],
    truth: Sequence[PhaseSegment],
    threshold: float,
) -> tuple[int, list[float]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    predicted_by_file = group_phase_segments_by_file(predicted)
    truth_by_file = group_phase_segments_by_file(truth)
    pred_offset = 0
    truth_offset = 0
    for file_path in sorted(set(predicted_by_file) | set(truth_by_file)):
        file_predicted = predicted_by_file.get(file_path, [])
        file_truth = truth_by_file.get(file_path, [])
        for pred_idx, pred_segment in enumerate(file_predicted):
            for true_idx, true_segment in enumerate(file_truth):
                if pred_segment.phase != true_segment.phase:
                    continue
                iou = phase_iou(pred_segment, true_segment)
                if iou >= threshold:
                    candidate_pairs.append((iou, pred_offset + pred_idx, truth_offset + true_idx))
        pred_offset += len(file_predicted)
        truth_offset += len(file_truth)
    candidate_pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_truth: set[int] = set()
    matched_ious: list[float] = []
    for iou, pred_idx, true_idx in candidate_pairs:
        if pred_idx in matched_pred or true_idx in matched_truth:
            continue
        matched_pred.add(pred_idx)
        matched_truth.add(true_idx)
        matched_ious.append(iou)
    return len(matched_ious), matched_ious


def phase_metric_rows(
    predicted: Sequence[PhaseSegment],
    truth: Sequence[PhaseSegment],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        tp, matched_ious = greedy_phase_match_count(predicted, truth, threshold)
        fp = len(predicted) - tp
        fn = len(truth) - tp
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "iou_threshold": threshold,
                "true_phase_segments": len(truth),
                "predicted_phase_segments": len(predicted),
                "matched_phase_segments": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
            }
        )
    return rows


def phase_metric_rows_by_phase(
    predicted: Sequence[PhaseSegment],
    truth: Sequence[PhaseSegment],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    phases = sorted({segment.phase for segment in truth})
    for phase in phases:
        truth_subset = [segment for segment in truth if segment.phase == phase]
        predicted_subset = [segment for segment in predicted if segment.phase == phase]
        for row in phase_metric_rows(predicted_subset, truth_subset, thresholds):
            rows.append({"phase": phase, **row})
    return rows


def plot_segmentation_metrics(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    thresholds = [float(row["iou_threshold"]) for row in rows]
    x = np.arange(len(thresholds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [float(row["precision"]) for row in rows], width, label="Precision")
    ax.bar(x, [float(row["recall"]) for row in rows], width, label="Recall")
    ax.bar(x + width, [float(row["f1"]) for row in rows], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU >= {threshold:.2f}" for threshold in thresholds])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Rep Segmentation Accuracy by IoU Threshold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rep_segmentation_iou_metrics.png", dpi=180)
    plt.close(fig)


def plot_segmentation_metrics_by_exercise(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    exercises = sorted({str(row["exercise"]) for row in rows})
    thresholds = sorted({float(row["iou_threshold"]) for row in rows})
    matrix = np.zeros((len(exercises), len(thresholds)), dtype=np.float64)
    row_lookup = {
        (str(row["exercise"]), float(row["iou_threshold"])): float(row["f1"])
        for row in rows
    }
    for exercise_idx, exercise in enumerate(exercises):
        for threshold_idx, threshold in enumerate(thresholds):
            matrix[exercise_idx, threshold_idx] = row_lookup.get((exercise, threshold), 0.0)

    fig, ax = plt.subplots(figsize=(max(7, len(thresholds) * 1.6), max(6, len(exercises) * 0.45)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{threshold:.2f}" for threshold in thresholds])
    ax.set_yticks(np.arange(len(exercises)))
    ax.set_yticklabels(exercises)
    ax.set_xlabel("IoU Threshold")
    ax.set_title("Rep Segmentation F1 by Exercise")
    for exercise_idx in range(len(exercises)):
        for threshold_idx in range(len(thresholds)):
            value = matrix[exercise_idx, threshold_idx]
            ax.text(
                threshold_idx,
                exercise_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(output_dir / "rep_segmentation_iou_f1_by_exercise.png", dpi=180)
    plt.close(fig)


def plot_phase_metrics(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    thresholds = [float(row["iou_threshold"]) for row in rows]
    x = np.arange(len(thresholds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [float(row["precision"]) for row in rows], width, label="Precision")
    ax.bar(x, [float(row["recall"]) for row in rows], width, label="Recall")
    ax.bar(x + width, [float(row["f1"]) for row in rows], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU >= {threshold:.2f}" for threshold in thresholds])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Concentric / Eccentric Phase Split Accuracy by IoU")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_split_iou_metrics.png", dpi=180)
    plt.close(fig)


def plot_phase_metrics_by_phase(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    phases = sorted({str(row["phase"]) for row in rows})
    thresholds = sorted({float(row["iou_threshold"]) for row in rows})
    matrix = np.zeros((len(phases), len(thresholds)), dtype=np.float64)
    row_lookup = {
        (str(row["phase"]), float(row["iou_threshold"])): float(row["f1"])
        for row in rows
    }
    for phase_idx, phase in enumerate(phases):
        for threshold_idx, threshold in enumerate(thresholds):
            matrix[phase_idx, threshold_idx] = row_lookup.get((phase, threshold), 0.0)

    fig, ax = plt.subplots(figsize=(max(7, len(thresholds) * 1.6), max(4, len(phases) * 0.7)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{threshold:.2f}" for threshold in thresholds])
    ax.set_yticks(np.arange(len(phases)))
    ax.set_yticklabels(phases)
    ax.set_xlabel("IoU Threshold")
    ax.set_title("Phase Split F1 by Phase")
    for phase_idx in range(len(phases)):
        for threshold_idx in range(len(thresholds)):
            value = matrix[phase_idx, threshold_idx]
            ax.text(
                threshold_idx,
                phase_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(output_dir / "phase_split_iou_f1_by_phase.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rep segmentation followed by subject-wise K-fold exercise classification.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--include-other", action="store_true", help="Add an 'other' class for unmatched/non-top exercise segments.")
    parser.add_argument(
        "--block-source",
        choices=["action-label", "active-phase-span"],
        default="action-label",
        help="Set candidate block source. active-phase-span removes pre/post rest and evaluates only labeled movement spans.",
    )
    parser.add_argument(
        "--segment-method",
        choices=[
            "labels",
            "dominant-axis",
            "short-time-energy",
            "pca-extrema",
            "pca-autocorr",
            "pca-extrema-fft",
        ],
        default="labels",
    )
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--peak-prominence-scale", type=float, default=0.35)
    parser.add_argument("--peak-distance-scale", type=float, default=3.0)
    parser.add_argument("--fft-min-period-samples", type=int, default=25)
    parser.add_argument("--fft-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--fft-peak-distance-scale", type=float, default=1.2)
    parser.add_argument("--autocorr-min-period-samples", type=int, default=25)
    parser.add_argument("--autocorr-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--autocorr-peak-distance-scale", type=float, default=0.75)
    parser.add_argument("--min-label-iou", type=float, default=0.25)
    parser.add_argument("--segmentation-iou-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--evaluate-phase-split", action="store_true")
    parser.add_argument("--phase-split-method", choices=["midpoint", "pca-reversal"], default="pca-reversal")
    parser.add_argument("--min-phase-segment-samples", type=int, default=10)
    parser.add_argument("--phase-iou-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session_cache: dict[Path, pd.DataFrame] = {}
    truth: list[RepSegment] = []
    phase_truth: list[PhaseSegment] = []
    for path in whole_session_files(args.data_dirs):
        df = read_session(path, args.data_dirs)
        if not all(col in df.columns for col in IMU_COLUMNS):
            continue
        session_cache[path] = df
        truth.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))
        if args.evaluate_phase_split:
            phase_truth.extend(true_phase_segments(df, path, min_samples=args.min_phase_segment_samples))

    if not truth:
        raise RuntimeError("No labeled repetitions found. Check phase/rep annotations and data paths.")

    truth_labels = [segment.exercise for segment in truth]
    class_names = select_classes(truth_labels, args.num_classes, args.include_other)
    truth_by_file: dict[Path, list[RepSegment]] = {}
    for segment in truth:
        truth_by_file.setdefault(segment.file_path, []).append(segment)

    if args.segment_method == "labels":
        predicted = list(truth)
    elif args.segment_method == "pca-extrema":
        predicted = []
        for path, df in session_cache.items():
            predicted.extend(
                pca_extrema_segments(
                    df,
                    path,
                    truth_by_file.get(path, []),
                    smooth_window=args.smooth_window,
                    min_samples=args.min_segment_samples,
                    peak_prominence_scale=args.peak_prominence_scale,
                    peak_distance_scale=args.peak_distance_scale,
                    block_source=args.block_source,
                )
            )
    elif args.segment_method == "dominant-axis":
        predicted = []
        for path, df in session_cache.items():
            predicted.extend(
                extrema_segments_from_signal(
                    df,
                    path,
                    signal_fn=dominant_axis_signal,
                    source="dominant_axis",
                    smooth_window=args.smooth_window,
                    min_samples=args.min_segment_samples,
                    peak_prominence_scale=args.peak_prominence_scale,
                    peak_distance_scale=args.peak_distance_scale,
                    true_segments=truth_by_file.get(path, []),
                    block_source=args.block_source,
                )
            )
    elif args.segment_method == "short-time-energy":
        predicted = []
        for path, df in session_cache.items():
            predicted.extend(
                short_time_energy_segments(
                    df,
                    path,
                    smooth_window=args.smooth_window,
                    min_samples=args.min_segment_samples,
                    peak_prominence_scale=args.peak_prominence_scale,
                    peak_distance_scale=args.peak_distance_scale,
                    true_segments=truth_by_file.get(path, []),
                    block_source=args.block_source,
                )
            )
    elif args.segment_method == "pca-autocorr":
        predicted = []
        for path, df in session_cache.items():
            predicted.extend(
                autocorr_guided_pca_extrema_segments(
                    df,
                    path,
                    smooth_window=args.smooth_window,
                    min_samples=args.min_segment_samples,
                    peak_prominence_scale=args.peak_prominence_scale,
                    autocorr_min_period_samples=args.autocorr_min_period_samples,
                    autocorr_max_period_fraction=args.autocorr_max_period_fraction,
                    autocorr_peak_distance_scale=args.autocorr_peak_distance_scale,
                    true_segments=truth_by_file.get(path, []),
                    block_source=args.block_source,
                )
            )
    else:
        predicted = []
        for path, df in session_cache.items():
            predicted.extend(
                fft_guided_pca_extrema_segments(
                    df,
                    path,
                    smooth_window=args.smooth_window,
                    min_samples=args.min_segment_samples,
                    peak_prominence_scale=args.peak_prominence_scale,
                    fft_min_period_samples=args.fft_min_period_samples,
                    fft_max_period_fraction=args.fft_max_period_fraction,
                    fft_peak_distance_scale=args.fft_peak_distance_scale,
                    true_segments=truth_by_file.get(path, []),
                    block_source=args.block_source,
                )
            )

    segments, labels, manifest_rows = label_predicted_segments(
        predicted,
        truth,
        class_names=class_names,
        include_other=args.include_other,
        min_iou=args.min_label_iou,
    )
    if not segments:
        raise RuntimeError("No predicted repetition segments could be labeled for classification.")

    x, y, groups = build_feature_table(segments, labels, session_cache)
    metrics = run_group_kfold(x, y, groups, class_names, args.folds, args.seed, args.output_dir)

    write_csv(args.output_dir / "rep_segments_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "rep_segmentation_matches.csv", segmentation_summary(predicted, truth))
    write_csv(args.output_dir / "rep_segmentation_truth_matches.csv", best_truth_match_rows(predicted, truth))
    segmentation_rows = segmentation_metric_rows(predicted, truth, args.segmentation_iou_thresholds)
    segmentation_by_exercise_rows = segmentation_metric_rows_by_exercise(predicted, truth, args.segmentation_iou_thresholds)
    write_csv(args.output_dir / "rep_segmentation_metrics.csv", segmentation_rows)
    write_csv(args.output_dir / "rep_segmentation_metrics_by_exercise.csv", segmentation_by_exercise_rows)
    plot_segmentation_metrics(segmentation_rows, args.output_dir)
    plot_segmentation_metrics_by_exercise(segmentation_by_exercise_rows, args.output_dir)
    phase_rows: list[dict[str, object]] = []
    phase_by_phase_rows: list[dict[str, object]] = []
    predicted_phases: list[PhaseSegment] = []
    if args.evaluate_phase_split:
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
        "segment_method": args.segment_method,
        "block_source": args.block_source,
        "num_truth_reps": len(truth),
        "num_predicted_reps": len(predicted),
        "num_classified_reps": len(segments),
        "class_names": class_names,
        "segmentation_metrics": segmentation_rows,
        "phase_split_method": args.phase_split_method if args.evaluate_phase_split else None,
        "num_truth_phase_segments": len(phase_truth),
        "num_predicted_phase_segments": len(predicted_phases),
        "phase_split_metrics": phase_rows,
        **metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
