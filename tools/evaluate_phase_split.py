from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")
PHASES = {"concentric", "eccentric"}


@dataclass(frozen=True)
class RepResult:
    path: Path
    session: str
    exercise: str
    set_name: str
    rep_name: str
    n_samples: int
    true_cut: int
    pred_cut: int
    abs_error: int
    rel_error: float
    sample_period_seconds: float
    abs_error_seconds: float
    start_phase: str
    end_phase: str

    def is_hit(self, tolerance: int) -> bool:
        return self.abs_error <= tolerance


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


def principal_motion_signal(df: pd.DataFrame, imu_columns: Sequence[str], smooth_window: int) -> np.ndarray:
    available = [col for col in imu_columns if col in df.columns]
    if not available:
        raise ValueError(f"Missing IMU columns: {imu_columns}")

    x = df.loc[:, available].to_numpy(dtype=np.float64)
    x = np.apply_along_axis(robust_zscore, 0, x)
    variances = np.var(x, axis=0)
    x = x[:, variances > 1e-9]
    if x.shape[1] == 0:
        return np.zeros(len(df), dtype=np.float64)

    # Literature on wearable resistance-training repetition analysis commonly
    # smooths IMU data and follows the dominant motion direction. PCA gives a
    # data-driven dominant axis instead of hard-coding one exercise axis.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    signal = x @ vt[0]
    return moving_average(signal, smooth_window)


def candidate_turning_points(signal: np.ndarray, min_fraction: float, max_fraction: float) -> Iterable[int]:
    n = len(signal)
    lo = max(1, int(math.floor(n * min_fraction)))
    hi = min(n - 2, int(math.ceil(n * max_fraction)))
    if lo > hi:
        return []

    points: list[int] = []
    for idx in range(lo, hi + 1):
        prev_v = signal[idx - 1]
        curr_v = signal[idx]
        next_v = signal[idx + 1]
        if (curr_v >= prev_v and curr_v >= next_v) or (curr_v <= prev_v and curr_v <= next_v):
            points.append(idx)

    if points:
        return points
    return range(lo, hi + 1)


def split_score(signal: np.ndarray, idx: int) -> float:
    left = signal[:idx]
    right = signal[idx:]
    if len(left) < 2 or len(right) < 2:
        return -np.inf

    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    left_var = float(np.var(left))
    right_var = float(np.var(right))
    separation = abs(left_mean - right_mean)

    local_lo = max(0, idx - 3)
    local_hi = min(len(signal), idx + 4)
    local = signal[local_lo:local_hi]
    local_prominence = abs(float(signal[idx]) - float(np.median(local)))

    balance = 1.0 - abs(idx - len(signal) / 2.0) / max(len(signal), 1)
    return separation + 0.25 * local_prominence - 0.05 * (left_var + right_var) + 0.1 * balance


def predict_signal_phase_cut(df: pd.DataFrame, smooth_window: int, min_fraction: float, max_fraction: float) -> int:
    signal = principal_motion_signal(df, IMU_COLUMNS, smooth_window)
    candidates = list(candidate_turning_points(signal, min_fraction, max_fraction))
    if not candidates:
        return max(1, len(df) // 2)
    return int(max(candidates, key=lambda idx: split_score(signal, idx)))


def predict_phase_cut(
    df: pd.DataFrame,
    method: str,
    smooth_window: int,
    min_fraction: float,
    max_fraction: float,
) -> int:
    if method == "midpoint":
        return max(1, len(df) // 2)
    if method == "signal":
        return predict_signal_phase_cut(df, smooth_window, min_fraction, max_fraction)
    if method == "learned-fraction":
        return max(1, len(df) // 2)
    if method == "supervised-regression":
        return max(1, len(df) // 2)
    if method == "phase-column":
        cut = true_phase_cut(df)
        if cut is None:
            return max(1, len(df) // 2)
        return cut
    raise ValueError(f"Unknown method: {method}")


def estimate_sample_period_seconds(df: pd.DataFrame) -> float:
    for column in ("sensor_ts", "host_ts"):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(values) < 2:
            continue
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            continue
        median_diff = float(np.median(diffs))
        if median_diff > 100000:
            return median_diff / 1_000_000.0
        if median_diff > 1000:
            return median_diff / 1_000_000.0
        if median_diff > 1:
            return median_diff / 1000.0
        return median_diff

    if "pc_time" in df.columns:
        times = pd.to_datetime(df["pc_time"], errors="coerce").dropna()
        if len(times) >= 2:
            diffs = times.diff().dropna().dt.total_seconds().to_numpy(dtype=np.float64)
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                return float(np.median(diffs))

    return 0.02


def true_phase_cut(df: pd.DataFrame) -> int | None:
    if "phase" not in df.columns:
        return None
    phases = df["phase"].astype(str).str.strip().str.lower().to_numpy()
    valid = np.isin(phases, list(PHASES))
    if not valid.all():
        return None
    changes = np.flatnonzero(phases[1:] != phases[:-1]) + 1
    if len(changes) != 1:
        return None
    return int(changes[0])


def parse_rep_path(path: Path, data_dir: Path) -> tuple[str, str, str, str]:
    rel = path.relative_to(data_dir)
    parts = rel.parts
    session = parts[0] if len(parts) > 0 else ""
    exercise = parts[1] if len(parts) > 1 else ""
    set_name = parts[2] if len(parts) > 2 else ""
    rep_name = path.stem
    return session, exercise, set_name, rep_name


def iter_rep_csvs(data_dir: Path, session_filter: str | None) -> list[Path]:
    paths = sorted(data_dir.rglob("rep*.csv"))
    if session_filter:
        wanted = session_filter.lower()
        paths = [p for p in paths if wanted in str(p.relative_to(data_dir)).lower()]
    return paths


def evaluate(
    data_dir: Path,
    session_filter: str | None,
    method: str,
    smooth_window: int,
    min_fraction: float,
    max_fraction: float,
) -> tuple[list[RepResult], dict[str, int]]:
    results: list[RepResult] = []
    skipped = {"missing_phase": 0, "multi_or_no_cut": 0, "too_short": 0, "read_error": 0}

    for path in iter_rep_csvs(data_dir, session_filter):
        try:
            df = pd.read_csv(path)
        except Exception:
            skipped["read_error"] += 1
            continue

        if len(df) < 6:
            skipped["too_short"] += 1
            continue
        if "phase" not in df.columns:
            skipped["missing_phase"] += 1
            continue

        cut = true_phase_cut(df)
        if cut is None:
            skipped["multi_or_no_cut"] += 1
            continue

        pred = predict_phase_cut(df, method, smooth_window, min_fraction, max_fraction)
        session, exercise, set_name, rep_name = parse_rep_path(path, data_dir)
        phases = df["phase"].astype(str).str.strip().str.lower().to_numpy()
        abs_error = abs(pred - cut)
        sample_period_seconds = estimate_sample_period_seconds(df)
        results.append(
            RepResult(
                path=path,
                session=session,
                exercise=exercise,
                set_name=set_name,
                rep_name=rep_name,
                n_samples=len(df),
                true_cut=cut,
                pred_cut=pred,
                abs_error=abs_error,
                rel_error=abs_error / float(len(df)),
                sample_period_seconds=sample_period_seconds,
                abs_error_seconds=abs_error * sample_period_seconds,
                start_phase=str(phases[0]),
                end_phase=str(phases[-1]),
            )
        )

    return results, skipped


def phase_iou_scores(n_samples: int, true_cut: int, pred_cut: int) -> tuple[float, float]:
    true_cut = min(max(1, true_cut), n_samples - 1)
    pred_cut = min(max(1, pred_cut), n_samples - 1)

    first_iou = min(true_cut, pred_cut) / float(max(true_cut, pred_cut))
    second_intersection = n_samples - max(true_cut, pred_cut)
    second_union = n_samples - min(true_cut, pred_cut)
    second_iou = second_intersection / float(second_union) if second_union > 0 else 0.0
    return (first_iou + second_iou) / 2.0, min(first_iou, second_iou)


def result_mean_iou(result: RepResult) -> float:
    return phase_iou_scores(result.n_samples, result.true_cut, result.pred_cut)[0]


def result_min_iou(result: RepResult) -> float:
    return phase_iou_scores(result.n_samples, result.true_cut, result.pred_cut)[1]


def summarize(
    results: Sequence[RepResult],
    tolerances: Sequence[int],
    relative_tolerances: Sequence[float],
    iou_thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[RepResult]] = {}
    for result in results:
        groups.setdefault((result.session, result.exercise), []).append(result)
    groups[("ALL", "ALL")] = list(results)

    for (session, exercise), group in sorted(groups.items()):
        if not group:
            continue
        row: dict[str, object] = {
            "session": session,
            "exercise": exercise,
            "reps": len(group),
            "mae_samples": round(float(np.mean([r.abs_error for r in group])), 2),
            "median_error_samples": round(float(np.median([r.abs_error for r in group])), 2),
            "mae_pct_rep": round(100.0 * float(np.mean([r.rel_error for r in group])), 2),
            "mae_seconds": round(float(np.mean([r.abs_error_seconds for r in group])), 4),
            "median_error_seconds": round(float(np.median([r.abs_error_seconds for r in group])), 4),
            "mean_iou": round(float(np.mean([result_mean_iou(r) for r in group])), 4),
            "median_iou": round(float(np.median([result_mean_iou(r) for r in group])), 4),
            "mean_min_iou": round(float(np.mean([result_min_iou(r) for r in group])), 4),
        }
        for tolerance in tolerances:
            hits = sum(r.is_hit(tolerance) for r in group)
            row[f"acc_<=_{tolerance}_samples"] = round(100.0 * hits / len(group), 2)
        for tolerance in relative_tolerances:
            hits = sum(r.rel_error <= tolerance for r in group)
            row[f"acc_<=_{int(tolerance * 100)}pct_rep"] = round(100.0 * hits / len(group), 2)
        for threshold in iou_thresholds:
            hits = sum(result_mean_iou(r) >= threshold for r in group)
            row[f"acc_>=_{int(threshold * 100)}pct_iou"] = round(100.0 * hits / len(group), 2)
        rows.append(row)
    return rows


def summarize_people(
    results: Sequence[RepResult],
    tolerances: Sequence[int],
    relative_tolerances: Sequence[float],
    iou_thresholds: Sequence[float],
    split_name: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[str, list[RepResult]] = {}
    for result in results:
        groups.setdefault(result.session, []).append(result)

    for person, group in sorted(groups.items()):
        row: dict[str, object] = {
            "person": person,
            "split": split_name or "",
            "reps": len(group),
            "mae_samples": round(float(np.mean([r.abs_error for r in group])), 2),
            "median_error_samples": round(float(np.median([r.abs_error for r in group])), 2),
            "mae_pct_rep": round(100.0 * float(np.mean([r.rel_error for r in group])), 2),
            "mae_seconds": round(float(np.mean([r.abs_error_seconds for r in group])), 4),
            "median_error_seconds": round(float(np.median([r.abs_error_seconds for r in group])), 4),
            "mean_iou": round(float(np.mean([result_mean_iou(r) for r in group])), 4),
            "median_iou": round(float(np.median([result_mean_iou(r) for r in group])), 4),
            "mean_min_iou": round(float(np.mean([result_min_iou(r) for r in group])), 4),
        }
        for tolerance in tolerances:
            hits = sum(r.is_hit(tolerance) for r in group)
            row[f"acc_<=_{tolerance}_samples"] = round(100.0 * hits / len(group), 2)
        for tolerance in relative_tolerances:
            hits = sum(r.rel_error <= tolerance for r in group)
            row[f"acc_<=_{int(tolerance * 100)}pct_rep"] = round(100.0 * hits / len(group), 2)
        for threshold in iou_thresholds:
            hits = sum(result_mean_iou(r) >= threshold for r in group)
            row[f"acc_>=_{int(threshold * 100)}pct_iou"] = round(100.0 * hits / len(group), 2)
        rows.append(row)
    return rows


def summarize_exercises(
    results: Sequence[RepResult],
    tolerances: Sequence[int],
    relative_tolerances: Sequence[float],
    iou_thresholds: Sequence[float],
    split_name: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[str, list[RepResult]] = {}
    for result in results:
        groups.setdefault(result.exercise, []).append(result)

    for exercise, group in sorted(groups.items()):
        row: dict[str, object] = {
            "exercise": exercise,
            "split": split_name or "",
            "reps": len(group),
            "mae_samples": round(float(np.mean([r.abs_error for r in group])), 2),
            "median_error_samples": round(float(np.median([r.abs_error for r in group])), 2),
            "mae_pct_rep": round(100.0 * float(np.mean([r.rel_error for r in group])), 2),
            "mae_seconds": round(float(np.mean([r.abs_error_seconds for r in group])), 4),
            "median_error_seconds": round(float(np.median([r.abs_error_seconds for r in group])), 4),
            "mean_iou": round(float(np.mean([result_mean_iou(r) for r in group])), 4),
            "median_iou": round(float(np.median([result_mean_iou(r) for r in group])), 4),
            "mean_min_iou": round(float(np.mean([result_min_iou(r) for r in group])), 4),
        }
        for tolerance in tolerances:
            hits = sum(r.is_hit(tolerance) for r in group)
            row[f"acc_<=_{tolerance}_samples"] = round(100.0 * hits / len(group), 2)
        for tolerance in relative_tolerances:
            hits = sum(r.rel_error <= tolerance for r in group)
            row[f"acc_<=_{int(tolerance * 100)}pct_rep"] = round(100.0 * hits / len(group), 2)
        for threshold in iou_thresholds:
            hits = sum(result_mean_iou(r) >= threshold for r in group)
            row[f"acc_>=_{int(threshold * 100)}pct_iou"] = round(100.0 * hits / len(group), 2)
        rows.append(row)
    return rows


def result_phase_confusion_counts(result: RepResult) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    true_segments = [
        (0, result.true_cut, result.start_phase),
        (result.true_cut, result.n_samples, result.end_phase),
    ]
    pred_segments = [
        (0, result.pred_cut, result.start_phase),
        (result.pred_cut, result.n_samples, result.end_phase),
    ]
    for true_start, true_end, true_phase in true_segments:
        for pred_start, pred_end, pred_phase in pred_segments:
            overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start))
            if overlap:
                counts[(true_phase, pred_phase)] = counts.get((true_phase, pred_phase), 0) + overlap
    return counts


def exercise_confusion_matrix_rows(results: Sequence[RepResult], split_name: str | None = None) -> list[dict[str, object]]:
    matrix: dict[tuple[str, str, str], int] = {}
    totals_by_exercise: dict[str, int] = {}
    correct_by_exercise: dict[str, int] = {}

    for result in results:
        totals_by_exercise[result.exercise] = totals_by_exercise.get(result.exercise, 0) + result.n_samples
        for (true_phase, pred_phase), samples in result_phase_confusion_counts(result).items():
            key = (result.exercise, true_phase, pred_phase)
            matrix[key] = matrix.get(key, 0) + samples
            if true_phase == pred_phase:
                correct_by_exercise[result.exercise] = correct_by_exercise.get(result.exercise, 0) + samples

    rows: list[dict[str, object]] = []
    for exercise in sorted(totals_by_exercise):
        total = totals_by_exercise[exercise]
        sample_accuracy = 100.0 * correct_by_exercise.get(exercise, 0) / total if total else 0.0
        for true_phase in sorted(PHASES):
            true_total = sum(matrix.get((exercise, true_phase, pred_phase), 0) for pred_phase in sorted(PHASES))
            for pred_phase in sorted(PHASES):
                samples = matrix.get((exercise, true_phase, pred_phase), 0)
                rows.append(
                    {
                        "exercise": exercise,
                        "split": split_name or "",
                        "true_phase": true_phase,
                        "pred_phase": pred_phase,
                        "samples": samples,
                        "percent_of_true_phase": round(100.0 * samples / true_total, 2) if true_total else 0.0,
                        "exercise_sample_accuracy": round(sample_accuracy, 2),
                    }
                )
    return rows


def fit_cut_fractions(results: Sequence[RepResult]) -> tuple[dict[tuple[str, str, str], float], dict[str, float], float]:
    exact: dict[tuple[str, str, str], list[float]] = {}
    by_exercise: dict[str, list[float]] = {}
    all_values: list[float] = []

    for result in results:
        fraction = result.true_cut / float(result.n_samples)
        key = (result.exercise, result.start_phase, result.end_phase)
        exact.setdefault(key, []).append(fraction)
        by_exercise.setdefault(result.exercise, []).append(fraction)
        all_values.append(fraction)

    exact_median = {key: float(np.median(values)) for key, values in exact.items()}
    exercise_median = {key: float(np.median(values)) for key, values in by_exercise.items()}
    global_median = float(np.median(all_values)) if all_values else 0.5
    return exact_median, exercise_median, global_median


def apply_cut_fractions(
    results: Sequence[RepResult],
    exact_fractions: dict[tuple[str, str, str], float],
    exercise_fractions: dict[str, float],
    global_fraction: float,
    use_transition_labels: bool = False,
) -> list[RepResult]:
    adjusted: list[RepResult] = []
    for result in results:
        key = (result.exercise, result.start_phase, result.end_phase)
        if use_transition_labels:
            fraction = exact_fractions.get(key, exercise_fractions.get(result.exercise, global_fraction))
        else:
            fraction = exercise_fractions.get(result.exercise, global_fraction)
        pred_cut = int(round(fraction * result.n_samples))
        pred_cut = min(max(1, pred_cut), result.n_samples - 1)
        abs_error = abs(pred_cut - result.true_cut)
        adjusted.append(
            replace(
                result,
                pred_cut=pred_cut,
                abs_error=abs_error,
                rel_error=abs_error / float(result.n_samples),
                abs_error_seconds=abs_error * result.sample_period_seconds,
            )
        )
    return adjusted


def rep_sort_key(result: RepResult) -> tuple[str, str, str, str]:
    return (result.exercise, result.set_name, result.rep_name, str(result.path))


def apply_fraction_bias(result: RepResult, fraction_bias: float) -> RepResult:
    pred_fraction = result.pred_cut / float(result.n_samples)
    corrected = float(np.clip(pred_fraction + fraction_bias, 0.15, 0.85))
    pred_cut = int(round(corrected * result.n_samples))
    pred_cut = min(max(1, pred_cut), result.n_samples - 1)
    abs_error = abs(pred_cut - result.true_cut)
    return replace(
        result,
        pred_cut=pred_cut,
        abs_error=abs_error,
        rel_error=abs_error / float(result.n_samples),
        abs_error_seconds=abs_error * result.sample_period_seconds,
    )


def apply_personal_calibration(
    predicted_results: Sequence[RepResult],
    calibration_reps: int,
    scope: str,
    shrink: float,
) -> tuple[list[RepResult], list[RepResult], list[RepResult], list[dict[str, object]]]:
    calibration_results: list[RepResult] = []
    baseline_test_results: list[RepResult] = []
    calibrated_test_results: list[RepResult] = []
    rows: list[dict[str, object]] = []

    by_person: dict[str, list[RepResult]] = {}
    for result in predicted_results:
        by_person.setdefault(result.session, []).append(result)

    for person, group in sorted(by_person.items()):
        ordered = sorted(group, key=rep_sort_key)
        calibration = ordered[:calibration_reps]
        test = ordered[calibration_reps:]
        calibration_results.extend(calibration)
        baseline_test_results.extend(test)

        residuals = [
            result.true_cut / float(result.n_samples) - result.pred_cut / float(result.n_samples)
            for result in calibration
        ]
        raw_global_bias = float(np.median(residuals)) if residuals else 0.0
        global_bias = shrink * raw_global_bias
        exercise_biases: dict[str, float] = {}
        if scope == "exercise":
            by_exercise: dict[str, list[float]] = {}
            for result in calibration:
                residual = result.true_cut / float(result.n_samples) - result.pred_cut / float(result.n_samples)
                by_exercise.setdefault(result.exercise, []).append(residual)
            exercise_biases = {
                exercise: shrink * float(np.median(values))
                for exercise, values in by_exercise.items()
            }

        for result in test:
            bias = exercise_biases.get(result.exercise, global_bias) if scope == "exercise" else global_bias
            calibrated_test_results.append(apply_fraction_bias(result, bias))

        rows.append(
            {
                "person": person,
                "calibration_reps": len(calibration),
                "test_reps": len(test),
                "scope": scope,
                "shrink": shrink,
                "raw_global_fraction_bias": round(raw_global_bias, 5),
                "global_fraction_bias": round(global_bias, 5),
                "exercise_biases": ";".join(
                    f"{exercise}:{bias:.5f}" for exercise, bias in sorted(exercise_biases.items())
                ),
            }
        )

    return calibration_results, baseline_test_results, calibrated_test_results, rows


def personal_calibration_comparison_rows(
    baseline_test_results: Sequence[RepResult],
    calibrated_test_results: Sequence[RepResult],
    tolerances: Sequence[int],
    relative_tolerances: Sequence[float],
    iou_thresholds: Sequence[float],
) -> list[dict[str, object]]:
    before_rows = summarize_people(
        baseline_test_results, tolerances, relative_tolerances, iou_thresholds, "test_uncalibrated"
    )
    after_rows = summarize_people(
        calibrated_test_results, tolerances, relative_tolerances, iou_thresholds, "test_calibrated"
    )
    after_by_person = {str(row["person"]): row for row in after_rows}
    comparison: list[dict[str, object]] = []
    for before in before_rows:
        person = str(before["person"])
        after = after_by_person.get(person)
        if after is None:
            continue
        comparison.append(
            {
                "person": person,
                "test_reps": before["reps"],
                "before_mean_iou": before["mean_iou"],
                "after_mean_iou": after["mean_iou"],
                "delta_mean_iou": round(float(after["mean_iou"]) - float(before["mean_iou"]), 4),
                "before_acc_<=_10pct_rep": before.get("acc_<=_10pct_rep", ""),
                "after_acc_<=_10pct_rep": after.get("acc_<=_10pct_rep", ""),
                "delta_acc_<=_10pct_rep": round(
                    float(after.get("acc_<=_10pct_rep", 0.0)) - float(before.get("acc_<=_10pct_rep", 0.0)), 2
                ),
                "before_mae_seconds": before["mae_seconds"],
                "after_mae_seconds": after["mae_seconds"],
                "delta_mae_seconds": round(float(after["mae_seconds"]) - float(before["mae_seconds"]), 4),
            }
        )
    return comparison


def fraction_rows(
    exact_fractions: dict[tuple[str, str, str], float],
    exercise_fractions: dict[str, float],
    global_fraction: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "scope": "global",
            "exercise": "ALL",
            "start_phase": "",
            "end_phase": "",
            "cut_fraction": round(global_fraction, 4),
        }
    ]
    for exercise, fraction in sorted(exercise_fractions.items()):
        rows.append(
            {
                "scope": "exercise",
                "exercise": exercise,
                "start_phase": "",
                "end_phase": "",
                "cut_fraction": round(fraction, 4),
            }
        )
    for (exercise, start_phase, end_phase), fraction in sorted(exact_fractions.items()):
        rows.append(
            {
                "scope": "exercise_transition",
                "exercise": exercise,
                "start_phase": start_phase,
                "end_phase": end_phase,
                "cut_fraction": round(fraction, 4),
            }
        )
    return rows


def regression_features(result: RepResult, smooth_window: int) -> dict[str, float]:
    df = pd.read_csv(result.path)
    signal = principal_motion_signal(df, IMU_COLUMNS, smooth_window)
    n = len(signal)
    features: dict[str, float] = {
        f"exercise={result.exercise}": 1.0,
        "n_samples": float(result.n_samples),
        "sample_period_seconds": float(result.sample_period_seconds),
    }
    if n < 3:
        return features

    quantiles = np.quantile(signal, [0.05, 0.25, 0.5, 0.75, 0.95])
    for name, value in zip(("q05", "q25", "q50", "q75", "q95"), quantiles):
        features[name] = float(value)

    features.update(
        {
            "signal_mean": float(np.mean(signal)),
            "signal_std": float(np.std(signal)),
            "signal_range": float(np.ptp(signal)),
            "argmax_frac": float(np.argmax(signal) / max(1, n - 1)),
            "argmin_frac": float(np.argmin(signal) / max(1, n - 1)),
        }
    )

    diff = np.diff(signal)
    features.update(
        {
            "diff_mean": float(np.mean(diff)),
            "diff_std": float(np.std(diff)),
            "diff_abs_mean": float(np.mean(np.abs(diff))),
        }
    )

    for idx in range(4):
        part = signal[int(n * idx / 4) : int(n * (idx + 1) / 4)]
        features[f"quartile_{idx}_mean"] = float(np.mean(part))
        features[f"quartile_{idx}_std"] = float(np.std(part))

    try:
        signal_cut = predict_signal_phase_cut(df, smooth_window, 0.15, 0.85)
    except Exception:
        signal_cut = n // 2
    features["signal_cut_frac"] = float(signal_cut / max(1, n))
    return features


def apply_supervised_regression(
    train_results: Sequence[RepResult],
    target_results: Sequence[RepResult],
    smooth_window: int,
    seed: int,
    bias_shrink: float = 0.25,
    tune_iou_bias: bool = False,
) -> tuple[list[RepResult], dict[str, object]]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import make_pipeline

    x_train = [regression_features(result, smooth_window) for result in train_results]
    y_train = np.asarray([result.true_cut / float(result.n_samples) for result in train_results], dtype=np.float64)
    x_target = [regression_features(result, smooth_window) for result in target_results]

    def build_model() -> object:
        return make_pipeline(
            DictVectorizer(sparse=False),
            GradientBoostingRegressor(random_state=seed, max_depth=2, n_estimators=200, learning_rate=0.03),
        )

    model = build_model()
    model.fit(x_train, y_train)
    pred_fractions = np.clip(model.predict(x_target), 0.15, 0.85)

    residual_by_exercise: dict[str, list[float]] = {}
    oof_records: list[tuple[RepResult, float]] = []
    train_people = sorted({result.session for result in train_results})
    if bias_shrink > 0 and len(train_people) > 1:
        for held_out_person in train_people:
            fold_train = [result for result in train_results if result.session != held_out_person]
            fold_holdout = [result for result in train_results if result.session == held_out_person]
            if not fold_train or not fold_holdout:
                continue
            fold_model = build_model()
            fold_model.fit(
                [regression_features(result, smooth_window) for result in fold_train],
                np.asarray([result.true_cut / float(result.n_samples) for result in fold_train], dtype=np.float64),
            )
            fold_pred = np.clip(
                fold_model.predict([regression_features(result, smooth_window) for result in fold_holdout]),
                0.15,
                0.85,
            )
            for result, pred_fraction in zip(fold_holdout, fold_pred):
                oof_records.append((result, float(pred_fraction)))
                residual_by_exercise.setdefault(result.exercise, []).append(
                    result.true_cut / float(result.n_samples) - float(pred_fraction)
                )

    exercise_bias = {exercise: float(np.median(values)) for exercise, values in residual_by_exercise.items()}
    selected_bias_shrink = bias_shrink
    if exercise_bias:
        all_residuals = [value for values in residual_by_exercise.values() for value in values]
        global_bias = float(np.median(all_residuals))
        if tune_iou_bias and oof_records:
            candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            best_score = -1.0
            best_shrink = selected_bias_shrink
            for candidate in candidates:
                scores: list[float] = []
                for result, pred_fraction in oof_records:
                    corrected = pred_fraction + candidate * exercise_bias.get(result.exercise, global_bias)
                    corrected = float(np.clip(corrected, 0.15, 0.85))
                    pred_cut = int(round(corrected * result.n_samples))
                    pred_cut = min(max(1, pred_cut), result.n_samples - 1)
                    scores.append(phase_iou_scores(result.n_samples, result.true_cut, pred_cut)[0])
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_shrink = candidate
            selected_bias_shrink = best_shrink

        pred_fractions = np.asarray(
            [
                fraction + selected_bias_shrink * exercise_bias.get(result.exercise, global_bias)
                for result, fraction in zip(target_results, pred_fractions)
            ],
            dtype=np.float64,
        )
        pred_fractions = np.clip(pred_fractions, 0.15, 0.85)


    adjusted: list[RepResult] = []
    for result, fraction in zip(target_results, pred_fractions):
        pred_cut = int(round(float(fraction) * result.n_samples))
        pred_cut = min(max(1, pred_cut), result.n_samples - 1)
        abs_error = abs(pred_cut - result.true_cut)
        adjusted.append(
            replace(
                result,
                pred_cut=pred_cut,
                abs_error=abs_error,
                rel_error=abs_error / float(result.n_samples),
                abs_error_seconds=abs_error * result.sample_period_seconds,
            )
        )

    info = {
        "model": "GradientBoostingRegressor",
        "n_train_reps": len(train_results),
        "n_features": len(x_train[0]) if x_train else 0,
        "target": "true_cut / n_samples",
        "bias_correction": "leave-one-person-out median residual by exercise",
        "bias_shrink": selected_bias_shrink,
        "bias_selection_metric": "OOF mean IoU" if tune_iou_bias else "fixed",
    }
    return adjusted, info


def model_info_rows(info: dict[str, object]) -> list[dict[str, object]]:
    return [{"key": key, "value": value} for key, value in info.items()]


def split_people(results: Sequence[RepResult], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    people = sorted({r.session for r in results})
    rng = random.Random(seed)
    rng.shuffle(people)
    if len(people) <= 1:
        return people, []

    n_val = max(1, int(round(len(people) * val_ratio)))
    n_val = min(n_val, len(people) - 1)
    val_people = sorted(people[:n_val])
    train_people = sorted(people[n_val:])
    return train_people, val_people


def split_people_by_validation_names(results: Sequence[RepResult], val_people: Sequence[str]) -> tuple[list[str], list[str]]:
    people = sorted({r.session for r in results})
    known = set(people)
    requested = set(val_people)
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown validation people: {unknown}. Available people: {people}")
    selected_val = sorted(requested)
    selected_train = [person for person in people if person not in requested]
    if not selected_train:
        raise ValueError("At least one person must remain in the training set.")
    return selected_train, selected_val


def filter_by_people(results: Sequence[RepResult], people: Sequence[str]) -> list[RepResult]:
    allowed = set(people)
    return [r for r in results if r.session in allowed]


def write_split_manifest(path: Path, train_people: Sequence[str], val_people: Sequence[str]) -> None:
    rows = [{"person": person, "split": "train"} for person in train_people]
    rows.extend({"person": person, "split": "val"} for person in val_people)
    write_csv(path, rows)


def iou_accuracy_key(iou_thresholds: Sequence[float]) -> tuple[str, str]:
    threshold = max(iou_thresholds) if iou_thresholds else 0.9
    pct = int(threshold * 100)
    return f"acc_>=_{pct}pct_iou", f"mean phase IoU >= {threshold:.2f}"


def scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if abs(domain_max - domain_min) < 1e-9:
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def write_person_comparison_svg(
    path: Path,
    rows: Sequence[dict[str, object]],
    title: str,
    metric_key: str = "acc_<=_10pct_rep",
    metric_label: str = "accuracy within 10% rep length",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120">'
            '<text x="24" y="56" font-family="Arial, sans-serif" font-size="16">No person metrics available.</text>'
            "</svg>\n",
            encoding="utf-8",
        )
        return
    if metric_key not in rows[0]:
        raise ValueError(f"Missing metric column for comparison SVG: {metric_key}")

    width = 1100
    row_h = 64
    pad_l = 190
    pad_r = 260
    pad_t = 88
    chart_w = width - pad_l - pad_r
    height = pad_t + row_h * len(rows) + 72
    max_err = max(float(row["mae_seconds"]) for row in rows)
    max_err = max(max_err, 0.001)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{escape(title)}</text>',
        '<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">'
        f'Bar = {escape(metric_label)}; red marker = mean absolute error in seconds.</text>',
        f'<line x1="{pad_l}" y1="{pad_t - 18}" x2="{pad_l + chart_w}" y2="{pad_t - 18}" stroke="#d1d5db"/>',
    ]

    for tick in range(0, 101, 25):
        x = pad_l + chart_w * tick / 100.0
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{pad_t - 24}" x2="{x:.2f}" y2="{height - 52}" stroke="#f3f4f6"/>',
                f'<text x="{x - 9:.2f}" y="{pad_t - 30}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{tick}%</text>',
            ]
        )

    for idx, row in enumerate(rows):
        y = pad_t + idx * row_h
        person = escape(str(row["person"]))
        split = escape(str(row.get("split", "")))
        reps = int(row["reps"])
        acc = float(row[metric_key])
        err_s = float(row["mae_seconds"])
        bar_w = chart_w * acc / 100.0
        err_x = scale_linear(err_s, 0.0, max_err, pad_l, pad_l + chart_w)
        fill = "#2563eb" if split == "val" else "#64748b"
        parts.extend(
            [
                f'<text x="32" y="{y + 22}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{person}</text>',
                f'<text x="32" y="{y + 42}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{split}, reps={reps}</text>',
                f'<rect x="{pad_l}" y="{y + 9}" width="{chart_w}" height="24" rx="4" fill="#eef2ff"/>',
                f'<rect x="{pad_l}" y="{y + 9}" width="{bar_w:.2f}" height="24" rx="4" fill="{fill}"/>',
                f'<line x1="{err_x:.2f}" y1="{y + 3}" x2="{err_x:.2f}" y2="{y + 41}" stroke="#dc2626" stroke-width="3"/>',
                f'<text x="{pad_l + chart_w + 18}" y="{y + 25}" font-family="Arial, sans-serif" font-size="12" fill="#111827">acc {acc:.2f}%</text>',
                f'<text x="{pad_l + chart_w + 110}" y="{y + 25}" font-family="Arial, sans-serif" font-size="12" fill="#dc2626">MAE {err_s:.3f}s</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{pad_l}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">0% accuracy</text>',
            f'<text x="{pad_l + chart_w - 78}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">100% accuracy</text>',
            f'<text x="{pad_l + chart_w + 18}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" fill="#dc2626">red marker scaled 0-{max_err:.3f}s</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_personal_calibration_delta_svg(path: Path, rows: Sequence[dict[str, object]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120">'
            '<text x="24" y="56" font-family="Arial, sans-serif" font-size="16">No comparison metrics available.</text>'
            "</svg>\n",
            encoding="utf-8",
        )
        return

    width = 1100
    row_h = 66
    pad_l = 190
    pad_r = 260
    pad_t = 88
    chart_w = width - pad_l - pad_r
    height = pad_t + row_h * len(rows) + 72
    max_delta = max(abs(float(row["delta_mean_iou"])) for row in rows)
    max_delta = max(max_delta, 0.01)
    zero_x = pad_l + chart_w / 2.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{escape(title)}</text>',
        '<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">'
        'Bar = after-before mean IoU; green improves, red worsens.</text>',
        f'<line x1="{zero_x:.2f}" y1="{pad_t - 18}" x2="{zero_x:.2f}" y2="{height - 52}" stroke="#9ca3af" stroke-width="2"/>',
    ]
    for idx, row in enumerate(rows):
        y = pad_t + idx * row_h
        person = escape(str(row["person"]))
        reps = int(row["test_reps"])
        delta = float(row["delta_mean_iou"])
        before = float(row["before_mean_iou"])
        after = float(row["after_mean_iou"])
        bar_w = abs(delta) / max_delta * (chart_w / 2.0)
        x = zero_x if delta >= 0 else zero_x - bar_w
        fill = "#16a34a" if delta >= 0 else "#dc2626"
        parts.extend(
            [
                f'<text x="32" y="{y + 22}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{person}</text>',
                f'<text x="32" y="{y + 42}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">test reps={reps}</text>',
                f'<rect x="{x:.2f}" y="{y + 10}" width="{bar_w:.2f}" height="24" rx="4" fill="{fill}"/>',
                f'<text x="{pad_l + chart_w + 18}" y="{y + 24}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{before:.4f} -> {after:.4f}</text>',
                f'<text x="{pad_l + chart_w + 142}" y="{y + 24}" font-family="Arial, sans-serif" font-size="12" fill="{fill}">delta {delta:+.4f}</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{pad_l}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">worse</text>',
            f'<text x="{zero_x + 8:.2f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">0</text>',
            f'<text x="{pad_l + chart_w - 48}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">better</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_split_outputs(
    output_dir: Path,
    data_dir: Path,
    results: Sequence[RepResult],
    train_people: Sequence[str],
    val_people: Sequence[str],
    tolerances: Sequence[int],
    relative_tolerances: Sequence[float],
    iou_thresholds: Sequence[float],
    smooth_window: int,
    plot_limit: int | None,
    method: str,
    seed: int,
    bias_shrink: float,
    tune_iou_bias: bool,
    personal_calibration_reps: int,
    personal_calibration_scope: str,
    personal_calibration_shrink: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output_dir / "person_split.csv", train_people, val_people)

    train_results = filter_by_people(results, train_people)
    val_results = filter_by_people(results, val_people)
    if method == "learned-fraction":
        exact_fractions, exercise_fractions, global_fraction = fit_cut_fractions(train_results)
        train_results = apply_cut_fractions(train_results, exact_fractions, exercise_fractions, global_fraction)
        val_results = apply_cut_fractions(val_results, exact_fractions, exercise_fractions, global_fraction)
        write_csv(output_dir / "learned_cut_fractions.csv", fraction_rows(exact_fractions, exercise_fractions, global_fraction))
    elif method == "supervised-regression":
        train_results, model_info = apply_supervised_regression(
            train_results, train_results, smooth_window, seed, bias_shrink=bias_shrink, tune_iou_bias=tune_iou_bias
        )
        val_results, _ = apply_supervised_regression(
            filter_by_people(results, train_people),
            val_results,
            smooth_window,
            seed,
            bias_shrink=bias_shrink,
            tune_iou_bias=tune_iou_bias,
        )
        write_csv(output_dir / "regression_model_info.csv", model_info_rows(model_info))

    train_rows = summarize_people(train_results, tolerances, relative_tolerances, iou_thresholds, "train")
    if personal_calibration_reps > 0:
        calibration_results, baseline_test_results, calibrated_test_results, calibration_rows = apply_personal_calibration(
            val_results,
            personal_calibration_reps,
            personal_calibration_scope,
            personal_calibration_shrink,
        )
        write_csv(output_dir / "personal_calibration.csv", calibration_rows)
        write_csv(
            output_dir / "val_calibration_person_metrics.csv",
            summarize_people(calibration_results, tolerances, relative_tolerances, iou_thresholds, "calibration"),
        )
        write_csv(
            output_dir / "val_test_person_metrics_uncalibrated.csv",
            summarize_people(baseline_test_results, tolerances, relative_tolerances, iou_thresholds, "test_uncalibrated"),
        )
        comparison_rows = personal_calibration_comparison_rows(
            baseline_test_results,
            calibrated_test_results,
            tolerances,
            relative_tolerances,
            iou_thresholds,
        )
        write_csv(output_dir / "personal_calibration_comparison.csv", comparison_rows)
        write_personal_calibration_delta_svg(
            output_dir / "personal_calibration_iou_delta.svg",
            comparison_rows,
            "Personal Calibration IoU Delta",
        )
        val_results = calibrated_test_results
        val_split_name = "test_calibrated"
    else:
        val_split_name = "val"

    val_rows = summarize_people(val_results, tolerances, relative_tolerances, iou_thresholds, val_split_name)
    train_exercise_rows = summarize_exercises(train_results, tolerances, relative_tolerances, iou_thresholds, "train")
    val_exercise_rows = summarize_exercises(val_results, tolerances, relative_tolerances, iou_thresholds, val_split_name)
    write_csv(output_dir / "train_person_metrics.csv", train_rows)
    write_csv(output_dir / "val_person_metrics.csv", val_rows)
    write_csv(output_dir / "all_person_metrics.csv", train_rows + val_rows)
    write_csv(output_dir / "train_exercise_metrics.csv", train_exercise_rows)
    write_csv(output_dir / "val_exercise_metrics.csv", val_exercise_rows)
    write_csv(output_dir / "all_exercise_metrics.csv", train_exercise_rows + val_exercise_rows)
    write_csv(output_dir / "train_exercise_confusion_matrix.csv", exercise_confusion_matrix_rows(train_results, "train"))
    write_csv(output_dir / "val_exercise_confusion_matrix.csv", exercise_confusion_matrix_rows(val_results, val_split_name))
    write_person_comparison_svg(output_dir / "val_person_accuracy_comparison.svg", val_rows, "Validation Person Accuracy")
    write_person_comparison_svg(output_dir / "all_person_accuracy_comparison.svg", train_rows + val_rows, "All Person Accuracy")
    iou_key, iou_label = iou_accuracy_key(iou_thresholds)
    write_person_comparison_svg(
        output_dir / "val_person_iou_comparison.svg",
        val_rows,
        "Validation Person IoU Accuracy",
        metric_key=iou_key,
        metric_label=iou_label,
    )
    write_person_comparison_svg(
        output_dir / "all_person_iou_comparison.svg",
        train_rows + val_rows,
        "All Person IoU Accuracy",
        metric_key=iou_key,
        metric_label=iou_label,
    )
    write_waveform_plots(data_dir, val_results, output_dir / "val_waveforms", smooth_window, plot_limit)


def write_leave_one_person_out_outputs(
    output_dir: Path,
    data_dir: Path,
    results: Sequence[RepResult],
    tolerances: Sequence[int],
    relative_tolerances: Sequence[float],
    iou_thresholds: Sequence[float],
    smooth_window: int,
    method: str,
    seed: int,
    bias_shrink: float,
    tune_iou_bias: bool,
    personal_calibration_reps: int,
    personal_calibration_scope: str,
    personal_calibration_shrink: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    exercise_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for person in sorted({r.session for r in results}):
        train_people, val_people = split_people_by_validation_names(results, [person])
        train_results = filter_by_people(results, train_people)
        val_results = filter_by_people(results, val_people)

        if method == "learned-fraction":
            exact_fractions, exercise_fractions, global_fraction = fit_cut_fractions(train_results)
            val_results = apply_cut_fractions(val_results, exact_fractions, exercise_fractions, global_fraction)
        elif method == "supervised-regression":
            val_results, _ = apply_supervised_regression(
                train_results,
                val_results,
                smooth_window,
                seed,
                bias_shrink=bias_shrink,
                tune_iou_bias=tune_iou_bias,
            )

        split_name = "val"
        if personal_calibration_reps > 0:
            _, baseline_test_results, val_results, calibration_rows = apply_personal_calibration(
                val_results,
                personal_calibration_reps,
                personal_calibration_scope,
                personal_calibration_shrink,
            )
            split_name = "test_calibrated"
            write_csv(output_dir / f"{person}_personal_calibration.csv", calibration_rows)
            write_csv(
                output_dir / f"{person}_test_metrics_uncalibrated.csv",
                summarize_people(baseline_test_results, tolerances, relative_tolerances, iou_thresholds, "test_uncalibrated"),
            )
            comparison_rows.extend(
                {
                    **row,
                    "held_out_person": person,
                }
                for row in personal_calibration_comparison_rows(
                    baseline_test_results,
                    val_results,
                    tolerances,
                    relative_tolerances,
                    iou_thresholds,
                )
            )

        person_rows = summarize_people(val_results, tolerances, relative_tolerances, iou_thresholds, split_name)
        for row in person_rows:
            row["train_people"] = ",".join(train_people)
            rows.append(row)
        for row in summarize_exercises(val_results, tolerances, relative_tolerances, iou_thresholds, split_name):
            row["held_out_person"] = person
            row["train_people"] = ",".join(train_people)
            exercise_rows.append(row)
        for row in exercise_confusion_matrix_rows(val_results, split_name):
            row["held_out_person"] = person
            confusion_rows.append(row)

    write_csv(output_dir / "leave_one_person_out_metrics.csv", rows)
    write_csv(output_dir / "leave_one_person_out_exercise_metrics.csv", exercise_rows)
    write_csv(output_dir / "leave_one_person_out_exercise_confusion_matrix.csv", confusion_rows)
    if comparison_rows:
        write_csv(output_dir / "leave_one_person_out_personal_calibration_comparison.csv", comparison_rows)
        write_personal_calibration_delta_svg(
            output_dir / "leave_one_person_out_personal_calibration_iou_delta.svg",
            comparison_rows,
            "Leave-One-Person-Out Personal Calibration IoU Delta",
        )
    write_person_comparison_svg(output_dir / "leave_one_person_out_accuracy_comparison.svg", rows, "Leave-One-Person-Out Accuracy")
    iou_key, iou_label = iou_accuracy_key(iou_thresholds)
    write_person_comparison_svg(
        output_dir / "leave_one_person_out_iou_comparison.svg",
        rows,
        "Leave-One-Person-Out IoU Accuracy",
        metric_key=iou_key,
        metric_label=iou_label,
    )


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase_splits(data_dir: Path, results: Sequence[RepResult], output_dir: Path) -> None:
    for result in results:
        df = pd.read_csv(result.path)
        rel = result.path.relative_to(data_dir)
        base = output_dir / rel.parent / result.path.stem
        base.mkdir(parents=True, exist_ok=True)

        first = df.iloc[: result.pred_cut].copy()
        second = df.iloc[result.pred_cut :].copy()
        if first.empty or second.empty:
            continue

        first_phase = result.start_phase if result.start_phase in PHASES else "phase_a"
        second_phase = result.end_phase if result.end_phase in PHASES else "phase_b"
        first.to_csv(base / f"{first_phase}.csv", index=False)
        second.to_csv(base / f"{second_phase}.csv", index=False)


def svg_points(values: np.ndarray, width: int, height: int, pad: int) -> str:
    if len(values) == 0:
        return ""

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0

    usable_w = max(1, width - 2 * pad)
    usable_h = max(1, height - 2 * pad)
    denom = max(1, len(values) - 1)
    points: list[str] = []
    for idx, value in enumerate(values):
        x = pad + usable_w * idx / denom
        y = pad + usable_h * (1.0 - (float(value) - y_min) / (y_max - y_min))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def cut_x(cut: int, n_samples: int, width: int, pad: int) -> float:
    usable_w = max(1, width - 2 * pad)
    denom = max(1, n_samples - 1)
    return pad + usable_w * cut / denom


def write_waveform_svg(
    path: Path,
    result: RepResult,
    signal: np.ndarray,
    source_rel: Path,
) -> None:
    width = 960
    height = 360
    pad = 56
    plot_h = height - 88
    pred_x = cut_x(result.pred_cut, result.n_samples, width, pad)
    true_x = cut_x(result.true_cut, result.n_samples, width, pad)
    points = svg_points(signal, width, plot_h, pad)

    title = escape(f"{result.session}/{result.exercise}/{result.set_name}/{result.rep_name}")
    subtitle = escape(
        f"pred={result.pred_cut}, true={result.true_cut}, abs_error={result.abs_error} samples, "
        f"error={result.abs_error_seconds:.3f}s, rel_error={result.rel_error * 100:.2f}%"
    )
    source = escape(str(source_rel))
    first_phase = escape(result.start_phase)
    second_phase = escape(result.end_phase)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="28" font-family="Arial, sans-serif" font-size="17" fill="#111827">{title}</text>
  <text x="{pad}" y="50" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{subtitle}</text>
  <text x="{pad}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{source}</text>
  <rect x="{pad}" y="{pad}" width="{max(0, pred_x - pad):.2f}" height="{plot_h - 2 * pad}" fill="#dbeafe" opacity="0.55"/>
  <rect x="{pred_x:.2f}" y="{pad}" width="{max(0, width - pad - pred_x):.2f}" height="{plot_h - 2 * pad}" fill="#dcfce7" opacity="0.55"/>
  <line x1="{pad}" y1="{plot_h - pad}" x2="{width - pad}" y2="{plot_h - pad}" stroke="#9ca3af" stroke-width="1"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{plot_h - pad}" stroke="#9ca3af" stroke-width="1"/>
  <polyline points="{points}" fill="none" stroke="#1d4ed8" stroke-width="2"/>
  <line x1="{pred_x:.2f}" y1="{pad}" x2="{pred_x:.2f}" y2="{plot_h - pad}" stroke="#dc2626" stroke-width="3"/>
  <line x1="{true_x:.2f}" y1="{pad}" x2="{true_x:.2f}" y2="{plot_h - pad}" stroke="#111827" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="{pad + 8}" y="{pad + 18}" font-family="Arial, sans-serif" font-size="12" fill="#1e40af">{first_phase}</text>
  <text x="{pred_x + 8:.2f}" y="{pad + 18}" font-family="Arial, sans-serif" font-size="12" fill="#166534">{second_phase}</text>
  <text x="{pred_x + 8:.2f}" y="{plot_h - pad - 10}" font-family="Arial, sans-serif" font-size="12" fill="#dc2626">predicted cut</text>
  <text x="{true_x + 8:.2f}" y="{plot_h - pad - 28}" font-family="Arial, sans-serif" font-size="12" fill="#111827">true phase cut</text>
</svg>
""",
        encoding="utf-8",
    )


def write_waveform_plots(
    data_dir: Path,
    results: Sequence[RepResult],
    output_dir: Path,
    smooth_window: int,
    limit: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected = list(results[:limit]) if limit is not None else list(results)
    for result in selected:
        df = pd.read_csv(result.path)
        signal = principal_motion_signal(df, IMU_COLUMNS, smooth_window)
        rel = result.path.relative_to(data_dir)
        plot_path = output_dir / rel.with_suffix(".svg")
        write_waveform_svg(plot_path, result, signal, rel)
        rows.append(
            {
                "session": result.session,
                "exercise": result.exercise,
                "set": result.set_name,
                "rep": result.rep_name,
                "source_csv": str(rel),
                "plot_svg": str(plot_path.relative_to(output_dir)),
                "pred_cut": result.pred_cut,
                "true_cut": result.true_cut,
                "abs_error": result.abs_error,
                "abs_error_seconds": round(result.abs_error_seconds, 4),
                "rel_error_pct": round(result.rel_error * 100.0, 2),
            }
        )

    write_csv(output_dir / "plot_manifest.csv", rows)


def markdown_table(rows: Sequence[dict[str, object]]) -> str:
    if not rows:
        return "No evaluable reps found.\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one-cut concentric/eccentric phase segmentation inside already-cut workout reps."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/workout"))
    parser.add_argument("--session", default=None, help="Case-insensitive substring filter for the session path.")
    parser.add_argument(
        "--method",
        choices=["midpoint", "signal", "learned-fraction", "supervised-regression", "phase-column"],
        default="midpoint",
        help=(
            "midpoint is the robust one-cut baseline when rep boundaries are assumed correct; "
            "learned-fraction fits per-exercise cut fractions from training people; "
            "supervised-regression fits IMU waveform features from training people; "
            "signal uses an unsupervised IMU turning point; phase-column uses labels as an oracle."
        ),
    )
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--min-fraction", type=float, default=0.15)
    parser.add_argument("--max-fraction", type=float, default=0.85)
    parser.add_argument("--tolerances", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--relative-tolerances", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.75, 0.90])
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--write-splits", type=Path, default=None)
    parser.add_argument("--write-plots", type=Path, default=None)
    parser.add_argument("--plot-limit", type=int, default=None, help="Limit number of waveform SVGs to write.")
    parser.add_argument(
        "--person-split-output",
        type=Path,
        default=None,
        help="Write train/validation person split, per-person metrics, and validation waveform plots.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument(
        "--val-people",
        nargs="+",
        default=None,
        help="Explicit validation people/session folder names. All other people become training data.",
    )
    parser.add_argument(
        "--leave-one-person-out-output",
        type=Path,
        default=None,
        help="Run every person as validation once and write aggregate metrics/accuracy comparison.",
    )
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="Disable leave-one-person-out residual bias correction for supervised-regression.",
    )
    parser.add_argument(
        "--tune-iou-bias",
        action="store_true",
        help="Select supervised-regression bias shrinkage using training-set OOF mean IoU.",
    )
    parser.add_argument(
        "--personal-calibration-reps",
        type=int,
        default=0,
        help=(
            "Use the first N predicted reps from each validation/test person as labeled calibration data, "
            "then report metrics only on that person's later reps."
        ),
    )
    parser.add_argument(
        "--personal-calibration-scope",
        choices=["global", "exercise"],
        default="global",
        help="Estimate one personal bias for all exercises, or exercise-specific personal biases with global fallback.",
    )
    parser.add_argument(
        "--personal-calibration-shrink",
        type=float,
        default=0.25,
        help="Shrink the personal calibration bias before applying it to later reps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, skipped = evaluate(
        data_dir=args.data_dir,
        session_filter=args.session,
        method=args.method,
        smooth_window=args.smooth_window,
        min_fraction=args.min_fraction,
        max_fraction=args.max_fraction,
    )
    rows = summarize(results, args.tolerances, args.relative_tolerances, args.iou_thresholds)
    print(markdown_table(rows))
    if skipped:
        skipped_text = ", ".join(f"{key}={value}" for key, value in skipped.items() if value)
        if skipped_text:
            print(f"Skipped: {skipped_text}")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, rows)
        print(f"Wrote {args.output_csv}")
    if args.write_splits:
        write_phase_splits(args.data_dir, results, args.write_splits)
        print(f"Wrote phase splits under {args.write_splits}")
    if args.write_plots:
        write_waveform_plots(args.data_dir, results, args.write_plots, args.smooth_window, args.plot_limit)
        print(f"Wrote waveform SVGs under {args.write_plots}")
    if args.person_split_output:
        if args.val_people:
            train_people, val_people = split_people_by_validation_names(results, args.val_people)
        else:
            train_people, val_people = split_people(results, args.val_ratio, args.seed)
        write_split_outputs(
            output_dir=args.person_split_output,
            data_dir=args.data_dir,
            results=results,
            train_people=train_people,
            val_people=val_people,
            tolerances=args.tolerances,
            relative_tolerances=args.relative_tolerances,
            iou_thresholds=args.iou_thresholds,
            smooth_window=args.smooth_window,
            plot_limit=args.plot_limit,
            method=args.method,
            seed=args.seed,
            bias_shrink=0.0 if args.no_bias_correction else 0.25,
            tune_iou_bias=args.tune_iou_bias,
            personal_calibration_reps=args.personal_calibration_reps,
            personal_calibration_scope=args.personal_calibration_scope,
            personal_calibration_shrink=args.personal_calibration_shrink,
        )
        print(f"Train people: {', '.join(train_people)}")
        print(f"Validation people: {', '.join(val_people)}")
        print(f"Wrote person split outputs under {args.person_split_output}")
    if args.leave_one_person_out_output:
        write_leave_one_person_out_outputs(
            output_dir=args.leave_one_person_out_output,
            data_dir=args.data_dir,
            results=results,
            tolerances=args.tolerances,
            relative_tolerances=args.relative_tolerances,
            iou_thresholds=args.iou_thresholds,
            smooth_window=args.smooth_window,
            method=args.method,
            seed=args.seed,
            bias_shrink=0.0 if args.no_bias_correction else 0.25,
            tune_iou_bias=args.tune_iou_bias,
            personal_calibration_reps=args.personal_calibration_reps,
            personal_calibration_scope=args.personal_calibration_scope,
            personal_calibration_shrink=args.personal_calibration_shrink,
        )
        print(f"Wrote leave-one-person-out outputs under {args.leave_one_person_out_output}")


if __name__ == "__main__":
    main()
