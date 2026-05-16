from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_rep_boundary_features import (
    FeatureSignal,
    derivative_abs,
    local_candidate,
    norm_columns,
    normalize,
    safe_name,
    search_radius,
    summarize_alignment,
)
from evaluate_rep_segmentation_classification import (
    estimate_autocorrelation_period,
    estimate_fft_period,
    moving_average,
    principal_motion_signal,
    robust_zscore,
)


@dataclass(frozen=True)
class PeriodSignal:
    name: str
    values: np.ndarray
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze universal waveform signals for rep boundary segmentation.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-fraction", type=float, default=0.35)
    parser.add_argument("--min-search-radius", type=int, default=20)
    parser.add_argument("--max-search-radius", type=int, default=160)
    parser.add_argument("--smooth-windows", type=str, default="9,21,51")
    parser.add_argument("--energy-windows", type=str, default="21,51,81")
    parser.add_argument("--autocorr-min-period", type=int, default=25)
    parser.add_argument("--autocorr-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--fft-min-period", type=int, default=25)
    parser.add_argument("--fft-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--example-sets-per-exercise", type=int, default=1)
    parser.add_argument("--top-signals-for-examples", type=int, default=5)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_signal_components(df: pd.DataFrame, smooth_window: int, energy_window: int) -> dict[str, np.ndarray]:
    pca = normalize(principal_motion_signal(df, smooth_window=smooth_window))
    acc_mag = normalize(moving_average(norm_columns(df, ("ax", "ay", "az")), smooth_window))
    gyro_mag = normalize(moving_average(norm_columns(df, ("gx", "gy", "gz")), smooth_window))
    pca_velocity = normalize(moving_average(derivative_abs(pca), energy_window))
    acc_jerk = normalize(moving_average(derivative_abs(acc_mag), energy_window))
    gyro_jerk = normalize(moving_average(derivative_abs(gyro_mag), energy_window))
    motion_energy = normalize(pca_velocity + acc_mag + gyro_mag)
    transition_energy = normalize(acc_jerk + gyro_jerk + pca_velocity)
    return {
        "pca_motion": pca,
        "pca_extreme": np.abs(pca),
        "acc_magnitude": acc_mag,
        "gyro_magnitude": gyro_mag,
        "pca_velocity": pca_velocity,
        "acc_jerk": acc_jerk,
        "gyro_jerk": gyro_jerk,
        "motion_energy": motion_energy,
        "transition_energy": transition_energy,
    }


def build_boundary_signals(df: pd.DataFrame, smooth_windows: list[int], energy_windows: list[int]) -> list[FeatureSignal]:
    signals: list[FeatureSignal] = []
    seen: set[str] = set()
    for smooth_window in smooth_windows:
        for energy_window in energy_windows:
            parts = build_signal_components(df, smooth_window=smooth_window, energy_window=energy_window)
            candidates = [
                FeatureSignal(f"pca_extreme_max_s{smooth_window}", parts["pca_extreme"], "max", "Absolute PCA motion peak"),
                FeatureSignal(f"acc_magnitude_min_s{smooth_window}", parts["acc_magnitude"], "min", "Low acceleration magnitude"),
                FeatureSignal(f"gyro_magnitude_min_s{smooth_window}", parts["gyro_magnitude"], "min", "Low gyroscope magnitude"),
                FeatureSignal(f"pca_velocity_min_s{smooth_window}_e{energy_window}", parts["pca_velocity"], "min", "Low PCA derivative / turn-around"),
                FeatureSignal(f"acc_jerk_max_s{smooth_window}_e{energy_window}", parts["acc_jerk"], "max", "High acceleration jerk"),
                FeatureSignal(f"gyro_jerk_max_s{smooth_window}_e{energy_window}", parts["gyro_jerk"], "max", "High gyroscope jerk"),
                FeatureSignal(f"motion_energy_min_s{smooth_window}_e{energy_window}", parts["motion_energy"], "min", "Low combined motion energy"),
                FeatureSignal(f"transition_energy_max_s{smooth_window}_e{energy_window}", parts["transition_energy"], "max", "High combined transition energy"),
            ]
            for signal in candidates:
                if signal.name in seen:
                    continue
                seen.add(signal.name)
                signals.append(signal)
    return signals


def build_period_signals(df: pd.DataFrame, smooth_window: int, energy_window: int) -> list[PeriodSignal]:
    parts = build_signal_components(df, smooth_window=smooth_window, energy_window=energy_window)
    return [
        PeriodSignal("pca_motion", parts["pca_motion"], "PCA motion signal"),
        PeriodSignal("abs_pca_motion", parts["pca_extreme"], "Absolute PCA motion"),
        PeriodSignal("acc_magnitude", parts["acc_magnitude"], "Acceleration magnitude"),
        PeriodSignal("gyro_magnitude", parts["gyro_magnitude"], "Gyroscope magnitude"),
        PeriodSignal("pca_velocity", parts["pca_velocity"], "PCA derivative energy"),
        PeriodSignal("motion_energy", parts["motion_energy"], "Combined motion energy"),
        PeriodSignal("transition_energy", parts["transition_energy"], "Combined jerk / transition energy"),
    ]


def analyze_boundary_alignment(
    group: pd.DataFrame,
    feature_signals: list[FeatureSignal],
    search_fraction: float,
    min_radius: int,
    max_radius: int,
) -> list[dict[str, object]]:
    group = group.sort_values("true_start").copy()
    if len(group) < 2:
        return []
    set_start = int(group["true_start"].min())
    radius = search_radius(group, search_fraction, min_radius, max_radius)
    rows: list[dict[str, object]] = []
    internal_boundaries = group["true_start"].astype(int).iloc[1:].tolist()
    meta = {
        "file": str(group["file"].iloc[0]),
        "subject": str(group["subject"].iloc[0]),
        "exercise": str(group["exercise"].iloc[0]),
        "set_id": str(group["set_id"].iloc[0]),
    }
    for boundary in internal_boundaries:
        local_boundary = int(boundary) - set_start
        for feature in feature_signals:
            candidate, score_value = local_candidate(feature.values, local_boundary, radius, feature.objective)
            signed_error = candidate - local_boundary
            rows.append(
                {
                    **meta,
                    "boundary": int(boundary),
                    "local_boundary": local_boundary,
                    "feature": feature.name,
                    "objective": feature.objective,
                    "description": feature.description,
                    "candidate": candidate + set_start,
                    "signed_error_samples": int(signed_error),
                    "abs_error_samples": int(abs(signed_error)),
                    "search_radius": radius,
                    "candidate_score": round(score_value, 4),
                }
            )
    return rows


def true_period_for_set(group: pd.DataFrame) -> float | None:
    starts = group.sort_values("true_start")["true_start"].astype(int).to_numpy()
    if len(starts) >= 2:
        diffs = np.diff(starts)
        diffs = diffs[diffs > 0]
        if len(diffs):
            return float(np.median(diffs))
    durations = (group["true_end"].astype(int) - group["true_start"].astype(int)).to_numpy()
    durations = durations[durations > 0]
    return float(np.median(durations)) if len(durations) else None


def analyze_periods_for_set(
    group: pd.DataFrame,
    period_signals: list[PeriodSignal],
    autocorr_min_period: int,
    autocorr_max_period_fraction: float,
    fft_min_period: int,
    fft_max_period_fraction: float,
) -> list[dict[str, object]]:
    group = group.sort_values("true_start").copy()
    if len(group) < 2:
        return []
    true_period = true_period_for_set(group)
    if true_period is None or true_period <= 0:
        return []
    set_length = int(group["true_end"].max() - group["true_start"].min())
    rows: list[dict[str, object]] = []
    meta = {
        "file": str(group["file"].iloc[0]),
        "subject": str(group["subject"].iloc[0]),
        "exercise": str(group["exercise"].iloc[0]),
        "set_id": str(group["set_id"].iloc[0]),
        "num_reps": int(len(group)),
        "set_length_samples": set_length,
        "true_period_samples": round(true_period, 4),
    }
    for signal in period_signals:
        for method in ("autocorr", "fft"):
            if method == "autocorr":
                max_period = max(autocorr_min_period, int(round(len(signal.values) * autocorr_max_period_fraction)))
                estimate = estimate_autocorrelation_period(signal.values, autocorr_min_period, max_period)
            else:
                max_period = max(fft_min_period, int(round(len(signal.values) * fft_max_period_fraction)))
                estimate = estimate_fft_period(signal.values, fft_min_period, max_period)
            if estimate is None:
                rows.append(
                    {
                        **meta,
                        "signal": signal.name,
                        "method": method,
                        "estimated_period_samples": np.nan,
                        "signed_error_samples": np.nan,
                        "abs_error_samples": np.nan,
                        "relative_abs_error": np.nan,
                        "within_10_percent": False,
                        "within_20_percent": False,
                    }
                )
                continue
            signed_error = float(estimate - true_period)
            abs_error = abs(signed_error)
            rel_error = abs_error / true_period
            rows.append(
                {
                    **meta,
                    "signal": signal.name,
                    "method": method,
                    "estimated_period_samples": round(float(estimate), 4),
                    "signed_error_samples": round(signed_error, 4),
                    "abs_error_samples": round(abs_error, 4),
                    "relative_abs_error": round(rel_error, 4),
                    "within_10_percent": bool(rel_error <= 0.10),
                    "within_20_percent": bool(rel_error <= 0.20),
                }
            )
    return rows


def summarize_periods(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    valid = rows.dropna(subset=["abs_error_samples"]).copy()
    if valid.empty:
        return pd.DataFrame()
    summary = (
        valid.groupby(group_cols, sort=True)
        .agg(
            sets=("abs_error_samples", "count"),
            mean_abs_error=("abs_error_samples", "mean"),
            median_abs_error=("abs_error_samples", "median"),
            p80_abs_error=("abs_error_samples", lambda x: float(np.percentile(x, 80))),
            median_relative_abs_error=("relative_abs_error", "median"),
            within_10_percent=("within_10_percent", "mean"),
            within_20_percent=("within_20_percent", "mean"),
        )
        .reset_index()
    )
    for col in [
        "mean_abs_error",
        "median_abs_error",
        "p80_abs_error",
        "median_relative_abs_error",
        "within_10_percent",
        "within_20_percent",
    ]:
        summary[col] = summary[col].astype(float).round(4)
    return summary


def universal_boundary_ranking(overall: pd.DataFrame, by_exercise: pd.DataFrame, by_subject: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    max_p80 = max(float(overall["p80_abs_error"].max()), 1.0)
    for row in overall.itertuples(index=False):
        feature = str(row.feature)
        objective = str(row.objective)
        exercise_rows = by_exercise[(by_exercise["feature"].eq(feature)) & (by_exercise["objective"].eq(objective))]
        subject_rows = by_subject[(by_subject["feature"].eq(feature)) & (by_subject["objective"].eq(objective))]
        worst_exercise_within_50 = float(exercise_rows["within_50_samples"].min()) if not exercise_rows.empty else 0.0
        exercise_within_50_std = float(exercise_rows["within_50_samples"].std(ddof=0)) if len(exercise_rows) else 1.0
        subject_within_50_std = float(subject_rows["within_50_samples"].std(ddof=0)) if len(subject_rows) else 1.0
        stability_score = max(0.0, 1.0 - exercise_within_50_std)
        subject_stability_score = max(0.0, 1.0 - subject_within_50_std)
        p80_score = max(0.0, 1.0 - float(row.p80_abs_error) / max_p80)
        universal_score = (
            0.34 * float(row.within_50_samples)
            + 0.18 * float(row.within_25_samples)
            + 0.20 * worst_exercise_within_50
            + 0.14 * p80_score
            + 0.08 * stability_score
            + 0.06 * subject_stability_score
        )
        rows.append(
            {
                "feature": feature,
                "objective": objective,
                "boundaries": int(row.boundaries),
                "median_abs_error": float(row.median_abs_error),
                "p80_abs_error": float(row.p80_abs_error),
                "within_25_samples": float(row.within_25_samples),
                "within_50_samples": float(row.within_50_samples),
                "within_100_samples": float(row.within_100_samples),
                "worst_exercise_within_50": round(worst_exercise_within_50, 4),
                "exercise_within_50_std": round(exercise_within_50_std, 4),
                "subject_within_50_std": round(subject_within_50_std, 4),
                "universal_score": round(float(universal_score), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["universal_score", "median_abs_error", "p80_abs_error"],
        ascending=[False, True, True],
    )


def plot_universal_ranking(ranking: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    if ranking.empty:
        return
    data = ranking.head(top_n).iloc[::-1]
    labels = data["feature"] + " (" + data["objective"] + ")"
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    ax.barh(labels, data["universal_score"], color="#4c78a8")
    ax.set_xlabel("Universal boundary score")
    ax.set_title("Universal Rep-Boundary Feature Ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_boundary_heatmap(summary: pd.DataFrame, value_col: str, output_path: Path, title: str, cmap: str) -> None:
    if summary.empty:
        return
    data = summary.copy()
    top_features = (
        data.groupby(["feature", "objective"], sort=True)["within_50_samples"]
        .mean()
        .sort_values(ascending=False)
        .head(14)
        .index
    )
    keep = pd.MultiIndex.from_frame(data[["feature", "objective"]]).isin(top_features)
    data = data.loc[keep].copy()
    data["feature_label"] = data["feature"] + " (" + data["objective"] + ")"
    pivot = data.pivot(index="exercise", columns="feature_label", values=value_col).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.85), max(5, len(pivot.index) * 0.5)))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            text = f"{value:.2f}" if "within" in value_col else f"{value:.0f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=6, color="white" if value > np.nanmax(values) * 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_period_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    data = summary.sort_values(["median_relative_abs_error", "median_abs_error"]).head(20).iloc[::-1]
    labels = data["signal"] + " / " + data["method"]
    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.34)))
    ax.barh(labels, data["median_relative_abs_error"], color="#f58518")
    ax.set_xlabel("Median relative period error")
    ax.set_title("Period Estimation Error by Waveform Signal")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_period_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    data = summary.copy()
    data["signal_method"] = data["signal"] + " / " + data["method"]
    top = (
        data.groupby("signal_method")["median_relative_abs_error"]
        .mean()
        .sort_values()
        .head(12)
        .index
    )
    data = data[data["signal_method"].isin(top)]
    pivot = data.pivot(index="exercise", columns="signal_method", values="median_relative_abs_error").fillna(1.0)
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.9), max(5, len(pivot.index) * 0.5)))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Reds", aspect="auto", vmin=0.0, vmax=min(1.0, max(0.2, float(np.nanmax(values)))))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Period Estimation Median Relative Error by Exercise")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > np.nanmax(values) * 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="median relative abs error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_examples(
    truth: pd.DataFrame,
    output_dir: Path,
    ranking: pd.DataFrame,
    smooth_window: int,
    energy_window: int,
    sets_per_exercise: int,
    top_signals: int,
) -> None:
    if ranking.empty:
        return
    example_dir = output_dir / "universal_feature_waveform_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    top_feature_names = ranking.head(top_signals)["feature"].astype(str).tolist()
    selected = (
        truth.groupby(["file", "subject", "exercise", "set_id"], sort=True)
        .size()
        .reset_index(name="reps")
        .sort_values(["exercise", "reps"], ascending=[True, False])
        .groupby("exercise", sort=True)
        .head(sets_per_exercise)
    )
    df_cache: dict[str, pd.DataFrame] = {}
    for row in selected.itertuples(index=False):
        group = truth[
            (truth["file"].astype(str) == str(row.file))
            & (truth["subject"].astype(str) == str(row.subject))
            & (truth["exercise"].astype(str) == str(row.exercise))
            & (truth["set_id"].astype(str) == str(row.set_id))
        ].sort_values("true_start")
        if len(group) < 2:
            continue
        file_path = str(row.file)
        if file_path not in df_cache:
            df_cache[file_path] = pd.read_csv(file_path)
        df = df_cache[file_path]
        set_start = int(group["true_start"].min())
        set_end = int(group["true_end"].max())
        segment_df = df.iloc[set_start:set_end].reset_index(drop=True)
        signals = {signal.name: signal for signal in build_boundary_signals(segment_df, [smooth_window], [energy_window])}
        plot_signals = [signals[name] for name in top_feature_names if name in signals]
        if not plot_signals:
            continue
        x = np.arange(set_start, set_end)
        boundaries = group["true_start"].astype(int).iloc[1:].tolist()
        fig, axes = plt.subplots(len(plot_signals), 1, figsize=(14, max(5.5, len(plot_signals) * 1.2)), sharex=True)
        if len(plot_signals) == 1:
            axes = [axes]
        for ax, signal in zip(axes, plot_signals, strict=True):
            ax.plot(x, signal.values[: len(x)], linewidth=0.85)
            for boundary in boundaries:
                ax.axvline(boundary, color="#0066cc", linewidth=1.0, alpha=0.9)
            ax.set_ylabel(signal.name, fontsize=8)
            ax.grid(axis="x", alpha=0.15)
        axes[-1].set_xlabel("Sample index")
        fig.suptitle(f"{row.subject} | {row.exercise} | set {row.set_id}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(example_dir / f"{safe_name(row.exercise)}_{safe_name(row.subject)}_set_{safe_name(row.set_id)}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smooth_windows = parse_ints(args.smooth_windows)
    energy_windows = parse_ints(args.energy_windows)
    default_smooth = smooth_windows[len(smooth_windows) // 2]
    default_energy = energy_windows[len(energy_windows) // 2]

    truth = read_required_csv(args.run_dir / "rep_segmentation_truth_matches.csv")
    rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    df_cache: dict[str, pd.DataFrame] = {}
    for _, group in truth.groupby(["file", "subject", "exercise", "set_id"], sort=True):
        group = group.sort_values("true_start").copy()
        if len(group) < 2:
            continue
        file_path = str(group["file"].iloc[0])
        if file_path not in df_cache:
            df_cache[file_path] = pd.read_csv(file_path)
        df = df_cache[file_path]
        set_start = int(group["true_start"].min())
        set_end = int(group["true_end"].max())
        segment_df = df.iloc[set_start:set_end].reset_index(drop=True)
        boundary_signals = build_boundary_signals(segment_df, smooth_windows=smooth_windows, energy_windows=energy_windows)
        rows.extend(
            analyze_boundary_alignment(
                group,
                boundary_signals,
                search_fraction=args.search_fraction,
                min_radius=args.min_search_radius,
                max_radius=args.max_search_radius,
            )
        )
        period_signals = build_period_signals(segment_df, smooth_window=default_smooth, energy_window=default_energy)
        period_rows.extend(
            analyze_periods_for_set(
                group,
                period_signals,
                autocorr_min_period=args.autocorr_min_period,
                autocorr_max_period_fraction=args.autocorr_max_period_fraction,
                fft_min_period=args.fft_min_period,
                fft_max_period_fraction=args.fft_max_period_fraction,
            )
        )

    alignment = pd.DataFrame(rows)
    alignment.to_csv(args.output_dir / "universal_boundary_alignment_samples.csv", index=False)
    by_exercise = summarize_alignment(alignment, ["exercise", "feature", "objective"])
    by_subject = summarize_alignment(alignment, ["subject", "feature", "objective"])
    overall = summarize_alignment(alignment, ["feature", "objective"])
    by_exercise.to_csv(args.output_dir / "universal_boundary_alignment_by_exercise.csv", index=False)
    by_subject.to_csv(args.output_dir / "universal_boundary_alignment_by_subject.csv", index=False)
    overall.to_csv(args.output_dir / "universal_boundary_alignment_overall.csv", index=False)
    ranking = universal_boundary_ranking(overall, by_exercise, by_subject)
    ranking.to_csv(args.output_dir / "universal_boundary_feature_ranking.csv", index=False)

    periods = pd.DataFrame(period_rows)
    periods.to_csv(args.output_dir / "period_estimation_samples.csv", index=False)
    period_summary = summarize_periods(periods, ["signal", "method"])
    period_by_exercise = summarize_periods(periods, ["exercise", "signal", "method"])
    if not period_summary.empty:
        period_summary = period_summary.sort_values(["median_relative_abs_error", "median_abs_error"]).reset_index(drop=True)
    if not period_by_exercise.empty:
        period_by_exercise = period_by_exercise.sort_values(["exercise", "median_relative_abs_error", "median_abs_error"]).reset_index(drop=True)
    period_summary.to_csv(args.output_dir / "period_estimation_summary.csv", index=False)
    period_by_exercise.to_csv(args.output_dir / "period_estimation_by_exercise.csv", index=False)

    plot_universal_ranking(ranking, args.output_dir / "universal_boundary_feature_ranking.png")
    plot_boundary_heatmap(by_exercise, "within_50_samples", args.output_dir / "universal_boundary_within_50_by_exercise.png", "Universal Boundary Feature Within 50 Samples by Exercise", "Blues")
    plot_boundary_heatmap(by_exercise, "median_abs_error", args.output_dir / "universal_boundary_median_error_by_exercise.png", "Universal Boundary Feature Median Error by Exercise", "Reds")
    plot_period_summary(period_summary, args.output_dir / "period_estimation_error_by_signal.png")
    plot_period_heatmap(period_by_exercise, args.output_dir / "period_estimation_error_by_exercise.png")
    plot_examples(
        truth,
        args.output_dir,
        ranking,
        smooth_window=default_smooth,
        energy_window=default_energy,
        sets_per_exercise=args.example_sets_per_exercise,
        top_signals=args.top_signals_for_examples,
    )

    best_boundary = ranking.iloc[0].to_dict() if not ranking.empty else {}
    best_period = period_summary.iloc[0].to_dict() if not period_summary.empty else {}
    summary = {
        "run_dir": str(args.run_dir),
        "num_boundary_rows": int(len(alignment)),
        "num_internal_boundaries": int(alignment[["file", "subject", "exercise", "set_id", "boundary"]].drop_duplicates().shape[0]) if not alignment.empty else 0,
        "num_period_rows": int(len(periods)),
        "smooth_windows": smooth_windows,
        "energy_windows": energy_windows,
        "best_universal_boundary_feature": best_boundary.get("feature"),
        "best_universal_boundary_objective": best_boundary.get("objective"),
        "best_universal_boundary_median_abs_error": best_boundary.get("median_abs_error"),
        "best_universal_boundary_within_50": best_boundary.get("within_50_samples"),
        "best_period_signal": best_period.get("signal"),
        "best_period_method": best_period.get("method"),
        "best_period_median_relative_abs_error": best_period.get("median_relative_abs_error"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not ranking.empty:
        print(ranking.head(15).to_string(index=False))
    if not period_summary.empty:
        print(period_summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
