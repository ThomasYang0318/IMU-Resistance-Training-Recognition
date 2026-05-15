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

from evaluate_rep_segmentation_classification import (
    IMU_COLUMNS,
    moving_average,
    principal_motion_signal,
    robust_zscore,
)


@dataclass(frozen=True)
class FeatureSignal:
    name: str
    values: np.ndarray
    objective: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature alignment to labeled rep boundaries.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set-summary", type=Path)
    parser.add_argument("--search-fraction", type=float, default=0.35)
    parser.add_argument("--min-search-radius", type=int, default=20)
    parser.add_argument("--max-search-radius", type=int, default=160)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--energy-window", type=int, default=51)
    parser.add_argument("--example-padding-fraction", type=float, default=0.15)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    return robust_zscore(np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))


def norm_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    available = [col for col in cols if col in df.columns]
    if not available:
        return np.zeros(len(df), dtype=np.float64)
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    x = np.apply_along_axis(robust_zscore, 0, x)
    return np.linalg.norm(x, axis=1)


def derivative_abs(values: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(values, prepend=values[:1]))


def build_feature_signals(df: pd.DataFrame, smooth_window: int, energy_window: int) -> list[FeatureSignal]:
    pca = normalize(principal_motion_signal(df, smooth_window=smooth_window))
    acc_mag = normalize(moving_average(norm_columns(df, ("ax", "ay", "az")), smooth_window))
    gyro_mag = normalize(moving_average(norm_columns(df, ("gx", "gy", "gz")), smooth_window))
    pca_velocity = normalize(moving_average(derivative_abs(pca), energy_window))
    acc_jerk = normalize(moving_average(derivative_abs(acc_mag), energy_window))
    gyro_jerk = normalize(moving_average(derivative_abs(gyro_mag), energy_window))
    motion_energy = normalize(pca_velocity + gyro_mag + acc_mag)
    transition_energy = normalize(acc_jerk + gyro_jerk + pca_velocity)

    return [
        FeatureSignal("pca_extreme_max", np.abs(pca), "max", "Absolute PCA motion peak"),
        FeatureSignal("pca_velocity_min", pca_velocity, "min", "Low PCA derivative / turn-around"),
        FeatureSignal("acc_magnitude_min", acc_mag, "min", "Low acceleration magnitude"),
        FeatureSignal("gyro_magnitude_min", gyro_mag, "min", "Low gyroscope magnitude"),
        FeatureSignal("acc_jerk_max", acc_jerk, "max", "High acceleration jerk"),
        FeatureSignal("gyro_jerk_max", gyro_jerk, "max", "High gyroscope jerk"),
        FeatureSignal("motion_energy_min", motion_energy, "min", "Low combined motion energy"),
        FeatureSignal("transition_energy_max", transition_energy, "max", "High combined transition energy"),
    ]


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def search_radius(group: pd.DataFrame, search_fraction: float, min_radius: int, max_radius: int) -> int:
    durations = (group["true_end"].astype(int) - group["true_start"].astype(int)).to_numpy(dtype=np.float64)
    median_duration = float(np.median(durations)) if len(durations) else min_radius
    return int(np.clip(round(median_duration * search_fraction), min_radius, max_radius))


def local_candidate(values: np.ndarray, center: int, radius: int, objective: str) -> tuple[int, float]:
    if len(values) == 0:
        return center, 0.0
    lo = max(0, center - radius)
    hi = min(len(values) - 1, center + radius)
    window = values[lo : hi + 1]
    if len(window) == 0:
        return center, 0.0
    if objective == "min":
        offset = int(np.argmin(window))
    elif objective == "max":
        offset = int(np.argmax(window))
    else:
        raise ValueError(f"Unknown objective: {objective}")
    candidate = lo + offset
    return candidate, float(values[candidate])


def analyze_boundaries_for_set(
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
                    "candidate": candidate + set_start,
                    "signed_error_samples": int(signed_error),
                    "abs_error_samples": int(abs(signed_error)),
                    "search_radius": radius,
                    "candidate_score": round(score_value, 4),
                }
            )
    return rows


def summarize_alignment(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(group_cols, sort=True)["abs_error_samples"]
    summary = grouped.agg(
        boundaries="count",
        mean_abs_error="mean",
        median_abs_error="median",
        p80_abs_error=lambda x: float(np.percentile(x, 80)),
        p90_abs_error=lambda x: float(np.percentile(x, 90)),
        within_25_samples=lambda x: float((x <= 25).mean()),
        within_50_samples=lambda x: float((x <= 50).mean()),
        within_100_samples=lambda x: float((x <= 100).mean()),
    ).reset_index()
    for col in [
        "mean_abs_error",
        "median_abs_error",
        "p80_abs_error",
        "p90_abs_error",
        "within_25_samples",
        "within_50_samples",
        "within_100_samples",
    ]:
        summary[col] = summary[col].astype(float).round(4)
    return summary


def feature_label(row: pd.Series) -> str:
    return f"{row['feature']} ({row['objective']})"


def plot_heatmap(summary: pd.DataFrame, value_col: str, output_path: Path, title: str, cmap: str, reverse: bool = False) -> None:
    if summary.empty:
        return
    data = summary.copy()
    data["feature_label"] = data.apply(feature_label, axis=1)
    pivot = data.pivot(index="exercise", columns="feature_label", values=value_col).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(4.8, len(pivot.index) * 0.5)))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if value_col.startswith("within"):
                text = f"{value:.2f}"
            else:
                text = f"{value:.0f}"
            color = "white" if (value > np.nanmax(values) * 0.55 if not reverse else value < np.nanmax(values) * 0.45) else "black"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def best_feature_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.sort_values(["exercise", "median_abs_error", "p80_abs_error", "within_50_samples"], ascending=[True, True, True, False])
    return ranked.groupby("exercise", as_index=False).head(1).reset_index(drop=True)


def plot_feature_examples(
    truth: pd.DataFrame,
    set_summary: pd.DataFrame | None,
    output_dir: Path,
    smooth_window: int,
    energy_window: int,
    padding_fraction: float,
) -> None:
    example_dir = output_dir / "feature_waveform_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    if set_summary is not None and not set_summary.empty:
        selected = (
            set_summary.sort_values("f1")
            .groupby("exercise", sort=True)
            .head(1)
            .loc[:, ["file", "subject", "exercise", "set_id", "f1"]]
        )
    else:
        selected = (
            truth.groupby(["file", "subject", "exercise", "set_id"], sort=True)
            .size()
            .reset_index(name="count")
            .groupby("exercise", sort=True)
            .head(1)
        )
        selected["f1"] = np.nan

    df_cache: dict[str, pd.DataFrame] = {}
    for row in selected.itertuples(index=False):
        group = truth[
            (truth["file"].astype(str) == str(row.file))
            & (truth["subject"].astype(str) == str(row.subject))
            & (truth["exercise"].astype(str) == str(row.exercise))
            & (truth["set_id"].astype(str) == str(row.set_id))
        ].sort_values("true_start")
        if group.empty:
            continue
        file_path = str(row.file)
        if file_path not in df_cache:
            df_cache[file_path] = pd.read_csv(file_path)
        df = df_cache[file_path]
        start = int(group["true_start"].min())
        end = int(group["true_end"].max())
        padding = int(round((end - start) * padding_fraction))
        window_start = max(0, start - padding)
        window_end = min(len(df), end + padding)
        segment_df = df.iloc[window_start:window_end].reset_index(drop=True)
        x = np.arange(window_start, window_end)
        features = build_feature_signals(segment_df, smooth_window=smooth_window, energy_window=energy_window)

        plot_features = [
            ("pca_motion", normalize(principal_motion_signal(segment_df, smooth_window=smooth_window)), "#303030"),
            ("acc_magnitude", normalize(norm_columns(segment_df, ("ax", "ay", "az"))), "#4c78a8"),
            ("gyro_magnitude", normalize(norm_columns(segment_df, ("gx", "gy", "gz"))), "#f58518"),
            ("pca_velocity", next(feature.values for feature in features if feature.name == "pca_velocity_min"), "#54a24b"),
            ("acc_jerk", next(feature.values for feature in features if feature.name == "acc_jerk_max"), "#b279a2"),
            ("gyro_jerk", next(feature.values for feature in features if feature.name == "gyro_jerk_max"), "#e45756"),
        ]
        fig, axes = plt.subplots(len(plot_features), 1, figsize=(14, max(7.0, len(plot_features) * 1.35)), sharex=True)
        boundaries = group["true_start"].astype(int).iloc[1:].tolist()
        for ax, (name, values, color) in zip(axes, plot_features, strict=True):
            ax.plot(x, values[: len(x)], color=color, linewidth=0.85)
            for boundary in boundaries:
                ax.axvline(boundary, color="#0066cc", linewidth=1.1, linestyle="-", alpha=0.9)
            ax.set_ylabel(name)
            ax.grid(axis="x", alpha=0.12)
        axes[-1].set_xlabel("Sample index")
        title = f"{row.subject} | {row.exercise} | set {row.set_id}"
        if not pd.isna(row.f1):
            title += f" | current F1={float(row.f1):.2f}"
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(example_dir / f"{safe_name(row.exercise)}_{safe_name(row.subject)}_set_{safe_name(row.set_id)}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth = read_required_csv(args.run_dir / "rep_segmentation_truth_matches.csv")
    set_summary = pd.read_csv(args.set_summary) if args.set_summary and args.set_summary.exists() else None

    rows: list[dict[str, object]] = []
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
        features = build_feature_signals(segment_df, smooth_window=args.smooth_window, energy_window=args.energy_window)
        rows.extend(
            analyze_boundaries_for_set(
                group,
                features,
                search_fraction=args.search_fraction,
                min_radius=args.min_search_radius,
                max_radius=args.max_search_radius,
            )
        )

    alignment = pd.DataFrame(rows)
    alignment.to_csv(args.output_dir / "boundary_feature_alignment_samples.csv", index=False)
    by_exercise = summarize_alignment(alignment, ["exercise", "feature", "objective"])
    by_subject = summarize_alignment(alignment, ["subject", "feature", "objective"])
    overall = summarize_alignment(alignment, ["feature", "objective"])
    by_exercise.to_csv(args.output_dir / "boundary_feature_alignment_by_exercise.csv", index=False)
    by_subject.to_csv(args.output_dir / "boundary_feature_alignment_by_subject.csv", index=False)
    overall.to_csv(args.output_dir / "boundary_feature_alignment_overall.csv", index=False)

    best = best_feature_table(by_exercise)
    best.to_csv(args.output_dir / "boundary_feature_recommendations_by_exercise.csv", index=False)
    plot_heatmap(
        by_exercise,
        "median_abs_error",
        args.output_dir / "boundary_feature_median_error_by_exercise.png",
        "Boundary Feature Median Absolute Error by Exercise",
        cmap="Reds",
    )
    plot_heatmap(
        by_exercise,
        "within_50_samples",
        args.output_dir / "boundary_feature_within_50_by_exercise.png",
        "Boundary Feature Within 50 Samples by Exercise",
        cmap="Blues",
        reverse=True,
    )
    plot_feature_examples(
        truth,
        set_summary,
        args.output_dir,
        smooth_window=args.smooth_window,
        energy_window=args.energy_window,
        padding_fraction=args.example_padding_fraction,
    )

    summary = {
        "run_dir": str(args.run_dir),
        "set_summary": str(args.set_summary) if args.set_summary else None,
        "num_boundary_feature_rows": int(len(alignment)),
        "num_internal_boundaries": int(alignment[["file", "subject", "exercise", "set_id", "boundary"]].drop_duplicates().shape[0])
        if not alignment.empty
        else 0,
        "search_fraction": args.search_fraction,
        "min_search_radius": args.min_search_radius,
        "max_search_radius": args.max_search_radius,
        "smooth_window": args.smooth_window,
        "energy_window": args.energy_window,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not best.empty:
        print(best[["exercise", "feature", "objective", "median_abs_error", "within_50_samples"]].to_string(index=False))


if __name__ == "__main__":
    main()
