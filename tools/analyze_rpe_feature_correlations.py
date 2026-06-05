from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


GROUP_COLUMNS = ["folder", "exercise", "set_id"]
TARGET = "borg"


FEATURE_GROUPS: dict[str, str] = {
    "kg": "load",
    "kg_x_rep": "load",
    "kg_x_cumulative_tut": "load",
    "kg_x_total_tut": "load",
    "kg_x_n_reps": "load",
    "rep_index": "progress",
    "rep_progress": "progress",
    "set_index_numeric": "progress",
    "n_reps": "progress",
    "rep_duration_sec": "tut",
    "concentric_sec": "tut",
    "eccentric_sec": "tut",
    "concentric_ratio": "phase",
    "eccentric_ratio": "phase",
    "phase_balance_abs": "phase",
    "total_tut_sec": "tut",
    "total_concentric_sec": "tut",
    "total_eccentric_sec": "tut",
    "cumulative_tut_sec": "tut",
    "cumulative_concentric_sec": "tut",
    "cumulative_eccentric_sec": "tut",
    "duration_gain_from_first2": "fatigue_drift",
    "concentric_gain_from_first2": "fatigue_drift",
    "eccentric_gain_from_first2": "fatigue_drift",
    "pca_range_change_from_first2": "fatigue_drift",
    "gyro_diff_change_from_first2": "fatigue_drift",
    "acc_diff_change_from_first2": "fatigue_drift",
    "movement_rate": "waveform_intensity",
    "movement_rate_change_from_first2": "fatigue_drift",
    "velocity_loss_proxy": "fatigue_drift",
    "similarity_decay_from_first": "similarity",
    "similarity_instability_prev": "similarity",
    "sim_to_first": "similarity",
    "sim_to_prev": "similarity",
    "duration_cv_so_far": "variability",
    "pca_range_cv_so_far": "variability",
    "movement_rate_cv_so_far": "variability",
    "similarity_std_so_far": "variability",
    "duration_gain_last2_vs_first2": "fatigue_drift",
    "concentric_gain_last2_vs_first2": "fatigue_drift",
    "eccentric_gain_last2_vs_first2": "fatigue_drift",
    "velocity_loss_last2_vs_first2": "fatigue_drift",
    "pca_range_loss_last2_vs_first2": "fatigue_drift",
    "gyro_diff_gain_last2_vs_first2": "fatigue_drift",
    "acc_diff_gain_last2_vs_first2": "fatigue_drift",
    "sim_to_first_last_minus_first": "similarity",
    "sim_to_first_mean": "similarity",
    "sim_to_first_min": "similarity",
    "sim_to_first_last": "similarity",
}


def safe_divide(numerator: pd.Series | np.ndarray | float, denominator: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def numeric_slope(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
    if len(y) < 2 or float(np.nanstd(y)) < 1e-12:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(y))
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def coefficient_of_variation(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return 0.0
    denom = abs(float(np.nanmean(y)))
    if denom < 1e-9:
        return 0.0
    return float(np.nanstd(y) / denom)


def mean_first(values: pd.Series, n: int = 2) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.head(n).mean())


def mean_last(values: pd.Series, n: int = 2) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.tail(n).mean())


def feature_group(feature: str) -> str:
    if feature in FEATURE_GROUPS:
        return FEATURE_GROUPS[feature]
    if feature.startswith(("pca_", "acc_mag_", "gyro_mag_", "mag_mag_", "ax_", "ay_", "az_", "gx_", "gy_", "gz_", "mx_", "my_", "mz_")):
        if feature.endswith(("_slope",)):
            return "shape_trend"
        if "diff" in feature or "range" in feature:
            return "waveform_intensity"
        return "waveform_shape"
    if feature.endswith("_slope"):
        return "fatigue_drift"
    if feature.endswith(("_std", "_cv")):
        return "variability"
    if "duration" in feature or "tut" in feature or feature.endswith("_sec"):
        return "tut"
    return "other"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["excluded_from_training"].eq(False)].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for col in df.columns:
        if col not in {"folder", "file", "subject", "exercise", "set_id", "rep_id", "raw_value", "excluded_from_training"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["set_id"] = df["set_id"].astype(str)
    df["rep_id"] = df["rep_id"].astype(str)
    df["set_index_numeric"] = pd.to_numeric(df["set_id"], errors="coerce")
    return df.sort_values(GROUP_COLUMNS + ["rep_index"]).reset_index(drop=True)


def add_rep_level_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in df.groupby(GROUP_COLUMNS, sort=False):
        group = group.sort_values("rep_index").copy()
        n_reps = len(group)
        group["n_reps"] = n_reps
        group["rep_order"] = np.arange(n_reps)
        group["rep_progress"] = group["rep_order"] / max(n_reps - 1, 1)
        group["movement_rate"] = safe_divide(group["pca_range"], group["rep_duration_sec"])
        group["kg_x_rep"] = group["kg"] * (group["rep_order"] + 1)

        cumulative_tut = group["rep_duration_sec"].cumsum()
        group["cumulative_tut_sec"] = cumulative_tut
        group["cumulative_concentric_sec"] = group["concentric_sec"].cumsum()
        group["cumulative_eccentric_sec"] = group["eccentric_sec"].cumsum()
        group["kg_x_cumulative_tut"] = group["kg"] * cumulative_tut

        baselines = {
            "rep_duration_sec": mean_first(group["rep_duration_sec"]),
            "concentric_sec": mean_first(group["concentric_sec"]),
            "eccentric_sec": mean_first(group["eccentric_sec"]),
            "pca_range": mean_first(group["pca_range"]),
            "gyro_mag_diff_rms": mean_first(group["gyro_mag_diff_rms"]),
            "acc_mag_diff_rms": mean_first(group["acc_mag_diff_rms"]),
            "movement_rate": mean_first(group["movement_rate"]),
        }
        group["duration_gain_from_first2"] = safe_divide(group["rep_duration_sec"], baselines["rep_duration_sec"]) - 1.0
        group["concentric_gain_from_first2"] = safe_divide(group["concentric_sec"], baselines["concentric_sec"]) - 1.0
        group["eccentric_gain_from_first2"] = safe_divide(group["eccentric_sec"], baselines["eccentric_sec"]) - 1.0
        group["pca_range_change_from_first2"] = safe_divide(group["pca_range"], baselines["pca_range"]) - 1.0
        group["gyro_diff_change_from_first2"] = safe_divide(group["gyro_mag_diff_rms"], baselines["gyro_mag_diff_rms"]) - 1.0
        group["acc_diff_change_from_first2"] = safe_divide(group["acc_mag_diff_rms"], baselines["acc_mag_diff_rms"]) - 1.0
        group["movement_rate_change_from_first2"] = safe_divide(group["movement_rate"], baselines["movement_rate"]) - 1.0
        group["velocity_loss_proxy"] = 1.0 - safe_divide(group["movement_rate"], baselines["movement_rate"])
        group["similarity_decay_from_first"] = 1.0 - group["sim_to_first"]
        group["similarity_instability_prev"] = 1.0 - group["sim_to_prev"]

        group["duration_cv_so_far"] = group["rep_duration_sec"].expanding().apply(coefficient_of_variation, raw=False)
        group["pca_range_cv_so_far"] = group["pca_range"].expanding().apply(coefficient_of_variation, raw=False)
        group["movement_rate_cv_so_far"] = group["movement_rate"].expanding().apply(coefficient_of_variation, raw=False)
        group["similarity_std_so_far"] = group["sim_to_first"].expanding().std().fillna(0.0)
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def build_set_level(rep_df: pd.DataFrame) -> pd.DataFrame:
    set_rows: list[dict[str, object]] = []
    for key, group in rep_df.groupby(GROUP_COLUMNS, sort=False):
        group = group.sort_values("rep_index").copy()
        first_rate = mean_first(group["movement_rate"])
        last_rate = mean_last(group["movement_rate"])
        first_range = mean_first(group["pca_range"])
        last_range = mean_last(group["pca_range"])
        first_gyro = mean_first(group["gyro_mag_diff_rms"])
        last_gyro = mean_last(group["gyro_mag_diff_rms"])
        first_acc = mean_first(group["acc_mag_diff_rms"])
        last_acc = mean_last(group["acc_mag_diff_rms"])
        row: dict[str, object] = {
            "folder": key[0],
            "exercise": key[1],
            "set_id": key[2],
            "set_index_numeric": group["set_index_numeric"].iloc[0],
            "borg": float(group["borg"].iloc[-1]),
            "kg": float(group["kg"].dropna().iloc[0]) if group["kg"].notna().any() else np.nan,
            "n_reps": len(group),
            "total_tut_sec": float(group["rep_duration_sec"].sum()),
            "total_concentric_sec": float(group["concentric_sec"].sum()),
            "total_eccentric_sec": float(group["eccentric_sec"].sum()),
            "mean_rep_duration_sec": float(group["rep_duration_sec"].mean()),
            "mean_concentric_sec": float(group["concentric_sec"].mean()),
            "mean_eccentric_sec": float(group["eccentric_sec"].mean()),
            "mean_concentric_ratio": float(group["concentric_ratio"].mean()),
            "mean_phase_balance_abs": float(group["phase_balance_abs"].mean()),
            "duration_gain_last2_vs_first2": safe_scalar_ratio(mean_last(group["rep_duration_sec"]), mean_first(group["rep_duration_sec"])) - 1.0,
            "concentric_gain_last2_vs_first2": safe_scalar_ratio(mean_last(group["concentric_sec"]), mean_first(group["concentric_sec"])) - 1.0,
            "eccentric_gain_last2_vs_first2": safe_scalar_ratio(mean_last(group["eccentric_sec"]), mean_first(group["eccentric_sec"])) - 1.0,
            "velocity_loss_last2_vs_first2": 1.0 - safe_scalar_ratio(last_rate, first_rate),
            "pca_range_loss_last2_vs_first2": 1.0 - safe_scalar_ratio(last_range, first_range),
            "gyro_diff_gain_last2_vs_first2": safe_scalar_ratio(last_gyro, first_gyro) - 1.0,
            "acc_diff_gain_last2_vs_first2": safe_scalar_ratio(last_acc, first_acc) - 1.0,
            "sim_to_first_mean": float(group["sim_to_first"].mean()),
            "sim_to_first_min": float(group["sim_to_first"].min()),
            "sim_to_first_last": float(group["sim_to_first"].iloc[-1]),
            "sim_to_first_last_minus_first": float(group["sim_to_first"].iloc[-1] - group["sim_to_first"].iloc[0]),
            "rep_duration_slope": numeric_slope(group["rep_duration_sec"]),
            "concentric_sec_slope": numeric_slope(group["concentric_sec"]),
            "eccentric_sec_slope": numeric_slope(group["eccentric_sec"]),
            "pca_range_slope": numeric_slope(group["pca_range"]),
            "movement_rate_slope": numeric_slope(group["movement_rate"]),
            "gyro_mag_diff_rms_slope": numeric_slope(group["gyro_mag_diff_rms"]),
            "acc_mag_diff_rms_slope": numeric_slope(group["acc_mag_diff_rms"]),
            "sim_to_first_slope": numeric_slope(group["sim_to_first"]),
            "rep_duration_cv": coefficient_of_variation(group["rep_duration_sec"]),
            "pca_range_cv": coefficient_of_variation(group["pca_range"]),
            "movement_rate_cv": coefficient_of_variation(group["movement_rate"]),
            "sim_to_first_std": float(group["sim_to_first"].std()) if len(group) > 1 else 0.0,
        }
        row["kg_x_total_tut"] = row["kg"] * row["total_tut_sec"] if np.isfinite(row["kg"]) else np.nan
        row["kg_x_n_reps"] = row["kg"] * row["n_reps"] if np.isfinite(row["kg"]) else np.nan
        for col in [
            "pca_range",
            "pca_diff_rms",
            "acc_mag_range",
            "acc_mag_diff_rms",
            "gyro_mag_range",
            "gyro_mag_diff_rms",
            "mag_mag_range",
            "gx_range",
            "gy_range",
            "gz_range",
        ]:
            if col in group:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_max"] = float(group[col].max())
                row[f"{col}_std"] = float(group[col].std()) if len(group) > 1 else 0.0
        set_rows.append(row)
    return pd.DataFrame(set_rows)


def safe_scalar_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return np.nan
    return float(numerator / denominator)


def centered(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").copy()
    frame = pd.DataFrame({"value": out})
    for idx, group in enumerate(groups):
        frame[f"group_{idx}"] = group.astype(str).to_numpy()
    group_cols = [col for col in frame.columns if col.startswith("group_")]
    means = frame.groupby(group_cols)["value"].transform("mean")
    return frame["value"] - means


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    try:
        if method == "spearman":
            value = spearmanr(frame["x"], frame["y"]).statistic
        else:
            value = pearsonr(frame["x"], frame["y"]).statistic
    except Exception:
        return np.nan
    return float(value) if np.isfinite(value) else np.nan


def correlation_table(df: pd.DataFrame, feature_cols: list[str], level: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(df[TARGET], errors="coerce")
    y_ex = centered(y, [df["exercise"]])
    y_sub = centered(y, [df["folder"]])
    y_sub_ex = centered(y, [df["folder"], df["exercise"]])
    for feature in feature_cols:
        x = pd.to_numeric(df[feature], errors="coerce")
        if x.notna().sum() < 5 or x.nunique(dropna=True) < 2:
            continue
        row = {
            "level": level,
            "feature": feature,
            "group": feature_group(feature),
            "n": int(pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
            "raw_spearman": corr_pair(x, y, "spearman"),
            "raw_pearson": corr_pair(x, y, "pearson"),
            "exercise_centered_spearman": corr_pair(centered(x, [df["exercise"]]), y_ex, "spearman"),
            "subject_centered_spearman": corr_pair(centered(x, [df["folder"]]), y_sub, "spearman"),
            "subject_exercise_centered_spearman": corr_pair(centered(x, [df["folder"], df["exercise"]]), y_sub_ex, "spearman"),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_raw_spearman"] = out["raw_spearman"].abs()
    out["abs_subject_exercise_centered_spearman"] = out["subject_exercise_centered_spearman"].abs()
    return out.sort_values(["abs_raw_spearman", "abs_subject_exercise_centered_spearman"], ascending=False).reset_index(drop=True)


def plot_top_bar(table: pd.DataFrame, output_path: Path, metric: str, title: str, top_n: int = 20) -> None:
    subset = table.dropna(subset=[metric]).copy()
    subset = subset.reindex(subset[metric].abs().sort_values(ascending=False).index).head(top_n)
    subset = subset.iloc[::-1]
    colors = ["#3f7cac" if value >= 0 else "#c45b4f" for value in subset[metric]]
    fig, ax = plt.subplots(figsize=(10, max(5.5, len(subset) * 0.36)))
    ax.barh(subset["feature"], subset[metric], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_summary(table: pd.DataFrame, output_path: Path, metric: str, title: str) -> pd.DataFrame:
    summary = (
        table.dropna(subset=[metric])
        .assign(abs_metric=lambda frame: frame[metric].abs())
        .groupby("group")
        .agg(features=("feature", "count"), mean_abs_corr=("abs_metric", "mean"), max_abs_corr=("abs_metric", "max"))
        .sort_values("max_abs_corr", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(summary) * 0.38)))
    ax.barh(summary["group"].iloc[::-1], summary["max_abs_corr"].iloc[::-1], color="#5f9f74")
    ax.set_xlabel(f"max |{metric}|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def candidate_features(df: pd.DataFrame) -> list[str]:
    banned = {
        "borg",
        "folder",
        "file",
        "subject",
        "exercise",
        "set_id",
        "rep_id",
        "raw_value",
        "excluded_from_training",
        "start",
        "end",
    }
    cols = []
    for col in df.columns:
        if col in banned:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RPE/Borg feature correlations from GT-segmented IMU features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts_rep_classification/018_borg_gt_waveform_relation_exclude_sparse/018_gt_rep_waveform_borg_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/020_rpe_feature_correlation_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_dataset(args.input)
    rep = add_rep_level_features(base)
    set_level = build_set_level(rep)

    rep_features = candidate_features(rep)
    set_features = candidate_features(set_level)
    rep_corr = correlation_table(rep, rep_features, "rep")
    set_corr = correlation_table(set_level, set_features, "set")

    rep.to_csv(args.output_dir / "020_rpe_rep_level_feature_dataset.csv", index=False)
    set_level.to_csv(args.output_dir / "020_rpe_set_level_feature_dataset.csv", index=False)
    rep_corr.to_csv(args.output_dir / "020_rpe_rep_level_correlations.csv", index=False)
    set_corr.to_csv(args.output_dir / "020_rpe_set_level_correlations.csv", index=False)

    plot_top_bar(
        rep_corr,
        args.output_dir / "020_rpe_rep_top_raw_spearman.png",
        "raw_spearman",
        "Rep-level features vs Borg/RPE: top raw Spearman correlations",
    )
    plot_top_bar(
        rep_corr,
        args.output_dir / "020_rpe_rep_top_subject_exercise_centered_spearman.png",
        "subject_exercise_centered_spearman",
        "Rep-level within-subject/exercise correlations",
    )
    plot_top_bar(
        set_corr,
        args.output_dir / "020_rpe_set_top_raw_spearman.png",
        "raw_spearman",
        "Set-level features vs final Borg/RPE: top raw Spearman correlations",
    )
    plot_top_bar(
        set_corr,
        args.output_dir / "020_rpe_set_top_subject_exercise_centered_spearman.png",
        "subject_exercise_centered_spearman",
        "Set-level within-subject/exercise correlations",
    )
    rep_group = plot_group_summary(
        rep_corr,
        args.output_dir / "020_rpe_rep_feature_group_summary.png",
        "raw_spearman",
        "Rep-level feature groups: strongest raw association with RPE",
    )
    set_group = plot_group_summary(
        set_corr,
        args.output_dir / "020_rpe_set_feature_group_summary.png",
        "raw_spearman",
        "Set-level feature groups: strongest raw association with final RPE",
    )
    rep_group.to_csv(args.output_dir / "020_rpe_rep_feature_group_summary.csv", index=False)
    set_group.to_csv(args.output_dir / "020_rpe_set_feature_group_summary.csv", index=False)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "trainable_subjects": sorted(base["folder"].astype(str).unique().tolist()),
        "rep_rows": int(len(rep)),
        "set_rows": int(len(set_level)),
        "rpe_range": [float(base[TARGET].min()), float(base[TARGET].max())],
        "top_rep_raw_spearman": rep_corr.head(12).to_dict(orient="records"),
        "top_set_raw_spearman": set_corr.head(12).to_dict(orient="records"),
        "notes": {
            "target": "Rep-level correlations use each rep's Borg label; set-level correlations use the last Borg label in the set.",
            "centered_metrics": "Centered Spearman values subtract exercise, subject, or subject+exercise means before correlation, reducing baseline confounds.",
            "source": "Uses ground-truth rep segmentation from 018 so the result tests feature-RPE association before automatic segmentation noise.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Rep rows:", len(rep), "Set rows:", len(set_level))
    print("\nTop rep-level raw Spearman:")
    print(rep_corr[["feature", "group", "n", "raw_spearman", "subject_exercise_centered_spearman"]].head(15).to_string(index=False))
    print("\nTop set-level raw Spearman:")
    print(set_corr[["feature", "group", "n", "raw_spearman", "subject_exercise_centered_spearman"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
