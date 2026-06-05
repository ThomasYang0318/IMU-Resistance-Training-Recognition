from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


TARGET = "borg"
KEYS = ["folder", "exercise", "set_id"]


FEATURE_GROUPS = {
    "set_index_numeric": "progress",
    "n_reps": "progress",
    "kg": "load",
    "kg_x_n_reps": "load",
    "kg_x_total_tut": "load",
    "total_tut_sec": "tut",
    "total_concentric_sec": "tut",
    "total_eccentric_sec": "tut",
    "mean_rep_duration_sec": "tut",
    "mean_concentric_sec": "tut",
    "mean_eccentric_sec": "tut",
    "mean_concentric_ratio": "phase",
    "mean_phase_balance_abs": "phase",
    "duration_gain_last2_vs_first2": "imu_fatigue_trend",
    "concentric_gain_last2_vs_first2": "imu_fatigue_trend",
    "eccentric_gain_last2_vs_first2": "imu_fatigue_trend",
    "velocity_loss_last2_vs_first2": "imu_fatigue_trend",
    "pca_range_loss_last2_vs_first2": "imu_fatigue_trend",
    "gyro_diff_gain_last2_vs_first2": "imu_fatigue_trend",
    "acc_diff_gain_last2_vs_first2": "imu_fatigue_trend",
    "sim_to_first_mean": "imu_similarity",
    "sim_to_first_min": "imu_similarity",
    "sim_to_first_last": "imu_similarity",
    "sim_to_first_last_minus_first": "imu_similarity",
    "rep_duration_cv": "imu_variability",
    "pca_range_cv": "imu_variability",
    "movement_rate_cv": "imu_variability",
    "sim_to_first_std": "imu_variability",
    "vo2_mean": "vo2",
    "vo2_peak": "vo2",
    "vo2_min": "vo2",
    "vo2_slope": "vo2",
    "vo2_range": "vo2",
    "vo2_peak_minus_mean": "vo2",
    "vo2_mean_delta_subject_min": "vo2_subject_relative",
    "vo2_peak_delta_subject_min": "vo2_subject_relative",
    "vo2_mean_z_subject": "vo2_subject_relative",
    "vo2_peak_z_subject": "vo2_subject_relative",
    "vo2_mean_x_total_tut": "vo2_load",
    "vo2_peak_x_total_tut": "vo2_load",
    "vo2_mean_x_n_reps": "vo2_load",
    "vo2_mean_per_rep": "vo2_load",
}


def normalize_set_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def centered(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce")})
    for idx, group in enumerate(groups):
        frame[f"group_{idx}"] = group.astype(str).to_numpy()
    group_cols = [col for col in frame.columns if col.startswith("group_")]
    return frame["value"] - frame.groupby(group_cols)["value"].transform("mean")


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 8 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    try:
        if method == "spearman":
            value = spearmanr(frame["x"], frame["y"]).statistic
        else:
            value = pearsonr(frame["x"], frame["y"]).statistic
    except Exception:
        return np.nan
    return float(value) if np.isfinite(value) else np.nan


def feature_group(feature: str) -> str:
    if feature in FEATURE_GROUPS:
        return FEATURE_GROUPS[feature]
    if feature.startswith(("vo2_",)):
        return "vo2"
    if feature.endswith("_slope") or "last2_vs_first2" in feature:
        return "imu_fatigue_trend"
    if "sim_to" in feature:
        return "imu_similarity"
    if "pca" in feature or "gyro" in feature or "acc" in feature or "range" in feature:
        return "imu_waveform"
    if "tut" in feature or feature.endswith("_sec"):
        return "tut"
    return "other"


def load_inputs(rpe_path: Path, vo2_path: Path) -> pd.DataFrame:
    rpe = pd.read_csv(rpe_path)
    vo2 = pd.read_csv(vo2_path)
    for df in [rpe, vo2]:
        for key in KEYS:
            df[key] = df[key].map(normalize_set_id) if key == "set_id" else df[key].astype(str)
    rpe = rpe.drop_duplicates(KEYS, keep="last").copy()
    vo2_cols = [
        "lag_sec",
        "folder",
        "exercise",
        "set_id",
        "vo2_points",
        "vo2_mean",
        "vo2_peak",
        "vo2_min",
        "vo2_slope",
    ]
    merged = vo2[vo2_cols].merge(rpe, on=KEYS, how="inner", suffixes=("", "_rpe"))
    merged[TARGET] = pd.to_numeric(merged[TARGET], errors="coerce")
    merged = merged[merged[TARGET].notna()].copy()

    for col in merged.columns:
        if col not in {"folder", "exercise", "set_id"}:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["vo2_range"] = merged["vo2_peak"] - merged["vo2_min"]
    merged["vo2_peak_minus_mean"] = merged["vo2_peak"] - merged["vo2_mean"]
    merged["vo2_mean_x_total_tut"] = merged["vo2_mean"] * merged["total_tut_sec"]
    merged["vo2_peak_x_total_tut"] = merged["vo2_peak"] * merged["total_tut_sec"]
    merged["vo2_mean_x_n_reps"] = merged["vo2_mean"] * merged["n_reps"]
    merged["vo2_mean_per_rep"] = merged["vo2_mean"] / merged["n_reps"].replace(0, np.nan)

    for col in ["vo2_mean", "vo2_peak"]:
        subject_min = merged.groupby("folder")[col].transform("min")
        subject_mean = merged.groupby("folder")[col].transform("mean")
        subject_std = merged.groupby("folder")[col].transform("std").replace(0, np.nan)
        merged[f"{col}_delta_subject_min"] = merged[col] - subject_min
        merged[f"{col}_z_subject"] = (merged[col] - subject_mean) / subject_std
    return merged


def candidate_features(df: pd.DataFrame) -> list[str]:
    banned = {
        TARGET,
        "folder",
        "exercise",
        "set_id",
        "lag_sec",
        "vo2_points",
        "file",
        "subject",
        "start_utc",
        "end_utc",
    }
    features: list[str] = []
    for col in df.columns:
        if col in banned:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 8 and df[col].nunique(dropna=True) > 1:
            features.append(col)
    return features


def correlation_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lag, sub in df.groupby("lag_sec"):
        y = pd.to_numeric(sub[TARGET], errors="coerce")
        y_ex = centered(y, [sub["exercise"]])
        y_sub = centered(y, [sub["folder"]])
        y_sub_ex = centered(y, [sub["folder"], sub["exercise"]])
        for feature in features:
            x = pd.to_numeric(sub[feature], errors="coerce")
            rows.append(
                {
                    "lag_sec": float(lag),
                    "feature": feature,
                    "group": feature_group(feature),
                    "n": int(pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
                    "raw_spearman": corr_pair(x, y, "spearman"),
                    "raw_pearson": corr_pair(x, y, "pearson"),
                    "exercise_centered_spearman": corr_pair(centered(x, [sub["exercise"]]), y_ex, "spearman"),
                    "subject_centered_spearman": corr_pair(centered(x, [sub["folder"]]), y_sub, "spearman"),
                    "subject_exercise_centered_spearman": corr_pair(centered(x, [sub["folder"], sub["exercise"]]), y_sub_ex, "spearman"),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_raw_spearman"] = out["raw_spearman"].abs()
    out["abs_subject_exercise_centered_spearman"] = out["subject_exercise_centered_spearman"].abs()
    return out.sort_values(["lag_sec", "abs_raw_spearman"], ascending=[True, False]).reset_index(drop=True)


def plot_top_by_lag(table: pd.DataFrame, output_path: Path, metric: str, title: str, include_groups: set[str] | None = None) -> None:
    lags = sorted(table["lag_sec"].dropna().unique())
    fig, axes = plt.subplots(len(lags), 1, figsize=(11, max(4, len(lags) * 3.2)), squeeze=False)
    for ax, lag in zip(axes[:, 0], lags):
        sub = table[table["lag_sec"].eq(lag)].dropna(subset=[metric]).copy()
        if include_groups is not None:
            sub = sub[sub["group"].isin(include_groups)].copy()
        sub = sub.reindex(sub[metric].abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        colors = ["#3f7cac" if value >= 0 else "#c45b4f" for value in sub[metric]]
        labels = [f"{row.feature} ({row.group})" for row in sub.itertuples(index=False)]
        ax.barh(labels, sub[metric], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"lag {lag:g}s")
        ax.grid(axis="x", alpha=0.25)
    axes[-1, 0].set_xlabel(metric)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_vo2_lag_lines(table: pd.DataFrame, output_path: Path) -> None:
    vo2 = table[table["group"].str.startswith("vo2", na=False)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for feature in ["vo2_mean", "vo2_peak", "vo2_mean_delta_subject_min", "vo2_peak_delta_subject_min", "vo2_slope"]:
        sub = vo2[vo2["feature"].eq(feature)].sort_values("lag_sec")
        if sub.empty:
            continue
        ax.plot(sub["lag_sec"], sub["raw_spearman"], marker="o", label=feature)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("VO2 lag after set (sec)")
    ax.set_ylabel("raw Spearman vs Borg/RPE")
    ax.set_title("VO2 feature correlation with set-level Borg/RPE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge set-level RPE features with lagged VO2 and rank real-time RPE predictors.")
    parser.add_argument(
        "--rpe-set-features",
        type=Path,
        default=Path("artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv"),
    )
    parser.add_argument(
        "--vo2-set-features",
        type=Path,
        default=Path("artifacts_rep_classification/019_vo2_gt_waveform_relation/019_vo2_set_waveform_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = load_inputs(args.rpe_set_features, args.vo2_set_features)
    features = candidate_features(merged)
    corr = correlation_table(merged, features)
    merged.to_csv(args.output_dir / "022_realtime_rpe_vo2_merged_set_dataset.csv", index=False)
    corr.to_csv(args.output_dir / "022_realtime_rpe_vo2_feature_correlations.csv", index=False)

    plot_top_by_lag(corr, args.output_dir / "022_top_realtime_features_by_lag_raw_spearman.png", "raw_spearman", "Top real-time IMU + VO2 features for Borg/RPE")
    plot_top_by_lag(
        corr,
        args.output_dir / "022_top_vo2_features_by_lag_raw_spearman.png",
        "raw_spearman",
        "Top VO2 features for Borg/RPE",
        include_groups={"vo2", "vo2_subject_relative", "vo2_load"},
    )
    plot_vo2_lag_lines(corr, args.output_dir / "022_vo2_feature_correlation_by_lag.png")

    group_summary = (
        corr.dropna(subset=["raw_spearman"])
        .assign(abs_raw=lambda frame: frame["raw_spearman"].abs())
        .groupby(["lag_sec", "group"])
        .agg(features=("feature", "count"), mean_abs_corr=("abs_raw", "mean"), max_abs_corr=("abs_raw", "max"))
        .sort_values(["lag_sec", "max_abs_corr"], ascending=[True, False])
        .reset_index()
    )
    group_summary.to_csv(args.output_dir / "022_realtime_rpe_vo2_feature_group_summary.csv", index=False)

    summary = {
        "output_dir": str(args.output_dir),
        "rpe_set_features": str(args.rpe_set_features),
        "vo2_set_features": str(args.vo2_set_features),
        "rows": int(len(merged)),
        "sets": int(merged[KEYS].drop_duplicates().shape[0]),
        "subjects": sorted(merged["folder"].astype(str).unique().tolist()),
        "lags_sec": sorted(float(x) for x in merged["lag_sec"].dropna().unique()),
        "top_raw_spearman_by_lag": corr.groupby("lag_sec").head(10).to_dict(orient="records"),
        "notes": {
            "target": "Set-level Borg/RPE from the updated 021 RPE set feature dataset.",
            "vo2_lag": "VO2 is evaluated at 0, 10, 20, 30, 45, and 60 seconds after each set because VO2 is delayed relative to movement.",
            "realtime_interpretation": "IMU fatigue state can update per rep; VO2 features should be treated as delayed physiological-load features.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Merged rows:", len(merged), "sets:", summary["sets"], "subjects:", ", ".join(summary["subjects"]))
    print("\nTop raw Spearman by lag:")
    print(corr[["lag_sec", "feature", "group", "n", "raw_spearman", "subject_exercise_centered_spearman"]].groupby("lag_sec").head(12).round(4).to_string(index=False))
    print("\nVO2-only features:")
    vo2 = corr[corr["group"].str.startswith("vo2", na=False)]
    print(vo2[["lag_sec", "feature", "group", "n", "raw_spearman", "subject_exercise_centered_spearman"]].groupby("lag_sec").head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
