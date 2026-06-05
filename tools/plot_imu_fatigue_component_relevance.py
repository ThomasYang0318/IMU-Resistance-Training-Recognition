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


COMPONENT_SPECS = [
    {
        "component": "Accumulated TUT",
        "display_feature": "cumulative active time",
        "source": "023_rep",
        "feature": "cumulative_rep_sec",
        "lag_sec": np.nan,
        "interpretation": "More time under tension within a set is associated with higher RPE.",
    },
    {
        "component": "CE phase range",
        "display_feature": "eccentric PCA range mean",
        "source": "023_set",
        "feature": "eccentric_pca_range_mean",
        "lag_sec": np.nan,
        "interpretation": "Larger eccentric phase waveform range is associated with higher final RPE.",
    },
    {
        "component": "CE phase similarity",
        "display_feature": "eccentric wave similarity drift",
        "source": "023_set",
        "feature": "eccentric_wave_sim_to_first2_last_minus_first",
        "lag_sec": np.nan,
        "interpretation": "Lower similarity to early reps is associated with higher final RPE.",
    },
    {
        "component": "Concentric gyro",
        "display_feature": "concentric gyro diff RMS last2",
        "source": "023_set",
        "feature": "concentric_gyro_diff_rms_last2",
        "lag_sec": np.nan,
        "interpretation": "Higher late-set concentric gyroscope variation is associated with higher final RPE.",
    },
    {
        "component": "Phase movement rate",
        "display_feature": "eccentric PCA movement rate mean",
        "source": "023_set",
        "feature": "eccentric_pca_movement_rate_mean",
        "lag_sec": np.nan,
        "interpretation": "Phase movement-rate statistics contain RPE-related signal.",
    },
    {
        "component": "Phase timing drift",
        "display_feature": "concentric duration last2/first2",
        "source": "023_set",
        "feature": "concentric_sec_last2_vs_first2",
        "lag_sec": np.nan,
        "interpretation": "Late-set concentric duration increase weakly supports the fatigue hypothesis.",
    },
    {
        "component": "CE ratio drift",
        "display_feature": "CE ratio slope",
        "source": "023_set",
        "feature": "ce_ratio_slope",
        "lag_sec": np.nan,
        "interpretation": "CE phase ratio drift is present but weaker than range, gyro, and similarity.",
    },
    {
        "component": "Delayed VO2",
        "display_feature": "VO2 slope at 45s",
        "source": "022_vo2",
        "feature": "vo2_slope",
        "lag_sec": 45.0,
        "interpretation": "VO2 slope after the set is a delayed physiological-load correlate.",
    },
    {
        "component": "VO2 baseline delta",
        "display_feature": "VO2 mean delta at 10s",
        "source": "022_vo2",
        "feature": "vo2_mean_delta_subject_min",
        "lag_sec": 10.0,
        "interpretation": "Subject-relative VO2 has signal, but raw direction is affected by rest and delay.",
    },
]

HEATMAP_FEATURES = [
    "concentric_gyro_diff_rms_slope",
    "concentric_gyro_diff_rms_last2_vs_first2",
    "concentric_sec_slope",
    "concentric_sec_last2_vs_first2",
    "concentric_pca_movement_rate_slope",
    "phase_vector_similarity_slope",
    "concentric_wave_sim_to_first2_slope",
    "eccentric_wave_sim_to_first2_slope",
    "ce_ratio_slope",
]

FEATURE_LABELS = {
    "concentric_gyro_diff_rms_slope": "Conc. gyro slope",
    "concentric_gyro_diff_rms_last2_vs_first2": "Conc. gyro last2/first2",
    "concentric_sec_slope": "Conc. time slope",
    "concentric_sec_last2_vs_first2": "Conc. time last2/first2",
    "concentric_pca_movement_rate_slope": "Conc. rate slope",
    "phase_vector_similarity_slope": "Phase similarity slope",
    "concentric_wave_sim_to_first2_slope": "Conc. wave sim slope",
    "eccentric_wave_sim_to_first2_slope": "Ecc. wave sim slope",
    "ce_ratio_slope": "CE ratio slope",
}


def load_row(table: pd.DataFrame, feature: str, lag_sec: float | None = None) -> pd.Series:
    sub = table[table["feature"].eq(feature)].copy()
    if lag_sec is not None and "lag_sec" in sub.columns and np.isfinite(lag_sec):
        sub = sub[sub["lag_sec"].eq(float(lag_sec))]
    if sub.empty:
        raise KeyError(f"Missing feature: {feature} lag={lag_sec}")
    return sub.iloc[0]


def build_component_table(rep_corr: pd.DataFrame, set_corr: pd.DataFrame, vo2_corr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in COMPONENT_SPECS:
        if spec["source"] == "023_rep":
            row = load_row(rep_corr, spec["feature"])
        elif spec["source"] == "023_set":
            row = load_row(set_corr, spec["feature"])
        elif spec["source"] == "022_vo2":
            row = load_row(vo2_corr, spec["feature"], float(spec["lag_sec"]))
        else:
            raise ValueError(spec["source"])
        rows.append(
            {
                **spec,
                "raw_spearman": float(row["raw_spearman"]),
                "subject_exercise_centered_spearman": float(row.get("subject_exercise_centered_spearman", np.nan)),
                "n": int(row["n"]),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_raw_spearman"] = out["raw_spearman"].abs()
    return out.sort_values("abs_raw_spearman", ascending=False).reset_index(drop=True)


def plot_component_summary(component_table: pd.DataFrame, exercise_heatmap: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.15], height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    sub = component_table.sort_values("raw_spearman").copy()
    y = np.arange(len(sub))
    colors = ["#3f7cac" if value >= 0 else "#c45b4f" for value in sub["raw_spearman"]]
    labels = [f"{row.component}\n{row.display_feature}" for row in sub.itertuples(index=False)]
    ax1.barh(y, sub["raw_spearman"], color=colors)
    ax1.axvline(0.0, color="#333333", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Spearman correlation with Borg/RPE")
    ax1.set_title("A. IMU/VO2 components associated with perceived fatigue")
    ax1.grid(axis="x", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    comp = component_table.sort_values("abs_raw_spearman", ascending=False).head(8).copy()
    x = np.arange(len(comp))
    ax2.bar(x - 0.18, comp["raw_spearman"], width=0.36, label="Raw", color="#5276a7")
    ax2.bar(
        x + 0.18,
        comp["subject_exercise_centered_spearman"],
        width=0.36,
        label="Within subject+exercise",
        color="#5f9f74",
    )
    ax2.axhline(0.0, color="#333333", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp["component"], rotation=28, ha="right")
    ax2.set_ylabel("Spearman correlation")
    ax2.set_title("B. Raw association vs. within-subject/exercise association")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :])
    matrix = exercise_heatmap.pivot(index="exercise", columns="feature_label", values="raw_spearman")
    feature_order = [FEATURE_LABELS[f] for f in HEATMAP_FEATURES if FEATURE_LABELS[f] in matrix.columns]
    matrix = matrix.reindex(columns=feature_order)
    im = ax3.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
    ax3.set_xticks(np.arange(matrix.shape[1]))
    ax3.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax3.set_yticks(np.arange(matrix.shape[0]))
    ax3.set_yticklabels(matrix.index)
    ax3.set_title("C. Exercise-specific CE phase feature correlations")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax3.text(j, i, f"{value:.2f}", ha="center", va="center", color="#111111", fontsize=8)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman correlation")

    fig.suptitle("Fatigue-Related Movement Components Measured by IMU and VO2", fontsize=15, y=0.99)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_component_group_summary(component_table: pd.DataFrame, output_path: Path) -> None:
    sub = component_table.sort_values("abs_raw_spearman", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#3f7cac" if value >= 0 else "#c45b4f" for value in sub["raw_spearman"]]
    ax.barh(sub["component"], sub["raw_spearman"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Spearman correlation with Borg/RPE")
    ax.set_title("IMU/VO2 Fatigue Component Relevance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paper-style figure summarizing IMU/VO2 components related to RPE.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/024_imu_fatigue_component_relevance_figure"))
    parser.add_argument("--rep-corr", type=Path, default=Path("artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_rep_correlations.csv"))
    parser.add_argument("--set-corr", type=Path, default=Path("artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_correlations.csv"))
    parser.add_argument("--exercise-corr", type=Path, default=Path("artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_correlations_by_exercise.csv"))
    parser.add_argument("--vo2-corr", type=Path, default=Path("artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_feature_correlations.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rep_corr = pd.read_csv(args.rep_corr)
    set_corr = pd.read_csv(args.set_corr)
    exercise_corr = pd.read_csv(args.exercise_corr)
    vo2_corr = pd.read_csv(args.vo2_corr)

    component_table = build_component_table(rep_corr, set_corr, vo2_corr)
    component_table.to_csv(args.output_dir / "024_imu_fatigue_component_relevance_table.csv", index=False)

    heatmap = exercise_corr[exercise_corr["feature"].isin(HEATMAP_FEATURES)].copy()
    heatmap["feature_label"] = heatmap["feature"].map(FEATURE_LABELS)
    heatmap.to_csv(args.output_dir / "024_exercise_phase_feature_heatmap_values.csv", index=False)

    plot_component_summary(component_table, heatmap, args.output_dir / "024_imu_fatigue_component_relevance_summary.png")
    plot_component_group_summary(component_table, args.output_dir / "024_imu_fatigue_component_bar.png")

    summary = {
        "output_dir": str(args.output_dir),
        "component_table": str(args.output_dir / "024_imu_fatigue_component_relevance_table.csv"),
        "main_figure": str(args.output_dir / "024_imu_fatigue_component_relevance_summary.png"),
        "component_bar": str(args.output_dir / "024_imu_fatigue_component_bar.png"),
        "top_components": component_table.head(8).to_dict(orient="records"),
        "notes": {
            "style": "Uses a correlation bar chart and exercise-feature heatmap, a common presentation for sensor feature relevance analyses.",
            "correlation": "Spearman correlation is used because Borg/RPE is ordinal.",
            "interpretation": "IMU components are fatigue-related movement correlates, not direct physiological fatigue measurements.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(component_table[["component", "display_feature", "n", "raw_spearman", "subject_exercise_centered_spearman"]].round(4).to_string(index=False))
    print("\nMain figure:", args.output_dir / "024_imu_fatigue_component_relevance_summary.png")


if __name__ == "__main__":
    main()
