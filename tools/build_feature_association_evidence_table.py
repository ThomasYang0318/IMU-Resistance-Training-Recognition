from __future__ import annotations

import argparse
import json
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
from scipy.stats import spearmanr


TARGET = "borg"
KEYS = ["folder", "exercise", "set_id"]
PRIMARY_VO2_LAG_SEC = 45.0

FEATURES: list[dict[str, object]] = [
    {
        "feature": "kg",
        "family": "workload_dose",
        "construct": "external load",
        "source": "rpe_only",
        "real_time_status": "known_before_set",
        "claim_role": "control covariate",
    },
    {
        "feature": "n_reps",
        "family": "workload_dose",
        "construct": "set volume",
        "source": "rpe_only",
        "real_time_status": "known_after_set_or_planned",
        "claim_role": "control covariate",
    },
    {
        "feature": "total_tut_sec",
        "family": "workload_dose",
        "construct": "time under tension dose",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "dose evidence",
    },
    {
        "feature": "cumulative_tut_exercise_sec",
        "family": "cumulative_dose",
        "construct": "cumulative TUT within the same subject and exercise up to the current set",
        "source": "rpe_only",
        "real_time_status": "known_after_current_set",
        "claim_role": "cumulative dose evidence",
    },
    {
        "feature": "set_index_numeric",
        "family": "set_order_diagnostic",
        "construct": "experimental order / accumulated exposure proxy",
        "source": "rpe_only",
        "real_time_status": "protocol_context",
        "claim_role": "diagnostic proxy only",
    },
    {
        "feature": "rep_duration_cv",
        "family": "lowdim_set_trend",
        "construct": "within-set repetition duration variability",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "IMU trend evidence",
    },
    {
        "feature": "movement_rate_cv",
        "family": "lowdim_set_trend",
        "construct": "within-set movement rate variability",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "IMU trend evidence",
    },
    {
        "feature": "gyro_diff_gain_last2_vs_first2",
        "family": "lowdim_set_trend",
        "construct": "late-vs-early gyroscope change",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "IMU trend evidence",
    },
    {
        "feature": "gyro_mag_diff_rms_slope",
        "family": "lowdim_set_trend",
        "construct": "gyroscope difference trend across reps",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "IMU trend evidence",
    },
    {
        "feature": "sim_to_first_slope",
        "family": "lowdim_set_trend",
        "construct": "trend in similarity to first repetition",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "IMU trend evidence",
    },
    {
        "feature": "pca_diff_rms_mean",
        "family": "lowdim_set_trend",
        "construct": "mean waveform difference magnitude",
        "source": "rpe_only",
        "real_time_status": "post_set_summary",
        "claim_role": "IMU trend evidence",
    },
    {
        "feature": "vo2_mean",
        "family": "delayed_vo2_45s",
        "construct": "mean VO2 in delayed post-set window",
        "source": "vo2_lag45",
        "real_time_status": "delayed_45s_physiology",
        "claim_role": "delayed physiological evidence",
    },
    {
        "feature": "vo2_peak",
        "family": "delayed_vo2_45s",
        "construct": "peak VO2 in delayed post-set window",
        "source": "vo2_lag45",
        "real_time_status": "delayed_45s_physiology",
        "claim_role": "delayed physiological evidence",
    },
    {
        "feature": "vo2_slope",
        "family": "delayed_vo2_45s",
        "construct": "VO2 slope in delayed post-set window",
        "source": "vo2_lag45",
        "real_time_status": "delayed_45s_physiology",
        "claim_role": "delayed physiological evidence",
    },
]


def normalize_set_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (list, dict, tuple, np.ndarray)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def bh_adjust(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return out.tolist()
    pv = values[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[mask] = tmp
    return out.tolist()


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for key in KEYS:
        if key in df.columns:
            df[key] = df[key].map(normalize_set_id) if key == "set_id" else df[key].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for col in df.columns:
        if col not in {"folder", "exercise", "set_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    if "total_tut_sec" not in df.columns:
        return df
    out = df.copy()
    sort_col = "set_index_numeric" if "set_index_numeric" in out.columns else "set_id"
    out["_original_order"] = np.arange(len(out))
    out = out.sort_values(["folder", "exercise", sort_col, "set_id"], kind="mergesort")
    out["cumulative_tut_exercise_sec"] = out.groupby(["folder", "exercise"], sort=False)["total_tut_sec"].cumsum()
    out = out.sort_values("_original_order").drop(columns="_original_order")
    return out.reset_index(drop=True)


def centered(frame: pd.DataFrame, column: str, groups: list[str]) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values - values.groupby([frame[group] for group in groups]).transform("mean")


def spearman_pair(x: pd.Series, y: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 4 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return {"n": int(len(data)), "rho": np.nan, "pvalue": np.nan}
    result = spearmanr(data["x"], data["y"])
    rho = float(result.statistic) if np.isfinite(result.statistic) else np.nan
    pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
    return {"n": int(len(data)), "rho": rho, "pvalue": pvalue}


def feature_correlation_rows(rpe: pd.DataFrame, vo2_lag45: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in FEATURES:
        feature = str(spec["feature"])
        source = str(spec["source"])
        frame = vo2_lag45 if source == "vo2_lag45" else rpe
        if feature not in frame.columns:
            continue
        y = pd.to_numeric(frame[TARGET], errors="coerce")
        x = pd.to_numeric(frame[feature], errors="coerce")
        scopes = [
            ("raw", x, y),
            ("subject_centered", centered(frame, feature, ["folder"]), centered(frame, TARGET, ["folder"])),
            ("exercise_centered", centered(frame, feature, ["exercise"]), centered(frame, TARGET, ["exercise"])),
            ("subject_exercise_centered", centered(frame, feature, ["folder", "exercise"]), centered(frame, TARGET, ["folder", "exercise"])),
        ]
        for scope, xs, ys in scopes:
            score = spearman_pair(xs, ys)
            rows.append(
                {
                    **spec,
                    "correlation_scope": "vo2_overlap_lag45_96_sets" if source == "vo2_lag45" else "rpe_only_143_sets",
                    "adjustment": scope,
                    "n": score["n"],
                    "subjects": int(frame["folder"].nunique()),
                    "spearman_rho": score["rho"],
                    "spearman_pvalue": score["pvalue"],
                }
            )
    out = pd.DataFrame(rows)
    out["spearman_qvalue_bh"] = np.nan
    for adjustment, idx in out.groupby("adjustment").groups.items():
        out.loc[idx, "spearman_qvalue_bh"] = bh_adjust(out.loc[idx, "spearman_pvalue"])
    return out


def primary_feature_evidence(correlations: pd.DataFrame, group_evidence: pd.DataFrame) -> pd.DataFrame:
    primary = correlations.pivot_table(
        index=["feature", "family", "construct", "source", "real_time_status", "claim_role", "correlation_scope", "subjects"],
        columns="adjustment",
        values=["n", "spearman_rho", "spearman_pvalue", "spearman_qvalue_bh"],
        aggfunc="first",
    )
    primary.columns = [f"{metric}_{scope}" for metric, scope in primary.columns]
    primary = primary.reset_index()

    group_best = group_evidence.sort_values(["family", "evidence_priority"]).drop_duplicates("family")
    primary = primary.merge(
        group_best[
            [
                "family",
                "model_evidence_dataset",
                "model_evidence_lag_sec",
                "model_evidence_comparison",
                "model_evidence_type",
                "model_mae_reduction",
                "model_spearman_gain",
                "model_pm1_acc_gain",
                "model_interpretation",
            ]
        ],
        on="family",
        how="left",
    )

    def claim_strength(row: pd.Series) -> str:
        family = row["family"]
        mae_gain = row.get("model_mae_reduction")
        q_raw = row.get("spearman_qvalue_bh_raw")
        rho_raw = row.get("spearman_rho_raw")
        if family == "set_order_diagnostic":
            return "strong diagnostic proxy, not a physiological or IMU claim"
        if family == "delayed_vo2_45s" and pd.notna(mae_gain) and mae_gain > 0.05:
            return "promising delayed auxiliary evidence"
        if family == "lowdim_set_trend" and pd.notna(mae_gain) and mae_gain > 0.05:
            return "promising group-level IMU evidence"
        if family == "workload_dose":
            return "control/dose evidence, not sufficient alone"
        if pd.notna(q_raw) and q_raw < 0.10 and pd.notna(rho_raw) and abs(rho_raw) >= 0.20:
            return "feature-level association evidence"
        return "weak or context-dependent evidence"

    primary["claim_strength"] = primary.apply(claim_strength, axis=1)
    return primary.sort_values(["family", "feature"]).reset_index(drop=True)


def build_group_evidence(delta_path: Path) -> pd.DataFrame:
    delta = pd.read_csv(delta_path)
    specs = [
        {
            "family": "set_order_diagnostic",
            "model_evidence_dataset": "rpe_lowdim_143_sets",
            "model_evidence_lag_sec": np.nan,
            "model_evidence_comparison": "order_diagnostic_gain",
            "model_evidence_type": "random_forest",
            "evidence_priority": 1,
            "model_interpretation": "Set order improves the workload model, but this is a protocol/progression proxy.",
        },
        {
            "family": "lowdim_set_trend",
            "model_evidence_dataset": "rpe_vo2_lowdim_96_sets",
            "model_evidence_lag_sec": PRIMARY_VO2_LAG_SEC,
            "model_evidence_comparison": "B_minus_A_lowdim_imu_gain",
            "model_evidence_type": "random_forest",
            "evidence_priority": 1,
            "model_interpretation": "Low-dimensional IMU trend improves the workload/dose model on the VO2-overlap subset.",
        },
        {
            "family": "lowdim_set_trend",
            "model_evidence_dataset": "rpe_lowdim_143_sets",
            "model_evidence_lag_sec": np.nan,
            "model_evidence_comparison": "B_minus_A_lowdim_imu_gain",
            "model_evidence_type": "random_forest",
            "evidence_priority": 2,
            "model_interpretation": "Full RPE-only evidence is weaker; the gain is relative to workload/dose, not to exercise mean.",
        },
        {
            "family": "delayed_vo2_45s",
            "model_evidence_dataset": "rpe_vo2_lowdim_96_sets",
            "model_evidence_lag_sec": PRIMARY_VO2_LAG_SEC,
            "model_evidence_comparison": "C_minus_B_vo2_gain",
            "model_evidence_type": "random_forest",
            "evidence_priority": 1,
            "model_interpretation": "45 s delayed VO2 gives the clearest additional gain after low-dimensional IMU trend.",
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in specs:
        mask = (
            delta["dataset"].eq(spec["model_evidence_dataset"])
            & delta["model_type"].eq(spec["model_evidence_type"])
            & delta["comparison"].eq(spec["model_evidence_comparison"])
        )
        lag = spec["model_evidence_lag_sec"]
        if pd.isna(lag):
            mask &= delta["lag_sec"].isna()
        else:
            mask &= np.isclose(pd.to_numeric(delta["lag_sec"], errors="coerce"), float(lag), equal_nan=False)
        match = delta[mask]
        values = match.iloc[0].to_dict() if not match.empty else {}
        rows.append(
            {
                **spec,
                "model_mae_reduction": values.get("mae_reduction", np.nan),
                "model_spearman_gain": values.get("spearman_gain", np.nan),
                "model_pm1_acc_gain": values.get("pm1_acc_gain", np.nan),
            }
        )
    rows.append(
        {
            "family": "workload_dose",
            "model_evidence_dataset": "rpe_lowdim_143_sets",
            "model_evidence_lag_sec": np.nan,
            "model_evidence_comparison": "included_as_Model_A_controls",
            "model_evidence_type": "ridge/random_forest",
            "evidence_priority": 1,
            "model_interpretation": "kg, n_reps, and total_tut_sec define the workload/dose control baseline.",
            "model_mae_reduction": np.nan,
            "model_spearman_gain": np.nan,
            "model_pm1_acc_gain": np.nan,
        }
    )
    return pd.DataFrame(rows)


def plot_feature_evidence(evidence: pd.DataFrame, output: Path) -> None:
    plot_cols = ["spearman_rho_raw", "spearman_rho_subject_centered", "spearman_rho_exercise_centered"]
    labels = ["raw", "subject-centered", "exercise-centered"]
    data = evidence.sort_values("spearman_rho_raw", key=lambda s: s.abs(), ascending=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, max(6, len(data) * 0.38)))
    offsets = [-0.23, 0.0, 0.23]
    colors = ["#356d9d", "#5b8e4d", "#a85f3d"]
    for col, label, offset, color in zip(plot_cols, labels, offsets, colors):
        ax.barh(y + offset, data[col], height=0.2, label=label, color=color, alpha=0.88)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Spearman rho with Borg/RPE")
    ax.set_title("Feature-Level Association Evidence")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_group_evidence(group_evidence: pd.DataFrame, output: Path) -> None:
    data = group_evidence[group_evidence["model_mae_reduction"].notna()].copy()
    data["label"] = data["family"] + "\n" + data["model_evidence_comparison"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#7a8f3a" if value >= 0 else "#9d4d4d" for value in data["model_mae_reduction"]]
    ax.bar(data["label"], data["model_mae_reduction"], color=colors, alpha=0.9)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("MAE reduction from ablation")
    ax.set_title("Controlled Model Evidence by Feature Group")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["tables", "metrics", "figures"]:
        (args.output_dir / subdir).mkdir(exist_ok=True)

    rpe = add_cumulative_features(load_table(args.rpe_set_features))
    vo2 = add_cumulative_features(load_table(args.vo2_merged_features))
    vo2_lag45 = vo2[np.isclose(vo2["lag_sec"], PRIMARY_VO2_LAG_SEC)].copy()

    correlations = feature_correlation_rows(rpe, vo2_lag45)
    group_evidence = build_group_evidence(args.model_delta_summary)
    feature_evidence = primary_feature_evidence(correlations, group_evidence)

    correlations.to_csv(args.output_dir / "metrics" / "feature_correlation_long.csv", index=False)
    group_evidence.to_csv(args.output_dir / "tables" / "group_model_evidence.csv", index=False)
    feature_evidence.to_csv(args.output_dir / "tables" / "feature_association_evidence.csv", index=False)

    plot_feature_evidence(feature_evidence, args.output_dir / "figures" / "feature_spearman_evidence.png")
    plot_group_evidence(group_evidence, args.output_dir / "figures" / "group_model_gain_evidence.png")

    run_config = "\n".join(
        [
            "experiment_id: '003'",
            "domain: fatigue_rpe_vo2",
            "name: feature_association_evidence_table",
            "created_at: '2026-05-17'",
            "primary_vo2_lag_sec: 45",
            "split_reference: leave-one-subject-out model deltas from experiment 002",
            "inputs:",
            f"  rpe_set_features: {args.rpe_set_features}",
            f"  vo2_merged_features: {args.vo2_merged_features}",
            f"  model_delta_summary: {args.model_delta_summary}",
            "outputs:",
            "  - tables/feature_association_evidence.csv",
            "  - tables/group_model_evidence.csv",
            "  - metrics/feature_correlation_long.csv",
            "  - figures/feature_spearman_evidence.png",
            "  - figures/group_model_gain_evidence.png",
            "",
        ]
    )
    (args.output_dir / "run_config.yaml").write_text(run_config, encoding="utf-8")

    top_abs = feature_evidence.assign(abs_raw=feature_evidence["spearman_rho_raw"].abs()).sort_values("abs_raw", ascending=False).head(8)
    summary = {
        "schema_version": "1.0",
        "experiment_id": "003",
        "domain": "fatigue_rpe_vo2",
        "name": "feature_association_evidence_table",
        "created_at": "2026-05-17",
        "status": "formal",
        "task": "feature-level and group-level evidence table for Borg/RPE associations",
        "question": "Which candidate workload, IMU trend, and delayed VO2 features can be cited as associated with Borg/RPE, and what are the limits of that claim?",
        "input_data": [str(args.rpe_set_features), str(args.vo2_merged_features), str(args.model_delta_summary)],
        "output_dir": str(args.output_dir),
        "command": f".venv311/bin/python tools/build_feature_association_evidence_table.py --output-dir {args.output_dir}",
        "primary_metrics": {
            "rpe_only_rows": int(len(rpe)),
            "rpe_only_subjects": int(rpe["folder"].nunique()),
            "vo2_lag45_rows": int(len(vo2_lag45)),
            "vo2_lag45_subjects": int(vo2_lag45["folder"].nunique()),
            "candidate_features": int(feature_evidence["feature"].nunique()),
        },
        "top_raw_associations_by_abs_spearman": top_abs[
            ["feature", "family", "spearman_rho_raw", "spearman_qvalue_bh_raw", "claim_strength"]
        ].to_dict(orient="records"),
        "key_group_model_evidence": group_evidence.to_dict(orient="records"),
        "key_files": {
            "feature_table": "tables/feature_association_evidence.csv",
            "group_model_table": "tables/group_model_evidence.csv",
            "long_correlation_table": "metrics/feature_correlation_long.csv",
            "feature_figure": "figures/feature_spearman_evidence.png",
            "group_gain_figure": "figures/group_model_gain_evidence.png",
        },
        "notes": "This artifact supports association claims only. set_index_numeric is a diagnostic proxy; delayed VO2 is not real-time; IMU trend features are post-set summaries.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_manifest(args.output_dir)

    print("Feature association evidence written to:", args.output_dir)
    print("\nTop raw associations:")
    print(top_abs[["feature", "family", "spearman_rho_raw", "spearman_qvalue_bh_raw", "claim_strength"]].round(4).to_string(index=False))
    print("\nGroup model evidence:")
    print(group_evidence[["family", "model_evidence_dataset", "model_evidence_lag_sec", "model_evidence_comparison", "model_mae_reduction", "model_spearman_gain", "model_pm1_acc_gain"]].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature association evidence table for Borg/RPE claims.")
    parser.add_argument(
        "--rpe-set-features",
        type=Path,
        default=Path("artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv"),
    )
    parser.add_argument(
        "--vo2-merged-features",
        type=Path,
        default=Path("artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv"),
    )
    parser.add_argument(
        "--model-delta-summary",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/metrics/model_delta_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
