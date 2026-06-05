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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "borg"
KEYS = ["folder", "exercise", "set_id"]
PRIMARY_VO2_LAG_SEC = 45.0

PROGRESSION_CONTROLS = ["set_index_numeric", "cumulative_tut_exercise_sec"]

RPE_CANDIDATES: list[dict[str, str]] = [
    {"feature": "kg", "family": "workload_dose", "claim_role": "load control"},
    {"feature": "n_reps", "family": "workload_dose", "claim_role": "volume control"},
    {"feature": "total_tut_sec", "family": "workload_dose", "claim_role": "single-set TUT"},
    {"feature": "rep_duration_cv", "family": "lowdim_set_trend", "claim_role": "IMU trend"},
    {"feature": "movement_rate_cv", "family": "lowdim_set_trend", "claim_role": "IMU trend"},
    {"feature": "gyro_diff_gain_last2_vs_first2", "family": "lowdim_set_trend", "claim_role": "IMU trend"},
    {"feature": "gyro_mag_diff_rms_slope", "family": "lowdim_set_trend", "claim_role": "IMU trend"},
    {"feature": "sim_to_first_slope", "family": "lowdim_set_trend", "claim_role": "IMU trend"},
    {"feature": "pca_diff_rms_mean", "family": "lowdim_set_trend", "claim_role": "IMU trend"},
]

VO2_CANDIDATES: list[dict[str, str]] = [
    *RPE_CANDIDATES,
    {"feature": "vo2_mean", "family": "delayed_vo2_45s", "claim_role": "delayed VO2"},
    {"feature": "vo2_peak", "family": "delayed_vo2_45s", "claim_role": "delayed VO2"},
    {"feature": "vo2_slope", "family": "delayed_vo2_45s", "claim_role": "delayed VO2"},
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


def safe_spearman(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    frame = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["y"].nunique() < 2 or frame["pred"].nunique() < 2:
        return np.nan
    value = spearmanr(frame["y"], frame["pred"]).statistic
    return float(value) if np.isfinite(value) else np.nan


def regression_scores(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
            "rounded_exact_acc": np.nan,
            "rounded_pm1_acc": np.nan,
        }
    rounded = np.rint(pred)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and len(np.unique(y)) > 1 else np.nan,
        "spearman": safe_spearman(y, pred),
        "rounded_exact_acc": float(np.mean(rounded == y)),
        "rounded_pm1_acc": float(np.mean(np.abs(rounded - y) <= 1.0)),
    }


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
    return df.reset_index(drop=True)


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


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    out: list[str] = []
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]) and df[feature].notna().sum() >= 8 and df[feature].nunique(dropna=True) > 1:
            out.append(feature)
    return out


def build_matrix(train: pd.DataFrame, test: pd.DataFrame, numeric_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    kept = available_features(train, numeric_features)
    x_train = train[kept].apply(pd.to_numeric, errors="coerce").copy()
    x_test = test[kept].apply(pd.to_numeric, errors="coerce").copy()
    train_ex = pd.get_dummies(train["exercise"].astype(str), prefix="exercise", dtype=float)
    test_ex = pd.get_dummies(test["exercise"].astype(str), prefix="exercise", dtype=float)
    train_ex, test_ex = train_ex.align(test_ex, join="outer", axis=1, fill_value=0.0)
    x_train = pd.concat([x_train.reset_index(drop=True), train_ex.reset_index(drop=True)], axis=1)
    x_test = pd.concat([x_test.reset_index(drop=True), test_ex.reset_index(drop=True)], axis=1)
    return x_train, x_test, list(x_train.columns)


def ridge_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def evaluate_exercise_mean(df: pd.DataFrame, dataset: str, stage: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    preds: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        global_mean = float(train[TARGET].mean())
        exercise_mean = train.groupby("exercise")[TARGET].mean().to_dict()
        pred = test["exercise"].map(exercise_mean).fillna(global_mean).astype(float).to_numpy()
        truth = test[TARGET].to_numpy(dtype=float)
        score = regression_scores(truth, pred)
        folds.append(
            {
                "dataset": dataset,
                "model_stage": stage,
                "fold": fold_idx,
                "val_subject": folder,
                "n_features": 0,
                "added_feature_coef": np.nan,
                **score,
            }
        )
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append(
                {
                    "dataset": dataset,
                    "model_stage": stage,
                    "fold": fold_idx,
                    "folder": row.folder,
                    "exercise": row.exercise,
                    "set_id": row.set_id,
                    "borg_true": float(y),
                    "borg_pred": float(p),
                    "added_feature": None,
                }
            )
    pred_df = pd.DataFrame(preds)
    metric = {
        "dataset": dataset,
        "model_stage": stage,
        "model_type": "exercise_mean",
        "subjects": int(df["folder"].nunique()),
        "n_features": 0,
        "added_feature": None,
        "added_feature_family": None,
        "added_feature_coef_mean": np.nan,
        "added_feature_coef_median": np.nan,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, pd.DataFrame(folds)


def evaluate_ridge(
    df: pd.DataFrame,
    dataset: str,
    stage: str,
    features: list[str],
    added_feature: str | None,
    added_feature_family: str | None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    preds: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    coef_values: list[float] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        x_train, x_test, columns = build_matrix(train, test, features)
        model = ridge_model()
        model.fit(x_train, train[TARGET].to_numpy(dtype=float))
        pred = model.predict(x_test)
        truth = test[TARGET].to_numpy(dtype=float)
        coef = np.nan
        if added_feature is not None and added_feature in columns:
            idx = columns.index(added_feature)
            coef = float(model.named_steps["ridge"].coef_[idx])
            coef_values.append(coef)
        score = regression_scores(truth, pred)
        folds.append(
            {
                "dataset": dataset,
                "model_stage": stage,
                "fold": fold_idx,
                "val_subject": folder,
                "n_features": len(columns),
                "added_feature_coef": coef,
                **score,
            }
        )
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append(
                {
                    "dataset": dataset,
                    "model_stage": stage,
                    "fold": fold_idx,
                    "folder": row.folder,
                    "exercise": row.exercise,
                    "set_id": row.set_id,
                    "borg_true": float(y),
                    "borg_pred": float(p),
                    "added_feature": added_feature,
                }
            )
    pred_df = pd.DataFrame(preds)
    fold_df = pd.DataFrame(folds)
    metric = {
        "dataset": dataset,
        "model_stage": stage,
        "model_type": "ridge",
        "subjects": int(df["folder"].nunique()),
        "n_features": int(fold_df["n_features"].median()) if not fold_df.empty else 0,
        "added_feature": added_feature,
        "added_feature_family": added_feature_family,
        "added_feature_coef_mean": float(np.nanmean(coef_values)) if coef_values else np.nan,
        "added_feature_coef_median": float(np.nanmedian(coef_values)) if coef_values else np.nan,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, fold_df


def run_dataset(
    df: pd.DataFrame,
    dataset: str,
    candidates: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []

    for metric, pred_df, fold_df in [
        evaluate_exercise_mean(df, dataset, "M0_exercise_mean"),
        evaluate_ridge(df, dataset, "M1_exercise_plus_set_index", ["set_index_numeric"], None, None),
        evaluate_ridge(df, dataset, "M2_exercise_plus_cumulative_tut", ["cumulative_tut_exercise_sec"], None, None),
        evaluate_ridge(df, dataset, "M3_progression_control", PROGRESSION_CONTROLS, None, None),
    ]:
        metrics.append(metric)
        preds.append(pred_df)
        folds.append(fold_df)

    base_features = list(PROGRESSION_CONTROLS)
    for spec in candidates:
        feature = spec["feature"]
        if feature in base_features or feature not in df.columns:
            continue
        stage = f"M4_add_one__{feature}"
        metric, pred_df, fold_df = evaluate_ridge(
            df,
            dataset,
            stage,
            base_features + [feature],
            feature,
            spec["family"],
        )
        metric["claim_role"] = spec["claim_role"]
        metrics.append(metric)
        preds.append(pred_df)
        folds.append(fold_df)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(preds, ignore_index=True)
    folds_df = pd.concat(folds, ignore_index=True)
    ablation_df = build_ablation_table(metrics_df, dataset)
    return metrics_df, ablation_df, predictions_df, folds_df


def build_ablation_table(metrics: pd.DataFrame, dataset: str) -> pd.DataFrame:
    base = metrics[(metrics["dataset"].eq(dataset)) & (metrics["model_stage"].eq("M3_progression_control"))]
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    progression_base = metrics[
        metrics["dataset"].eq(dataset)
        & metrics["model_stage"].isin(["M1_exercise_plus_set_index", "M2_exercise_plus_cumulative_tut", "M3_progression_control"])
    ].sort_values(["mae", "model_stage"])
    best_progression = progression_base.iloc[0]
    rows = []
    candidates = metrics[(metrics["dataset"].eq(dataset)) & metrics["model_stage"].str.startswith("M4_add_one__", na=False)]
    for _, row in candidates.iterrows():
        rows.append(
            {
                "dataset": dataset,
                "baseline_stage": "M3_progression_control",
                "candidate_stage": row["model_stage"],
                "added_feature": row["added_feature"],
                "family": row["added_feature_family"],
                "claim_role": row.get("claim_role"),
                "baseline_mae": base_row["mae"],
                "candidate_mae": row["mae"],
                "delta_mae_reduction": base_row["mae"] - row["mae"],
                "baseline_spearman": base_row["spearman"],
                "candidate_spearman": row["spearman"],
                "delta_spearman": row["spearman"] - base_row["spearman"],
                "baseline_pm1_acc": base_row["rounded_pm1_acc"],
                "candidate_pm1_acc": row["rounded_pm1_acc"],
                "delta_pm1_acc": row["rounded_pm1_acc"] - base_row["rounded_pm1_acc"],
                "best_progression_stage": best_progression["model_stage"],
                "best_progression_mae": best_progression["mae"],
                "delta_mae_vs_best_progression": best_progression["mae"] - row["mae"],
                "best_progression_spearman": best_progression["spearman"],
                "delta_spearman_vs_best_progression": row["spearman"] - best_progression["spearman"],
                "best_progression_pm1_acc": best_progression["rounded_pm1_acc"],
                "delta_pm1_acc_vs_best_progression": row["rounded_pm1_acc"] - best_progression["rounded_pm1_acc"],
                "added_feature_coef_mean": row["added_feature_coef_mean"],
                "added_feature_coef_median": row["added_feature_coef_median"],
                "n": row["n"],
                "subjects": row["subjects"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["evidence_label"] = np.select(
            [
                (out["delta_mae_reduction"] > 0.05) & (out["delta_spearman"] > 0.02),
                (out["delta_mae_reduction"] > 0.0) | (out["delta_spearman"] > 0.0),
            ],
            ["positive_after_progression_control", "mixed_or_small_after_progression_control"],
            default="no_incremental_gain_after_progression_control",
        )
        out["strict_evidence_label"] = np.select(
            [
                (out["delta_mae_vs_best_progression"] > 0.05) & (out["delta_spearman_vs_best_progression"] > 0.02),
                (out["delta_mae_vs_best_progression"] > 0.0) | (out["delta_spearman_vs_best_progression"] > 0.0),
            ],
            ["positive_vs_best_progression_baseline", "mixed_or_small_vs_best_progression_baseline"],
            default="does_not_beat_best_progression_baseline",
        )
        out = out.sort_values(["delta_mae_reduction", "delta_spearman"], ascending=False).reset_index(drop=True)
    return out


def plot_progression_baselines(metrics: pd.DataFrame, output: Path) -> None:
    base = metrics[metrics["model_stage"].isin(["M0_exercise_mean", "M1_exercise_plus_set_index", "M2_exercise_plus_cumulative_tut", "M3_progression_control"])].copy()
    base["label"] = base["dataset"] + "\n" + base["model_stage"].str.replace("M0_", "").str.replace("M1_", "").str.replace("M2_", "").str.replace("M3_", "").str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(base["label"], base["mae"], color="#557da3", alpha=0.9)
    ax.set_ylabel("LOSO MAE")
    ax.set_title("Progression Baselines")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_one_feature_ablation(ablation: pd.DataFrame, output: Path) -> None:
    if ablation.empty:
        return
    families = sorted(ablation["family"].dropna().unique())
    color_map = {
        "workload_dose": "#7b6d9f",
        "lowdim_set_trend": "#5f8f63",
        "delayed_vo2_45s": "#b36b48",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(ablation["added_feature"].unique()) * 0.32)), sharey=False)
    for ax, dataset in zip(axes, sorted(ablation["dataset"].unique())):
        sub = ablation[ablation["dataset"].eq(dataset)].sort_values("delta_mae_reduction", ascending=True)
        y = np.arange(len(sub))
        colors = [color_map.get(family, "#777777") for family in sub["family"]]
        ax.barh(y, sub["delta_mae_reduction"], color=colors, alpha=0.9)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["added_feature"])
        ax.set_xlabel("Delta MAE reduction vs progression control")
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(family, "#777777"), alpha=0.9) for family in families]
    fig.legend(handles, families, loc="lower center", ncols=min(len(families), 3))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_delta_vs_best_progression(ablation: pd.DataFrame, output: Path) -> None:
    if ablation.empty:
        return
    families = sorted(ablation["family"].dropna().unique())
    color_map = {
        "workload_dose": "#7b6d9f",
        "lowdim_set_trend": "#5f8f63",
        "delayed_vo2_45s": "#b36b48",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(ablation["added_feature"].unique()) * 0.32)), sharey=False)
    for ax, dataset in zip(axes, sorted(ablation["dataset"].unique())):
        sub = ablation[ablation["dataset"].eq(dataset)].sort_values("delta_mae_vs_best_progression", ascending=True)
        y = np.arange(len(sub))
        colors = [color_map.get(family, "#777777") for family in sub["family"]]
        ax.barh(y, sub["delta_mae_vs_best_progression"], color=colors, alpha=0.9)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["added_feature"])
        ax.set_xlabel("Delta MAE reduction vs best progression baseline")
        best_stage = sub["best_progression_stage"].iloc[0] if not sub.empty else ""
        ax.set_title(f"{dataset}\nbest baseline: {best_stage}")
        ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(family, "#777777"), alpha=0.9) for family in families]
    fig.legend(handles, families, loc="lower center", ncols=min(len(families), 3))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_coefficients(ablation: pd.DataFrame, output: Path) -> None:
    if ablation.empty:
        return
    sub = ablation.sort_values("added_feature_coef_median", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(6, len(sub) * 0.28)))
    colors = ["#5f8f63" if value >= 0 else "#b36b48" for value in sub["added_feature_coef_median"]]
    labels = sub["dataset"] + " / " + sub["added_feature"]
    y = np.arange(len(sub))
    ax.barh(y, sub["added_feature_coef_median"], color=colors, alpha=0.9)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Median standardized Ridge coefficient")
    ax.set_title("One-Feature Direction After Progression Control")
    ax.grid(axis="x", alpha=0.25)
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

    rpe_metrics, rpe_ablation, rpe_predictions, rpe_folds = run_dataset(rpe, "rpe_only_143_sets", RPE_CANDIDATES)
    vo2_metrics, vo2_ablation, vo2_predictions, vo2_folds = run_dataset(vo2_lag45, "vo2_lag45_96_sets", VO2_CANDIDATES)

    metrics = pd.concat([rpe_metrics, vo2_metrics], ignore_index=True)
    ablation = pd.concat([rpe_ablation, vo2_ablation], ignore_index=True)
    predictions = pd.concat([rpe_predictions, vo2_predictions], ignore_index=True)
    folds = pd.concat([rpe_folds, vo2_folds], ignore_index=True)

    metrics.to_csv(args.output_dir / "metrics" / "model_summary.csv", index=False)
    folds.to_csv(args.output_dir / "metrics" / "fold_metrics.csv", index=False)
    ablation.to_csv(args.output_dir / "tables" / "controlled_one_feature_ablation.csv", index=False)
    predictions.to_csv(args.output_dir / "tables" / "predictions.csv", index=False)

    plot_progression_baselines(metrics, args.output_dir / "figures" / "progression_baselines_mae.png")
    plot_one_feature_ablation(ablation, args.output_dir / "figures" / "one_feature_delta_mae.png")
    plot_delta_vs_best_progression(ablation, args.output_dir / "figures" / "one_feature_delta_vs_best_progression.png")
    plot_coefficients(ablation, args.output_dir / "figures" / "one_feature_coefficients.png")

    run_config = "\n".join(
        [
            "experiment_id: '004'",
            "domain: fatigue_rpe_vo2",
            "name: controlled_one_feature_ablation",
            "created_at: '2026-05-17'",
            "model: ridge_regression",
            "split: leave-one-subject-out",
            "categorical_controls:",
            "  - exercise",
            "progression_controls:",
            "  - set_index_numeric",
            "  - cumulative_tut_exercise_sec",
            "one_feature_rule: candidate models add exactly one numeric candidate to the progression control baseline",
            "inputs:",
            f"  rpe_set_features: {args.rpe_set_features}",
            f"  vo2_merged_features: {args.vo2_merged_features}",
            "outputs:",
            "  - tables/controlled_one_feature_ablation.csv",
            "  - tables/predictions.csv",
            "  - metrics/model_summary.csv",
            "  - metrics/fold_metrics.csv",
            "  - figures/progression_baselines_mae.png",
            "  - figures/one_feature_delta_mae.png",
            "  - figures/one_feature_delta_vs_best_progression.png",
            "  - figures/one_feature_coefficients.png",
            "",
        ]
    )
    (args.output_dir / "run_config.yaml").write_text(run_config, encoding="utf-8")

    positive = ablation[ablation["evidence_label"].eq("positive_after_progression_control")].copy()
    strict_positive = ablation[ablation["strict_evidence_label"].eq("positive_vs_best_progression_baseline")].copy()
    top = ablation.sort_values(["delta_mae_reduction", "delta_spearman"], ascending=False).head(10)
    strict_top = ablation.sort_values(["delta_mae_vs_best_progression", "delta_spearman_vs_best_progression"], ascending=False).head(10)
    summary = {
        "schema_version": "1.0",
        "experiment_id": "004",
        "domain": "fatigue_rpe_vo2",
        "name": "controlled_one_feature_ablation",
        "created_at": "2026-05-17",
        "status": "formal",
        "task": "one-feature-at-a-time controlled ablation for Borg/RPE association claims",
        "question": "After controlling exercise and within-exercise progression, does any single workload, IMU, or delayed VO2 feature add predictive value for Borg/RPE?",
        "input_data": [str(args.rpe_set_features), str(args.vo2_merged_features)],
        "output_dir": str(args.output_dir),
        "command": f".venv311/bin/python tools/evaluate_controlled_one_feature_ablation.py --output-dir {args.output_dir}",
        "primary_metrics": {
            "rpe_only_rows": int(len(rpe)),
            "rpe_only_subjects": int(rpe["folder"].nunique()),
            "vo2_lag45_rows": int(len(vo2_lag45)),
            "vo2_lag45_subjects": int(vo2_lag45["folder"].nunique()),
            "candidate_rows": int(len(ablation)),
            "positive_after_progression_control_rows": int(len(positive)),
            "positive_vs_best_progression_baseline_rows": int(len(strict_positive)),
        },
        "progression_baselines": metrics[metrics["model_stage"].isin(["M0_exercise_mean", "M1_exercise_plus_set_index", "M2_exercise_plus_cumulative_tut", "M3_progression_control"])].to_dict(orient="records"),
        "top_one_feature_gains": top.to_dict(orient="records"),
        "top_one_feature_gains_vs_best_progression": strict_top.to_dict(orient="records"),
        "positive_after_progression_control": positive.to_dict(orient="records"),
        "positive_vs_best_progression_baseline": strict_positive.to_dict(orient="records"),
        "key_files": {
            "ablation_table": "tables/controlled_one_feature_ablation.csv",
            "model_summary": "metrics/model_summary.csv",
            "fold_metrics": "metrics/fold_metrics.csv",
            "predictions": "tables/predictions.csv",
            "delta_mae_figure": "figures/one_feature_delta_mae.png",
            "delta_vs_best_progression_figure": "figures/one_feature_delta_vs_best_progression.png",
            "coefficient_figure": "figures/one_feature_coefficients.png",
            "progression_baseline_figure": "figures/progression_baselines_mae.png",
        },
        "notes": "All candidate rows add exactly one numeric feature to exercise + set_index_numeric + cumulative_tut_exercise_sec. This tests incremental association after controlling the within-exercise RPE progression pattern.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_manifest(args.output_dir)

    print("Controlled one-feature ablation written to:", args.output_dir)
    print("\nProgression baselines:")
    print(metrics[metrics["model_stage"].isin(["M0_exercise_mean", "M1_exercise_plus_set_index", "M2_exercise_plus_cumulative_tut", "M3_progression_control"])][["dataset", "model_stage", "n", "mae", "spearman", "rounded_pm1_acc"]].round(4).to_string(index=False))
    print("\nTop one-feature gains:")
    print(top[["dataset", "added_feature", "family", "delta_mae_reduction", "delta_spearman", "delta_pm1_acc", "added_feature_coef_median", "evidence_label"]].round(4).to_string(index=False))
    print("\nTop one-feature gains vs best progression baseline:")
    print(strict_top[["dataset", "added_feature", "family", "best_progression_stage", "delta_mae_vs_best_progression", "delta_spearman_vs_best_progression", "strict_evidence_label"]].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled one-feature-at-a-time ablation for Borg/RPE association evidence.")
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
