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

LOWDIM_IMU_FEATURES = [
    "rep_duration_cv",
    "movement_rate_cv",
    "gyro_diff_gain_last2_vs_first2",
    "gyro_mag_diff_rms_slope",
    "sim_to_first_slope",
    "pca_diff_rms_mean",
]

VO2_FEATURES = [
    "vo2_mean",
    "vo2_peak",
    "vo2_slope",
]

ID_OR_TARGET_COLUMNS = {
    "lag_sec",
    "folder",
    "exercise",
    "set_id",
    "borg",
    "vo2_points",
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
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan, "rounded_pm1_acc": np.nan}
    rounded = np.rint(pred)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and len(np.unique(y)) > 1 else np.nan,
        "spearman": safe_spearman(y, pred),
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


def build_matrix(train: pd.DataFrame, test: pd.DataFrame, numeric_features: list[str], use_exercise: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    kept = available_features(train, numeric_features)
    x_train = train[kept].apply(pd.to_numeric, errors="coerce").copy()
    x_test = test[kept].apply(pd.to_numeric, errors="coerce").copy()
    if use_exercise:
        train_ex = pd.get_dummies(train["exercise"].astype(str), prefix="exercise", dtype=float)
        test_ex = pd.get_dummies(test["exercise"].astype(str), prefix="exercise", dtype=float)
        train_ex, test_ex = train_ex.align(test_ex, join="outer", axis=1, fill_value=0.0)
        x_train = pd.concat([x_train.reset_index(drop=True), train_ex.reset_index(drop=True)], axis=1)
        x_test = pd.concat([x_test.reset_index(drop=True), test_ex.reset_index(drop=True)], axis=1)
    return x_train, x_test, list(x_train.columns)


def ridge_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def eval_global_mean(df: pd.DataFrame, dataset: str, model_stage: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    preds: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        pred = np.full(len(test), float(train[TARGET].mean()))
        truth = test[TARGET].to_numpy(dtype=float)
        score = regression_scores(truth, pred)
        folds.append({"dataset": dataset, "model_stage": model_stage, "fold": fold_idx, "val_subject": folder, "n_features": 0, **score})
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append({"dataset": dataset, "model_stage": model_stage, "fold": fold_idx, "folder": row.folder, "exercise": row.exercise, "set_id": row.set_id, "borg_true": float(y), "borg_pred": float(p)})
    pred_df = pd.DataFrame(preds)
    metric = {"dataset": dataset, "model_stage": model_stage, "model_type": "global_mean", "module_claim": "no segmentation / no exercise context", "subjects": int(df["folder"].nunique()), "n_features": 0, **regression_scores(pred_df["borg_true"], pred_df["borg_pred"])}
    return metric, pred_df, pd.DataFrame(folds)


def eval_exercise_mean(df: pd.DataFrame, dataset: str, model_stage: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
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
        folds.append({"dataset": dataset, "model_stage": model_stage, "fold": fold_idx, "val_subject": folder, "n_features": 0, **score})
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append({"dataset": dataset, "model_stage": model_stage, "fold": fold_idx, "folder": row.folder, "exercise": row.exercise, "set_id": row.set_id, "borg_true": float(y), "borg_pred": float(p)})
    pred_df = pd.DataFrame(preds)
    metric = {"dataset": dataset, "model_stage": model_stage, "model_type": "exercise_mean", "module_claim": "exercise segmentation / recognition context", "subjects": int(df["folder"].nunique()), "n_features": 0, **regression_scores(pred_df["borg_true"], pred_df["borg_pred"])}
    return metric, pred_df, pd.DataFrame(folds)


def eval_ridge(
    df: pd.DataFrame,
    dataset: str,
    model_stage: str,
    features: list[str],
    use_exercise: bool,
    module_claim: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    preds: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        x_train, x_test, columns = build_matrix(train, test, features, use_exercise=use_exercise)
        model = ridge_model()
        model.fit(x_train, train[TARGET].to_numpy(dtype=float))
        pred = model.predict(x_test)
        truth = test[TARGET].to_numpy(dtype=float)
        score = regression_scores(truth, pred)
        folds.append({"dataset": dataset, "model_stage": model_stage, "fold": fold_idx, "val_subject": folder, "n_features": len(columns), **score})
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append({"dataset": dataset, "model_stage": model_stage, "fold": fold_idx, "folder": row.folder, "exercise": row.exercise, "set_id": row.set_id, "borg_true": float(y), "borg_pred": float(p)})
    pred_df = pd.DataFrame(preds)
    fold_df = pd.DataFrame(folds)
    metric = {
        "dataset": dataset,
        "model_stage": model_stage,
        "model_type": "ridge",
        "module_claim": module_claim,
        "subjects": int(df["folder"].nunique()),
        "n_features": int(fold_df["n_features"].median()) if not fold_df.empty else 0,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, fold_df


def run_rpe_ladder(df: pd.DataFrame, dataset: str, include_vo2: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stages: list[tuple] = [
        eval_global_mean(df, dataset, "M0_global_mean"),
        eval_exercise_mean(df, dataset, "M1_exercise_mean"),
        eval_ridge(df, dataset, "M2_set_index_no_exercise", ["set_index_numeric"], False, "set order without exercise context"),
        eval_ridge(df, dataset, "M3_exercise_plus_set_index", ["set_index_numeric"], True, "exercise segmentation + ordinal progression"),
        eval_ridge(df, dataset, "M4_exercise_plus_cumulative_tut", ["cumulative_tut_exercise_sec"], True, "exercise segmentation + set/rep/TUT segmentation"),
        eval_ridge(df, dataset, "M5_plus_lowdim_imu_group", ["cumulative_tut_exercise_sec"] + LOWDIM_IMU_FEATURES, True, "add IMU set-trend features"),
    ]
    if include_vo2:
        stages.append(
            eval_ridge(
                df,
                dataset,
                "M6_plus_delayed_vo2",
                ["cumulative_tut_exercise_sec"] + LOWDIM_IMU_FEATURES + VO2_FEATURES,
                True,
                "add delayed VO2 physiology",
            )
        )
    metrics = pd.DataFrame([stage[0] for stage in stages])
    predictions = pd.concat([stage[1] for stage in stages], ignore_index=True)
    folds = pd.concat([stage[2] for stage in stages], ignore_index=True)
    return metrics, predictions, folds


def comparison_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("exercise_segmentation_gain", "M0_global_mean", "M1_exercise_mean", "Exercise context improves over global mean."),
        ("exercise_context_for_progression_gain", "M2_set_index_no_exercise", "M3_exercise_plus_set_index", "Set index is more interpretable when tied to exercise."),
        ("tut_segmentation_gain_vs_set_index", "M3_exercise_plus_set_index", "M4_exercise_plus_cumulative_tut", "Cumulative TUT tests whether segmentation-derived dose improves over ordinal set count."),
        ("lowdim_imu_group_gain_after_tut", "M4_exercise_plus_cumulative_tut", "M5_plus_lowdim_imu_group", "IMU trend tests whether waveform features add beyond cumulative TUT."),
        ("delayed_vo2_gain_after_imu", "M5_plus_lowdim_imu_group", "M6_plus_delayed_vo2", "VO2 tests whether delayed physiology adds beyond exercise, TUT, and IMU trend."),
    ]
    rows = []
    for dataset, group in metrics.groupby("dataset", sort=True):
        by_stage = group.set_index("model_stage")
        for comparison, base, augmented, interpretation in specs:
            if base not in by_stage.index or augmented not in by_stage.index:
                continue
            b = by_stage.loc[base]
            a = by_stage.loc[augmented]
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "baseline_stage": base,
                    "augmented_stage": augmented,
                    "baseline_mae": b["mae"],
                    "augmented_mae": a["mae"],
                    "delta_mae_reduction": b["mae"] - a["mae"],
                    "baseline_spearman": b["spearman"],
                    "augmented_spearman": a["spearman"],
                    "delta_spearman": a["spearman"] - b["spearman"],
                    "baseline_pm1_acc": b["rounded_pm1_acc"],
                    "augmented_pm1_acc": a["rounded_pm1_acc"],
                    "delta_pm1_acc": a["rounded_pm1_acc"] - b["rounded_pm1_acc"],
                    "interpretation": interpretation,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["evidence_strength"] = np.select(
            [
                (out["delta_mae_reduction"] > 0.10) & (out["delta_spearman"] > 0.05),
                (out["delta_mae_reduction"] > 0.03) | (out["delta_spearman"] > 0.02),
            ],
            ["strong_for_module", "mixed_or_small_for_module"],
            default="not_supported_in_this_setting",
        )
    return out


def centered(frame: pd.DataFrame, column: str, groups: list[str]) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values - values.groupby([frame[group] for group in groups]).transform("mean")


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 4 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return int(len(data)), np.nan
    value = spearmanr(data["x"], data["y"]).statistic
    return int(len(data)), float(value) if np.isfinite(value) else np.nan


def feature_family(feature: str) -> str:
    if feature == "exercise_context":
        return "exercise_segmentation_context"
    if feature in {"set_index_numeric", "cumulative_tut_exercise_sec"}:
        return "progression_or_cumulative_exposure"
    if feature in {"kg", "n_reps", "total_tut_sec", "kg_x_total_tut", "kg_x_n_reps"}:
        return "workload_or_tut"
    if feature.startswith("vo2_"):
        return "delayed_vo2"
    if "concentric" in feature or "eccentric" in feature or "phase" in feature:
        return "phase_timing"
    if "sim_to" in feature:
        return "rep_similarity"
    if "gyro" in feature:
        return "gyro_waveform"
    if "acc" in feature:
        return "acc_waveform"
    if "pca" in feature:
        return "pca_waveform"
    if feature.endswith("_cv") or feature.endswith("_std"):
        return "variability"
    if feature.endswith("_slope") or "last2_vs_first2" in feature or "last_minus_first" in feature:
        return "trend"
    return "other_numeric"


def numeric_feature_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if col in ID_OR_TARGET_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 8 and df[col].nunique(dropna=True) > 1:
            candidates.append(col)
    return sorted(candidates)


def single_numeric_feature_metric(df: pd.DataFrame, dataset: str, feature: str) -> dict[str, object]:
    metric, _, _ = eval_ridge(
        df,
        dataset,
        f"S1_numeric__{feature}",
        [feature],
        use_exercise=False,
        module_claim="single numeric feature only",
    )
    return metric


def single_feature_ranking(df: pd.DataFrame, dataset: str, module_metrics: pd.DataFrame) -> pd.DataFrame:
    base = module_metrics[module_metrics["dataset"].eq(dataset)].set_index("model_stage")
    global_mae = float(base.loc["M0_global_mean", "mae"]) if "M0_global_mean" in base.index else np.nan
    exercise_mae = float(base.loc["M1_exercise_mean", "mae"]) if "M1_exercise_mean" in base.index else np.nan
    global_spearman = float(base.loc["M0_global_mean", "spearman"]) if "M0_global_mean" in base.index else np.nan
    exercise_spearman = float(base.loc["M1_exercise_mean", "spearman"]) if "M1_exercise_mean" in base.index else np.nan
    rows: list[dict[str, object]] = []

    if "M1_exercise_mean" in base.index:
        ex = base.loc["M1_exercise_mean"]
        rows.append(
            {
                "dataset": dataset,
                "feature": "exercise_context",
                "feature_type": "categorical_context",
                "family": feature_family("exercise_context"),
                "n": int(ex["n"]),
                "subjects": int(ex["subjects"]),
                "raw_spearman": np.nan,
                "abs_raw_spearman": np.nan,
                "subject_centered_spearman": np.nan,
                "exercise_centered_spearman": np.nan,
                "subject_exercise_centered_spearman": np.nan,
                "single_feature_mae": float(ex["mae"]),
                "single_feature_spearman": float(ex["spearman"]),
                "delta_mae_vs_global_mean": global_mae - float(ex["mae"]),
                "delta_spearman_vs_global_mean": float(ex["spearman"]) - global_spearman,
                "delta_mae_vs_exercise_context": 0.0,
                "delta_spearman_vs_exercise_context": 0.0,
            }
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    for feature in numeric_feature_candidates(df):
        x = pd.to_numeric(df[feature], errors="coerce")
        n_raw, raw = spearman_pair(x, y)
        _, subject_centered = spearman_pair(centered(df, feature, ["folder"]), centered(df, TARGET, ["folder"]))
        _, exercise_centered = spearman_pair(centered(df, feature, ["exercise"]), centered(df, TARGET, ["exercise"]))
        _, subject_exercise_centered = spearman_pair(centered(df, feature, ["folder", "exercise"]), centered(df, TARGET, ["folder", "exercise"]))
        metric = single_numeric_feature_metric(df, dataset, feature)
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "feature_type": "numeric",
                "family": feature_family(feature),
                "n": n_raw,
                "subjects": int(df["folder"].nunique()),
                "raw_spearman": raw,
                "abs_raw_spearman": abs(raw) if pd.notna(raw) else np.nan,
                "subject_centered_spearman": subject_centered,
                "exercise_centered_spearman": exercise_centered,
                "subject_exercise_centered_spearman": subject_exercise_centered,
                "single_feature_mae": metric["mae"],
                "single_feature_spearman": metric["spearman"],
                "delta_mae_vs_global_mean": global_mae - metric["mae"],
                "delta_spearman_vs_global_mean": metric["spearman"] - global_spearman,
                "delta_mae_vs_exercise_context": exercise_mae - metric["mae"],
                "delta_spearman_vs_exercise_context": metric["spearman"] - exercise_spearman,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank_by_abs_raw_spearman"] = out["abs_raw_spearman"].rank(ascending=False, method="min")
        out["rank_by_univariate_mae"] = out["single_feature_mae"].rank(ascending=True, method="min")
        out["rank_by_mae_gain_vs_global"] = out["delta_mae_vs_global_mean"].rank(ascending=False, method="min")
        out["evidence_note"] = np.select(
            [
                out["feature"].eq("exercise_context"),
                (out["delta_mae_vs_exercise_context"] > 0.05) & (out["delta_spearman_vs_exercise_context"] > 0.02),
                (out["abs_raw_spearman"] >= 0.30) & (out["delta_mae_vs_global_mean"] > 0.05),
            ],
            [
                "categorical exercise context; evidence comes from MAE gain over global mean",
                "single numeric feature beats exercise context",
                "strong raw association and useful univariate prediction",
            ],
            default="association exists but not stronger than exercise/progression context",
        )
        out = out.sort_values(["dataset", "rank_by_abs_raw_spearman", "rank_by_univariate_mae"], na_position="last").reset_index(drop=True)
    return out


def plot_single_feature_ranking(ranking: pd.DataFrame, output: Path) -> None:
    numeric = ranking[ranking["feature_type"].eq("numeric")].copy()
    if numeric.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=False)
    for ax, dataset in zip(axes, sorted(numeric["dataset"].unique())):
        sub = numeric[numeric["dataset"].eq(dataset)].sort_values("abs_raw_spearman", ascending=False).head(12)
        sub = sub.sort_values("raw_spearman", ascending=True)
        y = np.arange(len(sub))
        colors = ["#5f8f63" if value >= 0 else "#b36b48" for value in sub["raw_spearman"]]
        ax.barh(y, sub["raw_spearman"], color=colors, alpha=0.9)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["feature"])
        ax.set_xlabel("Raw Spearman rho with Borg/RPE")
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Top Single Numeric Features by Raw Association")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def vo2_estimability_table(vo2_summary_path: Path) -> pd.DataFrame:
    if not vo2_summary_path.exists():
        return pd.DataFrame()
    data = json.loads(vo2_summary_path.read_text(encoding="utf-8"))
    rows = data.get("best_models", [])
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["module_claim"] = "IMU set-level waveform features can estimate VO2 as an independent physiological load target."
    out["source_artifact"] = str(vo2_summary_path)
    return out.sort_values(["target", "mae", "spearman"], ascending=[True, True, False]).reset_index(drop=True)


def plot_ladder(metrics: pd.DataFrame, output: Path) -> None:
    order = ["M0_global_mean", "M1_exercise_mean", "M2_set_index_no_exercise", "M3_exercise_plus_set_index", "M4_exercise_plus_cumulative_tut", "M5_plus_lowdim_imu_group", "M6_plus_delayed_vo2"]
    metrics = metrics.copy()
    metrics["stage_order"] = metrics["model_stage"].map({stage: i for i, stage in enumerate(order)})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)
    for ax, dataset in zip(axes, sorted(metrics["dataset"].unique())):
        sub = metrics[metrics["dataset"].eq(dataset)].sort_values("stage_order")
        labels = sub["model_stage"].str.replace("M", "").str.replace("_", " ")
        ax.plot(labels, sub["mae"], marker="o", linewidth=2)
        ax.set_title(dataset)
        ax.set_ylabel("LOSO MAE")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.25)
    fig.suptitle("Module Necessity Ladder for Borg/RPE")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_comparisons(comparisons: pd.DataFrame, output: Path) -> None:
    if comparisons.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)
    color_map = {
        "strong_for_module": "#5f8f63",
        "mixed_or_small_for_module": "#b3924b",
        "not_supported_in_this_setting": "#9d4d4d",
    }
    for ax, dataset in zip(axes, sorted(comparisons["dataset"].unique())):
        sub = comparisons[comparisons["dataset"].eq(dataset)].sort_values("delta_mae_reduction", ascending=True)
        y = np.arange(len(sub))
        colors = [color_map.get(v, "#777777") for v in sub["evidence_strength"]]
        ax.barh(y, sub["delta_mae_reduction"], color=colors, alpha=0.9)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["comparison"].str.replace("_", " "))
        ax.set_xlabel("MAE reduction")
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Module-Level Evidence by Ablation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_vo2_estimability(vo2: pd.DataFrame, output: Path) -> None:
    if vo2.empty:
        return
    sub = vo2[vo2["target"].eq("vo2_mean")].copy()
    sub = sub[sub["model"].isin(["ridge", "random_forest"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, group in sub.groupby("model"):
        group = group.sort_values("lag_sec")
        ax.plot(group["lag_sec"], group["spearman"], marker="o", label=model)
    ax.set_xlabel("VO2 lag after set (sec)")
    ax.set_ylabel("Spearman rho for VO2 mean prediction")
    ax.set_title("VO2 Estimability from GT-Segmented IMU Set Features")
    ax.grid(alpha=0.25)
    ax.legend()
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

    rpe_metrics, rpe_predictions, rpe_folds = run_rpe_ladder(rpe, "rpe_only_143_sets", include_vo2=False)
    vo2_metrics, vo2_predictions, vo2_folds = run_rpe_ladder(vo2_lag45, "rpe_vo2_lag45_96_sets", include_vo2=True)
    metrics = pd.concat([rpe_metrics, vo2_metrics], ignore_index=True)
    predictions = pd.concat([rpe_predictions, vo2_predictions], ignore_index=True)
    folds = pd.concat([rpe_folds, vo2_folds], ignore_index=True)
    comparisons = comparison_rows(metrics)
    rpe_single = single_feature_ranking(rpe, "rpe_only_143_sets", metrics)
    vo2_single = single_feature_ranking(vo2_lag45, "rpe_vo2_lag45_96_sets", metrics)
    single_features = pd.concat([rpe_single, vo2_single], ignore_index=True)
    vo2_est = vo2_estimability_table(args.vo2_estimation_summary)

    metrics.to_csv(args.output_dir / "metrics" / "rpe_module_ladder.csv", index=False)
    folds.to_csv(args.output_dir / "metrics" / "rpe_module_fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "tables" / "rpe_module_predictions.csv", index=False)
    comparisons.to_csv(args.output_dir / "tables" / "module_necessity_comparisons.csv", index=False)
    single_features.to_csv(args.output_dir / "tables" / "single_feature_open_ranking.csv", index=False)
    vo2_est.to_csv(args.output_dir / "tables" / "vo2_estimability_evidence.csv", index=False)

    plot_ladder(metrics, args.output_dir / "figures" / "rpe_module_ladder_mae.png")
    plot_comparisons(comparisons, args.output_dir / "figures" / "module_comparison_delta_mae.png")
    plot_single_feature_ranking(single_features, args.output_dir / "figures" / "single_feature_raw_spearman_ranking.png")
    plot_vo2_estimability(vo2_est, args.output_dir / "figures" / "vo2_estimability_spearman.png")

    run_config = "\n".join(
        [
            "experiment_id: '005'",
            "domain: fatigue_rpe_vo2",
            "name: module_necessity_rpe_vo2",
            "created_at: '2026-05-17'",
            "split: leave-one-subject-out",
            "model: ridge_regression for RPE module ladder",
            "primary_vo2_lag_sec: 45",
            "inputs:",
            f"  rpe_set_features: {args.rpe_set_features}",
            f"  vo2_merged_features: {args.vo2_merged_features}",
            f"  vo2_estimation_summary: {args.vo2_estimation_summary}",
            "module_questions:",
            "  - Does exercise context improve RPE modeling?",
            "  - Does segmentation-derived cumulative TUT improve over ordinal set index?",
            "  - Do IMU trend features add beyond cumulative TUT?",
            "  - Does delayed VO2 add to RPE, and is VO2 estimable as its own physiological target?",
            "  - Which individual numeric features are most associated with RPE without assuming module importance?",
            "",
        ]
    )
    (args.output_dir / "run_config.yaml").write_text(run_config, encoding="utf-8")

    strong = comparisons[comparisons["evidence_strength"].eq("strong_for_module")].copy()
    vo2_best = vo2_est[vo2_est["model"].isin(["ridge", "random_forest"])].sort_values(["target", "mae"]).groupby("target").head(5) if not vo2_est.empty else pd.DataFrame()
    top_raw = (
        single_features[single_features["feature_type"].eq("numeric")]
        .sort_values(["dataset", "abs_raw_spearman"], ascending=[True, False])
        .groupby("dataset")
        .head(10)
    )
    top_univariate = (
        single_features.sort_values(["dataset", "single_feature_mae"], ascending=[True, True])
        .groupby("dataset")
        .head(10)
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": "005",
        "domain": "fatigue_rpe_vo2",
        "name": "module_necessity_rpe_vo2",
        "created_at": "2026-05-17",
        "status": "formal",
        "task": "module-level ablation to justify exercise segmentation, set/rep/TUT features, IMU trend features, and VO2 estimation",
        "question": "Which modules are supported by current data, and which should be described as auxiliary or future work?",
        "input_data": [str(args.rpe_set_features), str(args.vo2_merged_features), str(args.vo2_estimation_summary)],
        "output_dir": str(args.output_dir),
        "command": f".venv311/bin/python tools/analyze_module_necessity_rpe_vo2.py --output-dir {args.output_dir}",
        "primary_metrics": {
            "rpe_only_rows": int(len(rpe)),
            "rpe_only_subjects": int(rpe["folder"].nunique()),
            "vo2_overlap_lag45_rows": int(len(vo2_lag45)),
            "vo2_overlap_lag45_subjects": int(vo2_lag45["folder"].nunique()),
            "strong_module_comparisons": int(len(strong)),
            "single_feature_rows": int(len(single_features)),
        },
        "module_comparisons": comparisons.to_dict(orient="records"),
        "top_single_numeric_features_by_abs_raw_spearman": top_raw.to_dict(orient="records"),
        "top_single_features_by_univariate_loso_mae": top_univariate.to_dict(orient="records"),
        "vo2_estimability_best_models": vo2_best.to_dict(orient="records") if not vo2_best.empty else [],
        "key_files": {
            "rpe_ladder": "metrics/rpe_module_ladder.csv",
            "module_comparisons": "tables/module_necessity_comparisons.csv",
            "single_feature_ranking": "tables/single_feature_open_ranking.csv",
            "vo2_estimability": "tables/vo2_estimability_evidence.csv",
            "rpe_ladder_figure": "figures/rpe_module_ladder_mae.png",
            "module_delta_figure": "figures/module_comparison_delta_mae.png",
            "single_feature_ranking_figure": "figures/single_feature_raw_spearman_ranking.png",
            "vo2_estimability_figure": "figures/vo2_estimability_spearman.png",
        },
        "notes": "This analysis supports exercise context and segmentation-derived cumulative TUT for RPE. Delayed VO2 is better framed as an auxiliary physiological load target than as a necessary RPE predictor under the current labels.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_manifest(args.output_dir)

    print("Module necessity analysis written to:", args.output_dir)
    print("\nRPE module ladder:")
    print(metrics[["dataset", "model_stage", "module_claim", "n", "mae", "spearman", "rounded_pm1_acc"]].round(4).to_string(index=False))
    print("\nModule comparisons:")
    print(comparisons[["dataset", "comparison", "delta_mae_reduction", "delta_spearman", "evidence_strength"]].round(4).to_string(index=False))
    print("\nTop single numeric features by raw association:")
    print(top_raw[["dataset", "feature", "family", "raw_spearman", "subject_centered_spearman", "exercise_centered_spearman", "single_feature_mae", "delta_mae_vs_global_mean"]].round(4).to_string(index=False))
    print("\nTop single features by univariate LOSO MAE:")
    print(top_univariate[["dataset", "feature", "feature_type", "family", "single_feature_mae", "single_feature_spearman", "delta_mae_vs_global_mean", "delta_mae_vs_exercise_context"]].round(4).to_string(index=False))
    if not vo2_best.empty:
        print("\nVO2 estimability evidence:")
        print(vo2_best[["target", "lag_sec", "model", "n_sets", "subjects", "mae", "r2", "spearman"]].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze module necessity for exercise segmentation, TUT features, IMU trends, and VO2 estimation.")
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
        "--vo2-estimation-summary",
        type=Path,
        default=Path("artifacts_rep_classification/019_vo2_gt_waveform_relation/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
