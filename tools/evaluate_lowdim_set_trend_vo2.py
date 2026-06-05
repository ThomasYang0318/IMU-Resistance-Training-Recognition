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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "borg"
KEYS = ["folder", "exercise", "set_id"]

META_FEATURES = [
    "kg",
    "n_reps",
    "total_tut_sec",
]

ORDER_DIAGNOSTIC_FEATURES = [
    "set_index_numeric",
]

LOWDIM_TREND_FEATURES = [
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return clean_json(value.item())
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
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan, "rounded_exact_acc": np.nan, "rounded_pm1_acc": np.nan}
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
        df[key] = df[key].map(normalize_set_id) if key == "set_id" else df[key].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for col in df.columns:
        if col not in {"folder", "exercise", "set_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    out: list[str] = []
    for col in features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 8 and df[col].nunique(dropna=True) > 1:
            out.append(col)
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


def model_for(name: str, seed: int, n_estimators: int):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_baseline(df: pd.DataFrame, dataset: str, lag_sec: float | None) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
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
        folds.append({"dataset": dataset, "lag_sec": lag_sec, "model_stage": "baseline_exercise_mean", "model_type": "baseline", "fold": fold_idx, "val_subject": folder, **score})
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append(
                {
                    "dataset": dataset,
                    "lag_sec": lag_sec,
                    "model_stage": "baseline_exercise_mean",
                    "model_type": "baseline",
                    "fold": fold_idx,
                    "folder": row.folder,
                    "exercise": row.exercise,
                    "set_id": row.set_id,
                    "set_index_numeric": getattr(row, "set_index_numeric", np.nan),
                    "borg_true": float(y),
                    "borg_pred": float(p),
                    "is_calibration": False,
                    "calibration_offset": 0.0,
                }
            )
    pred_df = pd.DataFrame(preds)
    metric = {"dataset": dataset, "lag_sec": lag_sec, "model_stage": "baseline_exercise_mean", "model_type": "baseline", "subjects": int(df["folder"].nunique()), "n_features": 0, **regression_scores(pred_df["borg_true"], pred_df["borg_pred"])}
    return metric, pred_df, pd.DataFrame(folds)


def cross_subject_eval(
    df: pd.DataFrame,
    dataset: str,
    lag_sec: float | None,
    stage: str,
    model_name: str,
    features: list[str],
    seed: int,
    n_estimators: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    preds: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        if train.empty or test.empty:
            continue
        x_train, x_test, columns = build_matrix(train, test, features)
        model = model_for(model_name, seed + fold_idx, n_estimators)
        model.fit(x_train, train[TARGET].to_numpy(dtype=float))
        pred = model.predict(x_test)
        truth = test[TARGET].to_numpy(dtype=float)
        score = regression_scores(truth, pred)
        folds.append({"dataset": dataset, "lag_sec": lag_sec, "model_stage": stage, "model_type": model_name, "fold": fold_idx, "val_subject": folder, "n_features": len(columns), **score})
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append(
                {
                    "dataset": dataset,
                    "lag_sec": lag_sec,
                    "model_stage": stage,
                    "model_type": model_name,
                    "fold": fold_idx,
                    "folder": row.folder,
                    "exercise": row.exercise,
                    "set_id": row.set_id,
                    "set_index_numeric": getattr(row, "set_index_numeric", np.nan),
                    "borg_true": float(y),
                    "borg_pred": float(p),
                    "is_calibration": False,
                    "calibration_offset": 0.0,
                }
            )
    pred_df = pd.DataFrame(preds)
    n_features = int(pd.DataFrame(folds)["n_features"].median()) if folds else 0
    metric = {"dataset": dataset, "lag_sec": lag_sec, "model_stage": stage, "model_type": model_name, "subjects": int(df["folder"].nunique()), "n_features": n_features, **regression_scores(pred_df["borg_true"], pred_df["borg_pred"])}
    return metric, pred_df, pd.DataFrame(folds)


def apply_subject_calibration(predictions: pd.DataFrame, dataset: str, lag_sec: float | None) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for (_, folder), sub in predictions.groupby(["fold", "folder"], sort=True):
        sub = sub.copy()
        sub["set_index_numeric_sort"] = pd.to_numeric(sub["set_index_numeric"], errors="coerce").fillna(9999)
        calibration_idx: list[int] = []
        for _, ex_group in sub.sort_values(["exercise", "set_index_numeric_sort", "set_id"]).groupby("exercise", sort=True):
            calibration_idx.append(int(ex_group.index[0]))
        calib = sub.loc[calibration_idx]
        offset = float((calib["borg_true"] - calib["borg_pred"]).mean()) if not calib.empty else 0.0
        sub["is_calibration"] = sub.index.isin(calibration_idx)
        sub["calibration_offset"] = offset
        sub.loc[~sub["is_calibration"], "borg_pred"] = sub.loc[~sub["is_calibration"], "borg_pred"] + offset
        rows.append(sub.drop(columns=["set_index_numeric_sort"]))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    eval_rows = out[~out["is_calibration"]].copy()
    metric = {
        "dataset": dataset,
        "lag_sec": lag_sec,
        "model_stage": "D_subject_calibration",
        "model_type": predictions["model_type"].iloc[0] if not predictions.empty else "",
        "subjects": int(predictions["folder"].nunique()) if not predictions.empty else 0,
        "n_features": np.nan,
        "calibration": "first_set_per_exercise_offset",
        "calibration_rows": int(out["is_calibration"].sum()) if not out.empty else 0,
        **(regression_scores(eval_rows["borg_true"], eval_rows["borg_pred"]) if not eval_rows.empty else regression_scores([], [])),
    }
    out["model_stage"] = "D_subject_calibration"
    return metric, out


def run_dataset(df: pd.DataFrame, dataset: str, lag_sec: float | None, include_vo2: bool, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []

    metric, pred, fold = evaluate_baseline(df, dataset, lag_sec)
    metrics.append(metric)
    predictions.append(pred)
    folds.append(fold)

    stages = {
        "A_workload_dose": META_FEATURES,
        "A_plus_set_order_diagnostic": META_FEATURES + ORDER_DIAGNOSTIC_FEATURES,
        "B_lowdim_set_trend": META_FEATURES + LOWDIM_TREND_FEATURES,
    }
    if include_vo2:
        stages["C_lowdim_set_trend_vo2"] = META_FEATURES + LOWDIM_TREND_FEATURES + VO2_FEATURES

    for stage, features in stages.items():
        for model_name in ["ridge", "random_forest"]:
            metric, pred, fold = cross_subject_eval(df, dataset, lag_sec, stage, model_name, features, args.seed, args.n_estimators)
            metrics.append(metric)
            predictions.append(pred)
            folds.append(fold)
            if stage == "C_lowdim_set_trend_vo2":
                cal_metric, cal_pred = apply_subject_calibration(pred, dataset, lag_sec)
                metrics.append(cal_metric)
                predictions.append(cal_pred)

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.concat(folds, ignore_index=True)


def make_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparable = metrics[
        metrics["model_stage"].isin(["A_workload_dose", "A_plus_set_order_diagnostic", "B_lowdim_set_trend", "C_lowdim_set_trend_vo2"])
    ].copy()
    for key, group in comparable.groupby(["dataset", "lag_sec", "model_type"], dropna=False):
        by_stage = group.set_index("model_stage")
        for left, right, label in [
            ("A_workload_dose", "A_plus_set_order_diagnostic", "order_diagnostic_gain"),
            ("A_workload_dose", "B_lowdim_set_trend", "B_minus_A_lowdim_imu_gain"),
            ("B_lowdim_set_trend", "C_lowdim_set_trend_vo2", "C_minus_B_vo2_gain"),
        ]:
            if left in by_stage.index and right in by_stage.index:
                a = by_stage.loc[left]
                b = by_stage.loc[right]
                rows.append(
                    {
                        "dataset": key[0],
                        "lag_sec": key[1],
                        "model_type": key[2],
                        "comparison": label,
                        "mae_reduction": float(a["mae"] - b["mae"]),
                        "spearman_gain": float(b["spearman"] - a["spearman"]) if pd.notna(a["spearman"]) and pd.notna(b["spearman"]) else np.nan,
                        "pm1_acc_gain": float(b["rounded_pm1_acc"] - a["rounded_pm1_acc"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_rpe_only(metrics: pd.DataFrame, output_path: Path) -> None:
    sub = metrics[metrics["dataset"].eq("rpe_lowdim_143_sets")].copy()
    sub["label"] = sub["model_stage"] + " / " + sub["model_type"]
    order = {"baseline_exercise_mean": 0, "A_workload_dose": 1, "A_plus_set_order_diagnostic": 2, "B_lowdim_set_trend": 3}
    sub["stage_order"] = sub["model_stage"].map(order).fillna(99)
    sub = sub.sort_values(["stage_order", "model_type"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(sub["label"], sub["mae"], color="#4f86b7")
    ax.set_xlabel("Leave-subject-out MAE (Borg/RPE)")
    ax.set_title("Low-dimensional set-trend validation on 143 RPE sets")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_vo2(metrics: pd.DataFrame, output_path: Path) -> None:
    sub = metrics[
        metrics["dataset"].eq("rpe_vo2_lowdim_96_sets")
        & metrics["model_stage"].isin(["A_workload_dose", "A_plus_set_order_diagnostic", "B_lowdim_set_trend", "C_lowdim_set_trend_vo2"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for model_name, linestyle in [("ridge", "-"), ("random_forest", "--")]:
        model_sub = sub[sub["model_type"].eq(model_name)]
        for stage, color in [
            ("A_workload_dose", "#777777"),
            ("A_plus_set_order_diagnostic", "#999999"),
            ("B_lowdim_set_trend", "#4f86b7"),
            ("C_lowdim_set_trend_vo2", "#c05a4a"),
        ]:
            line = model_sub[model_sub["model_stage"].eq(stage)].sort_values("lag_sec")
            if line.empty:
                continue
            axes[0].plot(line["lag_sec"], line["mae"], marker="o", linestyle=linestyle, color=color, label=f"{stage} / {model_name}")
            axes[1].plot(line["lag_sec"], line["spearman"], marker="o", linestyle=linestyle, color=color, label=f"{stage} / {model_name}")
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Spearman")
    for ax in axes:
        ax.set_xlabel("VO2 lag after set (sec)")
        ax.grid(alpha=0.25)
    axes[0].set_title("Low-dimensional VO2 overlap MAE")
    axes[1].set_title("Low-dimensional VO2 overlap Spearman")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta(delta: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    vo2 = delta[delta["dataset"].eq("rpe_vo2_lowdim_96_sets")]
    for comparison, ax, title in [
        ("B_minus_A_lowdim_imu_gain", axes[0], "Low-dimensional IMU trend gain over workload/TUT"),
        ("C_minus_B_vo2_gain", axes[1], "Delayed VO2 gain over low-dimensional IMU"),
    ]:
        sub = vo2[vo2["comparison"].eq(comparison)]
        for model_name, color in [("ridge", "#4f86b7"), ("random_forest", "#c05a4a")]:
            line = sub[sub["model_type"].eq(model_name)].sort_values("lag_sec")
            if line.empty:
                continue
            ax.plot(line["lag_sec"], line["mae_reduction"], marker="o", label=model_name, color=color)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("VO2 lag after set (sec)")
        ax.set_ylabel("MAE reduction")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best(metrics: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> dict[str, object]:
    candidates = metrics[metrics["model_stage"].isin(["B_lowdim_set_trend", "C_lowdim_set_trend_vo2"])].dropna(subset=["mae"]).copy()
    best = candidates.sort_values(["mae", "model_stage"]).iloc[0].to_dict()
    mask = predictions["dataset"].eq(best["dataset"]) & predictions["model_stage"].eq(best["model_stage"]) & predictions["model_type"].eq(best["model_type"])
    if pd.notna(best.get("lag_sec")):
        mask &= predictions["lag_sec"].eq(best["lag_sec"])
    sub = predictions[mask].copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(sub["borg_true"], sub["borg_pred"], alpha=0.75, color="#4f86b7", edgecolor="white", linewidth=0.4)
    lo = float(min(sub["borg_true"].min(), sub["borg_pred"].min()))
    hi = float(max(sub["borg_true"].max(), sub["borg_pred"].max()))
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.0)
    ax.set_xlabel("True Borg/RPE")
    ax.set_ylabel("Predicted Borg/RPE")
    ax.set_title(f"Best low-dimensional model: {best['model_stage']} / {best['model_type']}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return best


def write_outputs(args: argparse.Namespace, metrics: pd.DataFrame, delta: pd.DataFrame, predictions: pd.DataFrame, folds: pd.DataFrame, best: dict[str, object], rpe: pd.DataFrame, vo2: pd.DataFrame) -> None:
    metrics.to_csv(args.output_dir / "metrics" / "nested_model_summary.csv", index=False)
    delta.to_csv(args.output_dir / "metrics" / "model_delta_summary.csv", index=False)
    folds.to_csv(args.output_dir / "metrics" / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "tables" / "nested_model_predictions.csv", index=False)

    manifest = pd.DataFrame(
        [
            {"path": "summary.json", "type": "json", "description": "Artifact summary and key validation results."},
            {"path": "run_config.yaml", "type": "yaml", "description": "Input paths and model configuration."},
            {"path": "metrics/nested_model_summary.csv", "type": "csv", "description": "LOSO metrics for low-dimensional nested models."},
            {"path": "metrics/model_delta_summary.csv", "type": "csv", "description": "Incremental gains from low-dimensional IMU and VO2 groups."},
            {"path": "metrics/fold_metrics.csv", "type": "csv", "description": "Per-held-out-subject metrics."},
            {"path": "tables/nested_model_predictions.csv", "type": "csv", "description": "Out-of-subject predictions."},
            {"path": "figures/rpe_lowdim_nested_mae.png", "type": "png", "description": "RPE-only low-dimensional MAE comparison."},
            {"path": "figures/vo2_lowdim_lag_metrics.png", "type": "png", "description": "VO2 lag metrics."},
            {"path": "figures/lowdim_incremental_mae_gain.png", "type": "png", "description": "Incremental MAE gains."},
            {"path": "figures/best_lowdim_predictions.png", "type": "png", "description": "Best model prediction scatter."},
        ]
    )
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)

    (args.output_dir / "run_config.yaml").write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                'created_at: "2026-05-17"',
                'domain: "fatigue_rpe_vo2"',
                'experiment_id: "002"',
                'name: "lowdim_set_trend_vo2_eval"',
                'split: "leave-one-subject-out"',
                f'rpe_set_features: "{args.rpe_set_features}"',
                f'vo2_merged_features: "{args.vo2_merged_features}"',
                f'output_dir: "{args.output_dir}"',
                f"seed: {args.seed}",
                f"n_estimators: {args.n_estimators}",
                "feature_groups:",
                "  workload_dose:",
                *[f"    - {feature}" for feature in META_FEATURES],
                "  set_order_diagnostic:",
                *[f"    - {feature}" for feature in ORDER_DIAGNOSTIC_FEATURES],
                "  lowdim_set_trend:",
                *[f"    - {feature}" for feature in LOWDIM_TREND_FEATURES],
                "  delayed_vo2:",
                *[f"    - {feature}" for feature in VO2_FEATURES],
                "constraints:",
                "  uses_ce_phase_specific_features: false",
                "  uses_subject_normalized_vo2_features: false",
                "  schema_changed: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "1.0",
        "experiment_id": "002",
        "domain": "fatigue_rpe_vo2",
        "name": "lowdim_set_trend_vo2_eval",
        "created_at": "2026-05-17",
        "status": "formal",
        "task": "low-dimensional set-level IMU trend and delayed VO2 validation",
        "question": "Can compact non-phase IMU set-trend features improve LOSO Borg/RPE prediction over metadata/progress baselines, and does delayed VO2 add further value?",
        "input_data": [str(args.rpe_set_features), str(args.vo2_merged_features)],
        "input_artifacts": [
            "artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan",
            "artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation",
        ],
        "output_dir": str(args.output_dir),
        "command": f".venv311/bin/python tools/evaluate_lowdim_set_trend_vo2.py --output-dir {args.output_dir}",
        "git_commit": "dirty-worktree",
        "split": "leave-one-subject-out",
        "primary_metrics": {
            "rpe_only_rows": int(len(rpe)),
            "rpe_only_subjects": int(rpe["folder"].nunique()),
            "vo2_overlap_sets": int(vo2[KEYS].drop_duplicates().shape[0]),
            "vo2_overlap_subjects": int(vo2["folder"].nunique()),
            "best_model_mae": float(best.get("mae", np.nan)),
            "best_model_spearman": float(best.get("spearman", np.nan)) if pd.notna(best.get("spearman", np.nan)) else None,
        },
        "feature_groups": {
            "categorical_controls": ["exercise"],
            "workload_dose": META_FEATURES,
            "set_order_diagnostic": ORDER_DIAGNOSTIC_FEATURES,
            "lowdim_set_trend": LOWDIM_TREND_FEATURES,
            "delayed_vo2": VO2_FEATURES,
        },
        "key_files": {
            "overall_metrics": "metrics/nested_model_summary.csv",
            "delta_metrics": "metrics/model_delta_summary.csv",
            "predictions": "tables/nested_model_predictions.csv",
            "rpe_only_figure": "figures/rpe_lowdim_nested_mae.png",
            "vo2_lag_figure": "figures/vo2_lowdim_lag_metrics.png",
            "best_scatter": "figures/best_lowdim_predictions.png",
        },
        "best_rpe_only_models": metrics[metrics["dataset"].eq("rpe_lowdim_143_sets")].sort_values("mae").head(6).to_dict(orient="records"),
        "best_vo2_overlap_models": metrics[metrics["dataset"].eq("rpe_vo2_lowdim_96_sets")].sort_values("mae").head(10).to_dict(orient="records"),
        "incremental_summary": delta.to_dict(orient="records"),
        "notes": "This experiment excludes CE phase-specific columns and subject-normalized VO2 features. Set-level trend features are still post-set summaries, not per-rep real-time inputs.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-dimensional non-phase set trend + delayed VO2 validation for Borg/RPE.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for subdir in ["metrics", "tables", "figures", "diagnostics", "logs"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    rpe = load_table(args.rpe_set_features)
    vo2 = load_table(args.vo2_merged_features)

    metrics_all: list[pd.DataFrame] = []
    predictions_all: list[pd.DataFrame] = []
    folds_all: list[pd.DataFrame] = []

    metrics, predictions, folds = run_dataset(rpe, "rpe_lowdim_143_sets", None, include_vo2=False, args=args)
    metrics_all.append(metrics)
    predictions_all.append(predictions)
    folds_all.append(folds)

    for lag in sorted(vo2["lag_sec"].dropna().unique()):
        sub = vo2[vo2["lag_sec"].eq(lag)].copy()
        metrics, predictions, folds = run_dataset(sub, "rpe_vo2_lowdim_96_sets", float(lag), include_vo2=True, args=args)
        metrics_all.append(metrics)
        predictions_all.append(predictions)
        folds_all.append(folds)

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    predictions_df = pd.concat(predictions_all, ignore_index=True)
    folds_df = pd.concat(folds_all, ignore_index=True)
    delta_df = make_delta_table(metrics_df)

    plot_rpe_only(metrics_df, args.output_dir / "figures" / "rpe_lowdim_nested_mae.png")
    plot_vo2(metrics_df, args.output_dir / "figures" / "vo2_lowdim_lag_metrics.png")
    plot_delta(delta_df, args.output_dir / "figures" / "lowdim_incremental_mae_gain.png")
    best = plot_best(metrics_df, predictions_df, args.output_dir / "figures" / "best_lowdim_predictions.png")

    write_outputs(args, metrics_df, delta_df, predictions_df, folds_df, best, rpe, vo2)

    print("RPE-only rows:", len(rpe), "subjects:", rpe["folder"].nunique())
    print("VO2 overlap sets:", vo2[KEYS].drop_duplicates().shape[0], "subjects:", vo2["folder"].nunique())
    print("\nFeature groups:")
    print("A workload:", ", ".join(available_features(rpe, META_FEATURES)))
    print("Order diagnostic:", ", ".join(available_features(rpe, ORDER_DIAGNOSTIC_FEATURES)))
    print("B lowdim:", ", ".join(available_features(rpe, LOWDIM_TREND_FEATURES)))
    print("C VO2:", ", ".join(available_features(vo2, VO2_FEATURES)))
    print("\nBest models by MAE:")
    print(metrics_df.sort_values("mae")[["dataset", "lag_sec", "model_stage", "model_type", "n", "mae", "spearman", "rounded_pm1_acc"]].head(12).round(4).to_string(index=False))
    print("\nIncremental gains:")
    print(delta_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
