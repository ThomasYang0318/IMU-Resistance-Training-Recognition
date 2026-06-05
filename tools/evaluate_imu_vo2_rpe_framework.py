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
META_FEATURES = ["set_index_numeric", "kg", "n_reps"]
VO2_BASE_FEATURES = [
    "vo2_mean",
    "vo2_peak",
    "vo2_min",
    "vo2_slope",
    "vo2_range",
    "vo2_peak_minus_mean",
    "vo2_mean_x_total_rep_sec",
    "vo2_peak_x_total_rep_sec",
    "vo2_mean_x_n_reps",
    "vo2_mean_per_rep",
]
BANNED_MODEL_COLUMNS = {
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


def normalize_set_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


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


def load_set_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for key in KEYS:
        df[key] = df[key].map(normalize_set_id) if key == "set_id" else df[key].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for col in df.columns:
        if col not in {"folder", "exercise", "set_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_vo2_overlap(phase_set: pd.DataFrame, vo2_merged_path: Path) -> pd.DataFrame:
    vo2 = pd.read_csv(vo2_merged_path)
    for key in KEYS:
        vo2[key] = vo2[key].map(normalize_set_id) if key == "set_id" else vo2[key].astype(str)
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
        "vo2_range",
        "vo2_peak_minus_mean",
        "vo2_mean_delta_subject_min",
        "vo2_mean_z_subject",
        "vo2_peak_delta_subject_min",
        "vo2_peak_z_subject",
    ]
    vo2_cols = [col for col in vo2_cols if col in vo2.columns]
    merged = phase_set.merge(vo2[vo2_cols], on=KEYS, how="inner").copy()
    for col in ["vo2_mean", "vo2_peak", "vo2_min", "vo2_slope"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    derived = pd.DataFrame(
        {
            "vo2_range": merged["vo2_peak"] - merged["vo2_min"],
            "vo2_peak_minus_mean": merged["vo2_peak"] - merged["vo2_mean"],
            "vo2_mean_x_total_rep_sec": merged["vo2_mean"] * merged["total_rep_sec"],
            "vo2_peak_x_total_rep_sec": merged["vo2_peak"] * merged["total_rep_sec"],
            "vo2_mean_x_n_reps": merged["vo2_mean"] * merged["n_reps"],
            "vo2_mean_per_rep": merged["vo2_mean"] / merged["n_reps"].replace(0, np.nan),
        }
    )
    merged = pd.concat([merged, derived], axis=1)
    return merged


def select_imu_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in BANNED_MODEL_COLUMNS or col in META_FEATURES or col.startswith("vo2_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 8 and df[col].nunique(dropna=True) > 1:
            features.append(col)
    return features


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    out: list[str] = []
    for col in features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 8 and df[col].nunique(dropna=True) > 1:
            out.append(col)
    return out


def build_matrix(train: pd.DataFrame, test: pd.DataFrame, numeric_features: list[str], include_exercise: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    kept: list[str] = []
    for col in numeric_features:
        if col not in train.columns:
            continue
        values = pd.to_numeric(train[col], errors="coerce")
        if values.notna().sum() >= 5 and values.nunique(dropna=True) > 1:
            kept.append(col)
    x_train = train[kept].apply(pd.to_numeric, errors="coerce").copy()
    x_test = test[kept].apply(pd.to_numeric, errors="coerce").copy()
    if include_exercise:
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


def evaluate_baseline(df: pd.DataFrame, dataset: str, stage: str, lag_sec: float | None = None) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    preds: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        if train.empty or test.empty:
            continue
        global_mean = float(train[TARGET].mean())
        exercise_mean = train.groupby("exercise")[TARGET].mean().to_dict()
        pred = test["exercise"].map(exercise_mean).fillna(global_mean).astype(float).to_numpy()
        truth = test[TARGET].to_numpy(dtype=float)
        fold_score = regression_scores(truth, pred)
        fold_rows.append({"dataset": dataset, "lag_sec": lag_sec, "model_stage": stage, "model_type": "baseline", "fold": fold_idx, "val_subject": folder, **fold_score})
        for row, y, p in zip(test.itertuples(index=False), truth, pred):
            preds.append(
                {
                    "dataset": dataset,
                    "lag_sec": lag_sec,
                    "model_stage": stage,
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
    metric = {"dataset": dataset, "lag_sec": lag_sec, "model_stage": stage, "model_type": "baseline", "subjects": int(df["folder"].nunique()), **regression_scores(pred_df["borg_true"], pred_df["borg_pred"])}
    return metric, pred_df, pd.DataFrame(fold_rows)


def cross_subject_eval(
    df: pd.DataFrame,
    dataset: str,
    stage: str,
    model_name: str,
    numeric_features: list[str],
    seed: int,
    n_estimators: int,
    lag_sec: float | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    preds: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fold_idx, folder in enumerate(sorted(df["folder"].astype(str).unique()), start=1):
        train = df[df["folder"].astype(str) != folder].copy()
        test = df[df["folder"].astype(str) == folder].copy()
        if train.empty or test.empty:
            continue
        x_train, x_test, feature_columns = build_matrix(train, test, numeric_features, include_exercise=True)
        model = model_for(model_name, seed + fold_idx, n_estimators)
        model.fit(x_train, train[TARGET].to_numpy(dtype=float))
        pred = model.predict(x_test)
        truth = test[TARGET].to_numpy(dtype=float)
        fold_score = regression_scores(truth, pred)
        fold_rows.append(
            {
                "dataset": dataset,
                "lag_sec": lag_sec,
                "model_stage": stage,
                "model_type": model_name,
                "fold": fold_idx,
                "val_subject": folder,
                "n_features": len(feature_columns),
                **fold_score,
            }
        )
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
    metric = {
        "dataset": dataset,
        "lag_sec": lag_sec,
        "model_stage": stage,
        "model_type": model_name,
        "subjects": int(df["folder"].nunique()),
        "n_features": int(pd.DataFrame(fold_rows)["n_features"].median()) if fold_rows else 0,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, pd.DataFrame(fold_rows)


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
        "calibration": "first_set_per_exercise_offset",
        "calibration_rows": int(out["is_calibration"].sum()) if not out.empty else 0,
        **(regression_scores(eval_rows["borg_true"], eval_rows["borg_pred"]) if not eval_rows.empty else regression_scores([], [])),
    }
    out["model_stage"] = "D_subject_calibration"
    return metric, out


def run_rpe_only(df: pd.DataFrame, imu_features: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    metric, pred, fold = evaluate_baseline(df, "rpe_only_143_sets", "baseline_exercise_mean", None)
    metrics.append(metric)
    predictions.append(pred)
    folds.append(fold)
    stages = {
        "A_metadata_progress": META_FEATURES,
        "B_metadata_imu_phase": META_FEATURES + imu_features,
    }
    for stage, features in stages.items():
        for model_name in ["ridge", "random_forest"]:
            metric, pred, fold = cross_subject_eval(df, "rpe_only_143_sets", stage, model_name, features, args.seed, args.n_estimators, None)
            metrics.append(metric)
            predictions.append(pred)
            folds.append(fold)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.concat(folds, ignore_index=True)


def run_vo2_overlap(df: pd.DataFrame, imu_features: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    vo2_features = available_features(df, VO2_BASE_FEATURES)
    stages = {
        "A_metadata_progress": META_FEATURES,
        "B_metadata_imu_phase": META_FEATURES + imu_features,
        "C_metadata_imu_phase_vo2": META_FEATURES + imu_features + vo2_features,
    }
    for lag in sorted(df["lag_sec"].dropna().unique()):
        sub = df[df["lag_sec"].eq(lag)].copy()
        metric, pred, fold = evaluate_baseline(sub, "rpe_vo2_overlap_96_sets", "baseline_exercise_mean", float(lag))
        metrics.append(metric)
        predictions.append(pred)
        folds.append(fold)
        for stage, features in stages.items():
            for model_name in ["ridge", "random_forest"]:
                metric, pred, fold = cross_subject_eval(sub, "rpe_vo2_overlap_96_sets", stage, model_name, features, args.seed, args.n_estimators, float(lag))
                metrics.append(metric)
                predictions.append(pred)
                folds.append(fold)
                if stage == "C_metadata_imu_phase_vo2":
                    cal_metric, cal_pred = apply_subject_calibration(pred, "rpe_vo2_overlap_96_sets", float(lag))
                    metrics.append(cal_metric)
                    predictions.append(cal_pred)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.concat(folds, ignore_index=True)


def make_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["dataset", "lag_sec", "model_type"]
    comparable = metrics[metrics["model_stage"].isin(["A_metadata_progress", "B_metadata_imu_phase", "C_metadata_imu_phase_vo2"])].copy()
    for key, group in comparable.groupby(keys, dropna=False):
        by_stage = group.set_index("model_stage")
        for left, right, label in [
            ("A_metadata_progress", "B_metadata_imu_phase", "B_minus_A_imu_gain"),
            ("B_metadata_imu_phase", "C_metadata_imu_phase_vo2", "C_minus_B_vo2_gain"),
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
    sub = metrics[metrics["dataset"].eq("rpe_only_143_sets")].copy()
    order = ["baseline_exercise_mean", "A_metadata_progress", "B_metadata_imu_phase"]
    sub["label"] = sub["model_stage"] + " / " + sub["model_type"]
    sub["stage_order"] = sub["model_stage"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    sub = sub.sort_values(["stage_order", "model_type"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(sub["label"], sub["mae"], color="#5b8db8")
    ax.set_xlabel("Leave-subject-out MAE (Borg/RPE)")
    ax.set_title("RPE-only nested validation on 143 GT phase-aware sets")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_vo2_lag(metrics: pd.DataFrame, output_path: Path) -> None:
    sub = metrics[
        metrics["dataset"].eq("rpe_vo2_overlap_96_sets")
        & metrics["model_stage"].isin(["A_metadata_progress", "B_metadata_imu_phase", "C_metadata_imu_phase_vo2"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for model_name, line_style in [("ridge", "-"), ("random_forest", "--")]:
        model_sub = sub[sub["model_type"].eq(model_name)]
        for stage, color in [
            ("A_metadata_progress", "#777777"),
            ("B_metadata_imu_phase", "#3f7cac"),
            ("C_metadata_imu_phase_vo2", "#c45b4f"),
        ]:
            line = model_sub[model_sub["model_stage"].eq(stage)].sort_values("lag_sec")
            if line.empty:
                continue
            label = f"{stage.replace('_', ' ')} / {model_name}"
            axes[0].plot(line["lag_sec"], line["mae"], marker="o", linestyle=line_style, color=color, label=label)
            axes[1].plot(line["lag_sec"], line["spearman"], marker="o", linestyle=line_style, color=color, label=label)
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Spearman")
    for ax in axes:
        ax.set_xlabel("VO2 lag after set (sec)")
        ax.grid(alpha=0.25)
    axes[0].set_title("VO2 overlap nested MAE")
    axes[1].set_title("VO2 overlap nested Spearman")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta(delta: pd.DataFrame, output_path: Path) -> None:
    sub = delta[delta["dataset"].eq("rpe_vo2_overlap_96_sets") & delta["comparison"].eq("C_minus_B_vo2_gain")].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model_name, color in [("ridge", "#3f7cac"), ("random_forest", "#c45b4f")]:
        line = sub[sub["model_type"].eq(model_name)].sort_values("lag_sec")
        if line.empty:
            continue
        ax.plot(line["lag_sec"], line["mae_reduction"], marker="o", label=model_name, color=color)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("VO2 lag after set (sec)")
    ax.set_ylabel("MAE reduction from adding VO2 (B - C)")
    ax.set_title("Incremental value of delayed VO2 over IMU phase features")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_scatter(metrics: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> dict[str, object]:
    candidates = metrics[metrics["model_stage"].isin(["B_metadata_imu_phase", "C_metadata_imu_phase_vo2"])].dropna(subset=["mae"]).copy()
    best = candidates.sort_values(["mae", "model_stage"]).iloc[0].to_dict()
    mask = (
        predictions["dataset"].eq(best["dataset"])
        & predictions["model_stage"].eq(best["model_stage"])
        & predictions["model_type"].eq(best["model_type"])
    )
    if pd.notna(best.get("lag_sec")):
        mask &= predictions["lag_sec"].eq(best["lag_sec"])
    sub = predictions[mask].copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(sub["borg_true"], sub["borg_pred"], alpha=0.7, color="#3f7cac", edgecolor="white", linewidth=0.4)
    lo = float(min(sub["borg_true"].min(), sub["borg_pred"].min()))
    hi = float(max(sub["borg_true"].max(), sub["borg_pred"].max()))
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.0)
    ax.set_xlabel("True Borg/RPE")
    ax.set_ylabel("Predicted Borg/RPE")
    ax.set_title(f"Best nested model: {best['model_stage']} / {best['model_type']}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return best


def write_summary(args: argparse.Namespace, metrics: pd.DataFrame, delta: pd.DataFrame, best: dict[str, object], rpe_rows: int, overlap_sets: int, subjects: list[str], overlap_subjects: list[str]) -> None:
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

    best_rpe = metrics[metrics["dataset"].eq("rpe_only_143_sets")].sort_values("mae").head(5).to_dict(orient="records")
    best_vo2 = metrics[metrics["dataset"].eq("rpe_vo2_overlap_96_sets")].sort_values("mae").head(10).to_dict(orient="records")
    summary = {
        "schema_version": "1.0",
        "experiment_id": "001",
        "domain": "fatigue_rpe_vo2",
        "name": "gt_phase_imu_vo2_rpe_framework_eval",
        "created_at": "2026-05-17",
        "status": "formal",
        "task": "IMU VO2 RPE formal validation",
        "question": "Do GT phase-aware IMU features and delayed VO2 features improve subject-wise Borg/RPE prediction over metadata/progress baselines?",
        "input_data": [str(args.phase_set_features), str(args.vo2_merged_features)],
        "input_artifacts": [
            "artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis",
            "artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation",
        ],
        "output_dir": str(args.output_dir),
        "command": " ".join(
            [
                ".venv311/bin/python",
                "tools/evaluate_imu_vo2_rpe_framework.py",
                "--output-dir",
                str(args.output_dir),
            ]
        ),
        "git_commit": "dirty-worktree",
        "split": "leave-one-subject-out",
        "primary_metrics": {
            "rpe_only_rows": int(rpe_rows),
            "rpe_only_subjects": int(len(subjects)),
            "vo2_overlap_sets": int(overlap_sets),
            "vo2_overlap_subjects": int(len(overlap_subjects)),
            "best_model_mae": float(best.get("mae", np.nan)),
            "best_model_spearman": float(best.get("spearman", np.nan)) if pd.notna(best.get("spearman", np.nan)) else None,
        },
        "key_files": {
            "overall_metrics": "metrics/nested_model_summary.csv",
            "delta_metrics": "metrics/model_delta_summary.csv",
            "predictions": "tables/nested_model_predictions.csv",
            "rpe_only_figure": "figures/rpe_only_nested_mae.png",
            "vo2_lag_figure": "figures/vo2_lag_nested_metrics.png",
            "best_scatter": "figures/best_model_predictions.png",
        },
        "best_rpe_only_models": best_rpe,
        "best_vo2_overlap_models": best_vo2,
        "vo2_incremental_summary": delta[delta["comparison"].eq("C_minus_B_vo2_gain")].to_dict(orient="records"),
        "notes": "This formal validation uses existing ground-truth phase-aware set features. Subject-normalized VO2 columns are excluded from primary model features to avoid LOSO leakage. It does not evaluate predicted segmentation deployment gap yet.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal validation for IMU, delayed VO2, and Borg/RPE framework.")
    parser.add_argument(
        "--phase-set-features",
        type=Path,
        default=Path("artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_feature_dataset.csv"),
    )
    parser.add_argument(
        "--vo2-merged-features",
        type=Path,
        default=Path("artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for subdir in ["metrics", "tables", "figures", "diagnostics", "logs"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    phase_set = load_set_features(args.phase_set_features)
    vo2_overlap = load_vo2_overlap(phase_set, args.vo2_merged_features)
    imu_features = select_imu_features(phase_set)

    rpe_metrics, rpe_predictions, rpe_folds = run_rpe_only(phase_set, imu_features, args)
    vo2_metrics, vo2_predictions, vo2_folds = run_vo2_overlap(vo2_overlap, imu_features, args)
    metrics = pd.concat([rpe_metrics, vo2_metrics], ignore_index=True)
    predictions = pd.concat([rpe_predictions, vo2_predictions], ignore_index=True)
    folds = pd.concat([rpe_folds, vo2_folds], ignore_index=True)
    delta = make_delta_table(metrics)

    metrics.to_csv(args.output_dir / "metrics" / "nested_model_summary.csv", index=False)
    delta.to_csv(args.output_dir / "metrics" / "model_delta_summary.csv", index=False)
    folds.to_csv(args.output_dir / "metrics" / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "tables" / "nested_model_predictions.csv", index=False)

    manifest = pd.DataFrame(
        [
            {"path": "summary.json", "type": "json", "description": "Artifact summary and key validation results."},
            {"path": "run_config.yaml", "type": "yaml", "description": "Input paths and validation configuration."},
            {"path": "metrics/nested_model_summary.csv", "type": "csv", "description": "Leave-subject-out metrics for nested models."},
            {"path": "metrics/model_delta_summary.csv", "type": "csv", "description": "Incremental gains from IMU and VO2 feature groups."},
            {"path": "metrics/fold_metrics.csv", "type": "csv", "description": "Per-held-out-subject fold metrics."},
            {"path": "tables/nested_model_predictions.csv", "type": "csv", "description": "Out-of-subject predictions for all evaluated models."},
            {"path": "figures/rpe_only_nested_mae.png", "type": "png", "description": "RPE-only MAE comparison on 143 sets."},
            {"path": "figures/vo2_lag_nested_metrics.png", "type": "png", "description": "VO2 lag nested MAE and Spearman comparison."},
            {"path": "figures/vo2_incremental_mae_gain.png", "type": "png", "description": "MAE reduction from adding VO2 to IMU phase features."},
            {"path": "figures/best_model_predictions.png", "type": "png", "description": "Scatter plot for the best nested model."},
        ]
    )
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)

    (args.output_dir / "run_config.yaml").write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                'created_at: "2026-05-17"',
                'domain: "fatigue_rpe_vo2"',
                'experiment_id: "001"',
                'name: "gt_phase_imu_vo2_rpe_framework_eval"',
                'split: "leave-one-subject-out"',
                f'phase_set_features: "{args.phase_set_features}"',
                f'vo2_merged_features: "{args.vo2_merged_features}"',
                f'output_dir: "{args.output_dir}"',
                f"seed: {args.seed}",
                f"n_estimators: {args.n_estimators}",
                "models:",
                "  - baseline_exercise_mean",
                "  - A_metadata_progress",
                "  - B_metadata_imu_phase",
                "  - C_metadata_imu_phase_vo2",
                "  - D_subject_calibration",
                "constraints:",
                "  uses_ground_truth_phase_features: true",
                "  predicted_segmentation_evaluated: false",
                "  schema_changed: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_rpe_only(metrics, args.output_dir / "figures" / "rpe_only_nested_mae.png")
    plot_vo2_lag(metrics, args.output_dir / "figures" / "vo2_lag_nested_metrics.png")
    plot_delta(delta, args.output_dir / "figures" / "vo2_incremental_mae_gain.png")
    best = plot_best_scatter(metrics, predictions, args.output_dir / "figures" / "best_model_predictions.png")

    write_summary(
        args,
        metrics,
        delta,
        best,
        rpe_rows=len(phase_set),
        overlap_sets=vo2_overlap[KEYS].drop_duplicates().shape[0],
        subjects=sorted(phase_set["folder"].astype(str).unique().tolist()),
        overlap_subjects=sorted(vo2_overlap["folder"].astype(str).unique().tolist()),
    )

    print("RPE-only rows:", len(phase_set), "subjects:", phase_set["folder"].nunique())
    print("VO2 overlap sets:", vo2_overlap[KEYS].drop_duplicates().shape[0], "subjects:", vo2_overlap["folder"].nunique())
    print("\nBest models by MAE:")
    print(metrics.sort_values("mae")[["dataset", "lag_sec", "model_stage", "model_type", "n", "mae", "spearman", "rounded_pm1_acc"]].head(12).round(4).to_string(index=False))
    print("\nVO2 incremental gains:")
    print(delta[delta["comparison"].eq("C_minus_B_vo2_gain")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
