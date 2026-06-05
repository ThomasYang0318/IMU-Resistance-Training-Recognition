from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

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
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

PROGRESSION = ["set_index_numeric", "cumulative_active_time_sec"]

IMU_COMPONENTS = [
    {
        "component": "CE phase range",
        "display_feature": "eccentric PCA range mean",
        "feature": "eccentric_pca_range_mean",
        "source": "IMU",
    },
    {
        "component": "Concentric gyro",
        "display_feature": "concentric gyro diff RMS last2",
        "feature": "concentric_gyro_diff_rms_last2",
        "source": "IMU",
    },
    {
        "component": "Phase movement rate",
        "display_feature": "eccentric PCA movement rate mean",
        "feature": "eccentric_pca_movement_rate_mean",
        "source": "IMU",
    },
    {
        "component": "Phase timing drift",
        "display_feature": "concentric duration last2/first2",
        "feature": "concentric_sec_last2_vs_first2",
        "source": "IMU",
    },
    {
        "component": "CE ratio drift",
        "display_feature": "CE ratio slope",
        "feature": "ce_ratio_slope",
        "source": "IMU",
    },
    {
        "component": "CE phase similarity",
        "display_feature": "eccentric wave similarity drift",
        "feature": "eccentric_wave_sim_to_first2_last_minus_first",
        "source": "IMU",
    },
]

VO2_COMPONENTS = [
    {
        "component": "Delayed VO2",
        "display_feature": "VO2 slope at 45s",
        "feature": "vo2_slope_lag45",
        "source": "VO2",
    },
    {
        "component": "VO2 baseline delta",
        "display_feature": "VO2 mean delta at 10s",
        "feature": "vo2_mean_delta_subject_min_lag10",
        "source": "VO2",
    },
]


def normalize_id(value: object) -> str:
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
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
            "rounded_exact_acc": np.nan,
            "rounded_pm1_acc": np.nan,
        }
    rounded = np.clip(np.rint(pred), 1, 10)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and len(np.unique(y)) > 1 else np.nan,
        "spearman": safe_spearman(y, pred),
        "rounded_exact_acc": float(np.mean(rounded == y)),
        "rounded_pm1_acc": float(np.mean(np.abs(rounded - y) <= 1.0)),
    }


def clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def add_cumulative_active_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["folder"] = out["folder"].astype(str)
    out["exercise"] = out["exercise"].astype(str)
    out["set_id"] = out["set_id"].map(normalize_id)
    out["set_index_numeric"] = pd.to_numeric(out["set_index_numeric"], errors="coerce")
    out["total_rep_sec"] = pd.to_numeric(out["total_rep_sec"], errors="coerce")
    out["_original_order"] = np.arange(len(out))
    out = out.sort_values(["folder", "exercise", "set_index_numeric", "set_id"], kind="mergesort")
    out["cumulative_active_time_sec"] = out.groupby(["folder", "exercise"], sort=False)["total_rep_sec"].cumsum()
    out = out.sort_values("_original_order").drop(columns="_original_order")
    return out.reset_index(drop=True)


def load_phase_set(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for key in KEYS:
        if key not in df.columns:
            raise ValueError(f"Missing required column {key} in {path}")
    if TARGET not in df.columns:
        raise ValueError(f"Missing {TARGET} in {path}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for key in KEYS:
        df[key] = df[key].astype(str) if key != "set_id" else df[key].map(normalize_id)
    for col in df.columns:
        if col not in {"folder", "exercise", "set_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_cumulative_active_time(df)


def load_vo2_features(path: Path) -> pd.DataFrame:
    vo2 = pd.read_csv(path)
    for key in KEYS:
        vo2[key] = vo2[key].astype(str) if key != "set_id" else vo2[key].map(normalize_id)
    lag45 = (
        vo2[vo2["lag_sec"].eq(45.0)][KEYS + ["vo2_slope"]]
        .drop_duplicates(KEYS)
        .rename(columns={"vo2_slope": "vo2_slope_lag45"})
    )
    lag10 = (
        vo2[vo2["lag_sec"].eq(10.0)][KEYS + ["vo2_mean_delta_subject_min"]]
        .drop_duplicates(KEYS)
        .rename(columns={"vo2_mean_delta_subject_min": "vo2_mean_delta_subject_min_lag10"})
    )
    return lag45.merge(lag10, on=KEYS, how="inner")


def make_datasets(phase_set: pd.DataFrame, vo2_path: Path | None) -> dict[str, pd.DataFrame]:
    datasets = {"imu_only_7subject_sets": phase_set.copy()}
    if vo2_path is not None and vo2_path.exists():
        vo2 = load_vo2_features(vo2_path)
        overlap = phase_set.merge(vo2, on=KEYS, how="inner")
        if not overlap.empty:
            datasets["imu_vo2_overlap_sets"] = overlap
    return datasets


def available_features(train: pd.DataFrame, features: Sequence[str]) -> list[str]:
    out: list[str] = []
    for feature in features:
        if feature not in train.columns:
            continue
        series = pd.to_numeric(train[feature], errors="coerce")
        if series.notna().sum() >= 8 and series.nunique(dropna=True) > 1:
            out.append(feature)
    return out


def build_matrix(train: pd.DataFrame, other: pd.DataFrame, numeric_features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    kept = available_features(train, numeric_features)
    x_train = train[kept].apply(pd.to_numeric, errors="coerce").copy()
    x_other = other[kept].apply(pd.to_numeric, errors="coerce").copy()
    train_ex = pd.get_dummies(train["exercise"].astype(str), prefix="exercise", dtype=float)
    other_ex = pd.get_dummies(other["exercise"].astype(str), prefix="exercise", dtype=float)
    train_ex, other_ex = train_ex.align(other_ex, join="outer", axis=1, fill_value=0.0)
    x_train = pd.concat([x_train.reset_index(drop=True), train_ex.reset_index(drop=True)], axis=1)
    x_other = pd.concat([x_other.reset_index(drop=True), other_ex.reset_index(drop=True)], axis=1)
    return x_train, x_other, list(x_train.columns)


def ridge_model(alpha: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(alpha)))


def cyclic_subject_splits(subjects: Sequence[str]) -> pd.DataFrame:
    ordered = list(subjects)
    rows: list[dict[str, object]] = []
    for idx, test_subject in enumerate(ordered):
        val_subject = ordered[(idx + 1) % len(ordered)]
        train_subjects = [subject for subject in ordered if subject not in {test_subject, val_subject}]
        rows.append(
            {
                "fold": idx + 1,
                "train_subjects": "|".join(train_subjects),
                "val_subject": val_subject,
                "test_subject": test_subject,
                "n_train_subjects": len(train_subjects),
            }
        )
    return pd.DataFrame(rows)


def predict_exercise_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_mean = float(train[TARGET].mean())
    exercise_mean = train.groupby("exercise")[TARGET].mean().to_dict()
    return test["exercise"].map(exercise_mean).fillna(global_mean).astype(float).to_numpy()


def tune_alpha(train: pd.DataFrame, val: pd.DataFrame, features: Sequence[str], alpha_grid: Sequence[float]) -> tuple[float, float, int]:
    best_alpha = float(alpha_grid[0])
    best_mae = np.inf
    best_features = 0
    for alpha in alpha_grid:
        x_train, x_val, columns = build_matrix(train, val, features)
        model = ridge_model(float(alpha))
        model.fit(x_train, train[TARGET].to_numpy(dtype=float))
        pred = model.predict(x_val)
        mae = mean_absolute_error(val[TARGET].to_numpy(dtype=float), pred)
        if mae < best_mae - 1e-12:
            best_mae = float(mae)
            best_alpha = float(alpha)
            best_features = len(columns)
    return best_alpha, best_mae, best_features


def evaluate_stage(
    df: pd.DataFrame,
    dataset: str,
    splits: pd.DataFrame,
    stage: str,
    model_type: str,
    features: Sequence[str],
    alpha_grid: Sequence[float],
    component: str | None = None,
    display_feature: str | None = None,
    source: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for split in splits.itertuples(index=False):
        train_subjects = str(split.train_subjects).split("|") if split.train_subjects else []
        train = df[df["folder"].isin(train_subjects)].copy()
        val = df[df["folder"].eq(split.val_subject)].copy()
        test = df[df["folder"].eq(split.test_subject)].copy()
        if model_type == "exercise_mean":
            selected_alpha = np.nan
            val_pred = predict_exercise_mean(train, val)
            test_pred = predict_exercise_mean(train, test)
            n_features = 0
            val_selection_mae = mean_absolute_error(val[TARGET].to_numpy(dtype=float), val_pred)
        else:
            selected_alpha, val_selection_mae, _ = tune_alpha(train, val, features, alpha_grid)
            x_train, x_test, columns = build_matrix(train, test, features)
            model = ridge_model(selected_alpha)
            model.fit(x_train, train[TARGET].to_numpy(dtype=float))
            test_pred = model.predict(x_test)
            x_train_val, x_val, _ = build_matrix(train, val, features)
            val_model = ridge_model(selected_alpha)
            val_model.fit(x_train_val, train[TARGET].to_numpy(dtype=float))
            val_pred = val_model.predict(x_val)
            n_features = len(columns)
        val_score = regression_scores(val[TARGET].to_numpy(dtype=float), val_pred)
        test_score = regression_scores(test[TARGET].to_numpy(dtype=float), test_pred)
        fold_rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                "model_type": model_type,
                "fold": int(split.fold),
                "train_subjects": split.train_subjects,
                "val_subject": split.val_subject,
                "test_subject": split.test_subject,
                "selected_alpha": selected_alpha,
                "val_selection_mae": val_selection_mae,
                "n_features": int(n_features),
                **{f"val_{key}": value for key, value in val_score.items()},
                **{f"test_{key}": value for key, value in test_score.items()},
            }
        )
        for row, pred in zip(test.itertuples(index=False), test_pred):
            pred_rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "model_type": model_type,
                    "fold": int(split.fold),
                    "train_subjects": split.train_subjects,
                    "val_subject": split.val_subject,
                    "test_subject": split.test_subject,
                    "folder": row.folder,
                    "exercise": row.exercise,
                    "set_id": row.set_id,
                    "borg_true": float(getattr(row, TARGET)),
                    "borg_pred": float(pred),
                    "component": component,
                    "display_feature": display_feature,
                    "source": source,
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)
    metric = {
        "dataset": dataset,
        "stage": stage,
        "model_type": model_type,
        "n_subjects": int(df["folder"].nunique()),
        "n_test_rows": int(len(pred_df)),
        "component": component,
        "display_feature": display_feature,
        "source": source,
        "selected_alpha_median": float(fold_df["selected_alpha"].median()) if fold_df["selected_alpha"].notna().any() else np.nan,
        "n_features_median": float(fold_df["n_features"].median()) if "n_features" in fold_df else np.nan,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, fold_df


def stage_specs(df: pd.DataFrame) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"stage": "M0_exercise_mean", "model_type": "exercise_mean", "features": [], "component": None, "display_feature": None, "source": None},
        {"stage": "M1_exercise_plus_set_index", "model_type": "ridge", "features": ["set_index_numeric"], "component": "Set progression", "display_feature": "set index", "source": "progression"},
        {"stage": "M2_exercise_plus_accumulated_TUT", "model_type": "ridge", "features": ["cumulative_active_time_sec"], "component": "Accumulated TUT", "display_feature": "cumulative active time", "source": "IMU"},
        {"stage": "M3_progression_control", "model_type": "ridge", "features": list(PROGRESSION), "component": "Progression control", "display_feature": "set index + cumulative active time", "source": "progression"},
    ]
    for item in IMU_COMPONENTS + VO2_COMPONENTS:
        if item["feature"] not in df.columns:
            continue
        specs.append(
            {
                "stage": f"M4_add_one__{item['feature']}",
                "model_type": "ridge",
                "features": list(PROGRESSION) + [item["feature"]],
                **item,
            }
        )
    imu_features = [item["feature"] for item in IMU_COMPONENTS if item["feature"] in df.columns]
    if imu_features:
        specs.append(
            {
                "stage": "M5_all_IMU_figure_components",
                "model_type": "ridge",
                "features": list(PROGRESSION) + imu_features,
                "component": "All IMU figure components",
                "display_feature": "CE/TUT/gyro/PCA/similarity group",
                "source": "IMU",
            }
        )
    vo2_features = [item["feature"] for item in VO2_COMPONENTS if item["feature"] in df.columns]
    if imu_features and vo2_features:
        specs.append(
            {
                "stage": "M6_all_IMU_plus_VO2_figure_components",
                "model_type": "ridge",
                "features": list(PROGRESSION) + imu_features + vo2_features,
                "component": "All IMU + VO2 figure components",
                "display_feature": "all figure components",
                "source": "IMU+VO2",
            }
        )
    return specs


def build_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in metrics.groupby("dataset", sort=False):
        base = group[group["stage"].eq("M3_progression_control")]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group[group["stage"].str.startswith(("M4_add_one__", "M5_", "M6_"), na=False)].iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "baseline_stage": "M3_progression_control",
                    "candidate_stage": row["stage"],
                    "component": row["component"],
                    "display_feature": row["display_feature"],
                    "source": row["source"],
                    "baseline_mae": base_row["mae"],
                    "candidate_mae": row["mae"],
                    "delta_mae_reduction": base_row["mae"] - row["mae"],
                    "baseline_spearman": base_row["spearman"],
                    "candidate_spearman": row["spearman"],
                    "delta_spearman": row["spearman"] - base_row["spearman"],
                    "baseline_pm1_acc": base_row["rounded_pm1_acc"],
                    "candidate_pm1_acc": row["rounded_pm1_acc"],
                    "delta_pm1_acc": row["rounded_pm1_acc"] - base_row["rounded_pm1_acc"],
                    "n_test_rows": row["n_test_rows"],
                    "n_subjects": row["n_subjects"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["evidence_label"] = np.select(
            [
                (out["delta_mae_reduction"] > 0.05) & (out["delta_spearman"] > 0.02),
                (out["delta_mae_reduction"] > 0.0) | (out["delta_spearman"] > 0.0),
            ],
            ["supported", "weak/mixed"],
            default="not_supported",
        )
        out = out.sort_values(["dataset", "delta_mae_reduction", "delta_pm1_acc"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def plot_delta_mae(ablation: pd.DataFrame, output_path: Path) -> None:
    if ablation.empty:
        return
    sub = ablation.copy()
    sub["label"] = sub["component"].fillna(sub["candidate_stage"])
    datasets = list(sub["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(9, max(4, 3.8 * len(datasets))), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        g = sub[sub["dataset"].eq(dataset)].sort_values("delta_mae_reduction")
        colors = ["#3f7cac" if value >= 0 else "#c45b4f" for value in g["delta_mae_reduction"]]
        ax.barh(g["label"], g["delta_mae_reduction"], color=colors)
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_xlabel("MAE reduction vs progression control")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    phase_set = load_phase_set(args.phase_set)
    datasets = make_datasets(phase_set, args.vo2_merged)
    alpha_grid = [float(value) for value in args.alpha_grid]

    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    input_rows: list[pd.DataFrame] = []

    for dataset_name, df in datasets.items():
        subjects = sorted(df["folder"].astype(str).unique())
        if len(subjects) < 3:
            continue
        splits = cyclic_subject_splits(subjects)
        split_frame = splits.copy()
        split_frame.insert(0, "dataset", dataset_name)
        split_frames.append(split_frame)
        input_frame = df.copy()
        input_frame.insert(0, "dataset", dataset_name)
        input_rows.append(input_frame)
        for spec in stage_specs(df):
            metric, pred, folds = evaluate_stage(
                df,
                dataset_name,
                splits,
                str(spec["stage"]),
                str(spec["model_type"]),
                list(spec["features"]),
                alpha_grid,
                component=spec.get("component"),
                display_feature=spec.get("display_feature"),
                source=spec.get("source"),
            )
            metric_rows.append(metric)
            pred_frames.append(pred)
            fold_frames.append(folds)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    splits = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
    inputs = pd.concat(input_rows, ignore_index=True, sort=False) if input_rows else pd.DataFrame()
    ablation = build_ablation(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(args.output_dir / "component_ablation_input_sets.csv", index=False)
    splits.to_csv(args.output_dir / "component_subject_split_rotation.csv", index=False)
    metrics.to_csv(args.output_dir / "component_model_summary.csv", index=False)
    folds.to_csv(args.output_dir / "component_fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "component_predictions.csv", index=False)
    ablation.to_csv(args.output_dir / "component_ablation_table.csv", index=False)
    plot_delta_mae(ablation, args.output_dir / "component_ablation_delta_mae.png")

    summary = {
        "phase_set": str(args.phase_set),
        "vo2_merged": str(args.vo2_merged) if args.vo2_merged else None,
        "output_dir": str(args.output_dir),
        "split_protocol": "cyclic subject-disjoint rotation; each fold uses n-2 train subjects, 1 validation subject, and 1 test subject.",
        "datasets": {
            name: {
                "n_rows": int(len(df)),
                "subjects": sorted(df["folder"].astype(str).unique().tolist()),
            }
            for name, df in datasets.items()
        },
        "files": {
            "input_sets": "component_ablation_input_sets.csv",
            "splits": "component_subject_split_rotation.csv",
            "model_summary": "component_model_summary.csv",
            "fold_metrics": "component_fold_metrics.csv",
            "predictions": "component_predictions.csv",
            "ablation": "component_ablation_table.csv",
            "delta_mae_plot": "component_ablation_delta_mae.png",
        },
        "stage_summary": metrics.to_dict(orient="records"),
        "top_ablation": ablation.head(15).to_dict(orient="records") if not ablation.empty else [],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["dataset", "stage", "n_subjects", "n_test_rows", "mae", "spearman", "rounded_exact_acc", "rounded_pm1_acc"]].to_string(index=False))
    print(f"\\nWrote {args.output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RPE ablation using the features shown in the IMU/VO2 fatigue component figure.")
    parser.add_argument(
        "--phase-set",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/024_same_name_xlsx_phase_aware_rpe_20260520/023_phase_aware_set_feature_dataset.csv"),
    )
    parser.add_argument(
        "--vo2-merged",
        type=Path,
        default=Path("artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/025_figure_component_rpe_ablation_20260520"),
    )
    parser.add_argument("--alpha-grid", nargs="*", type=float, default=list(DEFAULT_ALPHA_GRID))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
