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
KEYS = ("folder", "exercise", "set_id")
PROGRESSION_CONTROLS = ["set_index_numeric", "cumulative_tut_exercise_sec"]
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

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


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for key in KEYS:
        if key not in df.columns:
            raise ValueError(f"Missing required column: {key}")
    df["folder"] = df["folder"].astype(str)
    df["exercise"] = df["exercise"].astype(str)
    df["set_id"] = df["set_id"].map(normalize_set_id)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for col in df.columns:
        if col not in {"folder", "exercise", "set_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_cumulative_features(df.reset_index(drop=True))


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    if "total_tut_sec" not in df.columns:
        raise ValueError("Need total_tut_sec to compute cumulative_tut_exercise_sec.")
    out = df.copy()
    if "set_index_numeric" not in out.columns:
        out["set_index_numeric"] = pd.to_numeric(out["set_id"], errors="coerce")
    out["_original_order"] = np.arange(len(out))
    out = out.sort_values(["folder", "exercise", "set_index_numeric", "set_id"], kind="mergesort")
    out["cumulative_tut_exercise_sec"] = out.groupby(["folder", "exercise"], sort=False)["total_tut_sec"].cumsum()
    out = out.sort_values("_original_order").drop(columns="_original_order")
    return out.reset_index(drop=True)


def available_features(train: pd.DataFrame, features: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for feature in features:
        if feature not in train.columns:
            continue
        series = pd.to_numeric(train[feature], errors="coerce")
        if series.notna().sum() >= 8 and series.nunique(dropna=True) > 1:
            kept.append(feature)
    return kept


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
    splits: pd.DataFrame,
    stage: str,
    model_type: str,
    features: Sequence[str],
    alpha_grid: Sequence[float],
    added_feature: str | None = None,
    added_feature_family: str | None = None,
    claim_role: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for split in splits.itertuples(index=False):
        train_subjects = str(split.train_subjects).split("|") if split.train_subjects else []
        train = df[df["folder"].isin(train_subjects)].copy()
        val = df[df["folder"].eq(split.val_subject)].copy()
        test = df[df["folder"].eq(split.test_subject)].copy()
        truth_val = val[TARGET].to_numpy(dtype=float)
        truth_test = test[TARGET].to_numpy(dtype=float)

        if model_type == "exercise_mean":
            selected_alpha = np.nan
            n_features = 0
            val_pred = predict_exercise_mean(train, val)
            test_pred = predict_exercise_mean(train, test)
            val_mae = mean_absolute_error(truth_val, val_pred)
        else:
            selected_alpha, val_mae, n_features = tune_alpha(train, val, features, alpha_grid)
            x_train, x_test, columns = build_matrix(train, test, features)
            n_features = len(columns)
            model = ridge_model(selected_alpha)
            model.fit(x_train, train[TARGET].to_numpy(dtype=float))
            test_pred = model.predict(x_test)
            x_train_val, x_val, _ = build_matrix(train, val, features)
            val_model = ridge_model(selected_alpha)
            val_model.fit(x_train_val, train[TARGET].to_numpy(dtype=float))
            val_pred = val_model.predict(x_val)

        val_score = regression_scores(truth_val, val_pred)
        test_score = regression_scores(truth_test, test_pred)
        fold_rows.append(
            {
                "stage": stage,
                "model_type": model_type,
                "fold": int(split.fold),
                "train_subjects": split.train_subjects,
                "val_subject": split.val_subject,
                "test_subject": split.test_subject,
                "selected_alpha": selected_alpha,
                "val_selection_mae": val_mae,
                "n_features": int(n_features),
                **{f"val_{key}": value for key, value in val_score.items()},
                **{f"test_{key}": value for key, value in test_score.items()},
            }
        )
        for row, pred in zip(test.itertuples(index=False), test_pred):
            pred_rows.append(
                {
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
                    "added_feature": added_feature,
                    "added_feature_family": added_feature_family,
                    "claim_role": claim_role,
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)
    metric = {
        "stage": stage,
        "model_type": model_type,
        "n_subjects": int(df["folder"].nunique()),
        "n_test_rows": int(len(pred_df)),
        "added_feature": added_feature,
        "added_feature_family": added_feature_family,
        "claim_role": claim_role,
        "selected_alpha_median": float(fold_df["selected_alpha"].median()) if "selected_alpha" in fold_df and fold_df["selected_alpha"].notna().any() else np.nan,
        "n_features_median": float(fold_df["n_features"].median()) if "n_features" in fold_df else np.nan,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, fold_df


def build_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["stage"].eq("M3_progression_control")]
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    progression = metrics[metrics["stage"].isin(["M1_exercise_plus_set_index", "M2_exercise_plus_cumulative_tut", "M3_progression_control"])]
    best_progression = progression.sort_values(["mae", "stage"]).iloc[0]
    rows: list[dict[str, object]] = []
    for _, row in metrics[metrics["stage"].str.startswith("M4_add_one__", na=False)].iterrows():
        rows.append(
            {
                "baseline_stage": "M3_progression_control",
                "candidate_stage": row["stage"],
                "added_feature": row["added_feature"],
                "family": row["added_feature_family"],
                "claim_role": row["claim_role"],
                "baseline_mae": base_row["mae"],
                "candidate_mae": row["mae"],
                "delta_mae_reduction": base_row["mae"] - row["mae"],
                "baseline_spearman": base_row["spearman"],
                "candidate_spearman": row["spearman"],
                "delta_spearman": row["spearman"] - base_row["spearman"],
                "baseline_pm1_acc": base_row["rounded_pm1_acc"],
                "candidate_pm1_acc": row["rounded_pm1_acc"],
                "delta_pm1_acc": row["rounded_pm1_acc"] - base_row["rounded_pm1_acc"],
                "best_progression_stage": best_progression["stage"],
                "best_progression_mae": best_progression["mae"],
                "delta_mae_vs_best_progression": best_progression["mae"] - row["mae"],
                "best_progression_pm1_acc": best_progression["rounded_pm1_acc"],
                "delta_pm1_acc_vs_best_progression": row["rounded_pm1_acc"] - best_progression["rounded_pm1_acc"],
                "n_test_rows": row["n_test_rows"],
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
        out = out.sort_values(["delta_mae_reduction", "delta_pm1_acc"], ascending=False).reset_index(drop=True)
    return out


def plot_stage_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    stages = [
        "M0_exercise_mean",
        "M1_exercise_plus_set_index",
        "M2_exercise_plus_cumulative_tut",
        "M3_progression_control",
    ]
    sub = metrics[metrics["stage"].isin(stages)].set_index("stage").loc[stages].reset_index()
    labels = ["Exercise", "+ set", "+ cumulative TUT", "set + TUT"]
    x = np.arange(len(sub))
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(x, sub["rounded_pm1_acc"], marker="o", label="Within +/-1 accuracy")
    ax1.plot(x, sub["rounded_exact_acc"], marker="o", label="Exact accuracy")
    ax1.plot(x, sub["spearman"], marker="o", label="Spearman")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, sub["mae"], marker="s", color="#cc4c02", label="MAE")
    ax2.set_ylabel("MAE")
    ax2.set_ylim(0, max(2.5, float(sub["mae"].max()) * 1.15))
    lines1, names1 = ax1.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, names1 + names2, loc="upper center", ncol=2, frameon=False)
    ax1.set_title("RPE ablation with disjoint train/validation/test subject rotation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    df = load_table(args.input)
    if args.subjects:
        keep = set(args.subjects)
        df = df[df["folder"].isin(keep)].copy()
    subjects = sorted(df["folder"].astype(str).unique())
    if len(subjects) < 3:
        raise ValueError("Need at least three subjects for disjoint train/validation/test splits.")
    splits = cyclic_subject_splits(subjects)

    alpha_grid = [float(value) for value in args.alpha_grid]
    stages: list[tuple[str, str, list[str], str | None, str | None, str | None]] = [
        ("M0_exercise_mean", "exercise_mean", [], None, None, None),
        ("M1_exercise_plus_set_index", "ridge", ["set_index_numeric"], None, None, None),
        ("M2_exercise_plus_cumulative_tut", "ridge", ["cumulative_tut_exercise_sec"], None, None, None),
        ("M3_progression_control", "ridge", list(PROGRESSION_CONTROLS), None, None, None),
    ]
    base_features = list(PROGRESSION_CONTROLS)
    for spec in RPE_CANDIDATES:
        feature = spec["feature"]
        if feature in base_features or feature not in df.columns:
            continue
        stages.append((f"M4_add_one__{feature}", "ridge", base_features + [feature], feature, spec["family"], spec["claim_role"]))

    metric_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for stage, model_type, features, added_feature, family, claim_role in stages:
        metric, preds, folds = evaluate_stage(
            df,
            splits,
            stage,
            model_type,
            features,
            alpha_grid,
            added_feature=added_feature,
            added_feature_family=family,
            claim_role=claim_role,
        )
        metric_rows.append(metric)
        pred_frames.append(preds)
        fold_frames.append(folds)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    ablation = build_ablation_table(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "input_set_level_7subject_features.csv", index=False)
    splits.to_csv(args.output_dir / "subject_split_rotation.csv", index=False)
    metrics.to_csv(args.output_dir / "rpe_rotation_model_summary.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "rpe_rotation_fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "rpe_rotation_predictions.csv", index=False)
    ablation.to_csv(args.output_dir / "rpe_rotation_ablation_table.csv", index=False)
    plot_stage_comparison(metrics, args.output_dir / "rpe_rotation_stage_comparison.png")

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "split_protocol": "7 cyclic subject rotations; each fold uses 5 train subjects, 1 validation subject, and 1 test subject.",
        "validation_usage": "Ridge alpha is selected on the validation subject. The reported test metrics are computed on the held-out test subject only.",
        "subjects": subjects,
        "n_rows": int(len(df)),
        "files": {
            "input_set_level": "input_set_level_7subject_features.csv",
            "splits": "subject_split_rotation.csv",
            "model_summary": "rpe_rotation_model_summary.csv",
            "fold_metrics": "rpe_rotation_fold_metrics.csv",
            "predictions": "rpe_rotation_predictions.csv",
            "ablation": "rpe_rotation_ablation_table.csv",
            "stage_comparison": "rpe_rotation_stage_comparison.png",
        },
        "stage_summary": metrics.to_dict(orient="records"),
        "top_ablation": ablation.head(10).to_dict(orient="records") if not ablation.empty else [],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["stage", "n_test_rows", "mae", "spearman", "rounded_exact_acc", "rounded_pm1_acc"]].to_string(index=False))
    print(f"\\nWrote {args.output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-disjoint train/validation/test rotations for set-level RPE ablation.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/020_same_name_xlsx_7subject_set_features_20260520/020_rpe_set_level_feature_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/021_rpe_subject_rotation_ablation_20260520"),
    )
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional explicit subject folders to keep.")
    parser.add_argument("--alpha-grid", nargs="*", type=float, default=list(DEFAULT_ALPHA_GRID))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
