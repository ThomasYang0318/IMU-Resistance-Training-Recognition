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
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "borg"
ID_COLUMNS = {"folder", "file", "subject", "exercise", "set_id", "rep_id", "raw_value", "excluded_from_training"}
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
RPE_CLASSES = list(range(1, 11))

PROGRESSION_CONTROLS = ["set_index_numeric", "rep_progress", "cumulative_tut_sec"]

REP_CANDIDATES: list[dict[str, str]] = [
    {"feature": "kg", "family": "load", "claim_role": "load control"},
    {"feature": "n_reps", "family": "set_context", "claim_role": "set context"},
    {"feature": "rep_duration_sec", "family": "rep_tut", "claim_role": "single-rep TUT"},
    {"feature": "concentric_sec", "family": "phase_tut", "claim_role": "phase TUT"},
    {"feature": "eccentric_sec", "family": "phase_tut", "claim_role": "phase TUT"},
    {"feature": "movement_rate", "family": "velocity_proxy", "claim_role": "velocity proxy"},
    {"feature": "velocity_loss_proxy", "family": "velocity_proxy", "claim_role": "velocity loss"},
    {"feature": "duration_gain_from_first2", "family": "fatigue_drift", "claim_role": "duration drift"},
    {"feature": "movement_rate_change_from_first2", "family": "fatigue_drift", "claim_role": "velocity drift"},
    {"feature": "pca_diff_rms", "family": "imu_intensity", "claim_role": "IMU intensity"},
    {"feature": "gyro_mag_diff_rms", "family": "imu_intensity", "claim_role": "gyro intensity"},
    {"feature": "sim_to_first", "family": "waveform_similarity", "claim_role": "waveform consistency"},
    {"feature": "sim_to_prev", "family": "waveform_similarity", "claim_role": "waveform consistency"},
    {"feature": "similarity_decay_from_first", "family": "waveform_similarity", "claim_role": "waveform decay"},
    {"feature": "movement_rate_cv_so_far", "family": "variability", "claim_role": "variability"},
    {"feature": "similarity_std_so_far", "family": "variability", "claim_role": "variability"},
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
    for column in ["folder", "exercise", "set_id", "rep_id", TARGET]:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    df = df[df.get("excluded_from_training", False).eq(False) if "excluded_from_training" in df.columns else slice(None)].copy()
    df["folder"] = df["folder"].astype(str)
    df["exercise"] = df["exercise"].astype(str)
    df["set_id"] = df["set_id"].map(normalize_set_id)
    df["rep_id"] = df["rep_id"].map(normalize_set_id)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    for col in df.columns:
        if col not in ID_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rep_progress" not in df.columns:
        df = add_rep_progress(df)
    return df.reset_index(drop=True)


def add_rep_progress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rep_index"] = pd.to_numeric(out["rep_index"], errors="coerce")
    rows: list[pd.DataFrame] = []
    for _, group in out.groupby(["folder", "exercise", "set_id"], sort=False):
        group = group.sort_values("rep_index").copy()
        n_reps = len(group)
        group["n_reps"] = n_reps
        group["rep_order"] = np.arange(n_reps)
        group["rep_progress"] = group["rep_order"] / max(n_reps - 1, 1)
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def available_features(train: pd.DataFrame, features: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for feature in features:
        if feature not in train.columns:
            continue
        series = pd.to_numeric(train[feature], errors="coerce")
        if series.notna().sum() >= 20 and series.nunique(dropna=True) > 1:
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
    means = train.groupby("exercise")[TARGET].mean().to_dict()
    return test["exercise"].map(means).fillna(global_mean).astype(float).to_numpy()


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
            best_alpha = float(alpha)
            best_mae = float(mae)
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

        if model_type == "exercise_mean":
            selected_alpha = np.nan
            val_pred = predict_exercise_mean(train, val)
            test_pred = predict_exercise_mean(train, test)
            n_features = 0
            val_selection_mae = mean_absolute_error(val[TARGET].to_numpy(dtype=float), val_pred)
        else:
            selected_alpha, val_selection_mae, _ = tune_alpha(train, val, features, alpha_grid)
            x_train, x_test, columns = build_matrix(train, test, features)
            n_features = len(columns)
            model = ridge_model(selected_alpha)
            model.fit(x_train, train[TARGET].to_numpy(dtype=float))
            test_pred = model.predict(x_test)
            x_train_val, x_val, _ = build_matrix(train, val, features)
            val_model = ridge_model(selected_alpha)
            val_model.fit(x_train_val, train[TARGET].to_numpy(dtype=float))
            val_pred = val_model.predict(x_val)

        val_score = regression_scores(val[TARGET].to_numpy(dtype=float), val_pred)
        test_score = regression_scores(test[TARGET].to_numpy(dtype=float), test_pred)
        fold_rows.append(
            {
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
                    "stage": stage,
                    "model_type": model_type,
                    "fold": int(split.fold),
                    "train_subjects": split.train_subjects,
                    "val_subject": split.val_subject,
                    "test_subject": split.test_subject,
                    "folder": row.folder,
                    "exercise": row.exercise,
                    "set_id": row.set_id,
                    "rep_id": row.rep_id,
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
        "selected_alpha_median": float(fold_df["selected_alpha"].median()) if fold_df["selected_alpha"].notna().any() else np.nan,
        "n_features_median": float(fold_df["n_features"].median()) if "n_features" in fold_df else np.nan,
        **regression_scores(pred_df["borg_true"], pred_df["borg_pred"]),
    }
    return metric, pred_df, fold_df


def build_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["stage"].eq("M3_progression_control")]
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    best_progression = metrics[metrics["stage"].isin(["M1_exercise_plus_rep_progress", "M2_exercise_plus_cumulative_tut", "M3_progression_control"])].sort_values(["mae", "stage"]).iloc[0]
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


def write_confusion(predictions: pd.DataFrame, stage: str, output_dir: Path) -> None:
    sub = predictions[predictions["stage"].eq(stage)].copy()
    if sub.empty:
        return
    y_true = np.clip(np.rint(sub["borg_true"].to_numpy(dtype=float)), 1, 10).astype(int)
    y_pred = np.clip(np.rint(sub["borg_pred"].to_numpy(dtype=float)), 1, 10).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=RPE_CLASSES)
    row_sums = cm.sum(axis=1, keepdims=True)
    prop = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    pd.DataFrame(cm, index=RPE_CLASSES, columns=RPE_CLASSES).to_csv(output_dir / f"{stage}_rpe_1to10_confusion_matrix.csv")
    pd.DataFrame(prop, index=RPE_CLASSES, columns=RPE_CLASSES).to_csv(output_dir / f"{stage}_rpe_1to10_confusion_matrix_row_proportion.csv")

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    im = ax.imshow(prop, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Rep-level RPE 1-10 confusion matrix\\n{stage}, row proportion")
    ax.set_xlabel("Predicted RPE")
    ax.set_ylabel("True RPE")
    ax.set_xticks(range(len(RPE_CLASSES)), labels=RPE_CLASSES)
    ax.set_yticks(range(len(RPE_CLASSES)), labels=RPE_CLASSES)
    for r in range(len(RPE_CLASSES)):
        for c in range(len(RPE_CLASSES)):
            if cm[r, c] > 0:
                color = "white" if prop[r, c] > 0.55 else "black"
                ax.text(c, r, f"{prop[r, c]:.2f}\\n({cm[r, c]})", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stage}_rpe_1to10_confusion_matrix_row_proportion.png", dpi=180)
    plt.close(fig)


def plot_stage_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    stages = [
        "M0_exercise_mean",
        "M1_exercise_plus_rep_progress",
        "M2_exercise_plus_cumulative_tut",
        "M3_progression_control",
    ]
    sub = metrics[metrics["stage"].isin(stages)].set_index("stage").loc[stages].reset_index()
    labels = ["Exercise", "+ rep progress", "+ cumulative TUT", "progression"]
    x = np.arange(len(sub))
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
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
    ax1.set_title("Rep-level RPE subject-disjoint rotation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    df = load_table(args.input)
    if args.subjects:
        df = df[df["folder"].isin(set(args.subjects))].copy()
    subjects = sorted(df["folder"].astype(str).unique())
    if len(subjects) < 3:
        raise ValueError("Need at least three subjects for disjoint train/validation/test splits.")
    splits = cyclic_subject_splits(subjects)
    alpha_grid = [float(value) for value in args.alpha_grid]

    stages: list[tuple[str, str, list[str], str | None, str | None, str | None]] = [
        ("M0_exercise_mean", "exercise_mean", [], None, None, None),
        ("M1_exercise_plus_rep_progress", "ridge", ["rep_progress"], None, None, None),
        ("M2_exercise_plus_cumulative_tut", "ridge", ["cumulative_tut_sec"], None, None, None),
        ("M3_progression_control", "ridge", list(PROGRESSION_CONTROLS), None, None, None),
    ]
    for spec in REP_CANDIDATES:
        feature = spec["feature"]
        if feature in PROGRESSION_CONTROLS or feature not in df.columns:
            continue
        stages.append((f"M4_add_one__{feature}", "ridge", list(PROGRESSION_CONTROLS) + [feature], feature, spec["family"], spec["claim_role"]))

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
    folds = pd.concat(fold_frames, ignore_index=True)
    ablation = build_ablation_table(metrics)
    best_stage = str(metrics.sort_values(["mae", "stage"]).iloc[0]["stage"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "input_rep_level_7subject_features.csv", index=False)
    splits.to_csv(args.output_dir / "subject_split_rotation.csv", index=False)
    metrics.to_csv(args.output_dir / "rep_rpe_rotation_model_summary.csv", index=False)
    folds.to_csv(args.output_dir / "rep_rpe_rotation_fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "rep_rpe_rotation_predictions.csv", index=False)
    ablation.to_csv(args.output_dir / "rep_rpe_rotation_ablation_table.csv", index=False)
    plot_stage_comparison(metrics, args.output_dir / "rep_rpe_rotation_stage_comparison.png")
    write_confusion(predictions, best_stage, args.output_dir)

    label_counts = df[TARGET].round().astype(int).clip(1, 10).value_counts().reindex(RPE_CLASSES, fill_value=0).sort_index()
    label_counts.to_csv(args.output_dir / "rep_rpe_label_counts_1to10.csv", header=["count"])

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "split_protocol": "7 cyclic subject rotations; each fold uses 5 train subjects, 1 validation subject, and 1 test subject.",
        "subjects": subjects,
        "n_rows": int(len(df)),
        "best_stage_by_mae": best_stage,
        "label_counts_1to10": {int(k): int(v) for k, v in label_counts.items()},
        "files": {
            "input_rep_level": "input_rep_level_7subject_features.csv",
            "splits": "subject_split_rotation.csv",
            "model_summary": "rep_rpe_rotation_model_summary.csv",
            "fold_metrics": "rep_rpe_rotation_fold_metrics.csv",
            "predictions": "rep_rpe_rotation_predictions.csv",
            "ablation": "rep_rpe_rotation_ablation_table.csv",
            "stage_comparison": "rep_rpe_rotation_stage_comparison.png",
        },
        "stage_summary": metrics.to_dict(orient="records"),
        "top_ablation": ablation.head(10).to_dict(orient="records") if not ablation.empty else [],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["stage", "n_test_rows", "mae", "spearman", "rounded_exact_acc", "rounded_pm1_acc"]].to_string(index=False))
    print(f"\\nBest stage by MAE: {best_stage}")
    print(f"Wrote {args.output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate rep-level RPE models with subject-disjoint train/validation/test rotations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/020_same_name_xlsx_7subject_set_features_20260520/020_rpe_rep_level_feature_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/023_rep_level_rpe_subject_rotation_20260520"),
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--alpha-grid", nargs="*", type=float, default=list(DEFAULT_ALPHA_GRID))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
