from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_literature_inspired_rep_methods import (  # noqa: E402
    ACC_COLUMNS,
    GYRO_COLUMNS,
    IMU9_COLUMNS,
    MAG_COLUMNS,
    infer_sensor_period_seconds,
    principal_signal,
    read_session_9axis,
)
from tools.evaluate_rep_segmentation_classification import (  # noqa: E402
    PhaseSegment,
    RepSegment,
    true_phase_segments,
    true_rep_segments,
    whole_session_files,
)


WORKBOOK_SKIP = {
    "DataAverage.xlsx",
    "Report.xlsx",
    "SessionDiagnostics.xlsx",
    "VO2MasterUnit-Data.xlsx",
    "VO2 Master 5258Diagnostics.xlsx",
    "SpeedCadenceUnit-Data.xlsx",
}

EXERCISE_ORDER = (
    "db_bench_press",
    "one_arm_db_row",
    "db_rdl",
    "db_weighted_crunch",
    "db_shoulder_press",
    "db_biceps_curl",
    "db_triceps_curl",
    "db_squat",
)


def is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def numeric_or_none(value: object) -> float | None:
    if is_blank(value):
        return None
    text = str(value).strip()
    if text.upper() == "X":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def workbook_for_folder(folder: Path) -> Path | None:
    same_name = folder / f"{folder.name}.xlsx"
    if same_name.exists():
        return same_name
    candidates = [path for path in folder.glob("*.xlsx") if path.name not in WORKBOOK_SKIP]
    return candidates[0] if candidates else None


def read_borg_targets(data_dirs: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    workbook_rows: list[dict[str, object]] = []
    for root in data_dirs:
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            workbook = workbook_for_folder(folder)
            if workbook is None:
                workbook_rows.append({"folder": folder.name, "workbook": "", "status": "missing"})
                continue
            try:
                raw = pd.read_excel(workbook, sheet_name=0)
            except Exception as exc:
                workbook_rows.append({"folder": folder.name, "workbook": str(workbook), "status": f"read_error: {exc}"})
                continue
            if raw.empty:
                workbook_rows.append({"folder": folder.name, "workbook": str(workbook), "status": "empty"})
                continue

            first_col = raw.columns[0]
            kg_col = next((col for col in raw.columns if str(col).strip().lower() == "kg"), None)
            rep_cols = [col for col in raw.columns if str(col).strip().replace(".0", "").isdigit()]
            valid_count = 0
            for row in raw.itertuples(index=False):
                row_name = getattr(row, str(first_col).replace(" ", "_"), None)
                # itertuples renames invalid columns; direct Series is simpler and safer.
            for _, series in raw.iterrows():
                row_name = series.get(first_col)
                if is_blank(row_name):
                    continue
                match = re.match(r"^\s*(\d+)_([0-9]+)\s*$", str(row_name))
                if not match:
                    continue
                exercise_index = int(match.group(1))
                if exercise_index < 1 or exercise_index > len(EXERCISE_ORDER):
                    continue
                exercise = EXERCISE_ORDER[exercise_index - 1]
                set_id = str(int(match.group(2)))
                kg = numeric_or_none(series.get(kg_col)) if kg_col is not None else None

                last_borg: float | None = None
                stopped = False
                for rep_col in sorted(rep_cols, key=lambda col: int(str(col).strip().replace(".0", ""))):
                    rep_id = str(int(str(rep_col).strip().replace(".0", "")))
                    value = series.get(rep_col)
                    if isinstance(value, str) and value.strip().upper() == "X":
                        stopped = True
                        rows.append(
                            {
                                "folder": folder.name,
                                "workbook": str(workbook),
                                "exercise": exercise,
                                "set_id": set_id,
                                "rep_id": rep_id,
                                "kg": kg,
                                "borg": np.nan,
                                "completed": False,
                                "raw_value": "X",
                            }
                        )
                        break
                    parsed = numeric_or_none(value)
                    if parsed is not None:
                        last_borg = parsed
                    if last_borg is None:
                        continue
                    rows.append(
                        {
                            "folder": folder.name,
                            "workbook": str(workbook),
                            "exercise": exercise,
                            "set_id": set_id,
                            "rep_id": rep_id,
                            "kg": kg,
                            "borg": last_borg,
                            "completed": True,
                            "raw_value": "" if is_blank(value) else str(value).strip(),
                        }
                    )
                    valid_count += 1
                if stopped:
                    continue
            workbook_rows.append(
                {
                    "folder": folder.name,
                    "workbook": str(workbook),
                    "status": "ok" if valid_count else "no_targets",
                    "target_reps": valid_count,
                }
            )
    targets = pd.DataFrame(rows)
    workbooks = pd.DataFrame(workbook_rows)
    if not targets.empty:
        targets = targets.drop_duplicates(["folder", "exercise", "set_id", "rep_id"], keep="last").reset_index(drop=True)
    return targets, workbooks


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    scale = float(np.std(values))
    if scale < 1e-9:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / scale


def resample(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(points, dtype=np.float64)
    if len(values) == 1:
        return np.full(points, float(values[0]), dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(values))
    dst = np.linspace(0.0, 1.0, points)
    return np.interp(dst, src, values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def signal_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_diff_rms": 0.0,
            f"{prefix}_diff_abs_mean": 0.0,
            f"{prefix}_slope": 0.0,
        }
    diff = np.diff(values) if len(values) > 1 else np.zeros(1, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, len(values))
    slope = 0.0
    denom = float(np.sum((x - np.mean(x)) ** 2))
    if denom > 1e-12:
        slope = float(np.sum((x - np.mean(x)) * (values - np.mean(values))) / denom)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_diff_rms": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_diff_abs_mean": float(np.mean(np.abs(diff))),
        f"{prefix}_slope": slope,
    }


def magnitude(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    available = [col for col in columns if col in df.columns]
    if not available:
        return np.zeros(len(df), dtype=np.float64)
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    return np.linalg.norm(zscore(x), axis=1)


def phase_duration_features(rep: RepSegment, phases: Sequence[PhaseSegment], period: float) -> dict[str, float]:
    rows = [
        phase
        for phase in phases
        if phase.file_path == rep.file_path
        and phase.subject == rep.subject
        and phase.exercise == rep.exercise
        and str(phase.set_id) == str(rep.set_id)
        and str(phase.rep_id) == str(rep.rep_id)
    ]
    concentric_samples = sum(phase.n_samples for phase in rows if phase.phase == "concentric")
    eccentric_samples = sum(phase.n_samples for phase in rows if phase.phase == "eccentric")
    total = max(rep.n_samples, 1)
    return {
        "rep_duration_sec": rep.n_samples * period,
        "concentric_sec": concentric_samples * period,
        "eccentric_sec": eccentric_samples * period,
        "concentric_ratio": concentric_samples / float(total),
        "eccentric_ratio": eccentric_samples / float(total),
        "phase_balance_abs": abs(concentric_samples - eccentric_samples) / float(total),
    }


def extract_gt_feature_rows(data_dirs: Sequence[Path], resample_points: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    waveforms: dict[tuple[str, str, str, str], list[tuple[int, np.ndarray]]] = {}
    for path in whole_session_files(data_dirs):
        folder = path.parent.name
        df = read_session_9axis(path, data_dirs)
        period = infer_sensor_period_seconds(df)
        reps = true_rep_segments(df, path, min_samples=10)
        phases = true_phase_segments(df, path, min_samples=5)
        for rep in reps:
            local = df.iloc[rep.start : rep.end].reset_index(drop=True)
            pca = zscore(principal_signal(local, smooth_window=9, columns=IMU9_COLUMNS))
            pca_wave = resample(pca, resample_points)
            row: dict[str, object] = {
                "folder": folder,
                "file": str(path),
                "subject": rep.subject,
                "exercise": rep.exercise,
                "set_id": str(rep.set_id),
                "rep_id": str(rep.rep_id),
                "rep_index": int(float(rep.rep_id)) if str(rep.rep_id).replace(".", "", 1).isdigit() else 0,
                "start": rep.start,
                "end": rep.end,
            }
            row.update(phase_duration_features(rep, phases, period))
            row.update(signal_stats("pca", pca))
            row.update(signal_stats("acc_mag", magnitude(local, ACC_COLUMNS)))
            row.update(signal_stats("gyro_mag", magnitude(local, GYRO_COLUMNS)))
            row.update(signal_stats("mag_mag", magnitude(local, MAG_COLUMNS)))
            for col in IMU9_COLUMNS:
                if col in local.columns:
                    row.update(signal_stats(col, zscore(local[col].to_numpy(dtype=np.float64))))
            key = (folder, rep.exercise, str(rep.set_id), str(path))
            waveforms.setdefault(key, []).append((row["rep_index"], pca_wave))
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        return feature_df

    feature_df["sim_to_first"] = 0.0
    feature_df["sim_to_prev"] = 0.0
    lookup_index = {
        (row.folder, row.exercise, str(row.set_id), str(row.file), int(row.rep_index)): idx
        for idx, row in feature_df.iterrows()
    }
    for key, items in waveforms.items():
        items = sorted(items, key=lambda item: item[0])
        if not items:
            continue
        first_wave = items[0][1]
        prev_wave: np.ndarray | None = None
        for rep_index, wave in items:
            idx = lookup_index.get((*key[:3], key[3], int(rep_index)))
            if idx is None:
                continue
            feature_df.loc[idx, "sim_to_first"] = cosine_similarity(wave, first_wave)
            feature_df.loc[idx, "sim_to_prev"] = cosine_similarity(wave, prev_wave) if prev_wave is not None else 1.0
            prev_wave = wave
    return feature_df


def prepare_dataset(features: pd.DataFrame, targets: pd.DataFrame, excluded_training_folders: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = targets[targets["completed"].eq(True) & targets["borg"].notna()].copy()
    merged = features.merge(
        targets[["folder", "exercise", "set_id", "rep_id", "kg", "borg", "raw_value"]],
        on=["folder", "exercise", "set_id", "rep_id"],
        how="inner",
    )
    merged["borg"] = pd.to_numeric(merged["borg"], errors="coerce")
    merged["kg"] = pd.to_numeric(merged["kg"], errors="coerce")
    merged = merged[merged["borg"].notna()].copy()
    merged["excluded_from_training"] = merged["folder"].isin(excluded_training_folders)
    return merged, targets


def feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    metadata = ["rep_index", "kg"]
    tut = ["rep_duration_sec", "concentric_sec", "eccentric_sec", "concentric_ratio", "eccentric_ratio", "phase_balance_abs"]
    waveform = [
        col
        for col in df.columns
        if (
            col.startswith("pca_")
            or col.startswith("acc_mag_")
            or col.startswith("gyro_mag_")
            or col.startswith("mag_mag_")
            or col.startswith("ax_")
            or col.startswith("ay_")
            or col.startswith("az_")
            or col.startswith("gx_")
            or col.startswith("gy_")
            or col.startswith("gz_")
            or col.startswith("mx_")
            or col.startswith("my_")
            or col.startswith("mz_")
            or col in {"sim_to_first", "sim_to_prev"}
        )
    ]
    if mode == "metadata":
        cols = metadata
    elif mode == "tut":
        cols = metadata + tut
    elif mode == "waveform":
        cols = metadata + waveform
    elif mode == "combined":
        cols = metadata + tut + waveform
    else:
        raise ValueError(mode)
    return [col for col in dict.fromkeys(cols) if col in df.columns]


def design_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    x = df.loc[:, columns].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    exercise_dummies = pd.get_dummies(df["exercise"].astype(str), prefix="exercise", dtype=float)
    return pd.concat([x.reset_index(drop=True), exercise_dummies.reset_index(drop=True)], axis=1)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.rint(y_pred), 0, 11)
    with np.errstate(invalid="ignore"):
        rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4) if len(set(np.round(y_true, 6))) > 1 else 0.0,
        "spearman": round(float(rho), 4) if not np.isnan(rho) else 0.0,
        "rounded_exact_acc": round(float(np.mean(rounded == y_true)), 4),
        "rounded_pm1_acc": round(float(np.mean(np.abs(rounded - y_true) <= 1.0)), 4),
    }


def cross_subject_eval(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    trainable = df[~df["excluded_from_training"]].copy()
    trainable = trainable[trainable["folder"].notna()].copy()
    groups = trainable["folder"].astype(str).to_numpy()
    unique_groups = sorted(set(groups.tolist()))
    folds = min(args.folds, len(unique_groups))
    if folds < 2:
        raise ValueError("Need at least two folders with Borg targets for GroupKFold.")

    y = trainable["borg"].to_numpy(dtype=float)
    fold_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    group_mean = trainable.groupby("folder")["borg"].mean().to_dict()
    for mode in ["metadata", "tut", "waveform", "combined"]:
        cols = feature_columns(trainable, mode)
        x = design_matrix(trainable, cols)
        splitter = GroupKFold(n_splits=folds)
        for model_name in ["ridge", "random_forest"]:
            y_true_all: list[float] = []
            y_pred_all: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
                if model_name == "ridge":
                    model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
                else:
                    model = RandomForestRegressor(
                        n_estimators=args.n_estimators,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        random_state=args.seed + fold_idx,
                        n_jobs=-1,
                    )
                model.fit(x.iloc[train_idx], y[train_idx])
                pred = model.predict(x.iloc[val_idx])
                y_true_all.extend(y[val_idx].tolist())
                y_pred_all.extend(pred.tolist())
                val_meta = trainable.iloc[val_idx]
                for row, truth, value in zip(val_meta.itertuples(index=False), y[val_idx], pred):
                    pred_rows.append(
                        {
                            "feature_mode": mode,
                            "model": model_name,
                            "fold": fold_idx,
                            "folder": row.folder,
                            "subject": row.subject,
                            "exercise": row.exercise,
                            "set_id": row.set_id,
                            "rep_id": row.rep_id,
                            "kg": row.kg,
                            "borg_true": truth,
                            "borg_pred": round(float(value), 4),
                            "abs_error": round(abs(float(value) - float(truth)), 4),
                        }
                    )
            scores = regression_scores(np.asarray(y_true_all), np.asarray(y_pred_all))
            fold_rows.append(
                {
                    "feature_mode": mode,
                    "model": model_name,
                    "n_reps": len(y_true_all),
                    "subjects": len(unique_groups),
                    **scores,
                }
            )

    mean_pred = np.full_like(y, float(np.mean(y)), dtype=float)
    fold_rows.append({"feature_mode": "baseline", "model": "global_mean", "n_reps": len(y), "subjects": len(unique_groups), **regression_scores(y, mean_pred)})
    exercise_mean = trainable.groupby("exercise")["borg"].mean()
    ex_pred = trainable["exercise"].map(exercise_mean).fillna(float(np.mean(y))).to_numpy(dtype=float)
    fold_rows.append({"feature_mode": "baseline", "model": "exercise_mean_in_sample", "n_reps": len(y), "subjects": len(unique_groups), **regression_scores(y, ex_pred)})
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def plot_scores(summary: pd.DataFrame, output_dir: Path) -> None:
    subset = summary[summary["feature_mode"].ne("baseline")].copy()
    if subset.empty:
        return
    labels = [f"{row.feature_mode}\n{row.model}" for row in subset.itertuples(index=False)]
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(max(10, len(subset) * 1.2), 5))
    ax.bar(x, subset["mae"].to_numpy(dtype=float), color="#5276a7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Borg MAE")
    ax.set_title("Borg/RPE Prediction from Ground-Truth Segmented Waveform Features")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "018_borg_prediction_mae.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(subset) * 1.2), 5))
    ax.bar(x, subset["rounded_pm1_acc"].to_numpy(dtype=float), color="#5f9f74")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy within +/-1 Borg")
    ax.set_title("Borg/RPE +/-1 Accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "018_borg_prediction_pm1_accuracy.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether GT-segmented IMU waveform/TUT features predict Borg/RPE labels.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/018_borg_gt_waveform_relation"))
    parser.add_argument("--exclude-training-folders", nargs="*", default=["thomas0506workout"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--resample-points", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets, workbook_status = read_borg_targets(args.data_dirs)
    features = extract_gt_feature_rows(args.data_dirs, args.resample_points)
    merged, cleaned_targets = prepare_dataset(features, targets, set(args.exclude_training_folders))

    targets.to_csv(args.output_dir / "018_borg_targets_raw.csv", index=False)
    cleaned_targets.to_csv(args.output_dir / "018_borg_targets_completed.csv", index=False)
    workbook_status.to_csv(args.output_dir / "018_borg_workbook_status.csv", index=False)
    features.to_csv(args.output_dir / "018_gt_rep_waveform_features_all.csv", index=False)
    merged.to_csv(args.output_dir / "018_gt_rep_waveform_borg_dataset.csv", index=False)

    if merged.empty:
        raise SystemExit("No merged Borg targets and GT waveform features found.")

    summary, predictions = cross_subject_eval(merged, args)
    summary.to_csv(args.output_dir / "018_borg_prediction_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "018_borg_prediction_predictions.csv", index=False)
    by_exercise = (
        predictions[predictions["feature_mode"].eq("combined") & predictions["model"].eq("random_forest")]
        .groupby("exercise")
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), pm1=("abs_error", lambda values: float(np.mean(values <= 1.0))))
        .reset_index()
    )
    by_exercise.to_csv(args.output_dir / "018_borg_prediction_by_exercise_combined_rf.csv", index=False)
    plot_scores(summary, args.output_dir)

    summary_json = {
        "output_dir": str(args.output_dir),
        "target_reps_raw": int(len(targets)),
        "target_reps_completed": int(len(cleaned_targets)),
        "merged_gt_reps": int(len(merged)),
        "trainable_folders": sorted(set(merged.loc[~merged["excluded_from_training"], "folder"].astype(str))),
        "excluded_training_folders": sorted(args.exclude_training_folders),
        "summary_csv": str(args.output_dir / "018_borg_prediction_summary.csv"),
        "notes": {
            "x_handling": "X is treated as not completed and excluded from Borg target training.",
            "blank_handling": "Blank Borg cells are forward-filled from the previous rep value in the same set row.",
            "thomas": "thomas0506workout is excluded from training because no same-name Borg workbook exists.",
            "upper_bound": "This uses ground-truth rep/phase segmentation, so it tests whether waveform/TUT features contain Borg signal before automatic segmentation noise.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.sort_values(["feature_mode", "model"]).to_string(index=False))
    print()
    print("Merged dataset:", len(merged), "reps")
    print("Trainable folders:", summary_json["trainable_folders"])
    print("Excluded training folders:", summary_json["excluded_training_folders"])


if __name__ == "__main__":
    main()
