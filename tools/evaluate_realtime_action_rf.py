from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold


IMU_9AXIS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")
ACTIVE_PHASES = {"concentric", "eccentric"}
OTHER_LABEL = "Other"


@dataclass(frozen=True)
class SessionSummary:
    path: str
    subject: str
    samples: int
    duration_seconds: float
    windows: int


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    output_dir: str
    window_seconds: float
    stride_seconds: float
    label_majority: float
    min_active_fraction: float
    other_train_ratio: float
    n_splits: int
    n_estimators: int
    min_samples_leaf: int
    seed: int


def clean_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower()


def whole_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*whole_session*.csv"))


def infer_time_seconds(sensor_ts: pd.Series) -> np.ndarray:
    values = pd.to_numeric(sensor_ts, errors="coerce").to_numpy(dtype=np.float64)
    if len(values) == 0:
        return values
    valid = np.isfinite(values)
    if not valid.any():
        return np.arange(len(values), dtype=np.float64) / 100.0
    if not valid.all():
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=np.float64)

    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_delta = float(np.median(diffs)) if len(diffs) else 1.0
    if median_delta > 1000.0:
        scale = 1_000_000.0
    elif median_delta > 10.0:
        scale = 1000.0
    else:
        scale = 1.0
    times = (values - values[0]) / scale
    return np.maximum.accumulate(times)


def discover_exercises(files: Sequence[Path]) -> list[str]:
    exercises: set[str] = set()
    for path in files:
        df = pd.read_csv(path, usecols=lambda col: col in {"action_type", "phase"})
        actions = clean_series(df["action_type"])
        phases = clean_series(df["phase"])
        active = phases.isin(ACTIVE_PHASES)
        exercises.update(action for action in actions[active].unique().tolist() if action and action != "big_rest")
    return sorted(exercises)


def sample_labels(df: pd.DataFrame, exercises: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    exercise_set = set(exercises)
    actions = clean_series(df["action_type"]).to_numpy(dtype=object)
    phases = clean_series(df["phase"]).to_numpy(dtype=object)
    active = np.asarray(
        [(action in exercise_set and phase in ACTIVE_PHASES) for action, phase in zip(actions, phases)],
        dtype=bool,
    )
    labels = np.asarray([str(action) if is_active else OTHER_LABEL for action, is_active in zip(actions, active)], dtype=object)
    return labels, active


def window_label(
    labels: np.ndarray,
    active: np.ndarray,
    class_names: Sequence[str],
    majority_threshold: float,
    min_active_fraction: float,
) -> tuple[str, float, float]:
    if len(labels) == 0:
        return OTHER_LABEL, 0.0, 0.0
    counts = {name: int(np.sum(labels == name)) for name in class_names}
    best_label = max(counts, key=counts.get)
    best_fraction = counts[best_label] / float(len(labels))
    active_fraction = float(np.mean(active)) if len(active) else 0.0
    if best_label == OTHER_LABEL:
        return OTHER_LABEL, best_fraction, active_fraction
    if best_fraction >= majority_threshold and active_fraction >= min_active_fraction:
        return best_label, best_fraction, active_fraction
    return OTHER_LABEL, best_fraction, active_fraction


def nan_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (x_std * y_std))


def basic_stats(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0] * 10
    diff = np.diff(values)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.max(values) - np.min(values)),
        float(np.sqrt(np.mean(values * values))),
        float(np.median(values)),
        float(np.mean(np.abs(values))),
        float(values[-1] - values[0]),
        float(np.std(diff)) if len(diff) else 0.0,
    ]


def make_feature_names() -> list[str]:
    stats = ("mean", "std", "min", "max", "range", "rms", "median", "mean_abs", "delta", "diff_std")
    names: list[str] = []
    for col in IMU_9AXIS:
        names.extend([f"{col}_{stat}" for stat in stats])
    for group in ("acc_mag", "gyro_mag", "mag_mag"):
        names.extend([f"{group}_{stat}" for stat in stats])
    for group in ("acc", "gyro", "mag"):
        names.extend([f"{group}_corr_xy", f"{group}_corr_xz", f"{group}_corr_yz"])
    names.extend(["window_duration", "sample_count"])
    return names


def extract_features(block: np.ndarray, duration: float) -> np.ndarray:
    features: list[float] = []
    for axis_idx in range(block.shape[1]):
        features.extend(basic_stats(block[:, axis_idx]))
    for start in (0, 3, 6):
        magnitude = np.linalg.norm(block[:, start : start + 3], axis=1)
        features.extend(basic_stats(magnitude))
    for start in (0, 3, 6):
        group = block[:, start : start + 3]
        features.extend(
            [
                nan_corr(group[:, 0], group[:, 1]),
                nan_corr(group[:, 0], group[:, 2]),
                nan_corr(group[:, 1], group[:, 2]),
            ]
        )
    features.extend([float(duration), float(len(block))])
    return np.asarray(features, dtype=np.float32)


def build_windows(
    path: Path,
    exercises: Sequence[str],
    class_names: Sequence[str],
    window_seconds: float,
    stride_seconds: float,
    label_majority: float,
    min_active_fraction: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, SessionSummary]:
    required = set(IMU_9AXIS) | {"sensor_ts", "action_type", "phase", "subject_id"}
    df = pd.read_csv(path, usecols=lambda col: col in required)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.dropna(subset=list(IMU_9AXIS) + ["sensor_ts"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} has no usable rows")

    times = infer_time_seconds(df["sensor_ts"])
    values = df.loc[:, IMU_9AXIS].to_numpy(dtype=np.float32)
    labels, active = sample_labels(df, exercises)
    subject = str(df["subject_id"].dropna().astype(str).iloc[0]) if df["subject_id"].notna().any() else path.parent.name
    duration = float(times[-1] - times[0]) if len(times) else 0.0

    if duration < window_seconds:
        return (
            np.empty((0, len(make_feature_names())), dtype=np.float32),
            np.empty((0,), dtype=object),
            pd.DataFrame(),
            SessionSummary(str(path), subject, int(len(df)), duration, 0),
        )

    end_times = np.arange(times[0] + window_seconds, times[-1] + 1e-9, stride_seconds, dtype=np.float64)
    feature_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for window_idx, end_time in enumerate(end_times):
        start_time = end_time - window_seconds
        start_idx = int(np.searchsorted(times, start_time, side="left"))
        end_idx = int(np.searchsorted(times, end_time, side="right"))
        if end_idx <= start_idx:
            continue
        label, majority_fraction, active_fraction = window_label(
            labels[start_idx:end_idx],
            active[start_idx:end_idx],
            class_names,
            label_majority,
            min_active_fraction,
        )
        block = values[start_idx:end_idx]
        feature_rows.append(extract_features(block, float(times[end_idx - 1] - times[start_idx])))
        y_rows.append(label)
        manifest_rows.append(
            {
                "file": str(path),
                "subject": subject,
                "window_index": window_idx,
                "start_seconds": float(start_time - times[0]),
                "end_seconds": float(end_time - times[0]),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "label": label,
                "majority_fraction": majority_fraction,
                "active_fraction": active_fraction,
            }
        )
    features = np.vstack(feature_rows).astype(np.float32) if feature_rows else np.empty((0, len(make_feature_names())), dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    manifest = pd.DataFrame(manifest_rows)
    summary = SessionSummary(str(path), subject, int(len(df)), duration, int(len(y)))
    return features, y, manifest, summary


def balanced_train_indices(
    train_idx: np.ndarray,
    y: np.ndarray,
    other_train_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    other_idx = train_idx[y[train_idx] == OTHER_LABEL]
    active_idx = train_idx[y[train_idx] != OTHER_LABEL]
    max_other = int(round(len(active_idx) * other_train_ratio))
    if len(other_idx) > max_other:
        other_idx = rng.choice(other_idx, size=max_other, replace=False)
    selected = np.concatenate([active_idx, other_idx])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Realtime action recognition confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0)
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix_normalized.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format=".2f",
    )
    ax.set_title("Realtime action recognition confusion matrix (row-normalized)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)


def write_method_notes(output_dir: Path, labels: Sequence[str], config: RunConfig) -> None:
    text = f"""# Realtime 8-class + Other action recognition

## Online model

- Input: 9-axis IMU (`ax ay az gx gy gz mx my mz`) from whole-session CSV files.
- Windowing: causal {config.window_seconds:.2f}s window, updated every {config.stride_seconds:.2f}s.
- Labels: active `concentric/eccentric` samples become one of the 8 exercise labels; all rest, `none`, transition, or ambiguous windows become `Other`.
- Features: per-axis mean, standard deviation, min/max/range, RMS energy, median, mean absolute value, last-first delta, derivative variability, 3-axis magnitudes, and within-sensor axis correlations.
- Classifier: Random Forest, trained with subject-wise GroupKFold. Training folds downsample `Other` to {config.other_train_ratio:.2f}x active-window count; test folds keep all stream windows.

## Labels

{chr(10).join(f'- {label}' for label in labels)}

## References used

- Bao, L. and Intille, S. S. (2004). *Activity Recognition from User-Annotated Acceleration Data*. Used for sliding-window wearable acceleration features such as mean, energy, entropy/correlation style descriptors.
- Breiman, L. (2001). *Random Forests*. Used for the ensemble tree classifier.
"""
    (output_dir / "method_notes.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime 8-class + Other action recognition with causal IMU windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/workout"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/realtime_action_rf_8class_other_20260518"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--label-majority", type=float, default=0.55)
    parser.add_argument("--min-active-fraction", type=float, default=0.35)
    parser.add_argument("--other-train-ratio", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = whole_session_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No whole-session CSV files found under {args.data_dir}")
    exercises = discover_exercises(files)
    labels = [*exercises, OTHER_LABEL]
    feature_names = make_feature_names()

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    manifest_chunks: list[pd.DataFrame] = []
    session_summaries: list[SessionSummary] = []
    feature_start = time.perf_counter()
    for path in files:
        X_file, y_file, manifest_file, summary = build_windows(
            path=path,
            exercises=exercises,
            class_names=labels,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            label_majority=args.label_majority,
            min_active_fraction=args.min_active_fraction,
        )
        if len(y_file):
            feature_chunks.append(X_file)
            label_chunks.append(y_file)
            manifest_chunks.append(manifest_file)
        session_summaries.append(summary)
        print(f"[window] {summary.subject}: {summary.windows} windows from {Path(summary.path).name}", flush=True)

    if not feature_chunks:
        raise RuntimeError("No windows generated.")

    X = np.vstack(feature_chunks).astype(np.float32)
    y = np.concatenate(label_chunks).astype(object)
    manifest = pd.concat(manifest_chunks, ignore_index=True)
    groups = manifest["subject"].astype(str).to_numpy()
    feature_seconds = time.perf_counter() - feature_start

    label_counts = pd.Series(y).value_counts().reindex(labels, fill_value=0)
    label_counts.to_csv(args.output_dir / "window_label_counts.csv", header=["windows"])
    manifest.to_csv(args.output_dir / "window_manifest.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(args.output_dir / "feature_names.csv", index=False)

    n_splits = min(args.n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    train_seconds = 0.0
    predict_seconds = 0.0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        selected_train_idx = balanced_train_indices(train_idx, y, args.other_train_ratio, args.seed + fold_idx)
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.seed + fold_idx,
            n_jobs=-1,
        )
        fit_start = time.perf_counter()
        clf.fit(X[selected_train_idx], y[selected_train_idx])
        train_seconds += time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        pred = clf.predict(X[test_idx])
        predict_seconds += time.perf_counter() - pred_start

        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        pred_frame = manifest.iloc[test_idx].copy()
        pred_frame["fold"] = fold_idx
        pred_frame["prediction"] = pred
        pred_rows.append(pred_frame)

        fold_labels = sorted(set(y[test_idx].tolist()) | set(pred.tolist()))
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": ",".join(sorted(set(groups[train_idx].tolist()))),
                "test_subjects": ",".join(sorted(set(groups[test_idx].tolist()))),
                "train_windows": int(len(selected_train_idx)),
                "test_windows": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, labels=fold_labels, average="macro", zero_division=0)),
            }
        )
        print(
            f"[fold {fold_idx}/{n_splits}] test={len(test_idx)} acc={fold_rows[-1]['accuracy']:.4f} macro_f1={fold_rows[-1]['macro_f1']:.4f}",
            flush=True,
        )

    y_true = np.asarray(y_true_all, dtype=object)
    y_pred = np.asarray(y_pred_all, dtype=object)
    save_confusion(y_true, y_pred, labels, args.output_dir)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(args.output_dir / "window_predictions.csv", index=False)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    with (args.output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    config = RunConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_majority=args.label_majority,
        min_active_fraction=args.min_active_fraction,
        other_train_ratio=args.other_train_ratio,
        n_splits=n_splits,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    summary = {
        "config": asdict(config),
        "labels": labels,
        "sessions": [asdict(item) for item in session_summaries],
        "total_windows": int(len(y)),
        "window_label_counts": {str(k): int(v) for k, v in label_counts.to_dict().items()},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "feature_extraction_seconds": float(feature_seconds),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "predict_ms_per_window": float((predict_seconds / max(1, len(y_true))) * 1000.0),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_method_notes(args.output_dir, labels, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
