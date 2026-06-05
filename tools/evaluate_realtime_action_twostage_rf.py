from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, deque
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

from evaluate_realtime_action_rf import (
    ACTIVE_PHASES,
    IMU_9AXIS,
    OTHER_LABEL,
    basic_stats,
    clean_series,
    discover_exercises,
    infer_time_seconds,
    nan_corr,
    sample_labels,
    whole_session_files,
)


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
    scales_seconds: list[float]
    stride_seconds: float
    endpoint_seconds: float
    endpoint_min_active_fraction: float
    active_other_train_ratio: float
    active_threshold: float
    hysteresis_enter_threshold: float
    hysteresis_exit_threshold: float
    hysteresis_enter_windows: int
    hysteresis_exit_windows: int
    class_smooth_windows: int
    n_splits: int
    n_estimators: int
    min_samples_leaf: int
    seed: int


def magnitude(block: np.ndarray, start: int) -> np.ndarray:
    return np.linalg.norm(block[:, start : start + 3], axis=1)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0


def half_rms_features(values: np.ndarray) -> list[float]:
    if len(values) < 2:
        value = rms(values)
        return [value, value, 1.0]
    mid = len(values) // 2
    first = rms(values[:mid])
    second = rms(values[mid:])
    return [first, second, float(second / max(first, 1e-6))]


def enhanced_feature_names(prefix: str) -> list[str]:
    stats = ("mean", "std", "min", "max", "range", "rms", "median", "mean_abs", "delta", "diff_std")
    names: list[str] = []
    for col in IMU_9AXIS:
        names.extend([f"{prefix}_{col}_{stat}" for stat in stats])
    for group in ("acc_mag", "gyro_mag", "mag_mag"):
        names.extend([f"{prefix}_{group}_{stat}" for stat in stats])
        names.extend([f"{prefix}_{group}_rms_first", f"{prefix}_{group}_rms_second", f"{prefix}_{group}_rms_ratio"])
    for group in ("acc", "gyro", "mag"):
        names.extend([f"{prefix}_{group}_corr_xy", f"{prefix}_{group}_corr_xz", f"{prefix}_{group}_corr_yz"])
    for group in ("acc_jerk_mag", "gyro_jerk_mag", "mag_jerk_mag", "acc_dynamic_mag"):
        names.extend([f"{prefix}_{group}_{stat}" for stat in stats])
    names.extend(
        [
            f"{prefix}_gravity_unit_x",
            f"{prefix}_gravity_unit_y",
            f"{prefix}_gravity_unit_z",
            f"{prefix}_gravity_norm",
            f"{prefix}_window_duration",
            f"{prefix}_sample_count",
        ]
    )
    return names


def make_feature_names(scales: Sequence[float]) -> list[str]:
    names: list[str] = []
    for scale in scales:
        names.extend(enhanced_feature_names(f"w{scale:g}s"))
    return names


def extract_enhanced_features(block: np.ndarray, duration: float) -> np.ndarray:
    features: list[float] = []
    for axis_idx in range(block.shape[1]):
        features.extend(basic_stats(block[:, axis_idx]))

    for start in (0, 3, 6):
        mag = magnitude(block, start)
        features.extend(basic_stats(mag))
        features.extend(half_rms_features(mag))

    for start in (0, 3, 6):
        group = block[:, start : start + 3]
        features.extend(
            [
                nan_corr(group[:, 0], group[:, 1]),
                nan_corr(group[:, 0], group[:, 2]),
                nan_corr(group[:, 1], group[:, 2]),
            ]
        )

    for start in (0, 3, 6):
        group = block[:, start : start + 3]
        jerk = np.diff(group, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros((0,), dtype=np.float32)
        features.extend(basic_stats(jerk_mag))

    acc = block[:, 0:3]
    gravity = np.mean(acc, axis=0)
    gravity_norm = float(np.linalg.norm(gravity))
    gravity_unit = gravity / max(gravity_norm, 1e-6)
    dynamic_acc = acc - gravity
    features.extend(basic_stats(np.linalg.norm(dynamic_acc, axis=1)))
    features.extend([float(gravity_unit[0]), float(gravity_unit[1]), float(gravity_unit[2]), gravity_norm])
    features.extend([float(duration), float(len(block))])
    return np.asarray(features, dtype=np.float32)


def endpoint_label(
    labels: np.ndarray,
    active: np.ndarray,
    exercises: Sequence[str],
    min_active_fraction: float,
) -> tuple[str, float]:
    if len(labels) == 0:
        return OTHER_LABEL, 0.0
    active_fraction = float(np.mean(active))
    if active_fraction < min_active_fraction:
        return OTHER_LABEL, active_fraction
    active_labels = labels[(labels != OTHER_LABEL) & active]
    if len(active_labels) == 0:
        return OTHER_LABEL, active_fraction
    counts = {label: int(np.sum(active_labels == label)) for label in exercises}
    return max(counts, key=counts.get), active_fraction


def build_windows(
    path: Path,
    exercises: Sequence[str],
    scales_seconds: Sequence[float],
    stride_seconds: float,
    endpoint_seconds: float,
    endpoint_min_active_fraction: float,
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
    max_scale = max(scales_seconds)
    if duration < max_scale:
        return (
            np.empty((0, len(make_feature_names(scales_seconds))), dtype=np.float32),
            np.empty((0,), dtype=object),
            pd.DataFrame(),
            SessionSummary(str(path), subject, int(len(df)), duration, 0),
        )

    end_times = np.arange(times[0] + max_scale, times[-1] + 1e-9, stride_seconds, dtype=np.float64)
    feature_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for window_idx, end_time in enumerate(end_times):
        row_features: list[np.ndarray] = []
        valid = True
        for scale in scales_seconds:
            start_time = end_time - float(scale)
            start_idx = int(np.searchsorted(times, start_time, side="left"))
            end_idx = int(np.searchsorted(times, end_time, side="right"))
            if end_idx <= start_idx:
                valid = False
                break
            block = values[start_idx:end_idx]
            row_features.append(extract_enhanced_features(block, float(times[end_idx - 1] - times[start_idx])))
        if not valid:
            continue

        endpoint_start = end_time - endpoint_seconds
        tail_start_idx = int(np.searchsorted(times, endpoint_start, side="left"))
        tail_end_idx = int(np.searchsorted(times, end_time, side="right"))
        label, active_fraction = endpoint_label(
            labels[tail_start_idx:tail_end_idx],
            active[tail_start_idx:tail_end_idx],
            exercises,
            endpoint_min_active_fraction,
        )
        feature_rows.append(np.concatenate(row_features).astype(np.float32))
        y_rows.append(label)
        manifest_rows.append(
            {
                "file": str(path),
                "subject": subject,
                "window_index": window_idx,
                "end_seconds": float(end_time - times[0]),
                "endpoint_start_seconds": float(endpoint_start - times[0]),
                "endpoint_active_fraction": active_fraction,
                "label": label,
            }
        )

    X = np.vstack(feature_rows).astype(np.float32) if feature_rows else np.empty((0, len(make_feature_names(scales_seconds))), dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    manifest = pd.DataFrame(manifest_rows)
    return X, y, manifest, SessionSummary(str(path), subject, int(len(df)), duration, int(len(y)))


def balanced_binary_train_indices(
    train_idx: np.ndarray,
    y: np.ndarray,
    other_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active_idx = train_idx[y[train_idx] != OTHER_LABEL]
    other_idx = train_idx[y[train_idx] == OTHER_LABEL]
    max_other = int(round(len(active_idx) * other_ratio))
    if len(other_idx) > max_other:
        other_idx = rng.choice(other_idx, size=max_other, replace=False)
    selected = np.concatenate([active_idx, other_idx])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def majority_label(labels: Sequence[str], current: str) -> str:
    counts = Counter(labels)
    if not counts:
        return OTHER_LABEL
    max_count = max(counts.values())
    if counts[current] == max_count:
        return current
    return counts.most_common(1)[0][0]


def apply_online_hysteresis(
    frame: pd.DataFrame,
    enter_threshold: float,
    exit_threshold: float,
    enter_windows: int,
    exit_windows: int,
    class_smooth_windows: int,
) -> pd.Series:
    output: list[str] = []
    state_active = False
    enter_streak = 0
    exit_streak = 0
    class_history: deque[str] = deque(maxlen=class_smooth_windows)

    for row in frame.itertuples(index=False):
        active_probability = float(row.active_probability)
        action_prediction = str(row.action_prediction)
        if state_active:
            class_history.append(action_prediction)
            if active_probability < exit_threshold:
                exit_streak += 1
            else:
                exit_streak = 0
            if exit_streak >= exit_windows:
                state_active = False
                enter_streak = 0
                exit_streak = 0
                class_history.clear()
                output.append(OTHER_LABEL)
            else:
                output.append(majority_label(list(class_history), action_prediction))
        else:
            if active_probability >= enter_threshold:
                enter_streak += 1
                class_history.append(action_prediction)
            else:
                enter_streak = 0
                class_history.clear()
            if enter_streak >= enter_windows:
                state_active = True
                exit_streak = 0
                output.append(majority_label(list(class_history), action_prediction))
            else:
                output.append(OTHER_LABEL)
    return pd.Series(output, index=frame.index)


def save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{stem}.csv")
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0)
    pd.DataFrame(cm_prop, index=labels, columns=labels).to_csv(output_dir / f"{stem}_row_proportion.csv", float_format="%.6f")
    pd.DataFrame(cm_prop * 100.0, index=labels, columns=labels).to_csv(output_dir / f"{stem}_row_percent.csv", float_format="%.2f")

    short = {
        "db_bench_press": "Bench",
        "db_biceps_curl": "Biceps",
        "db_rdl": "RDL",
        "db_shoulder_press": "Shoulder",
        "db_squat": "Squat",
        "db_triceps_curl": "Triceps",
        "db_weighted_crunch": "Crunch",
        "one_arm_db_row": "Row",
        OTHER_LABEL: OTHER_LABEL,
    }
    display_labels = [short.get(label, label) for label in labels]

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    image = ax.imshow(cm_prop, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title} (row-normalized)")
    for row in range(cm_prop.shape[0]):
        for col in range(cm_prop.shape[1]):
            value = cm_prop[row, col]
            color = "white" if value >= 0.5 else "#1f2937"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within true class")
    ax.set_xticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_row_proportion.png", dpi=220)
    fig.savefig(output_dir / f"{stem}_row_proportion.pdf")
    plt.close(fig)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime two-stage 8-class + Other action recognition.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/workout"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/realtime_action_twostage_rf_v2_20260518"))
    parser.add_argument("--scales-seconds", type=float, nargs="+", default=[0.75, 2.0, 4.0])
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-min-active-fraction", type=float, default=0.25)
    parser.add_argument("--active-other-train-ratio", type=float, default=0.75)
    parser.add_argument("--active-threshold", type=float, default=0.35)
    parser.add_argument("--hysteresis-enter-threshold", type=float, default=0.30)
    parser.add_argument("--hysteresis-exit-threshold", type=float, default=0.18)
    parser.add_argument("--hysteresis-enter-windows", type=int, default=2)
    parser.add_argument("--hysteresis-exit-windows", type=int, default=4)
    parser.add_argument("--class-smooth-windows", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=240)
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
    feature_names = make_feature_names(args.scales_seconds)

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    manifest_chunks: list[pd.DataFrame] = []
    session_summaries: list[SessionSummary] = []
    feature_start = time.perf_counter()
    for path in files:
        X_file, y_file, manifest_file, summary = build_windows(
            path=path,
            exercises=exercises,
            scales_seconds=args.scales_seconds,
            stride_seconds=args.stride_seconds,
            endpoint_seconds=args.endpoint_seconds,
            endpoint_min_active_fraction=args.endpoint_min_active_fraction,
        )
        if len(y_file):
            feature_chunks.append(X_file)
            label_chunks.append(y_file)
            manifest_chunks.append(manifest_file)
        session_summaries.append(summary)
        print(f"[window-v2] {summary.subject}: {summary.windows} windows from {Path(summary.path).name}", flush=True)

    if not feature_chunks:
        raise RuntimeError("No windows generated.")

    X = np.vstack(feature_chunks).astype(np.float32)
    y = np.concatenate(label_chunks).astype(object)
    manifest = pd.concat(manifest_chunks, ignore_index=True)
    groups = manifest["subject"].astype(str).to_numpy()
    feature_seconds = time.perf_counter() - feature_start

    manifest.to_csv(args.output_dir / "window_manifest.csv", index=False)
    pd.Series(y).value_counts().reindex(labels, fill_value=0).to_csv(args.output_dir / "window_label_counts.csv", header=["windows"])
    pd.DataFrame({"feature": feature_names}).to_csv(args.output_dir / "feature_names.csv", index=False)

    n_splits = min(args.n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    train_seconds = 0.0
    predict_seconds = 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        active_train_idx = balanced_binary_train_indices(train_idx, y, args.active_other_train_ratio, args.seed + fold_idx)
        active_target = np.where(y[active_train_idx] == OTHER_LABEL, "Other", "Active")
        active_clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            class_weight={"Active": 1.35, "Other": 1.0},
            random_state=args.seed + fold_idx,
            n_jobs=-1,
        )

        action_train_idx = train_idx[y[train_idx] != OTHER_LABEL]
        action_clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.seed + 100 + fold_idx,
            n_jobs=-1,
        )

        fit_start = time.perf_counter()
        active_clf.fit(X[active_train_idx], active_target)
        action_clf.fit(X[action_train_idx], y[action_train_idx])
        train_seconds += time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        active_proba_all = active_clf.predict_proba(X[test_idx])
        active_col = list(active_clf.classes_).index("Active")
        active_probability = active_proba_all[:, active_col]
        action_prediction = action_clf.predict(X[test_idx])
        raw_prediction = np.where(active_probability >= args.active_threshold, action_prediction, OTHER_LABEL)
        predict_seconds += time.perf_counter() - pred_start

        pred_frame = manifest.iloc[test_idx].copy()
        pred_frame["fold"] = fold_idx
        pred_frame["active_probability"] = active_probability
        pred_frame["action_prediction"] = action_prediction
        pred_frame["prediction_raw"] = raw_prediction
        pred_rows.append(pred_frame)

        fold_metrics = metrics_dict(y[test_idx], raw_prediction, labels)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": ",".join(sorted(set(groups[train_idx].tolist()))),
                "test_subjects": ",".join(sorted(set(groups[test_idx].tolist()))),
                "train_windows_active_stage": int(len(active_train_idx)),
                "train_windows_action_stage": int(len(action_train_idx)),
                "test_windows": int(len(test_idx)),
                **fold_metrics,
            }
        )
        print(
            f"[fold-v2 {fold_idx}/{n_splits}] test={len(test_idx)} raw_acc={fold_metrics['accuracy']:.4f} raw_macro_f1={fold_metrics['macro_f1']:.4f}",
            flush=True,
        )

    predictions = pd.concat(pred_rows, ignore_index=True)
    predictions = predictions.sort_values(["file", "window_index"]).reset_index(drop=True)
    hysteresis_parts: list[pd.Series] = []
    for _file, group in predictions.groupby("file", sort=False):
        hysteresis_parts.append(
            apply_online_hysteresis(
                group,
                enter_threshold=args.hysteresis_enter_threshold,
                exit_threshold=args.hysteresis_exit_threshold,
                enter_windows=args.hysteresis_enter_windows,
                exit_windows=args.hysteresis_exit_windows,
                class_smooth_windows=args.class_smooth_windows,
            )
        )
    predictions["prediction_hysteresis"] = pd.concat(hysteresis_parts).sort_index().to_numpy()
    predictions.to_csv(args.output_dir / "window_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics_raw.csv", index=False)

    y_true = predictions["label"].to_numpy(dtype=object)
    y_raw = predictions["prediction_raw"].to_numpy(dtype=object)
    y_hyst = predictions["prediction_hysteresis"].to_numpy(dtype=object)

    save_confusion(y_true, y_raw, labels, args.output_dir, "confusion_matrix_raw_twostage", "Two-stage realtime action recognition")
    save_confusion(
        y_true,
        y_hyst,
        labels,
        args.output_dir,
        "confusion_matrix_online_hysteresis",
        "Two-stage realtime action recognition with online hysteresis",
    )
    raw_report = classification_report(y_true, y_raw, labels=labels, output_dict=True, zero_division=0)
    hyst_report = classification_report(y_true, y_hyst, labels=labels, output_dict=True, zero_division=0)
    (args.output_dir / "classification_report_raw_twostage.json").write_text(
        json.dumps(raw_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "classification_report_online_hysteresis.json").write_text(
        json.dumps(hyst_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = RunConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        scales_seconds=[float(item) for item in args.scales_seconds],
        stride_seconds=args.stride_seconds,
        endpoint_seconds=args.endpoint_seconds,
        endpoint_min_active_fraction=args.endpoint_min_active_fraction,
        active_other_train_ratio=args.active_other_train_ratio,
        active_threshold=args.active_threshold,
        hysteresis_enter_threshold=args.hysteresis_enter_threshold,
        hysteresis_exit_threshold=args.hysteresis_exit_threshold,
        hysteresis_enter_windows=args.hysteresis_enter_windows,
        hysteresis_exit_windows=args.hysteresis_exit_windows,
        class_smooth_windows=args.class_smooth_windows,
        n_splits=n_splits,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    summary = {
        "config": asdict(config),
        "labels": labels,
        "sessions": [asdict(item) for item in session_summaries],
        "total_windows": int(len(predictions)),
        "window_label_counts": {str(k): int(v) for k, v in predictions["label"].value_counts().reindex(labels, fill_value=0).to_dict().items()},
        "raw_twostage": metrics_dict(y_true, y_raw, labels),
        "online_hysteresis": metrics_dict(y_true, y_hyst, labels),
        "feature_extraction_seconds": float(feature_seconds),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "predict_ms_per_window": float((predict_seconds / max(1, len(predictions))) * 1000.0),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
