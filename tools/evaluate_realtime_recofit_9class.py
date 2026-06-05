from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from evaluate_active_only_8class_models import extract_posture_features
from evaluate_realtime_action_rf import IMU_9AXIS, OTHER_LABEL, discover_exercises, infer_time_seconds, sample_labels, whole_session_files
from evaluate_realtime_action_twostage_rf import (
    balanced_binary_train_indices,
    endpoint_label,
    extract_enhanced_features,
)


DISPLAY_LABELS = {
    "db_bench_press": "Bench",
    "db_biceps_curl": "Biceps",
    "db_rdl": "RDL",
    "db_shoulder_press": "Shoulder",
    "db_squat": "Squat",
    "db_triceps_curl": "Triceps",
    "db_weighted_crunch": "Crunch",
    "one_arm_db_row": "Row",
    OTHER_LABEL: "Other",
}


def majority_vote(values: Sequence[str], labels: Sequence[str]) -> str:
    counts = Counter(str(value) for value in values)
    return max(labels, key=lambda label: (counts.get(label, 0), -labels.index(label)))


def build_all_windows(
    path: Path,
    exercises: Sequence[str],
    scales_seconds: Sequence[float],
    endpoint_seconds: float,
    endpoint_min_active_fraction: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    required = set(IMU_9AXIS) | {"sensor_ts", "action_type", "phase", "subject_id"}
    df = pd.read_csv(path, usecols=lambda col: col in required)
    df = df.dropna(subset=list(IMU_9AXIS) + ["sensor_ts"]).reset_index(drop=True)
    if df.empty:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object), pd.DataFrame()

    times = infer_time_seconds(df["sensor_ts"])
    values = df.loc[:, IMU_9AXIS].to_numpy(dtype=np.float32)
    labels, active = sample_labels(df, exercises)
    subject = str(df["subject_id"].dropna().astype(str).iloc[0]) if df["subject_id"].notna().any() else path.parent.name
    max_scale = max(scales_seconds)
    if len(times) == 0 or float(times[-1] - times[0]) < max_scale:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object), pd.DataFrame()

    end_times = np.arange(times[0] + max_scale, times[-1] + 1e-9, stride_seconds, dtype=np.float64)
    base_rows: list[np.ndarray] = []
    posture_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for window_idx, end_time in enumerate(end_times):
        base_features: list[np.ndarray] = []
        posture_features: list[np.ndarray] = []
        valid = True
        for scale in scales_seconds:
            start_idx = int(np.searchsorted(times, end_time - float(scale), side="left"))
            end_idx = int(np.searchsorted(times, end_time, side="right"))
            if end_idx <= start_idx:
                valid = False
                break
            block = values[start_idx:end_idx]
            duration = float(times[end_idx - 1] - times[start_idx])
            enhanced = extract_enhanced_features(block, duration)
            base_features.append(enhanced)
            posture_features.append(np.concatenate([enhanced, extract_posture_features(block)]).astype(np.float32))
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
        base_rows.append(np.concatenate(base_features).astype(np.float32))
        posture_rows.append(np.concatenate(posture_features).astype(np.float32))
        y_rows.append(label)
        manifest_rows.append(
            {
                "file": str(path),
                "subject": subject,
                "window_index": int(window_idx),
                "end_seconds": float(end_time - times[0]),
                "endpoint_start_seconds": float(endpoint_start - times[0]),
                "endpoint_active_fraction": active_fraction,
                "label": label,
            }
        )

    return (
        np.vstack(base_rows).astype(np.float32),
        np.vstack(posture_rows).astype(np.float32),
        np.asarray(y_rows, dtype=object),
        pd.DataFrame(manifest_rows),
    )


def causal_majority_mask(active_raw: Sequence[bool], history_windows: int) -> np.ndarray:
    history: deque[bool] = deque(maxlen=history_windows)
    out: list[bool] = []
    for value in active_raw:
        history.append(bool(value))
        out.append(sum(history) >= math.ceil(len(history) / 2.0))
    return np.asarray(out, dtype=bool)


def pooled_window_vote(base_pred: Sequence[str], posture_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    return np.asarray([majority_vote([base, posture], labels) for base, posture in zip(base_pred, posture_pred)], dtype=object)


def warmup_segment_vote(
    frame: pd.DataFrame,
    active_column: str,
    labels: Sequence[str],
    warmup_windows: int,
) -> pd.Series:
    output = pd.Series(OTHER_LABEL, index=frame.index, dtype=object)
    for _, group in frame.groupby("file", sort=False):
        active = group[active_column].to_numpy(dtype=bool)
        start = 0
        while start < len(group):
            if not active[start]:
                start += 1
                continue
            end = start + 1
            while end < len(group) and active[end]:
                end += 1
            segment = group.iloc[start:end]
            warmup = segment.head(max(1, warmup_windows))
            votes = [*warmup["base_action_prediction"].astype(str).tolist(), *warmup["posture_action_prediction"].astype(str).tolist()]
            chosen = majority_vote(votes, labels)
            output.loc[segment.index] = chosen
            start = end
    return output.astype(str)


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{stem}.csv")
    pd.DataFrame(prop, index=labels, columns=labels).to_csv(output_dir / f"{stem}_row_proportion.csv", float_format="%.6f")
    pd.DataFrame(prop * 100.0, index=labels, columns=labels).to_csv(output_dir / f"{stem}_row_percent.csv", float_format="%.2f")

    display = [DISPLAY_LABELS.get(label, label) for label in labels]
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    image = ax.imshow(prop, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_yticklabels(display)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = prop[row, col]
            ax.text(col, row, f"{value * 100:.1f}%", ha="center", va="center", color=("white" if value >= 0.55 else "#1f2937"), fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row proportion")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_row_proportion.png", dpi=220)
    fig.savefig(output_dir / f"{stem}_row_proportion.pdf")
    plt.close(fig)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "min_recall": float(np.diag(prop).min()),
        "per_class_recall": {label: float(prop[idx, idx]) for idx, label in enumerate(labels)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 8 exercise classes + Other with RecoFit-style temporal voting.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/workout"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/realtime_recofit_9class_other_20260519"))
    parser.add_argument("--scales-seconds", type=float, nargs="+", default=[0.75, 2.0, 4.0])
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-min-active-fraction", type=float, default=0.25)
    parser.add_argument("--active-other-train-ratio", type=float, default=0.75)
    parser.add_argument("--active-thresholds", type=float, nargs="+", default=[0.10, 0.25, 0.525])
    parser.add_argument("--active-smooth-windows", type=int, default=3)
    parser.add_argument("--warmup-windows", type=int, nargs="+", default=[7, 10, 11])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = whole_session_files(args.data_dir)
    exercises = discover_exercises(files)
    labels = [*exercises, OTHER_LABEL]

    base_chunks: list[np.ndarray] = []
    posture_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    manifest_chunks: list[pd.DataFrame] = []
    feature_start = time.perf_counter()
    for path in files:
        X_base_file, X_posture_file, y_file, manifest_file = build_all_windows(
            path,
            exercises,
            [float(scale) for scale in args.scales_seconds],
            args.endpoint_seconds,
            args.endpoint_min_active_fraction,
            args.stride_seconds,
        )
        if len(y_file):
            base_chunks.append(X_base_file)
            posture_chunks.append(X_posture_file)
            label_chunks.append(y_file)
            manifest_chunks.append(manifest_file)
        print(f"[9class-window] {path.name}: {len(y_file)}", flush=True)

    X_base = np.vstack(base_chunks).astype(np.float32)
    X_posture = np.vstack(posture_chunks).astype(np.float32)
    y = np.concatenate(label_chunks).astype(object)
    manifest = pd.concat(manifest_chunks, ignore_index=True)
    groups = manifest["subject"].astype(str).to_numpy()
    feature_seconds = time.perf_counter() - feature_start

    n_splits = min(args.n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    pred_parts: list[pd.DataFrame] = []
    train_seconds = 0.0
    predict_seconds = 0.0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_base, y, groups), start=1):
        active_train_idx = balanced_binary_train_indices(train_idx, y, args.active_other_train_ratio, args.seed + fold_idx)
        active_target = np.where(y[active_train_idx] == OTHER_LABEL, OTHER_LABEL, "Active")
        active_clf = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            class_weight={"Active": 1.35, OTHER_LABEL: 1.0},
            random_state=args.seed + fold_idx,
            n_jobs=-1,
        )
        action_train_idx = train_idx[y[train_idx] != OTHER_LABEL]
        base_action_clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=args.seed + 100 + fold_idx,
        )
        posture_action_clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=args.seed + 200 + fold_idx,
        )

        fit_start = time.perf_counter()
        active_clf.fit(X_base[active_train_idx], active_target)
        base_action_clf.fit(X_base[action_train_idx], y[action_train_idx])
        posture_action_clf.fit(X_posture[action_train_idx], y[action_train_idx])
        train_seconds += time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        active_proba = active_clf.predict_proba(X_base[test_idx])
        active_col = list(active_clf.classes_).index("Active")
        active_probability = active_proba[:, active_col]
        base_action = base_action_clf.predict(X_base[test_idx])
        posture_action = posture_action_clf.predict(X_posture[test_idx])
        predict_seconds += time.perf_counter() - pred_start

        frame = manifest.iloc[test_idx].copy()
        frame["fold"] = fold_idx
        frame["active_probability"] = active_probability
        frame["base_action_prediction"] = base_action
        frame["posture_action_prediction"] = posture_action
        pred_parts.append(frame)
        print(f"[9class-fold {fold_idx}/{n_splits}] test={len(test_idx)}", flush=True)

    predictions = pd.concat(pred_parts, ignore_index=True).sort_values(["file", "window_index"]).reset_index(drop=True)
    predictions["pooled_window_action"] = pooled_window_vote(
        predictions["base_action_prediction"].astype(str),
        predictions["posture_action_prediction"].astype(str),
        exercises,
    )

    y_true = predictions["label"].astype(str).to_numpy()
    summaries: dict[str, object] = {}
    for threshold in args.active_thresholds:
        raw_col = f"active_t{threshold:g}"
        smooth_col = f"{raw_col}_smooth{args.active_smooth_windows}"
        predictions[raw_col] = predictions["active_probability"] >= threshold
        smooth_parts: list[pd.Series] = []
        for _, group in predictions.groupby("file", sort=False):
            smooth = causal_majority_mask(group[raw_col].to_numpy(dtype=bool), args.active_smooth_windows)
            smooth_parts.append(pd.Series(smooth, index=group.index))
        predictions[smooth_col] = pd.concat(smooth_parts).sort_index().astype(bool)

        window_stem = f"threshold{str(threshold).replace('.', 'p')}_smooth{args.active_smooth_windows}_window_vote"
        predictions[f"prediction_{window_stem}"] = np.where(predictions[smooth_col], predictions["pooled_window_action"], OTHER_LABEL)
        y_pred_window = predictions[f"prediction_{window_stem}"].astype(str).to_numpy()
        summaries[window_stem] = summarize(y_true, y_pred_window, labels)
        save_confusion_outputs(
            y_true,
            y_pred_window,
            labels,
            args.output_dir,
            f"confusion_matrix_{window_stem}",
            f"9-class recognition: threshold {threshold:g}, window vote",
        )

        for warmup in args.warmup_windows:
            warmup_stem = f"threshold{str(threshold).replace('.', 'p')}_smooth{args.active_smooth_windows}_warmup{warmup}"
            predictions[f"prediction_{warmup_stem}"] = warmup_segment_vote(predictions, smooth_col, exercises, warmup)
            y_pred = predictions[f"prediction_{warmup_stem}"].astype(str).to_numpy()
            summaries[warmup_stem] = summarize(y_true, y_pred, labels)
            save_confusion_outputs(
                y_true,
                y_pred,
                labels,
                args.output_dir,
                f"confusion_matrix_{warmup_stem}",
                f"9-class recognition: threshold {threshold:g}, warm-up {warmup}",
            )
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            (args.output_dir / f"classification_report_{warmup_stem}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    predictions.to_csv(args.output_dir / "window_predictions_9class_recofit.csv", index=False)
    summary = {
        "labels": labels,
        "display_labels": {label: DISPLAY_LABELS.get(label, label) for label in labels},
        "total_windows": int(len(predictions)),
        "window_label_counts": {str(k): int(v) for k, v in predictions["label"].value_counts().reindex(labels, fill_value=0).to_dict().items()},
        "feature_extraction_seconds": float(feature_seconds),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "predict_ms_per_window": float((predict_seconds / max(1, len(predictions))) * 1000.0),
        "methods": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
