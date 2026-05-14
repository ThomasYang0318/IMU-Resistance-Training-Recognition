from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")
ACTIVE_PHASES = {"concentric", "eccentric"}
REST_LABELS = {"big_rest", "rest", "none", "nan", ""}


@dataclass(frozen=True)
class RepSegment:
    file_path: Path
    subject: str
    exercise: str
    set_id: str
    rep_id: str
    start: int
    end: int
    source: str

    @property
    def n_samples(self) -> int:
        return self.end - self.start


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def whole_session_files(data_dirs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for data_dir in data_dirs:
        files.extend(sorted(data_dir.rglob("*whole_session*.csv")))
    return sorted(set(files))


def subject_from_path(path: Path, data_dirs: Sequence[Path]) -> str:
    for data_dir in data_dirs:
        try:
            return path.relative_to(data_dir).parts[0]
        except ValueError:
            continue
    return path.parent.name


def clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_session(path: Path, data_dirs: Sequence[Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject_id" not in df.columns:
        df["subject_id"] = subject_from_path(path, data_dirs)
    if "action_type" not in df.columns:
        df["action_type"] = path.parent.name
    if "set" not in df.columns:
        df["set"] = 0
    if "rep" not in df.columns:
        df["rep"] = np.arange(len(df))
    if "phase" not in df.columns:
        df["phase"] = "none"
    return df.reset_index(drop=True)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / scale


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(np.float64, copy=True)
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(values.astype(np.float64), pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def principal_motion_signal(df: pd.DataFrame, smooth_window: int) -> np.ndarray:
    available = [col for col in IMU_COLUMNS if col in df.columns]
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    x = np.apply_along_axis(robust_zscore, 0, x)
    variances = np.var(x, axis=0)
    x = x[:, variances > 1e-9]
    if x.shape[1] == 0:
        return np.zeros(len(df), dtype=np.float64)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return moving_average(x @ vt[0], smooth_window)


def true_rep_segments(df: pd.DataFrame, path: Path, min_samples: int) -> list[RepSegment]:
    phases = df["phase"].map(clean_label).str.lower()
    active = phases.isin(ACTIVE_PHASES).to_numpy()
    if not active.any():
        return []

    subject_values = df["subject_id"].map(clean_label)
    exercise_values = df["action_type"].map(clean_label)
    set_values = df["set"].map(clean_label)
    rep_values = df["rep"].map(clean_label)

    segments: list[RepSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str, str] | None = None

    for idx, is_active in enumerate(active):
        key = (
            subject_values.iloc[idx],
            exercise_values.iloc[idx],
            set_values.iloc[idx],
            rep_values.iloc[idx],
        )
        if is_active and start is None:
            start = idx
            last_key = key
        elif is_active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                segments.append(RepSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], start, idx, "label"))
            start = idx
            last_key = key
        elif (not is_active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                segments.append(RepSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], start, idx, "label"))
            start = None
            last_key = None

    if start is not None and last_key is not None and len(df) - start >= min_samples:
        segments.append(RepSegment(path, last_key[0], last_key[1], last_key[2], last_key[3], start, len(df), "label"))
    return segments


def set_blocks_from_labels(df: pd.DataFrame, path: Path, min_samples: int) -> list[RepSegment]:
    actions = df["action_type"].map(clean_label)
    sets = df["set"].map(clean_label)
    subjects = df["subject_id"].map(clean_label)
    non_rest = ~actions.str.lower().isin(REST_LABELS)

    blocks: list[RepSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str] | None = None
    for idx, active in enumerate(non_rest.to_numpy()):
        key = (subjects.iloc[idx], actions.iloc[idx], sets.iloc[idx])
        if active and start is None:
            start = idx
            last_key = key
        elif active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                blocks.append(RepSegment(path, last_key[0], last_key[1], last_key[2], "set", start, idx, "set_block"))
            start = idx
            last_key = key
        elif (not active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                blocks.append(RepSegment(path, last_key[0], last_key[1], last_key[2], "set", start, idx, "set_block"))
            start = None
            last_key = None
    if start is not None and last_key is not None and len(df) - start >= min_samples:
        blocks.append(RepSegment(path, last_key[0], last_key[1], last_key[2], "set", start, len(df), "set_block"))
    return blocks


def pca_extrema_segments(
    df: pd.DataFrame,
    path: Path,
    true_segments: Sequence[RepSegment],
    smooth_window: int,
    min_samples: int,
    peak_prominence_scale: float,
) -> list[RepSegment]:
    by_block: dict[tuple[str, str, str], list[RepSegment]] = {}
    for segment in true_segments:
        by_block.setdefault((segment.subject, segment.exercise, segment.set_id), []).append(segment)

    predicted: list[RepSegment] = []
    for block in set_blocks_from_labels(df, path, min_samples=min_samples):
        truth = sorted(by_block.get((block.subject, block.exercise, block.set_id), []), key=lambda s: s.start)
        if not truth:
            continue
        expected = len(truth)
        segment_df = df.iloc[block.start : block.end]
        signal = principal_motion_signal(segment_df, smooth_window)
        prominence = max(float(np.std(signal)) * peak_prominence_scale, 1e-6)
        distance = max(min_samples // 2, 1)

        candidates: list[np.ndarray] = []
        for candidate_signal in (signal, -signal):
            peaks, _ = find_peaks(candidate_signal, distance=distance, prominence=prominence)
            if len(peaks) >= 2:
                candidates.append(peaks)

        if not candidates:
            continue
        peaks = min(candidates, key=lambda p: abs(len(p) - expected))
        if len(peaks) == 0:
            continue

        centers = np.sort(peaks)
        boundaries = [0]
        if len(centers) > 1:
            boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(len(centers) - 1))
        boundaries.append(len(segment_df))

        for rep_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < min_samples:
                continue
            predicted.append(
                RepSegment(
                    path,
                    block.subject,
                    block.exercise,
                    block.set_id,
                    str(rep_idx),
                    block.start + start,
                    block.start + end,
                    "pca_extrema",
                )
            )
    return predicted


def segment_iou(a: RepSegment, b: RepSegment) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / float(union) if union > 0 else 0.0


def label_predicted_segments(
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    class_names: Sequence[str],
    include_other: bool,
    min_iou: float,
) -> tuple[list[RepSegment], list[str], list[dict[str, object]]]:
    by_file: dict[Path, list[RepSegment]] = {}
    for segment in truth:
        by_file.setdefault(segment.file_path, []).append(segment)

    labeled_segments: list[RepSegment] = []
    labels: list[str] = []
    rows: list[dict[str, object]] = []
    class_set = set(class_names)

    for segment in predicted:
        candidates = by_file.get(segment.file_path, [])
        best = max(candidates, key=lambda true_segment: segment_iou(segment, true_segment), default=None)
        best_iou = segment_iou(segment, best) if best is not None else 0.0
        label = best.exercise if best is not None and best_iou >= min_iou else "other"
        if label not in class_set:
            if include_other:
                label = "other"
            else:
                continue
        labeled_segments.append(segment)
        labels.append(label)
        rows.append(
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "pred_start": segment.start,
                "pred_end": segment.end,
                "matched_exercise": label,
                "matched_iou": round(best_iou, 4),
                "source": segment.source,
            }
        )
    return labeled_segments, labels, rows


def segment_features(df: pd.DataFrame, segment: RepSegment) -> dict[str, float]:
    x = df.iloc[segment.start : segment.end].loc[:, IMU_COLUMNS].to_numpy(dtype=np.float64)
    features: dict[str, float] = {
        "duration_samples": float(len(x)),
    }
    if len(x) == 0:
        return features
    diff = np.diff(x, axis=0) if len(x) > 1 else np.zeros_like(x)
    for col_idx, col in enumerate(IMU_COLUMNS):
        values = x[:, col_idx]
        d_values = diff[:, col_idx] if len(diff) else np.zeros(1)
        features[f"{col}_mean"] = float(np.mean(values))
        features[f"{col}_std"] = float(np.std(values))
        features[f"{col}_min"] = float(np.min(values))
        features[f"{col}_max"] = float(np.max(values))
        features[f"{col}_range"] = float(np.ptp(values))
        features[f"{col}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{col}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        features[f"{col}_diff_abs_mean"] = float(np.mean(np.abs(d_values)))
        features[f"{col}_diff_std"] = float(np.std(d_values))

    acc_norm = np.linalg.norm(x[:, :3], axis=1)
    gyro_norm = np.linalg.norm(x[:, 3:6], axis=1)
    features["acc_norm_mean"] = float(np.mean(acc_norm))
    features["acc_norm_std"] = float(np.std(acc_norm))
    features["gyro_norm_mean"] = float(np.mean(gyro_norm))
    features["gyro_norm_std"] = float(np.std(gyro_norm))
    try:
        signal = principal_motion_signal(pd.DataFrame(x, columns=IMU_COLUMNS), smooth_window=7)
        features["principal_range"] = float(np.ptp(signal))
        features["principal_turning_points"] = float(len(find_peaks(signal)[0]) + len(find_peaks(-signal)[0]))
    except Exception:
        features["principal_range"] = 0.0
        features["principal_turning_points"] = 0.0
    return features


def build_feature_table(segments: Sequence[RepSegment], labels: Sequence[str], session_cache: dict[Path, pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, float]] = []
    subjects: list[str] = []
    for segment in segments:
        rows.append(segment_features(session_cache[segment.file_path], segment))
        subjects.append(segment.subject)
    return pd.DataFrame(rows).fillna(0.0), np.asarray(labels, dtype=object), np.asarray(subjects, dtype=object)


def select_classes(labels: Sequence[str], num_classes: int, include_other: bool) -> list[str]:
    counts = pd.Series(labels).value_counts()
    counts = counts.drop(labels=["big_rest", "rest", "other"], errors="ignore")
    classes = counts.head(num_classes).index.astype(str).tolist()
    if include_other:
        classes.append("other")
    return classes


def run_group_kfold(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Sequence[str],
    folds: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    unique_groups = sorted(set(groups.tolist()))
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two subjects for subject-wise K-fold validation.")

    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
        train_subjects = sorted(set(groups[train_idx].tolist()))
        val_subjects = sorted(set(groups[val_idx].tolist()))
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed + fold_idx,
                n_jobs=-1,
            ),
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        pred = model.predict(x.iloc[val_idx])
        y_true_all.extend(y[val_idx].tolist())
        y_pred_all.extend(pred.tolist())
        cm = confusion_matrix(y[val_idx], pred, labels=class_names)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": ",".join(train_subjects),
                "val_subjects": ",".join(val_subjects),
                "val_samples": len(val_idx),
                "accuracy": round(float(np.trace(cm) / max(1, cm.sum())), 4),
            }
        )

    labels = list(class_names)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    report = classification_report(y_true_all, y_pred_all, labels=labels, output_dict=True, zero_division=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_manifest.csv", fold_rows)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true_label": true_label, "pred_label": pred_label, "count": int(cm[i, j])}
            for i, true_label in enumerate(labels)
            for j, pred_label in enumerate(labels)
        ],
    )
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(7, len(labels) * 1.1)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True, xticks_rotation=45)
    ax.set_title("Subject-wise K-fold Exercise Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(7, len(labels) * 1.1)))
    display = ConfusionMatrixDisplay(confusion_matrix=norm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True, xticks_rotation=45)
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)

    return {
        "folds": n_splits,
        "subjects": unique_groups,
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def segmentation_summary(predicted: Sequence[RepSegment], truth: Sequence[RepSegment]) -> list[dict[str, object]]:
    by_file_truth: dict[Path, list[RepSegment]] = {}
    for segment in truth:
        by_file_truth.setdefault(segment.file_path, []).append(segment)
    rows: list[dict[str, object]] = []
    for segment in predicted:
        candidates = by_file_truth.get(segment.file_path, [])
        best_iou = max((segment_iou(segment, candidate) for candidate in candidates), default=0.0)
        rows.append(
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "exercise_hint": segment.exercise,
                "start": segment.start,
                "end": segment.end,
                "samples": segment.n_samples,
                "best_true_iou": round(best_iou, 4),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rep segmentation followed by subject-wise K-fold exercise classification.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--include-other", action="store_true", help="Add an 'other' class for unmatched/non-top exercise segments.")
    parser.add_argument("--segment-method", choices=["labels", "pca-extrema"], default="labels")
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--peak-prominence-scale", type=float, default=0.35)
    parser.add_argument("--min-label-iou", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session_cache: dict[Path, pd.DataFrame] = {}
    truth: list[RepSegment] = []
    for path in whole_session_files(args.data_dirs):
        df = read_session(path, args.data_dirs)
        if not all(col in df.columns for col in IMU_COLUMNS):
            continue
        session_cache[path] = df
        truth.extend(true_rep_segments(df, path, min_samples=args.min_segment_samples))

    if not truth:
        raise RuntimeError("No labeled repetitions found. Check phase/rep annotations and data paths.")

    truth_labels = [segment.exercise for segment in truth]
    class_names = select_classes(truth_labels, args.num_classes, args.include_other)

    if args.segment_method == "labels":
        predicted = list(truth)
    else:
        predicted = []
        truth_by_file: dict[Path, list[RepSegment]] = {}
        for segment in truth:
            truth_by_file.setdefault(segment.file_path, []).append(segment)
        for path, df in session_cache.items():
            predicted.extend(
                pca_extrema_segments(
                    df,
                    path,
                    truth_by_file.get(path, []),
                    smooth_window=args.smooth_window,
                    min_samples=args.min_segment_samples,
                    peak_prominence_scale=args.peak_prominence_scale,
                )
            )

    segments, labels, manifest_rows = label_predicted_segments(
        predicted,
        truth,
        class_names=class_names,
        include_other=args.include_other,
        min_iou=args.min_label_iou,
    )
    if not segments:
        raise RuntimeError("No predicted repetition segments could be labeled for classification.")

    x, y, groups = build_feature_table(segments, labels, session_cache)
    metrics = run_group_kfold(x, y, groups, class_names, args.folds, args.seed, args.output_dir)

    write_csv(args.output_dir / "rep_segments_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "rep_segmentation_matches.csv", segmentation_summary(predicted, truth))
    summary = {
        "data_dirs": [str(path) for path in args.data_dirs],
        "segment_method": args.segment_method,
        "num_truth_reps": len(truth),
        "num_predicted_reps": len(predicted),
        "num_classified_reps": len(segments),
        "class_names": class_names,
        **metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
