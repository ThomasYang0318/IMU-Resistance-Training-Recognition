from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from evaluate_realtime_action_rf import (
    IMU_9AXIS,
    OTHER_LABEL,
    basic_stats,
    discover_exercises,
    infer_time_seconds,
    sample_labels,
    whole_session_files,
)
from evaluate_realtime_action_twostage_rf import (
    endpoint_label,
    extract_enhanced_features,
    make_feature_names,
    save_confusion,
)


def lowpass_gravity(acc: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    if len(acc) == 0:
        return acc.astype(np.float32)
    out = np.zeros_like(acc, dtype=np.float32)
    out[0] = acc[0]
    for idx in range(1, len(acc)):
        out[idx] = alpha * acc[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def unit_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-6)


def angle_stats(unit_values: np.ndarray, mean_unit: np.ndarray) -> list[float]:
    if len(unit_values) == 0:
        return [0.0, 0.0, 0.0]
    dots = np.clip(unit_values @ mean_unit, -1.0, 1.0)
    angles = np.arccos(dots)
    return [float(np.mean(angles)), float(np.std(angles)), float(np.max(angles))]


def projection_stats(values: np.ndarray, gravity_unit: np.ndarray) -> list[float]:
    vertical = values @ gravity_unit
    horizontal = values - np.outer(vertical, gravity_unit)
    horizontal_mag = np.linalg.norm(horizontal, axis=1)
    return [
        *basic_stats(vertical),
        *basic_stats(horizontal_mag),
        float(np.sqrt(np.mean(vertical * vertical)) / max(float(np.sqrt(np.mean(horizontal_mag * horizontal_mag))), 1e-6)),
    ]


def posture_feature_names(prefix: str) -> list[str]:
    stats = ("mean", "std", "min", "max", "range", "rms", "median", "mean_abs", "delta", "diff_std")
    names: list[str] = []
    for signal in ("pitch", "roll", "inclination", "gravity_norm"):
        names.extend([f"{prefix}_{signal}_{stat}" for stat in stats])
    for component in ("x", "y", "z"):
        names.extend([f"{prefix}_gravity_unit_{component}_mean", f"{prefix}_gravity_unit_{component}_std"])
    names.extend(
        [
            f"{prefix}_gravity_angle_mean",
            f"{prefix}_gravity_angle_std",
            f"{prefix}_gravity_angle_max",
            f"{prefix}_pitch_delta",
            f"{prefix}_roll_delta",
            f"{prefix}_inclination_delta",
        ]
    )
    for group in ("acc", "gyro", "mag"):
        names.extend([f"{prefix}_{group}_vertical_{stat}" for stat in stats])
        names.extend([f"{prefix}_{group}_horizontal_{stat}" for stat in stats])
        names.append(f"{prefix}_{group}_vertical_horizontal_rms_ratio")
    return names


def make_all_feature_names(scales_seconds: list[float], include_posture: bool) -> list[str]:
    names = make_feature_names(scales_seconds)
    if include_posture:
        for scale in scales_seconds:
            names.extend(posture_feature_names(f"w{scale:g}s"))
    return names


def extract_posture_features(block: np.ndarray) -> np.ndarray:
    acc = block[:, 0:3]
    gyro = block[:, 3:6]
    mag = block[:, 6:9]
    gravity = lowpass_gravity(acc)
    gravity_unit_series = unit_vectors(gravity)
    gravity_mean = np.mean(gravity, axis=0)
    gravity_norm = float(np.linalg.norm(gravity_mean))
    gravity_unit = gravity_mean / max(gravity_norm, 1e-6)

    gx = gravity_unit_series[:, 0]
    gy = gravity_unit_series[:, 1]
    gz = gravity_unit_series[:, 2]
    pitch = np.arctan2(-gx, np.sqrt(gy * gy + gz * gz))
    roll = np.arctan2(gy, gz)
    inclination = np.arccos(np.clip(np.abs(gz), 0.0, 1.0))
    gravity_norm_series = np.linalg.norm(gravity, axis=1)

    features: list[float] = []
    for signal in (pitch, roll, inclination, gravity_norm_series):
        features.extend(basic_stats(signal))
    for component_idx in range(3):
        component = gravity_unit_series[:, component_idx]
        features.extend([float(np.mean(component)), float(np.std(component))])
    features.extend(angle_stats(gravity_unit_series, gravity_unit))
    features.extend(
        [
            float(pitch[-1] - pitch[0]) if len(pitch) else 0.0,
            float(roll[-1] - roll[0]) if len(roll) else 0.0,
            float(inclination[-1] - inclination[0]) if len(inclination) else 0.0,
        ]
    )
    for values in (acc, gyro, mag):
        features.extend(projection_stats(values, gravity_unit))
    return np.asarray(features, dtype=np.float32)


def build_active_windows(
    path: Path,
    exercises: list[str],
    scales_seconds: list[float],
    stride_seconds: float,
    endpoint_seconds: float,
    endpoint_min_active_fraction: float,
    include_posture: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_count = len(make_all_feature_names(scales_seconds, include_posture))
    required = set(IMU_9AXIS) | {"sensor_ts", "action_type", "phase", "subject_id"}
    df = pd.read_csv(path, usecols=lambda col: col in required)
    df = df.dropna(subset=list(IMU_9AXIS) + ["sensor_ts"]).reset_index(drop=True)
    if df.empty:
        return np.empty((0, feature_count), dtype=np.float32), np.empty((0,), dtype=object), pd.DataFrame()

    times = infer_time_seconds(df["sensor_ts"])
    values = df.loc[:, IMU_9AXIS].to_numpy(dtype=np.float32)
    labels, active = sample_labels(df, exercises)
    subject = str(df["subject_id"].dropna().astype(str).iloc[0]) if df["subject_id"].notna().any() else path.parent.name
    max_scale = max(scales_seconds)
    if len(times) == 0 or float(times[-1] - times[0]) < max_scale:
        return np.empty((0, feature_count), dtype=np.float32), np.empty((0,), dtype=object), pd.DataFrame()

    end_times = np.arange(times[0] + max_scale, times[-1] + 1e-9, stride_seconds, dtype=np.float64)
    feature_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for window_idx, end_time in enumerate(end_times):
        tail_start_idx = int(np.searchsorted(times, end_time - endpoint_seconds, side="left"))
        tail_end_idx = int(np.searchsorted(times, end_time, side="right"))
        label, active_fraction = endpoint_label(
            labels[tail_start_idx:tail_end_idx],
            active[tail_start_idx:tail_end_idx],
            exercises,
            endpoint_min_active_fraction,
        )
        if label == OTHER_LABEL:
            continue

        row_features: list[np.ndarray] = []
        valid = True
        for scale in scales_seconds:
            start_idx = int(np.searchsorted(times, end_time - float(scale), side="left"))
            end_idx = int(np.searchsorted(times, end_time, side="right"))
            if end_idx <= start_idx:
                valid = False
                break
            block = values[start_idx:end_idx]
            scale_features = [extract_enhanced_features(block, float(times[end_idx - 1] - times[start_idx]))]
            if include_posture:
                scale_features.append(extract_posture_features(block))
            row_features.append(np.concatenate(scale_features).astype(np.float32))
        if not valid:
            continue
        feature_rows.append(np.concatenate(row_features).astype(np.float32))
        y_rows.append(label)
        manifest_rows.append(
            {
                "file": str(path),
                "subject": subject,
                "window_index": int(window_idx),
                "end_seconds": float(end_time - times[0]),
                "label": label,
                "endpoint_active_fraction": active_fraction,
            }
        )
    X = np.vstack(feature_rows).astype(np.float32) if feature_rows else np.empty((0, len(make_feature_names(scales_seconds))), dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    return X, y, pd.DataFrame(manifest_rows)


def classifier_factory(name: str, seed: int):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest_deep":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest_leaf2":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


def plot_active_matrix(prop: pd.DataFrame, output_path: Path, title: str) -> None:
    short = {
        "db_bench_press": "Bench",
        "db_biceps_curl": "Biceps",
        "db_rdl": "RDL",
        "db_shoulder_press": "Shoulder",
        "db_squat": "Squat",
        "db_triceps_curl": "Triceps",
        "db_weighted_crunch": "Crunch",
        "one_arm_db_row": "Row",
    }
    labels = list(prop.index)
    display_labels = [short.get(label, label) for label in labels]
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(prop.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted exercise")
    ax.set_ylabel("True exercise")
    ax.set_title(title)
    for row in range(prop.shape[0]):
        for col in range(prop.shape[1]):
            value = prop.iat[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=("white" if value >= 0.5 else "#1f2937"), fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within true exercise")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active-only 8-class classifiers on causal multi-scale IMU windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/workout"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/realtime_action_active_only_8class_models_20260518"))
    parser.add_argument("--scales-seconds", type=float, nargs="+", default=[0.75, 2.0, 4.0])
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-min-active-fraction", type=float, default=0.25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-posture", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["extra_trees", "random_forest_deep", "random_forest_leaf2", "hist_gradient_boosting"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = whole_session_files(args.data_dir)
    exercises = discover_exercises(files)
    feature_names = make_all_feature_names([float(scale) for scale in args.scales_seconds], args.include_posture)

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    manifest_chunks: list[pd.DataFrame] = []
    feature_start = time.perf_counter()
    for path in files:
        X_file, y_file, manifest_file = build_active_windows(
            path,
            exercises,
            [float(scale) for scale in args.scales_seconds],
            args.stride_seconds,
            args.endpoint_seconds,
            args.endpoint_min_active_fraction,
            args.include_posture,
        )
        if len(y_file):
            feature_chunks.append(X_file)
            label_chunks.append(y_file)
            manifest_chunks.append(manifest_file)
        print(f"[active-window] {path.name}: {len(y_file)}", flush=True)
    if not feature_chunks:
        raise RuntimeError("No active windows generated.")

    X = np.vstack(feature_chunks).astype(np.float32)
    y = np.concatenate(label_chunks).astype(object)
    manifest = pd.concat(manifest_chunks, ignore_index=True)
    groups = manifest["subject"].astype(str).to_numpy()
    np.savez_compressed(args.output_dir / "active_window_features.npz", X=X, y=y, groups=groups, labels=np.asarray(exercises, dtype=object))
    manifest.to_csv(args.output_dir / "active_window_manifest.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(args.output_dir / "feature_names.csv", index=False)
    pd.Series(y).value_counts().reindex(exercises, fill_value=0).to_csv(args.output_dir / "label_counts.csv", header=["windows"])

    n_splits = min(args.n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    all_model_summaries: dict[str, object] = {}
    best_name = ""
    best_macro = -1.0
    for model_name in args.models:
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        pred_rows: list[pd.DataFrame] = []
        fold_rows: list[dict[str, object]] = []
        train_seconds = 0.0
        predict_seconds = 0.0
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            clf = classifier_factory(model_name, args.seed + fold_idx)
            fit_start = time.perf_counter()
            clf.fit(X[train_idx], y[train_idx])
            train_seconds += time.perf_counter() - fit_start
            pred_start = time.perf_counter()
            pred = clf.predict(X[test_idx])
            predict_seconds += time.perf_counter() - pred_start

            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "test_subjects": ",".join(sorted(set(groups[test_idx].tolist()))),
                    "accuracy": float(accuracy_score(y[test_idx], pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                    "macro_f1": float(f1_score(y[test_idx], pred, labels=exercises, average="macro", zero_division=0)),
                }
            )
            frame = manifest.iloc[test_idx].copy()
            frame["prediction"] = pred
            frame["fold"] = fold_idx
            pred_rows.append(frame)
            print(f"[{model_name}] fold={fold_idx} acc={fold_rows[-1]['accuracy']:.4f} macro={fold_rows[-1]['macro_f1']:.4f}", flush=True)

        y_true = np.asarray(y_true_all, dtype=object)
        y_pred = np.asarray(y_pred_all, dtype=object)
        cm = confusion_matrix(y_true, y_pred, labels=exercises)
        row_sum = cm.sum(axis=1, keepdims=True)
        prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
        report = classification_report(y_true, y_pred, labels=exercises, output_dict=True, zero_division=0)
        summary = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=exercises, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=exercises, average="weighted", zero_division=0)),
            "min_recall": float(np.diag(prop).min()),
            "per_class_recall": {label: float(prop[idx, idx]) for idx, label in enumerate(exercises)},
            "train_seconds": float(train_seconds),
            "predict_seconds": float(predict_seconds),
            "predict_ms_per_window": float((predict_seconds / max(1, len(y_true))) * 1000.0),
        }
        all_model_summaries[model_name] = summary
        if summary["macro_f1"] > best_macro:
            best_macro = summary["macro_f1"]
            best_name = model_name

        stem = f"confusion_matrix_{model_name}"
        pd.DataFrame(cm, index=exercises, columns=exercises).to_csv(args.output_dir / f"{stem}.csv")
        pd.DataFrame(prop, index=exercises, columns=exercises).to_csv(args.output_dir / f"{stem}_row_proportion.csv", float_format="%.6f")
        pd.DataFrame(prop * 100.0, index=exercises, columns=exercises).to_csv(args.output_dir / f"{stem}_row_percent.csv", float_format="%.2f")
        plot_active_matrix(pd.DataFrame(prop, index=exercises, columns=exercises), args.output_dir / f"{stem}_row_proportion.png", f"Active-only 8-class recognition ({model_name})")
        (args.output_dir / f"classification_report_{model_name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(fold_rows).to_csv(args.output_dir / f"fold_metrics_{model_name}.csv", index=False)
        pd.concat(pred_rows, ignore_index=True).to_csv(args.output_dir / f"window_predictions_{model_name}.csv", index=False)

    run_summary = {
        "labels": exercises,
        "total_active_windows": int(len(y)),
        "include_posture": bool(args.include_posture),
        "feature_extraction_seconds": float(time.perf_counter() - feature_start),
        "best_model_by_macro_f1": best_name,
        "models": all_model_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
