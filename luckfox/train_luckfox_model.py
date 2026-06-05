from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from features import (  # noqa: E402
    IMU_COLUMNS,
    OTHER_LABEL,
    active_labels_from_rows,
    clean_label,
    endpoint_label,
    extract_features,
    infer_time_seconds,
    safe_float,
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


def whole_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*whole_session*.csv"))


def read_session(path: Path) -> dict[str, object]:
    raw_times: list[float] = []
    samples: list[list[float]] = []
    actions: list[str] = []
    phases: list[str] = []
    subject = ""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_times.append(safe_float(row.get("sensor_ts"), float(len(raw_times)) / 100.0))
            samples.append([safe_float(row.get(col)) for col in IMU_COLUMNS])
            actions.append(clean_label(row.get("action_type")))
            phases.append(clean_label(row.get("phase")))
            if not subject:
                subject = str(row.get("subject_id") or "").strip()
    return {
        "path": path,
        "subject": subject or path.parent.name,
        "times": infer_time_seconds(raw_times),
        "samples": samples,
        "actions": actions,
        "phases": phases,
    }


def discover_exercises(files: list[Path]) -> list[str]:
    exercises: set[str] = set()
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                action = clean_label(row.get("action_type"))
                phase = clean_label(row.get("phase"))
                if action and action != "big_rest" and phase in {"concentric", "eccentric"}:
                    exercises.add(action)
    return sorted(exercises)


def build_windows_for_session(
    session: dict[str, object],
    exercise_labels: list[str],
    scales_seconds: list[float],
    stride_seconds: float,
    endpoint_seconds: float,
    endpoint_min_active_fraction: float,
    train_window_step: int,
) -> tuple[list[list[float]], list[str], list[dict[str, object]]]:
    times = session["times"]
    samples = session["samples"]
    actions = session["actions"]
    phases = session["phases"]
    labels, active = active_labels_from_rows(actions, phases, exercise_labels)
    max_scale = max(scales_seconds)
    if not times or times[-1] < max_scale:
        return [], [], []

    rows: list[list[float]] = []
    y: list[str] = []
    manifest: list[dict[str, object]] = []
    end_time = max_scale
    window_idx = 0
    while end_time <= times[-1] + 1e-9:
        if window_idx % max(1, train_window_step) != 0:
            end_time += stride_seconds
            window_idx += 1
            continue
        features = extract_features(times, samples, end_time, scales_seconds, include_posture=True)
        if features is not None:
            label, active_fraction = endpoint_label(
                times,
                labels,
                active,
                end_time,
                endpoint_seconds,
                exercise_labels,
                endpoint_min_active_fraction,
            )
            rows.append(features)
            y.append(label)
            manifest.append(
                {
                    "file": str(session["path"]),
                    "subject": str(session["subject"]),
                    "window_index": window_idx,
                    "end_seconds": round(end_time, 6),
                    "label": label,
                    "endpoint_active_fraction": active_fraction,
                }
            )
        end_time += stride_seconds
        window_idx += 1
    return rows, y, manifest


def export_tree(tree) -> dict[str, object]:
    raw_values = tree.tree_.value
    if raw_values.ndim == 3:
        raw_values = raw_values[:, 0, :]
    return {
        "children_left": tree.tree_.children_left.astype(int).tolist(),
        "children_right": tree.tree_.children_right.astype(int).tolist(),
        "feature": tree.tree_.feature.astype(int).tolist(),
        "threshold": tree.tree_.threshold.astype(float).tolist(),
        "value": raw_values.astype(float).tolist(),
    }


def export_forest(model: RandomForestClassifier) -> dict[str, object]:
    return {
        "classes": [str(item) for item in model.classes_.tolist()],
        "trees": [export_tree(tree) for tree in model.estimators_],
    }


def balanced_active_indices(y: np.ndarray, other_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active_idx = np.flatnonzero(y != OTHER_LABEL)
    other_idx = np.flatnonzero(y == OTHER_LABEL)
    max_other = int(round(len(active_idx) * other_ratio))
    if len(other_idx) > max_other:
        other_idx = rng.choice(other_idx, size=max_other, replace=False)
    selected = np.concatenate([active_idx, other_idx])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export pure-Python LuckFox realtime IMU model.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/workout"))
    parser.add_argument("--output-dir", type=Path, default=Path("luckfox/artifacts/holdout_yentsen"))
    parser.add_argument("--holdout-subject", default="yentsen0515workout")
    parser.add_argument("--scales-seconds", type=float, nargs="+", default=[0.75, 2.0, 4.0])
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-seconds", type=float, default=0.5)
    parser.add_argument("--endpoint-min-active-fraction", type=float, default=0.25)
    parser.add_argument("--active-threshold", type=float, default=0.30)
    parser.add_argument("--class-active-threshold", action="append", default=["db_rdl=0.12"])
    parser.add_argument("--active-smooth-windows", type=int, default=5)
    parser.add_argument("--action-smooth-windows", type=int, default=7)
    parser.add_argument("--confirmation-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirmation-window-seconds", type=float, default=4.0)
    parser.add_argument("--confirmation-min-action-windows", type=int, default=8)
    parser.add_argument("--confirmation-min-action-ratio", type=float, default=0.9)
    parser.add_argument("--confirmation-streak-windows", type=int, default=3)
    parser.add_argument("--confirmation-min-peaks", type=int, default=2)
    parser.add_argument("--confirmation-peak-z", type=float, default=0.7)
    parser.add_argument("--confirmation-min-peak-distance-seconds", type=float, default=0.55)
    parser.add_argument("--mad-gate-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mad-gate-window-seconds", type=float, default=2.0)
    parser.add_argument("--mad-gate-min-acc", type=float, default=0.018)
    parser.add_argument("--mad-gate-min-gyro", type=float, default=1.0)
    parser.add_argument("--mad-gate-mode", choices=["or", "and"], default="or")
    parser.add_argument("--active-other-train-ratio", type=float, default=0.75)
    parser.add_argument("--active-estimators", type=int, default=90)
    parser.add_argument("--action-estimators", type=int, default=140)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument(
        "--train-window-step",
        type=int,
        default=1,
        help="Use every Nth training window. Inference still runs at the configured stride.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = whole_session_files(args.data_dir)
    exercise_labels = discover_exercises(files)
    class_thresholds: dict[str, float] = {}
    for item in args.class_active_threshold:
        if "=" not in item:
            raise ValueError(f"--class-active-threshold must be label=value, got {item!r}")
        label, threshold = item.split("=", 1)
        class_thresholds[label.strip()] = float(threshold)

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    train_files: list[str] = []
    holdout_files: list[str] = []

    start = time.perf_counter()
    for path in files:
        session = read_session(path)
        is_holdout = str(session["subject"]) == args.holdout_subject
        if is_holdout:
            holdout_files.append(str(path))
            continue
        train_files.append(str(path))
        X_file, y_file, manifest_file = build_windows_for_session(
            session,
            exercise_labels,
            [float(item) for item in args.scales_seconds],
            args.stride_seconds,
            args.endpoint_seconds,
            args.endpoint_min_active_fraction,
            args.train_window_step,
        )
        X_rows.extend(X_file)
        y_rows.extend(y_file)
        manifest_rows.extend(manifest_file)
        print(f"[train-window] {session['subject']}: {len(y_file)} from {path.name}", flush=True)
    if not X_rows:
        raise RuntimeError("No training windows generated.")
    if not holdout_files:
        raise RuntimeError(f"No files matched holdout subject {args.holdout_subject!r}.")

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    active_idx = balanced_active_indices(y, args.active_other_train_ratio, args.seed)
    y_active = np.where(y[active_idx] == OTHER_LABEL, OTHER_LABEL, "Active")
    active_model = RandomForestClassifier(
        n_estimators=args.active_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        class_weight={"Active": 1.35, OTHER_LABEL: 1.0},
        random_state=args.seed,
        n_jobs=-1,
    )
    action_idx = np.flatnonzero(y != OTHER_LABEL)
    action_model = RandomForestClassifier(
        n_estimators=args.action_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed + 100,
        n_jobs=-1,
    )
    fit_start = time.perf_counter()
    active_model.fit(X_scaled[active_idx], y_active)
    action_model.fit(X_scaled[action_idx], y[action_idx])
    fit_seconds = time.perf_counter() - fit_start

    payload = {
        "version": 1,
        "labels": [*exercise_labels, OTHER_LABEL],
        "display_labels": {label: DISPLAY_LABELS.get(label, label) for label in [*exercise_labels, OTHER_LABEL]},
        "config": {
            "scales_seconds": [float(item) for item in args.scales_seconds],
            "stride_seconds": float(args.stride_seconds),
            "endpoint_seconds": float(args.endpoint_seconds),
            "endpoint_min_active_fraction": float(args.endpoint_min_active_fraction),
            "active_threshold": float(args.active_threshold),
            "class_active_thresholds": class_thresholds,
            "active_smooth_windows": int(args.active_smooth_windows),
            "action_smooth_windows": int(args.action_smooth_windows),
            "confirmation_enabled": bool(args.confirmation_enabled),
            "confirmation_window_seconds": float(args.confirmation_window_seconds),
            "confirmation_min_action_windows": int(args.confirmation_min_action_windows),
            "confirmation_min_action_ratio": float(args.confirmation_min_action_ratio),
            "confirmation_streak_windows": int(args.confirmation_streak_windows),
            "confirmation_min_peaks": int(args.confirmation_min_peaks),
            "confirmation_peak_z": float(args.confirmation_peak_z),
            "confirmation_min_peak_distance_seconds": float(args.confirmation_min_peak_distance_seconds),
            "mad_gate_enabled": bool(args.mad_gate_enabled),
            "mad_gate_window_seconds": float(args.mad_gate_window_seconds),
            "mad_gate_min_acc": float(args.mad_gate_min_acc),
            "mad_gate_min_gyro": float(args.mad_gate_min_gyro),
            "mad_gate_mode": str(args.mad_gate_mode),
            "imu_columns": list(IMU_COLUMNS),
            "holdout_subject": args.holdout_subject,
        },
        "scaler": {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
        },
        "active_model": export_forest(active_model),
        "action_model": export_forest(action_model),
    }
    model_path = args.output_dir / "luckfox_model.json"
    model_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    write_csv(args.output_dir / "train_window_manifest.csv", manifest_rows)
    summary = {
        "holdout_subject": args.holdout_subject,
        "train_files": train_files,
        "holdout_files": holdout_files,
        "labels": payload["labels"],
        "training_windows": int(len(y)),
        "train_window_step": int(args.train_window_step),
        "active_training_windows": int(np.sum(y != OTHER_LABEL)),
        "other_training_windows": int(np.sum(y == OTHER_LABEL)),
        "feature_count": int(X.shape[1]),
        "feature_extraction_seconds": float(fit_start - start),
        "fit_seconds": float(fit_seconds),
        "model_path": str(model_path),
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
