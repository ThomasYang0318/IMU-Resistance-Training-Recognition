from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodRun:
    method_id: str
    method_name: str
    method_description: str
    domain: str
    run_dir: Path
    pred_kind: str
    sample_rate_hz: float | None = None


DEFAULT_RUNS = (
    MethodRun(
        "010_universal_gyro_valley",
        "UGV: Universal Gyro Valley",
        "Classical periodic rep segmentation using PCA/autocorrelation period prior and gyro-magnitude valley boundaries.",
        "active-only",
        Path("artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold"),
        "matches",
        None,
    ),
    MethodRun(
        "011_multifeature_boundary_score",
        "MFBS: Multi-Feature Boundary Score",
        "Classical active-only boundary scoring that fuses multiple IMU features and duration priors for high-IoU rep cuts.",
        "active-only",
        Path("artifacts_rep_classification/011_multifeature_boundary_score_high_iou"),
        "matches",
        None,
    ),
    MethodRun(
        "012_exercise_only_ds_ms_tcn",
        "9A-DS-MS-TCN-EO",
        "9-axis adapted dual-scale multi-stage TCN trained on exercise-only slices with micro phase labels.",
        "exercise-only",
        Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_exercise_only/ds_ms_tcn"),
        "segments",
        50.0,
    ),
    MethodRun(
        "012_exercise_only_ms_tcn",
        "9A-MS-TCN-EO",
        "9-axis multi-stage TCN exercise-only baseline without micro phase labels.",
        "exercise-only",
        Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_exercise_only/ms_tcn"),
        "segments",
        50.0,
    ),
    MethodRun(
        "012_full_session_ds_ms_tcn",
        "9A-DS-MS-TCN-FS+O",
        "9-axis adapted DS-MS-TCN trained on full-session slices with an additional other class.",
        "full-session+other",
        Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_full_session_other/ds_ms_tcn"),
        "segments",
        50.0,
    ),
    MethodRun(
        "012_full_session_ms_tcn",
        "9A-MS-TCN-FS+O",
        "9-axis MS-TCN full-session baseline without micro phase labels, including other class.",
        "full-session+other",
        Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_full_session_other/ms_tcn"),
        "segments",
        50.0,
    ),
)


def infer_sensor_period_seconds(path: Path) -> float:
    try:
        sensor_ts = pd.read_csv(path, usecols=["sensor_ts"], nrows=50000)["sensor_ts"]
    except Exception:
        return 0.01
    values = pd.to_numeric(sensor_ts, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(values) < 2:
        return 0.01
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.01
    median_delta = float(np.median(diffs))
    if median_delta > 1000.0:
        return median_delta / 1_000_000.0
    if median_delta > 10.0:
        return median_delta / 1000.0
    return median_delta


def read_truth(run: MethodRun) -> pd.DataFrame:
    if run.pred_kind == "matches":
        path = run.run_dir / "rep_segmentation_truth_matches.csv"
        df = pd.read_csv(path)
        out = pd.DataFrame(
            {
                "file": df["file"].astype(str),
                "subject": df["subject"].astype(str),
                "exercise": df["exercise"].astype(str),
                "set_id": df["set_id"].astype(str),
                "rep_id": df["rep_id"].astype(str),
                "start": df["true_start"].astype(int),
                "end": df["true_end"].astype(int),
            }
        )
    else:
        path = run.run_dir / "rep_segmentation_truth_segments.csv"
        df = pd.read_csv(path)
        out = pd.DataFrame(
            {
                "file": df["file"].astype(str),
                "subject": df["subject"].astype(str),
                "exercise": df["exercise"].astype(str),
                "set_id": df["set_id"].astype(str),
                "rep_id": df["rep_id"].astype(str),
                "start": df["start"].astype(int),
                "end": df["end"].astype(int),
            }
        )
    out["samples"] = out["end"] - out["start"]
    return out


def read_pred(run: MethodRun) -> pd.DataFrame:
    if run.pred_kind == "matches":
        path = run.run_dir / "rep_segmentation_matches.csv"
        df = pd.read_csv(path)
        out = pd.DataFrame(
            {
                "file": df["file"].astype(str),
                "subject": df["subject"].astype(str),
                "exercise": df["exercise_hint"].astype(str),
                "start": df["start"].astype(int),
                "end": df["end"].astype(int),
            }
        )
    else:
        path = run.run_dir / "rep_segmentation_pred_segments.csv"
        df = pd.read_csv(path)
        out = pd.DataFrame(
            {
                "file": df["file"].astype(str),
                "subject": df["subject"].astype(str),
                "exercise": df["exercise"].astype(str),
                "start": df["start"].astype(int),
                "end": df["end"].astype(int),
            }
        )
    out["samples"] = out["end"] - out["start"]
    return out


def sample_periods_for_run(run: MethodRun, files: Sequence[str]) -> dict[str, float]:
    if run.sample_rate_hz is not None:
        period = 1.0 / float(run.sample_rate_hz)
        return {file: period for file in files}
    return {file: infer_sensor_period_seconds(Path(file)) for file in files}


def set_table_from_truth(truth: pd.DataFrame, periods: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in truth.groupby(["file", "subject", "exercise", "set_id"], sort=True):
        file, subject, exercise, set_id = key
        period = periods.get(str(file), 0.01)
        rows.append(
            {
                "file": str(file),
                "subject": str(subject),
                "exercise": str(exercise),
                "set_id": str(set_id),
                "set_start": int(group["start"].min()),
                "set_end": int(group["end"].max()),
                "true_count": int(len(group)),
                "true_tut_samples": int(group["samples"].sum()),
                "true_tut_sec": float(group["samples"].sum()) * period,
            }
        )
    return pd.DataFrame(rows)


def assign_predictions_to_sets(pred: pd.DataFrame, sets: pd.DataFrame, periods: dict[str, float]) -> tuple[pd.DataFrame, int]:
    set_rows = sets.copy()
    set_rows["pred_count"] = 0
    set_rows["pred_tut_samples"] = 0
    set_rows["pred_tut_sec"] = 0.0

    by_file_exercise: dict[tuple[str, str], list[tuple[int, pd.Series]]] = {}
    for idx, row in set_rows.iterrows():
        by_file_exercise.setdefault((str(row["file"]), str(row["exercise"])), []).append((idx, row))

    unassigned = 0
    for pred_row in pred.itertuples(index=False):
        key = (str(pred_row.file), str(pred_row.exercise))
        candidates = by_file_exercise.get(key, [])
        best_idx: int | None = None
        best_overlap = 0
        for set_idx, set_row in candidates:
            overlap = max(0, min(int(pred_row.end), int(set_row["set_end"])) - max(int(pred_row.start), int(set_row["set_start"])))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = int(set_idx)
        if best_idx is None or best_overlap <= 0:
            unassigned += 1
            continue
        samples = int(pred_row.end) - int(pred_row.start)
        period = periods.get(str(pred_row.file), 0.01)
        set_rows.loc[best_idx, "pred_count"] += 1
        set_rows.loc[best_idx, "pred_tut_samples"] += samples
        set_rows.loc[best_idx, "pred_tut_sec"] += samples * period

    return set_rows, unassigned


def read_iou_metrics(run: MethodRun) -> dict[str, float]:
    path = run.run_dir / "rep_segmentation_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    values: dict[str, float] = {}
    for _, row in df.iterrows():
        threshold = float(row["iou_threshold"])
        values[f"rep_f1_iou_{threshold:.2f}"] = float(row["f1"])
    return values


def read_phase_iou_050(run: MethodRun) -> float | None:
    path = run.run_dir / "phase_split_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[np.isclose(df["iou_threshold"].astype(float), 0.5)]
    if row.empty:
        return None
    return float(row.iloc[0]["f1"])


def summarize_run(run: MethodRun, output_dir: Path) -> dict[str, object]:
    truth = read_truth(run)
    pred = read_pred(run)
    periods = sample_periods_for_run(run, sorted(set(truth["file"].astype(str)) | set(pred["file"].astype(str))))
    sets = set_table_from_truth(truth, periods)
    set_counts, unassigned = assign_predictions_to_sets(pred, sets, periods)
    set_counts["count_error"] = set_counts["pred_count"] - set_counts["true_count"]
    set_counts["abs_count_error"] = set_counts["count_error"].abs()
    set_counts["tut_error_sec"] = set_counts["pred_tut_sec"] - set_counts["true_tut_sec"]
    set_counts["abs_tut_error_sec"] = set_counts["tut_error_sec"].abs()
    set_counts["tut_abs_pct_error"] = set_counts["abs_tut_error_sec"] / set_counts["true_tut_sec"].clip(lower=1e-9) * 100.0

    output_dir.mkdir(parents=True, exist_ok=True)
    set_counts.to_csv(output_dir / f"{run.method_id}_set_count_tut_details.csv", index=False)

    iou_metrics = read_iou_metrics(run)
    phase_050 = read_phase_iou_050(run)
    row: dict[str, object] = {
        "method_name": run.method_name,
        "method_id": run.method_id,
        "method_description": run.method_description,
        "domain": run.domain,
        "true_sets": int(len(set_counts)),
        "true_reps": int(truth.shape[0]),
        "predicted_reps": int(pred.shape[0]),
        "unassigned_pred_reps": int(unassigned),
        "count_exact_acc": round(float((set_counts["abs_count_error"] == 0).mean()), 4),
        "count_pm1_acc": round(float((set_counts["abs_count_error"] <= 1).mean()), 4),
        "count_mae_reps": round(float(set_counts["abs_count_error"].mean()), 4),
        "count_bias_reps": round(float(set_counts["count_error"].mean()), 4),
        "tut_mae_sec": round(float(set_counts["abs_tut_error_sec"].mean()), 4),
        "tut_mape_pct": round(float(set_counts["tut_abs_pct_error"].mean()), 4),
        "phase_f1_iou_0.50": round(float(phase_050), 4) if phase_050 is not None else np.nan,
    }
    for threshold in (0.5, 0.75, 0.9):
        row[f"rep_f1_iou_{threshold:.2f}"] = round(float(iou_metrics.get(f"rep_f1_iou_{threshold:.2f}", np.nan)), 4)
    return row


def plot_table(df: pd.DataFrame, output_dir: Path) -> None:
    display_cols = [
        "method_name",
        "count_exact_acc",
        "count_pm1_acc",
        "count_mae_reps",
        "rep_f1_iou_0.50",
        "rep_f1_iou_0.75",
        "rep_f1_iou_0.90",
        "phase_f1_iou_0.50",
        "tut_mae_sec",
    ]
    labels = [
        "Method name",
        "Count exact",
        "Count +/-1",
        "Count MAE",
        "Rep F1@0.50",
        "Rep F1@0.75",
        "Rep F1@0.90",
        "Phase F1@0.50",
        "TUT MAE (s)",
    ]
    table = df.loc[:, display_cols].copy()
    for col in table.columns:
        if col == "method_name":
            continue
        table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")

    fig, ax = plt.subplots(figsize=(16, max(3.5, 0.5 * len(table) + 1.8)))
    ax.axis("off")
    artists = ax.table(
        cellText=table.to_numpy(),
        colLabels=labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    artists.auto_set_font_size(False)
    artists.set_fontsize(8)
    artists.scale(1, 1.35)
    for (row_idx, _col_idx), cell in artists.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#dbe8f6")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    ax.set_title("Count / Rep IoU / Phase IoU / TUT Method Comparison", pad=18)
    fig.tight_layout()
    fig.savefig(output_dir / "count_iou_tut_method_table.png", dpi=180)
    plt.close(fig)


def plot_bars(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("count_pm1_acc", "Count +/-1 acc"),
        ("rep_f1_iou_0.50", "Rep F1@0.50"),
        ("rep_f1_iou_0.75", "Rep F1@0.75"),
        ("rep_f1_iou_0.90", "Rep F1@0.90"),
    ]
    methods = df["method_name"].tolist()
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 1.45), 5.6))
    for idx, (col, label) in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, df[col].fillna(0.0).astype(float).to_numpy(), width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Count Accuracy and Rep Boundary IoU")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "count_iou_method_bars.png", dpi=180)
    plt.close(fig)


def build_table(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(run, args.output_dir / "set_details") for run in DEFAULT_RUNS if run.run_dir.exists()]
    df = pd.DataFrame(rows)
    ordered_cols = [
        "method_name",
        "method_id",
        "method_description",
        "domain",
        "true_sets",
        "true_reps",
        "predicted_reps",
        "unassigned_pred_reps",
        "count_exact_acc",
        "count_pm1_acc",
        "count_mae_reps",
        "count_bias_reps",
        "tut_mae_sec",
        "tut_mape_pct",
        "phase_f1_iou_0.50",
        "rep_f1_iou_0.50",
        "rep_f1_iou_0.75",
        "rep_f1_iou_0.90",
    ]
    df = df.loc[:, ordered_cols]
    df.to_csv(args.output_dir / "count_iou_tut_method_comparison.csv", index=False)
    plot_table(df, args.output_dir)
    plot_bars(df, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_csv": str(args.output_dir / "count_iou_tut_method_comparison.csv"),
                "definition": {
                    "method_name": "Human-readable method name used in report figures.",
                    "method_id": "Artifact/version identifier used to trace the original run directory.",
                    "count_exact_acc": "Per true set, predicted rep count exactly equals true rep count.",
                    "count_pm1_acc": "Per true set, absolute count error is <= 1 rep.",
                    "tut_mae_sec": "Per true set, mean absolute error between summed predicted rep duration and summed true rep duration.",
                    "rep_f1_iou": "Greedy one-to-one rep segment matching F1 at the specified IoU threshold.",
                },
                "method_legend": {
                    str(row["method_name"]): {
                        "method_id": str(row["method_id"]),
                        "description": str(row["method_description"]),
                    }
                    for _, row in df.iterrows()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build count, IoU, phase, and TUT comparison table across methods.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/013_count_iou_tut_method_table"))
    return parser.parse_args()


if __name__ == "__main__":
    build_table(parse_args())
