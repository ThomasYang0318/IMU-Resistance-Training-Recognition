from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_ds_rows(run_dir: Path, domain: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    path = run_dir / "ds_ms_tcn_method_comparison.csv"
    if not path.exists():
        return rows
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        method = str(row["method"])
        rep_values: dict[float, float] = {}
        rep_metrics_path = run_dir / method / "rep_segmentation_metrics.csv"
        if rep_metrics_path.exists():
            rep_df = pd.read_csv(rep_metrics_path)
            rep_values = {
                round(float(metric_row["iou_threshold"]), 2): float(metric_row["f1"])
                for _, metric_row in rep_df.iterrows()
            }
        rows.append(
            {
                "domain": domain,
                "method": method,
                "metric_family": "sequence_model",
                "macro_sample_f1": float(row.get("macro_sample_f1", 0.0)),
                "macro_segment_f1_iou_0.50": float(row.get("macro_segment_f1_iou_0.50", 0.0)),
                "rep_f1_iou_0.50": rep_values.get(0.5, float(row.get("rep_f1_iou_0.50", 0.0))),
                "rep_f1_iou_0.75": rep_values.get(0.75, float(row.get("rep_f1_iou_0.75", np.nan))),
                "rep_f1_iou_0.90": rep_values.get(0.9, float(row.get("rep_f1_iou_0.90", 0.0))),
                "source_dir": str(run_dir),
            }
        )
    return rows


def read_classical_rows(run_dir: Path, method: str) -> list[dict[str, object]]:
    path = run_dir / "rep_segmentation_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    values: dict[float, float] = {}
    for _, row in df.iterrows():
        values[round(float(row["iou_threshold"]), 2)] = float(row["f1"])
    return [
        {
            "domain": "classical_active_only",
            "method": method,
            "metric_family": "rep_boundary_only",
            "macro_sample_f1": np.nan,
            "macro_segment_f1_iou_0.50": np.nan,
            "rep_f1_iou_0.50": values.get(0.5, np.nan),
            "rep_f1_iou_0.75": values.get(0.75, np.nan),
            "rep_f1_iou_0.90": values.get(0.9, np.nan),
            "source_dir": str(run_dir),
        }
    ]


def plot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    labels = [f"{row.domain}\n{row.method}" for row in df.itertuples()]
    x = np.arange(len(labels))
    metrics = [
        ("macro_sample_f1", "Macro sample F1"),
        ("macro_segment_f1_iou_0.50", "Macro segment F1@0.50"),
        ("rep_f1_iou_0.50", "Rep F1@0.50"),
        ("rep_f1_iou_0.90", "Rep F1@0.90"),
    ]
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.35), 6))
    for idx, (col, label) in enumerate(metrics):
        values = df[col].fillna(0.0).astype(float).to_numpy()
        ax.bar(x + (idx - 1.5) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("012 DS-MS-TCN 9-axis vs Existing Rep Boundary Methods")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "012_ds_ms_tcn_vs_existing_methods.png", dpi=180)
    plt.close(fig)


def plot_table(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    display = df[
        [
            "domain",
            "method",
            "macro_sample_f1",
            "macro_segment_f1_iou_0.50",
            "rep_f1_iou_0.50",
            "rep_f1_iou_0.75",
            "rep_f1_iou_0.90",
        ]
    ].copy()
    for col in display.columns:
        if col in {"domain", "method"}:
            continue
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    fig, ax = plt.subplots(figsize=(13, max(3, 0.45 * len(display) + 1.6)))
    ax.axis("off")
    table = ax.table(
        cellText=display.to_numpy(),
        colLabels=[
            "Domain",
            "Method",
            "Macro sample F1",
            "Macro segment F1@0.50",
            "Rep F1@0.50",
            "Rep F1@0.75",
            "Rep F1@0.90",
        ],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#dbe8f6")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    ax.set_title("Paper-style Method Result Table", pad=18)
    fig.tight_layout()
    fig.savefig(output_dir / "012_ds_ms_tcn_method_table.png", dpi=180)
    plt.close(fig)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_comparison(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rows.extend(read_ds_rows(args.exercise_only_dir, "exercise_only"))
    rows.extend(read_ds_rows(args.full_session_dir, "full_session_other"))
    rows.extend(read_classical_rows(args.universal_gyro_dir, "010_universal_gyro_valley"))
    rows.extend(read_classical_rows(args.multifeature_dir, "011_multifeature_boundary_score"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No comparison inputs found.")
    df.to_csv(args.output_dir / "012_ds_ms_tcn_method_comparison.csv", index=False)
    plot_comparison(df, args.output_dir)
    plot_table(df, args.output_dir)
    write_json(
        args.output_dir / "summary.json",
        {
            "rows": rows,
            "note": (
                "DS-MS-TCN rows use 9-axis sequence models. Existing 010/011 rows are 6-axis/classical "
                "rep-boundary metrics, so macro sample/segment metrics are intentionally blank."
            ),
        },
    )
    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 012 DS-MS-TCN outputs with existing rep segmentation methods.")
    parser.add_argument("--exercise-only-dir", type=Path, default=Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_exercise_only"))
    parser.add_argument("--full-session-dir", type=Path, default=Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_full_session_other"))
    parser.add_argument("--universal-gyro-dir", type=Path, default=Path("artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold"))
    parser.add_argument("--multifeature-dir", type=Path, default=Path("artifacts_rep_classification/011_multifeature_boundary_score_high_iou"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_method_comparison"))
    return parser.parse_args()


if __name__ == "__main__":
    build_comparison(parse_args())
