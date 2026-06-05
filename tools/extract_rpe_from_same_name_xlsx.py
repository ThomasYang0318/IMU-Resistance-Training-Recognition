from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


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

RPE_CLASSES = tuple(range(1, 11))


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


def rpe_class(value: object) -> int | None:
    number = numeric_or_none(value)
    if number is None:
        return None
    return int(np.clip(np.rint(number), 1, 10))


def same_name_xlsx(folder: Path) -> Path | None:
    path = folder / f"{folder.name}.xlsx"
    return path if path.exists() else None


def digit_column_index(column: object) -> int | None:
    text = str(column).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return None
    return int(text)


def read_first_sheet(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def parse_workbook(folder: Path, source_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = read_first_sheet(source_file)
    status: dict[str, object] = {
        "folder": folder.name,
        "source_file": str(source_file),
        "status": "ok",
        "raw_rows": int(len(raw)),
        "parsed_set_rows": 0,
        "parsed_rep_rows": 0,
        "duplicate_row_keys": 0,
    }
    if raw.empty:
        status["status"] = "empty"
        return pd.DataFrame(), pd.DataFrame(), status

    first_col = raw.columns[0]
    kg_col = next((col for col in raw.columns if str(col).strip().lower() == "kg"), None)
    rep_cols = sorted(
        [col for col in raw.columns if digit_column_index(col) is not None],
        key=lambda col: digit_column_index(col) or -1,
    )
    if not rep_cols:
        status["status"] = "missing_rep_columns"
        return pd.DataFrame(), pd.DataFrame(), status

    rows_rep: list[dict[str, object]] = []
    rows_set: list[dict[str, object]] = []
    seen_row_keys: dict[str, int] = {}

    for source_row_index, series in raw.iterrows():
        row_name = series.get(first_col)
        if is_blank(row_name):
            continue
        row_key = str(row_name).strip()
        match = re.match(r"^\s*(\d+)_([0-9]+)\s*$", row_key)
        if not match:
            continue

        exercise_index = int(match.group(1))
        set_id = int(match.group(2))
        if exercise_index < 1 or exercise_index > len(EXERCISE_ORDER):
            continue

        exercise = EXERCISE_ORDER[exercise_index - 1]
        row_instance = seen_row_keys.get(row_key, 0)
        seen_row_keys[row_key] = row_instance + 1
        kg = numeric_or_none(series.get(kg_col)) if kg_col is not None else None

        rep_values: list[float] = []
        raw_values: list[str] = []
        stopped_at_rep: int | None = None
        last_rpe: float | None = None
        completed_reps = 0

        for rep_col in rep_cols:
            rep_id = digit_column_index(rep_col)
            if rep_id is None:
                continue
            value = series.get(rep_col)
            raw_value = "" if is_blank(value) else str(value).strip()

            if isinstance(value, str) and value.strip().upper() == "X":
                stopped_at_rep = rep_id
                rows_rep.append(
                    {
                        "folder": folder.name,
                        "source_file": str(source_file),
                        "source_row_index": int(source_row_index),
                        "row_key": row_key,
                        "row_instance": int(row_instance),
                        "exercise_index": int(exercise_index),
                        "exercise": exercise,
                        "set_id": int(set_id),
                        "rep_id": int(rep_id),
                        "kg": kg,
                        "rpe": np.nan,
                        "rpe_class": np.nan,
                        "completed": False,
                        "raw_value": "X",
                        "filled_from_previous": False,
                    }
                )
                break

            parsed = numeric_or_none(value)
            filled = False
            if parsed is None and last_rpe is not None and is_blank(value):
                parsed = last_rpe
                filled = True
            if parsed is None:
                continue

            last_rpe = parsed
            rep_values.append(float(parsed))
            raw_values.append(raw_value)
            completed_reps += 1
            rows_rep.append(
                {
                    "folder": folder.name,
                    "source_file": str(source_file),
                    "source_row_index": int(source_row_index),
                    "row_key": row_key,
                    "row_instance": int(row_instance),
                    "exercise_index": int(exercise_index),
                    "exercise": exercise,
                    "set_id": int(set_id),
                    "rep_id": int(rep_id),
                    "kg": kg,
                    "rpe": float(parsed),
                    "rpe_class": rpe_class(parsed),
                    "completed": True,
                    "raw_value": raw_value,
                    "filled_from_previous": filled,
                }
            )

        if rep_values:
            rows_set.append(
                {
                    "folder": folder.name,
                    "source_file": str(source_file),
                    "source_row_index": int(source_row_index),
                    "row_key": row_key,
                    "row_instance": int(row_instance),
                    "exercise_index": int(exercise_index),
                    "exercise": exercise,
                    "set_id": int(set_id),
                    "kg": kg,
                    "n_completed_reps": int(completed_reps),
                    "first_rpe": float(rep_values[0]),
                    "final_rpe": float(rep_values[-1]),
                    "final_rpe_class": rpe_class(rep_values[-1]),
                    "mean_rpe": float(np.mean(rep_values)),
                    "median_rpe": float(np.median(rep_values)),
                    "min_rpe": float(np.min(rep_values)),
                    "max_rpe": float(np.max(rep_values)),
                    "rpe_slope_last_minus_first": float(rep_values[-1] - rep_values[0]),
                    "stopped_at_rep": stopped_at_rep,
                    "has_x": stopped_at_rep is not None,
                    "raw_rpe_sequence": "|".join(raw_values),
                }
            )

    duplicate_keys = sum(count - 1 for count in seen_row_keys.values() if count > 1)
    status["parsed_set_rows"] = int(len(rows_set))
    status["parsed_rep_rows"] = int(sum(1 for row in rows_rep if row["completed"]))
    status["duplicate_row_keys"] = int(duplicate_keys)
    if not rows_set:
        status["status"] = "no_rpe_rows"

    return pd.DataFrame(rows_rep), pd.DataFrame(rows_set), status


def build_exercise_summary(set_level: pd.DataFrame, rep_level: pd.DataFrame) -> pd.DataFrame:
    if set_level.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    completed_reps = rep_level[rep_level["completed"].eq(True)].copy() if not rep_level.empty else pd.DataFrame()
    for (folder, exercise), sets in set_level.groupby(["folder", "exercise"], sort=True):
        reps = (
            completed_reps[(completed_reps["folder"].eq(folder)) & (completed_reps["exercise"].eq(exercise))]
            if not completed_reps.empty
            else pd.DataFrame()
        )
        row: dict[str, object] = {
            "folder": folder,
            "exercise": exercise,
            "n_sets": int(len(sets)),
            "n_completed_reps": int(len(reps)),
            "final_rpe_mean": float(sets["final_rpe"].mean()),
            "final_rpe_median": float(sets["final_rpe"].median()),
            "final_rpe_min": float(sets["final_rpe"].min()),
            "final_rpe_max": float(sets["final_rpe"].max()),
            "rep_rpe_mean": float(reps["rpe"].mean()) if not reps.empty else np.nan,
            "rep_rpe_median": float(reps["rpe"].median()) if not reps.empty else np.nan,
        }
        final_counts = sets["final_rpe_class"].value_counts().to_dict()
        rep_counts = reps["rpe_class"].value_counts().to_dict() if not reps.empty else {}
        for cls in RPE_CLASSES:
            row[f"final_rpe_count_{cls}"] = int(final_counts.get(cls, 0))
            row[f"rep_rpe_count_{cls}"] = int(rep_counts.get(cls, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_table(df: pd.DataFrame, value_col: str, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(list(group_cols), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_col].dropna().astype(int)
        counts = values.value_counts().reindex(RPE_CLASSES, fill_value=0).astype(int)
        total = int(counts.sum())
        for cls, count in counts.items():
            row = {col: key for col, key in zip(group_cols, keys)}
            row.update({"rpe_class": int(cls), "count": int(count), "percent": float(count / total * 100 if total else 0.0), "total": total})
            rows.append(row)
    return pd.DataFrame(rows)


def run(data_dirs: Sequence[Path], output_dir: Path) -> dict[str, object]:
    rep_frames: list[pd.DataFrame] = []
    set_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for root in data_dirs:
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            workbook = same_name_xlsx(folder)
            if workbook is None:
                status_rows.append(
                    {
                        "folder": folder.name,
                        "source_file": "",
                        "status": "missing_same_name_xlsx",
                        "raw_rows": 0,
                        "parsed_set_rows": 0,
                        "parsed_rep_rows": 0,
                        "duplicate_row_keys": 0,
                    }
                )
                continue
            rep, sets, status = parse_workbook(folder, workbook)
            status_rows.append(status)
            if not rep.empty:
                rep_frames.append(rep)
            if not sets.empty:
                set_frames.append(sets)

    output_dir.mkdir(parents=True, exist_ok=True)
    rep_level = pd.concat(rep_frames, ignore_index=True) if rep_frames else pd.DataFrame()
    set_level = pd.concat(set_frames, ignore_index=True) if set_frames else pd.DataFrame()
    exercise_level = build_exercise_summary(set_level, rep_level)
    status = pd.DataFrame(status_rows)

    set_dist = distribution_table(set_level, "final_rpe_class", ["folder"])
    rep_dist = distribution_table(rep_level[rep_level["completed"].eq(True)].copy() if not rep_level.empty else rep_level, "rpe_class", ["folder"])
    set_dist_all = distribution_table(set_level.assign(scope="all"), "final_rpe_class", ["scope"])
    rep_dist_all = distribution_table(
        (rep_level[rep_level["completed"].eq(True)].copy() if not rep_level.empty else rep_level).assign(scope="all"),
        "rpe_class",
        ["scope"],
    )

    rep_level.to_csv(output_dir / "rpe_rep_level_from_same_name_xlsx.csv", index=False)
    set_level.to_csv(output_dir / "rpe_set_level_from_same_name_xlsx.csv", index=False)
    exercise_level.to_csv(output_dir / "rpe_exercise_level_from_same_name_xlsx.csv", index=False)
    status.to_csv(output_dir / "rpe_workbook_status.csv", index=False)
    set_dist.to_csv(output_dir / "rpe_final_set_distribution_by_folder.csv", index=False)
    rep_dist.to_csv(output_dir / "rpe_rep_distribution_by_folder.csv", index=False)
    set_dist_all.to_csv(output_dir / "rpe_final_set_distribution_overall.csv", index=False)
    rep_dist_all.to_csv(output_dir / "rpe_rep_distribution_overall.csv", index=False)

    summary = {
        "data_dirs": [str(path) for path in data_dirs],
        "output_dir": str(output_dir),
        "same_name_xlsx_only": True,
        "n_folders": int(len(status)),
        "n_ok_workbooks": int(status["status"].eq("ok").sum()) if not status.empty else 0,
        "n_set_rows": int(len(set_level)),
        "n_completed_rep_rows": int(rep_level["completed"].eq(True).sum()) if not rep_level.empty else 0,
        "files": {
            "rep_level": "rpe_rep_level_from_same_name_xlsx.csv",
            "set_level": "rpe_set_level_from_same_name_xlsx.csv",
            "exercise_level": "rpe_exercise_level_from_same_name_xlsx.csv",
            "status": "rpe_workbook_status.csv",
            "set_distribution_overall": "rpe_final_set_distribution_overall.csv",
            "rep_distribution_overall": "rpe_rep_distribution_overall.csv",
        },
        "notes": {
            "row_key": "Workbook rows are parsed from labels like exerciseIndex_setId, e.g. 3_2.",
            "duplicate_handling": "Duplicate workbook row keys are preserved with row_instance instead of being dropped.",
            "x_handling": "X marks an incomplete rep; it is retained in rep-level output with completed=False and excluded from RPE distributions.",
            "blank_handling": "Blank cells after an RPE value are forward-filled within the same workbook row and flagged in filled_from_previous.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-rep, per-set, and per-exercise RPE from same-name workout XLSX files.")
    parser.add_argument("--data-dir", type=Path, action="append", default=[Path("datasets/workout")], help="Root containing workout subject folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fatigue_rpe_vo2/018_same_name_xlsx_rpe_20260520"),
        help="Directory for extracted RPE CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.data_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
