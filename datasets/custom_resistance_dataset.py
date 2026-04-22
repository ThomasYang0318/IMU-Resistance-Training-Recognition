from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from preprocessing.window_pipeline import resample_sequence


@dataclass
class FeatureConfig:
    imu_columns: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
    label_column: str = "action_type"
    subject_column: str = "subject_id"
    time_column: str = "sensor_ts"


def load_csv_sequence(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _infer_subject_id_from_path(csv_path: Path, data_dir: Path) -> str | None:
    try:
        rel = csv_path.relative_to(data_dir)
    except ValueError:
        rel = csv_path
    parts = rel.parts
    if not parts:
        return None
    return str(parts[0])


def _infer_action_type_from_path(csv_path: Path, data_dir: Path) -> str | None:
    try:
        rel = csv_path.relative_to(data_dir)
    except ValueError:
        rel = csv_path
    parts = rel.parts
    if len(parts) < 2:
        return None
    return str(parts[1])


def prepare_sequences_from_folder(
    data_dir: Path,
    feature_cfg: FeatureConfig,
    sample_rate_hz: int,
    csv_glob: str = "*.csv",
) -> Tuple[List[pd.DataFrame], List[str]]:
    sequences: List[pd.DataFrame] = []
    subjects: List[str] = []
    skipped_files = 0

    for csv_path in sorted(data_dir.rglob(csv_glob)):
        df = load_csv_sequence(csv_path)

        # Best-effort recovery for datasets where some metadata columns are absent.
        if feature_cfg.subject_column not in df.columns:
            inferred_subject = _infer_subject_id_from_path(csv_path, data_dir)
            if inferred_subject is not None:
                df[feature_cfg.subject_column] = inferred_subject

        if feature_cfg.label_column not in df.columns:
            inferred_action = _infer_action_type_from_path(csv_path, data_dir)
            if inferred_action is not None:
                df[feature_cfg.label_column] = inferred_action

        required = set(feature_cfg.imu_columns) | {feature_cfg.label_column, feature_cfg.subject_column, feature_cfg.time_column}
        missing = required - set(df.columns)
        if missing:
            skipped_files += 1
            print(f"[WARN] Skipping {csv_path} missing required columns: {sorted(missing)}")
            continue

        df = df.dropna(subset=list(feature_cfg.imu_columns) + [feature_cfg.label_column, feature_cfg.subject_column])
        if df.empty:
            skipped_files += 1
            print(f"[WARN] Skipping empty/invalid file after dropna: {csv_path}")
            continue

        df = resample_sequence(
            df=df,
            imu_columns=feature_cfg.imu_columns,
            time_column=feature_cfg.time_column,
            target_rate_hz=sample_rate_hz,
        )

        sequences.append(df)
        subjects.append(str(df.iloc[0][feature_cfg.subject_column]))

    if not sequences:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    if skipped_files:
        print(f"[INFO] Skipped files: {skipped_files}")

    return sequences, subjects


def filter_sequences_by_subject(
    sequences: Sequence[pd.DataFrame],
    allowed_subjects: Sequence[str],
    subject_column: str,
) -> List[pd.DataFrame]:
    allowed = set(str(s) for s in allowed_subjects)
    return [s for s in sequences if str(s.iloc[0][subject_column]) in allowed]
