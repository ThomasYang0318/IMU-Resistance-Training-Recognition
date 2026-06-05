from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_borg_from_gt_waveform_features import cosine_similarity, resample, zscore  # noqa: E402
from tools.evaluate_literature_inspired_rep_methods import (  # noqa: E402
    ACC_COLUMNS,
    GYRO_COLUMNS,
    IMU9_COLUMNS,
    infer_sensor_period_seconds,
    magnitude_signal,
    principal_signal,
    read_session_9axis,
)
from tools.evaluate_rep_segmentation_classification import (  # noqa: E402
    PhaseSegment,
    RepSegment,
    true_phase_segments,
    true_rep_segments,
    whole_session_files,
)


TARGET = "borg"
GROUP_COLUMNS = ["folder", "exercise", "set_id"]


def normalize_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return np.nan
    return float(numerator / denominator)


def safe_change(current: float, baseline: float) -> float:
    value = safe_ratio(current, baseline)
    return value - 1.0 if np.isfinite(value) else np.nan


def numeric_slope(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 2 or float(np.std(y)) < 1e-12:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(y))
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def coefficient_of_variation(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return 0.0
    denom = abs(float(np.mean(y)))
    if denom < 1e-9:
        return 0.0
    return float(np.std(y) / denom)


def first_mean(values: pd.Series, n: int = 2) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.head(n).mean()) if not clean.empty else np.nan


def last_mean(values: pd.Series, n: int = 2) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.tail(n).mean()) if not clean.empty else np.nan


def centered(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce")})
    for idx, group in enumerate(groups):
        frame[f"group_{idx}"] = group.astype(str).to_numpy()
    group_cols = [col for col in frame.columns if col.startswith("group_")]
    return frame["value"] - frame.groupby(group_cols)["value"].transform("mean")


def corr_pair(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 8 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    try:
        value = spearmanr(frame["x"], frame["y"]).statistic if method == "spearman" else pearsonr(frame["x"], frame["y"]).statistic
    except Exception:
        return np.nan
    return float(value) if np.isfinite(value) else np.nan


def phase_segments_by_key(phases: Sequence[PhaseSegment]) -> dict[tuple[Path, str, str, str, str], dict[str, list[PhaseSegment]]]:
    by_key: dict[tuple[Path, str, str, str, str], dict[str, list[PhaseSegment]]] = {}
    for phase in phases:
        key = (phase.file_path, phase.subject, phase.exercise, normalize_id(phase.set_id), normalize_id(phase.rep_id))
        by_key.setdefault(key, {}).setdefault(phase.phase, []).append(phase)
    return by_key


def signal_metrics(values: np.ndarray, seconds: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "range": np.nan,
            "rms": np.nan,
            "diff_rms": np.nan,
            "diff_abs_mean": np.nan,
            "peak_abs": np.nan,
            "movement_rate": np.nan,
        }
    diff = np.diff(values) if len(values) > 1 else np.zeros(1, dtype=np.float64)
    signal_range = float(np.ptp(values))
    return {
        "range": signal_range,
        "rms": float(np.sqrt(np.mean(values**2))),
        "diff_rms": float(np.sqrt(np.mean(diff**2))),
        "diff_abs_mean": float(np.mean(np.abs(diff))),
        "peak_abs": float(np.max(np.abs(values))),
        "movement_rate": signal_range / seconds if seconds > 1e-9 else np.nan,
    }


def extract_phase_values(
    rep: RepSegment,
    phase_rows: list[PhaseSegment],
    rep_pca: np.ndarray,
    rep_gyro: np.ndarray,
    rep_acc: np.ndarray,
    period: float,
) -> tuple[dict[str, float], np.ndarray]:
    if not phase_rows:
        return {
            "samples": 0.0,
            "sec": 0.0,
            "pca_range": np.nan,
            "pca_movement_rate": np.nan,
            "pca_diff_rms": np.nan,
            "gyro_range": np.nan,
            "gyro_movement_rate": np.nan,
            "gyro_diff_rms": np.nan,
            "gyro_peak_abs": np.nan,
            "acc_range": np.nan,
            "acc_diff_rms": np.nan,
        }, np.zeros(0, dtype=np.float64)
    masks: list[np.ndarray] = []
    wave_parts: list[np.ndarray] = []
    for segment in sorted(phase_rows, key=lambda item: item.start):
        local_start = max(int(segment.start) - rep.start, 0)
        local_end = min(int(segment.end) - rep.start, len(rep_pca))
        if local_end <= local_start:
            continue
        mask = np.arange(local_start, local_end)
        masks.append(mask)
        wave_parts.append(rep_pca[local_start:local_end])
    if not masks:
        return extract_phase_values(rep, [], rep_pca, rep_gyro, rep_acc, period)
    idx = np.concatenate(masks)
    pca = rep_pca[idx]
    gyro = rep_gyro[idx]
    acc = rep_acc[idx]
    seconds = float(len(idx) * period)
    pca_m = signal_metrics(pca, seconds)
    gyro_m = signal_metrics(gyro, seconds)
    acc_m = signal_metrics(acc, seconds)
    return {
        "samples": float(len(idx)),
        "sec": seconds,
        "pca_range": pca_m["range"],
        "pca_movement_rate": pca_m["movement_rate"],
        "pca_diff_rms": pca_m["diff_rms"],
        "gyro_range": gyro_m["range"],
        "gyro_movement_rate": gyro_m["movement_rate"],
        "gyro_diff_rms": gyro_m["diff_rms"],
        "gyro_peak_abs": gyro_m["peak_abs"],
        "acc_range": acc_m["range"],
        "acc_diff_rms": acc_m["diff_rms"],
    }, np.concatenate(wave_parts)


def load_targets(path: Path) -> pd.DataFrame:
    targets = pd.read_csv(path)
    targets = targets[targets["completed"].eq(True) & targets["borg"].notna()].copy()
    targets["folder"] = targets["folder"].astype(str)
    targets["exercise"] = targets["exercise"].astype(str)
    targets["set_id"] = targets["set_id"].map(normalize_id)
    targets["rep_id"] = targets["rep_id"].map(normalize_id)
    targets["borg"] = pd.to_numeric(targets["borg"], errors="coerce")
    targets["kg"] = pd.to_numeric(targets["kg"], errors="coerce")
    return targets.dropna(subset=["borg"]).copy()


def extract_rep_phase_features(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in whole_session_files(args.data_dirs):
        df = read_session_9axis(path, args.data_dirs)
        period = infer_sensor_period_seconds(df)
        reps = true_rep_segments(df, path, min_samples=args.min_rep_samples)
        phases = true_phase_segments(df, path, min_samples=args.min_phase_samples)
        phase_lookup = phase_segments_by_key(phases)
        folder = path.parent.name
        session_pca = zscore(principal_signal(df, smooth_window=args.smooth_window, columns=IMU9_COLUMNS))
        session_gyro = zscore(magnitude_signal(df, GYRO_COLUMNS, smooth_window=args.smooth_window))
        session_acc = zscore(magnitude_signal(df, ACC_COLUMNS, smooth_window=args.smooth_window))
        for rep in reps:
            if rep.n_samples < args.min_rep_samples:
                continue
            local = df.iloc[rep.start : rep.end].reset_index(drop=True)
            if len(local) < 3:
                continue
            rep_pca = session_pca[rep.start : rep.end]
            rep_gyro = session_gyro[rep.start : rep.end]
            rep_acc = session_acc[rep.start : rep.end]
            key = (path, rep.subject, rep.exercise, normalize_id(rep.set_id), normalize_id(rep.rep_id))
            phase_map = phase_lookup.get(key, {})

            row: dict[str, object] = {
                "folder": folder,
                "file": str(path),
                "subject": rep.subject,
                "exercise": rep.exercise,
                "set_id": normalize_id(rep.set_id),
                "rep_id": normalize_id(rep.rep_id),
                "rep_index": int(float(normalize_id(rep.rep_id))) if normalize_id(rep.rep_id).replace(".", "", 1).isdigit() else np.nan,
                "rep_samples": rep.n_samples,
                "rep_sec": rep.n_samples * period,
            }
            wave_by_phase: dict[str, np.ndarray] = {}
            for phase_name in ["concentric", "eccentric"]:
                metrics, wave = extract_phase_values(rep, phase_map.get(phase_name, []), rep_pca, rep_gyro, rep_acc, period)
                wave_by_phase[phase_name] = wave
                row[f"__{phase_name}_wave"] = resample(wave, args.resample_points) if len(wave) else np.zeros(args.resample_points)
                for metric_name, value in metrics.items():
                    row[f"{phase_name}_{metric_name}"] = value
            row["ce_ratio"] = safe_ratio(float(row["concentric_sec"]), float(row["rep_sec"]))
            row["ec_ratio"] = safe_ratio(float(row["eccentric_sec"]), float(row["rep_sec"]))
            row["ce_time_balance_abs"] = abs(float(row["concentric_sec"]) - float(row["eccentric_sec"])) / max(float(row["rep_sec"]), 1e-9)
            row["concentric_minus_eccentric_sec"] = float(row["concentric_sec"]) - float(row["eccentric_sec"])
            row["concentric_to_eccentric_sec_ratio"] = safe_ratio(float(row["concentric_sec"]), float(row["eccentric_sec"]))
            row["concentric_to_eccentric_rate_ratio"] = safe_ratio(float(row["concentric_pca_movement_rate"]), float(row["eccentric_pca_movement_rate"]))
            row["concentric_to_eccentric_gyro_ratio"] = safe_ratio(float(row["concentric_gyro_diff_rms"]), float(row["eccentric_gyro_diff_rms"]))
            rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        return features
    return features.sort_values(GROUP_COLUMNS + ["rep_index"]).reset_index(drop=True)


def add_similarity_and_progress(rep_df: pd.DataFrame, resample_points: int, raw_features: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in rep_df.groupby(GROUP_COLUMNS, sort=False):
        group = group.sort_values("rep_index").copy()
        n = len(group)
        group["rep_order"] = np.arange(n)
        group["rep_progress"] = group["rep_order"] / max(n - 1, 1)
        group["n_reps_in_set"] = n
        group["cumulative_rep_sec"] = group["rep_sec"].cumsum()
        group["cumulative_concentric_sec"] = group["concentric_sec"].cumsum()
        group["cumulative_eccentric_sec"] = group["eccentric_sec"].cumsum()

        vector_cols = [
            "concentric_sec",
            "eccentric_sec",
            "ce_ratio",
            "concentric_pca_movement_rate",
            "eccentric_pca_movement_rate",
            "concentric_gyro_diff_rms",
            "eccentric_gyro_diff_rms",
            "concentric_pca_range",
            "eccentric_pca_range",
        ]
        available = [col for col in vector_cols if col in group.columns]
        x = np.array(group[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True)
        col_mean = np.nanmean(x, axis=0)
        inds = np.where(~np.isfinite(x))
        x[inds] = np.take(col_mean, inds[1]) if len(inds[0]) else x[inds]
        scale = np.nanstd(x, axis=0)
        scale[scale < 1e-9] = 1.0
        x = (x - np.nanmean(x, axis=0)) / scale
        template = np.nanmean(x[: min(2, len(x))], axis=0)
        sims = []
        for row in x:
            sims.append(cosine_similarity(row, template))
        group["phase_vector_sim_to_first2"] = sims
        group["phase_vector_similarity_decay"] = 1.0 - group["phase_vector_sim_to_first2"]

        for phase_name in ["concentric", "eccentric"]:
            wave_col = f"__{phase_name}_wave"
            if wave_col not in group.columns:
                continue
            waves = [np.asarray(wave, dtype=np.float64) for wave in group[wave_col]]
            if not waves:
                continue
            template = np.mean(np.vstack(waves[: min(2, len(waves))]), axis=0)
            sim_values = [cosine_similarity(wave, template) for wave in waves]
            group[f"{phase_name}_wave_sim_to_first2"] = sim_values
            group[f"{phase_name}_wave_similarity_decay"] = 1.0 - group[f"{phase_name}_wave_sim_to_first2"]

        first = {col: first_mean(group[col]) for col in group.columns if col not in {"folder", "file", "subject", "exercise", "set_id", "rep_id"}}
        group["concentric_duration_gain_from_first2"] = group["concentric_sec"].map(lambda value: safe_change(float(value), first.get("concentric_sec", np.nan)))
        group["eccentric_duration_gain_from_first2"] = group["eccentric_sec"].map(lambda value: safe_change(float(value), first.get("eccentric_sec", np.nan)))
        group["ce_ratio_drift_from_first2"] = group["ce_ratio"].map(lambda value: float(value) - first.get("ce_ratio", np.nan))
        group["concentric_rate_loss_from_first2"] = group["concentric_pca_movement_rate"].map(lambda value: 1.0 - safe_ratio(float(value), first.get("concentric_pca_movement_rate", np.nan)))
        group["eccentric_rate_loss_from_first2"] = group["eccentric_pca_movement_rate"].map(lambda value: 1.0 - safe_ratio(float(value), first.get("eccentric_pca_movement_rate", np.nan)))
        group["concentric_gyro_gain_from_first2"] = group["concentric_gyro_diff_rms"].map(lambda value: safe_change(float(value), first.get("concentric_gyro_diff_rms", np.nan)))
        group["eccentric_gyro_gain_from_first2"] = group["eccentric_gyro_diff_rms"].map(lambda value: safe_change(float(value), first.get("eccentric_gyro_diff_rms", np.nan)))
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def build_set_level(rep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in rep_df.groupby(GROUP_COLUMNS, sort=False):
        group = group.sort_values("rep_index").copy()
        row: dict[str, object] = {
            "folder": key[0],
            "exercise": key[1],
            "set_id": key[2],
            "borg": float(group["borg"].iloc[-1]),
            "kg": float(group["kg"].dropna().iloc[0]) if group["kg"].notna().any() else np.nan,
            "n_reps": len(group),
            "set_index_numeric": float(key[2]) if str(key[2]).replace(".", "", 1).isdigit() else np.nan,
            "total_rep_sec": float(group["rep_sec"].sum()),
            "total_concentric_sec": float(group["concentric_sec"].sum()),
            "total_eccentric_sec": float(group["eccentric_sec"].sum()),
            "mean_ce_ratio": float(group["ce_ratio"].mean()),
            "mean_ce_time_balance_abs": float(group["ce_time_balance_abs"].mean()),
        }
        for phase in ["concentric", "eccentric"]:
            for metric in ["sec", "pca_movement_rate", "pca_range", "gyro_diff_rms", "gyro_peak_abs", "acc_diff_rms"]:
                col = f"{phase}_{metric}"
                if col in group:
                    row[f"{col}_mean"] = float(group[col].mean())
                    row[f"{col}_first2"] = first_mean(group[col])
                    row[f"{col}_last2"] = last_mean(group[col])
                    row[f"{col}_slope"] = numeric_slope(group[col])
                    row[f"{col}_cv"] = coefficient_of_variation(group[col])
                    row[f"{col}_last2_vs_first2"] = safe_change(last_mean(group[col]), first_mean(group[col]))
        row["ce_ratio_slope"] = numeric_slope(group["ce_ratio"])
        row["ce_ratio_last_minus_first"] = float(group["ce_ratio"].iloc[-1] - group["ce_ratio"].iloc[0])
        row["ce_time_balance_slope"] = numeric_slope(group["ce_time_balance_abs"])
        row["phase_vector_similarity_slope"] = numeric_slope(group["phase_vector_sim_to_first2"])
        row["phase_vector_similarity_last_minus_first"] = float(group["phase_vector_sim_to_first2"].iloc[-1] - group["phase_vector_sim_to_first2"].iloc[0])
        row["phase_vector_similarity_mean"] = float(group["phase_vector_sim_to_first2"].mean())
        row["phase_vector_similarity_min"] = float(group["phase_vector_sim_to_first2"].min())
        for phase in ["concentric", "eccentric"]:
            col = f"{phase}_wave_sim_to_first2"
            if col in group:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_min"] = float(group[col].min())
                row[f"{col}_slope"] = numeric_slope(group[col])
                row[f"{col}_last_minus_first"] = float(group[col].iloc[-1] - group[col].iloc[0])
        row["kg_x_total_rep_sec"] = row["kg"] * row["total_rep_sec"] if np.isfinite(row["kg"]) else np.nan
        row["kg_x_n_reps"] = row["kg"] * row["n_reps"] if np.isfinite(row["kg"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def feature_group(feature: str) -> str:
    if feature in {"rep_progress", "rep_order", "rep_index", "set_index_numeric", "n_reps", "n_reps_in_set"}:
        return "progress"
    if feature.startswith("cumulative") or feature.startswith("total") or feature.endswith("_sec") or "_sec_" in feature:
        return "phase_tut"
    if "rate_loss" in feature or "movement_rate" in feature or "pca_range" in feature:
        return "phase_velocity"
    if "gyro" in feature:
        return "phase_gyro"
    if "ratio" in feature or "balance" in feature:
        return "ce_ratio"
    if "similarity" in feature or "sim_to" in feature:
        return "phase_similarity"
    if feature.startswith("kg"):
        return "load"
    if "cv" in feature:
        return "variability"
    return "other"


def candidate_features(df: pd.DataFrame) -> list[str]:
    banned = {
        "borg",
        "folder",
        "file",
        "subject",
        "exercise",
        "set_id",
        "rep_id",
        "raw_value",
        "completed",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in banned:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 8 and df[col].nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def correlation_table(df: pd.DataFrame, features: list[str], level: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(df[TARGET], errors="coerce")
    y_ex = centered(y, [df["exercise"]])
    y_sub = centered(y, [df["folder"]])
    y_sub_ex = centered(y, [df["folder"], df["exercise"]])
    for feature in features:
        x = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "level": level,
                "feature": feature,
                "group": feature_group(feature),
                "n": int(pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
                "raw_spearman": corr_pair(x, y, "spearman"),
                "raw_pearson": corr_pair(x, y, "pearson"),
                "exercise_centered_spearman": corr_pair(centered(x, [df["exercise"]]), y_ex, "spearman"),
                "subject_centered_spearman": corr_pair(centered(x, [df["folder"]]), y_sub, "spearman"),
                "subject_exercise_centered_spearman": corr_pair(centered(x, [df["folder"], df["exercise"]]), y_sub_ex, "spearman"),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_raw_spearman"] = out["raw_spearman"].abs()
    out["abs_subject_exercise_centered_spearman"] = out["subject_exercise_centered_spearman"].abs()
    return out.sort_values(["abs_raw_spearman", "abs_subject_exercise_centered_spearman"], ascending=False).reset_index(drop=True)


def by_exercise_table(set_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = [
        "concentric_sec_slope",
        "concentric_sec_last2_vs_first2",
        "concentric_pca_movement_rate_slope",
        "concentric_pca_movement_rate_last2_vs_first2",
        "concentric_gyro_diff_rms_slope",
        "concentric_gyro_diff_rms_last2_vs_first2",
        "ce_ratio_slope",
        "phase_vector_similarity_slope",
        "concentric_wave_sim_to_first2_slope",
        "eccentric_wave_sim_to_first2_slope",
    ]
    selected = [feature for feature in selected if feature in set_df.columns]
    for exercise, sub in set_df.groupby("exercise"):
        y = sub["borg"]
        for feature in selected:
            rows.append(
                {
                    "exercise": exercise,
                    "feature": feature,
                    "group": feature_group(feature),
                    "n_sets": len(sub),
                    "raw_spearman": corr_pair(sub[feature], y, "spearman"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_raw_spearman"] = out["raw_spearman"].abs()
    return out.sort_values(["exercise", "abs_raw_spearman"], ascending=[True, False])


def plot_top(table: pd.DataFrame, path: Path, metric: str, title: str, top_n: int = 24) -> None:
    sub = table.dropna(subset=[metric]).copy()
    sub = sub.reindex(sub[metric].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    colors = ["#3f7cac" if value >= 0 else "#c45b4f" for value in sub[metric]]
    labels = [f"{row.feature} ({row.group})" for row in sub.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(12, max(6, len(sub) * 0.34)))
    ax.barh(labels, sub[metric], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hypothesis_bars(set_corr: pd.DataFrame, output_path: Path) -> None:
    features = [
        "concentric_sec_slope",
        "concentric_sec_last2_vs_first2",
        "concentric_pca_movement_rate_slope",
        "concentric_pca_movement_rate_last2_vs_first2",
        "concentric_gyro_diff_rms_slope",
        "concentric_gyro_diff_rms_last2_vs_first2",
        "ce_ratio_slope",
        "phase_vector_similarity_slope",
        "concentric_wave_sim_to_first2_slope",
        "eccentric_wave_sim_to_first2_slope",
    ]
    sub = set_corr[set_corr["feature"].isin(features)].copy()
    sub = sub.set_index("feature").reindex(features).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(sub))
    ax.bar(x - 0.18, sub["raw_spearman"], width=0.36, label="raw", color="#5276a7")
    ax.bar(x + 0.18, sub["subject_exercise_centered_spearman"], width=0.36, label="subject+exercise centered", color="#5f9f74")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["feature"], rotation=30, ha="right")
    ax.set_ylabel("Spearman vs Borg/RPE")
    ax.set_title("Phase-aware fatigue hypothesis checks")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether CE phase-aware IMU fatigue features relate to Borg/RPE.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--targets", type=Path, default=Path("artifacts_rep_classification/018_borg_gt_waveform_relation/018_borg_targets_completed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis"))
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--min-rep-samples", type=int, default=10)
    parser.add_argument("--min-phase-samples", type=int, default=5)
    parser.add_argument("--resample-points", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.targets)
    phase_features = extract_rep_phase_features(args)
    merged = phase_features.merge(
        targets[["folder", "exercise", "set_id", "rep_id", "kg", "borg", "raw_value"]],
        on=["folder", "exercise", "set_id", "rep_id"],
        how="inner",
    )
    merged = merged[merged["borg"].notna()].copy()
    merged = add_similarity_and_progress(merged, resample_points=80)
    set_level = build_set_level(merged)
    csv_rep = merged.drop(columns=[col for col in merged.columns if col.startswith("__")], errors="ignore")

    rep_corr = correlation_table(csv_rep, candidate_features(csv_rep), "rep")
    set_corr = correlation_table(set_level, candidate_features(set_level), "set")
    exercise_corr = by_exercise_table(set_level, candidate_features(set_level))

    csv_rep.to_csv(args.output_dir / "023_phase_aware_rep_feature_dataset.csv", index=False)
    set_level.to_csv(args.output_dir / "023_phase_aware_set_feature_dataset.csv", index=False)
    rep_corr.to_csv(args.output_dir / "023_phase_aware_rep_correlations.csv", index=False)
    set_corr.to_csv(args.output_dir / "023_phase_aware_set_correlations.csv", index=False)
    exercise_corr.to_csv(args.output_dir / "023_phase_aware_set_correlations_by_exercise.csv", index=False)

    plot_top(rep_corr, args.output_dir / "023_phase_aware_rep_top_raw_spearman.png", "raw_spearman", "Rep-level CE phase-aware features vs Borg/RPE")
    plot_top(set_corr, args.output_dir / "023_phase_aware_set_top_raw_spearman.png", "raw_spearman", "Set-level CE phase-aware features vs final Borg/RPE")
    plot_top(
        set_corr,
        args.output_dir / "023_phase_aware_set_top_subject_exercise_centered_spearman.png",
        "subject_exercise_centered_spearman",
        "Set-level CE phase-aware features vs final Borg/RPE, within subject+exercise",
    )
    plot_hypothesis_bars(set_corr, args.output_dir / "023_phase_aware_hypothesis_bars.png")

    summary = {
        "output_dir": str(args.output_dir),
        "rep_rows": int(len(merged)),
        "set_rows": int(len(set_level)),
        "subjects": sorted(merged["folder"].astype(str).unique().tolist()),
        "top_rep_raw_spearman": rep_corr.head(15).to_dict(orient="records"),
        "top_set_raw_spearman": set_corr.head(15).to_dict(orient="records"),
        "hypothesis_features": set_corr[
            set_corr["feature"].isin(
                [
                    "concentric_sec_slope",
                    "concentric_sec_last2_vs_first2",
                    "concentric_pca_movement_rate_slope",
                    "concentric_pca_movement_rate_last2_vs_first2",
                    "concentric_gyro_diff_rms_slope",
                    "concentric_gyro_diff_rms_last2_vs_first2",
                    "ce_ratio_slope",
                    "phase_vector_similarity_slope",
                ]
            )
        ][["feature", "raw_spearman", "subject_exercise_centered_spearman", "n"]].to_dict(orient="records"),
        "notes": {
            "target": "Rep-level uses per-rep Borg labels; set-level uses the last Borg label in each set.",
            "segmentation": "Uses ground-truth concentric/eccentric phase labels, so this tests the fatigue-feature hypothesis without predicted segmentation noise.",
            "interpretation": "Positive slope/gain features mean the phase metric increases toward later reps; movement-rate last2_vs_first2 is a gain, so negative values can indicate velocity loss.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Rep rows:", len(merged), "Set rows:", len(set_level))
    print("Subjects:", ", ".join(summary["subjects"]))
    print("\nTop set-level phase-aware features:")
    print(set_corr[["feature", "group", "n", "raw_spearman", "subject_exercise_centered_spearman"]].head(20).round(4).to_string(index=False))
    print("\nHypothesis checks:")
    print(pd.DataFrame(summary["hypothesis_features"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
