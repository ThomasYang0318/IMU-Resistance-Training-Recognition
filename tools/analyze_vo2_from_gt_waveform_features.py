from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_borg_from_gt_waveform_features import (  # noqa: E402
    ACC_COLUMNS,
    GYRO_COLUMNS,
    IMU9_COLUMNS,
    cosine_similarity,
    magnitude,
    resample,
    signal_stats,
    zscore,
)
from tools.evaluate_literature_inspired_rep_methods import principal_signal  # noqa: E402
from tools.evaluate_rep_segmentation_classification import ACTIVE_PHASES, clean_label_series, whole_session_files  # noqa: E402


READ_COLUMNS = set(IMU9_COLUMNS) | {"pc_time", "subject_id", "action_type", "set", "rep", "phase", "rpe", "weight_kg"}
VO2_COLUMNS = ("VO2[mL/kg/min]", "VO2[mL/min]", "Rf[bpm]", "Tv[L]", "Ve[L/min]")


def read_imu_session(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda column: column in READ_COLUMNS)
    df["pc_time_utc"] = pd.to_datetime(df["pc_time"], errors="coerce").dt.tz_localize(
        "Asia/Taipei", nonexistent="NaT", ambiguous="NaT"
    ).dt.tz_convert("UTC")
    return df.reset_index(drop=True)


def read_vo2(folder: Path) -> pd.DataFrame | None:
    path = folder / "VO2MasterUnit-Data.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name=0)
    if "Time[utc]" not in df.columns or "VO2[mL/kg/min]" not in df.columns:
        return None
    df["time_utc"] = pd.to_datetime(df["Time[utc]"], utc=True, errors="coerce")
    for col in VO2_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["time_utc"].notna()].sort_values("time_utc").reset_index(drop=True)
    return df


def contiguous_set_blocks(df: pd.DataFrame, min_samples: int) -> list[dict[str, object]]:
    phases = clean_label_series(df["phase"]).str.lower()
    active = phases.isin(ACTIVE_PHASES).to_numpy()
    if not active.any():
        return []
    subjects = clean_label_series(df["subject_id"]).to_numpy(dtype=object)
    exercises = clean_label_series(df["action_type"]).to_numpy(dtype=object)
    sets = clean_label_series(df["set"]).to_numpy(dtype=object)
    reps = clean_label_series(df["rep"]).to_numpy(dtype=object)

    rep_segments: list[dict[str, object]] = []
    start: int | None = None
    last_key: tuple[str, str, str, str] | None = None
    for idx, is_active in enumerate(active):
        key = (subjects[idx], exercises[idx], sets[idx], reps[idx])
        if is_active and start is None:
            start = idx
            last_key = key
        elif is_active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                rep_segments.append({"subject": last_key[0], "exercise": last_key[1], "set_id": str(last_key[2]), "rep_id": str(last_key[3]), "start": start, "end": idx})
            start = idx
            last_key = key
        elif (not is_active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                rep_segments.append({"subject": last_key[0], "exercise": last_key[1], "set_id": str(last_key[2]), "rep_id": str(last_key[3]), "start": start, "end": idx})
            start = None
            last_key = None
    if start is not None and last_key is not None and len(df) - start >= min_samples:
        rep_segments.append({"subject": last_key[0], "exercise": last_key[1], "set_id": str(last_key[2]), "rep_id": str(last_key[3]), "start": start, "end": len(df)})

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for rep in rep_segments:
        grouped.setdefault((str(rep["subject"]), str(rep["exercise"]), str(rep["set_id"])), []).append(rep)

    blocks: list[dict[str, object]] = []
    for (subject, exercise, set_id), reps_in_set in grouped.items():
        reps_in_set = sorted(reps_in_set, key=lambda row: int(float(row["rep_id"])) if str(row["rep_id"]).replace(".", "", 1).isdigit() else row["start"])
        start_idx = min(int(row["start"]) for row in reps_in_set)
        end_idx = max(int(row["end"]) for row in reps_in_set)
        if end_idx - start_idx < min_samples:
            continue
        blocks.append(
            {
                "subject": subject,
                "exercise": exercise,
                "set_id": set_id,
                "start": start_idx,
                "end": end_idx,
                "reps": reps_in_set,
            }
        )
    return sorted(blocks, key=lambda row: (row["subject"], row["exercise"], row["set_id"], row["start"]))


def time_seconds(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series, utc=True, errors="coerce")
    if values.isna().all():
        return np.arange(len(series), dtype=np.float64) * 0.01
    start = values.iloc[0]
    return (values - start).dt.total_seconds().to_numpy(dtype=np.float64)


def set_waveform_features(df: pd.DataFrame, block: dict[str, object], resample_points: int) -> dict[str, float]:
    start = int(block["start"])
    end = int(block["end"])
    local = df.iloc[start:end].reset_index(drop=True)
    out: dict[str, float] = {}
    set_seconds = max(float((local["pc_time_utc"].iloc[-1] - local["pc_time_utc"].iloc[0]).total_seconds()), 0.0) if len(local) > 1 else 0.0
    out["set_duration_sec"] = set_seconds
    out["n_reps"] = float(len(block["reps"]))
    out["reps_per_min"] = float(len(block["reps"]) / (set_seconds / 60.0)) if set_seconds > 0 else 0.0

    phases = clean_label_series(local["phase"]).str.lower()
    concentric_count = int(phases.eq("concentric").sum())
    eccentric_count = int(phases.eq("eccentric").sum())
    total_active = max(concentric_count + eccentric_count, 1)
    out["concentric_ratio_samples"] = concentric_count / float(total_active)
    out["eccentric_ratio_samples"] = eccentric_count / float(total_active)

    pca = zscore(principal_signal(local, smooth_window=9, columns=IMU9_COLUMNS))
    out.update(signal_stats("set_pca", pca))
    out.update(signal_stats("set_acc_mag", magnitude(local, ACC_COLUMNS)))
    out.update(signal_stats("set_gyro_mag", magnitude(local, GYRO_COLUMNS)))

    rep_durations: list[float] = []
    rep_sim_first: list[float] = []
    rep_sim_prev: list[float] = []
    rep_ranges: list[float] = []
    rep_waves: list[np.ndarray] = []
    for rep in block["reps"]:
        rep_df = df.iloc[int(rep["start"]) : int(rep["end"])].reset_index(drop=True)
        if len(rep_df) < 2:
            continue
        rep_durations.append(float((rep_df["pc_time_utc"].iloc[-1] - rep_df["pc_time_utc"].iloc[0]).total_seconds()))
        rep_pca = zscore(principal_signal(rep_df, smooth_window=9, columns=IMU9_COLUMNS))
        rep_ranges.append(float(np.ptp(rep_pca)) if len(rep_pca) else 0.0)
        rep_waves.append(resample(rep_pca, resample_points))
    if rep_waves:
        first = rep_waves[0]
        prev = None
        for wave in rep_waves:
            rep_sim_first.append(cosine_similarity(wave, first))
            rep_sim_prev.append(cosine_similarity(wave, prev) if prev is not None else 1.0)
            prev = wave
    for prefix, values in {
        "rep_duration": rep_durations,
        "rep_pca_range": rep_ranges,
        "sim_to_first": rep_sim_first,
        "sim_to_prev": rep_sim_prev,
    }.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            out[f"{prefix}_mean"] = 0.0
            out[f"{prefix}_slope"] = 0.0
            out[f"{prefix}_last_minus_first"] = 0.0
            continue
        x = np.arange(len(arr), dtype=np.float64)
        slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) >= 2 else 0.0
        out[f"{prefix}_mean"] = float(np.mean(arr))
        out[f"{prefix}_slope"] = slope
        out[f"{prefix}_last_minus_first"] = float(arr[-1] - arr[0])
    return out


def vo2_stats(vo2: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp, lag_sec: float) -> dict[str, float | int]:
    start = start_time + pd.Timedelta(seconds=lag_sec)
    end = end_time + pd.Timedelta(seconds=lag_sec)
    sub = vo2[(vo2["time_utc"] >= start) & (vo2["time_utc"] <= end)].copy()
    values = pd.to_numeric(sub["VO2[mL/kg/min]"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    out: dict[str, float | int] = {
        "vo2_points": int(len(values)),
        "vo2_mean": float(np.mean(values)) if len(values) else np.nan,
        "vo2_peak": float(np.max(values)) if len(values) else np.nan,
        "vo2_min": float(np.min(values)) if len(values) else np.nan,
        "vo2_slope": 0.0,
    }
    if len(values) >= 2:
        times = (sub.loc[pd.to_numeric(sub["VO2[mL/kg/min]"], errors="coerce").notna(), "time_utc"] - start).dt.total_seconds().to_numpy(dtype=np.float64)
        if len(times) == len(values) and float(np.ptp(times)) > 0:
            out["vo2_slope"] = float(np.polyfit(times, values, 1)[0])
    return out


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for csv_path in whole_session_files(args.data_dirs):
        folder = csv_path.parent
        vo2 = read_vo2(folder)
        if vo2 is None or vo2.empty:
            status_rows.append({"folder": folder.name, "csv": str(csv_path), "status": "missing_vo2"})
            continue
        imu = read_imu_session(csv_path)
        imu_start = imu["pc_time_utc"].min()
        imu_end = imu["pc_time_utc"].max()
        vo2_start = vo2["time_utc"].min()
        vo2_end = vo2["time_utc"].max()
        overlap = max(pd.Timedelta(0), min(imu_end, vo2_end) - max(imu_start, vo2_start))
        valid_vo2 = int(pd.to_numeric(vo2["VO2[mL/kg/min]"], errors="coerce").notna().sum())
        if overlap.total_seconds() <= 0:
            status_rows.append(
                {
                    "folder": folder.name,
                    "csv": str(csv_path),
                    "status": "no_time_overlap",
                    "valid_vo2_points": valid_vo2,
                    "overlap_sec": 0.0,
                }
            )
            continue
        blocks = contiguous_set_blocks(imu, args.min_segment_samples)
        set_count = 0
        for block in blocks:
            start_idx = int(block["start"])
            end_idx = int(block["end"]) - 1
            if start_idx < 0 or end_idx >= len(imu):
                continue
            start_time = imu.loc[start_idx, "pc_time_utc"]
            end_time = imu.loc[end_idx, "pc_time_utc"]
            if pd.isna(start_time) or pd.isna(end_time):
                continue
            features = set_waveform_features(imu, block, args.resample_points)
            base_row: dict[str, object] = {
                "folder": folder.name,
                "file": str(csv_path),
                "subject": block["subject"],
                "exercise": block["exercise"],
                "set_id": block["set_id"],
                "start_utc": str(start_time),
                "end_utc": str(end_time),
                **features,
            }
            weight = pd.to_numeric(imu.iloc[start_idx : end_idx + 1].get("weight_kg", pd.Series(dtype=float)), errors="coerce")
            rpe = pd.to_numeric(imu.iloc[start_idx : end_idx + 1].get("rpe", pd.Series(dtype=float)), errors="coerce")
            base_row["weight_kg"] = float(weight.replace(0, np.nan).dropna().median()) if len(weight.replace(0, np.nan).dropna()) else np.nan
            base_row["rpe_label"] = float(rpe.replace(0, np.nan).dropna().median()) if len(rpe.replace(0, np.nan).dropna()) else np.nan
            for lag in args.lags_sec:
                stats = vo2_stats(vo2, start_time, end_time, lag)
                if int(stats["vo2_points"]) < args.min_vo2_points:
                    continue
                rows.append({"lag_sec": float(lag), **base_row, **stats})
                set_count += 1
        status_rows.append(
            {
                "folder": folder.name,
                "csv": str(csv_path),
                "status": "ok",
                "valid_vo2_points": valid_vo2,
                "overlap_sec": round(overlap.total_seconds(), 4),
                "sets_with_vo2_rows": set_count,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = []
    skip = {
        "lag_sec",
        "vo2_points",
        "vo2_mean",
        "vo2_peak",
        "vo2_min",
        "vo2_slope",
        "rpe_label",
    }
    for col in df.columns:
        if col in skip:
            continue
        if col in {"folder", "file", "subject", "exercise", "set_id", "start_utc", "end_utc"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
    return numeric


def x_matrix(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = df.loc[:, cols].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    ex = pd.get_dummies(df["exercise"].astype(str), prefix="exercise", dtype=float)
    return pd.concat([x.reset_index(drop=True), ex.reset_index(drop=True)], axis=1)


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4) if len(set(np.round(y_true, 6))) > 1 else 0.0,
        "spearman": round(float(rho), 4) if not np.isnan(rho) else 0.0,
    }


def eval_prediction(dataset: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in ["vo2_mean", "vo2_peak"]:
        for lag, sub in dataset.groupby("lag_sec"):
            sub = sub[pd.to_numeric(sub[target], errors="coerce").notna()].copy()
            groups = sub["folder"].astype(str).to_numpy()
            unique = sorted(set(groups.tolist()))
            if len(unique) < 2 or len(sub) < 20:
                continue
            folds = min(args.folds, len(unique))
            y = sub[target].to_numpy(dtype=float)
            mean_pred = np.full_like(y, float(np.mean(y)), dtype=float)
            rows.append({"target": target, "lag_sec": lag, "model": "global_mean", "n_sets": len(sub), "subjects": len(unique), **scores(y, mean_pred)})
            exercise_mean = sub.groupby("exercise")[target].mean()
            ex_pred = sub["exercise"].map(exercise_mean).fillna(float(np.mean(y))).to_numpy(dtype=float)
            rows.append({"target": target, "lag_sec": lag, "model": "exercise_mean_in_sample", "n_sets": len(sub), "subjects": len(unique), **scores(y, ex_pred)})

            cols = feature_columns(sub)
            x = x_matrix(sub, cols)
            splitter = GroupKFold(n_splits=folds)
            for model_name in ["ridge", "random_forest"]:
                y_true: list[float] = []
                y_pred: list[float] = []
                for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
                    if model_name == "ridge":
                        model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
                    else:
                        model = RandomForestRegressor(
                            n_estimators=args.n_estimators,
                            min_samples_leaf=3,
                            max_features="sqrt",
                            random_state=args.seed + fold_idx,
                            n_jobs=-1,
                        )
                    model.fit(x.iloc[train_idx], y[train_idx])
                    pred = model.predict(x.iloc[val_idx])
                    y_true.extend(y[val_idx].tolist())
                    y_pred.extend(pred.tolist())
                rows.append({"target": target, "lag_sec": lag, "model": model_name, "n_sets": len(y_true), "subjects": len(unique), **scores(np.asarray(y_true), np.asarray(y_pred))})
    return pd.DataFrame(rows)


def correlation_table(dataset: pd.DataFrame) -> pd.DataFrame:
    feature_candidates = [
        "set_duration_sec",
        "n_reps",
        "reps_per_min",
        "concentric_ratio_samples",
        "rep_duration_mean",
        "rep_duration_slope",
        "rep_pca_range_mean",
        "rep_pca_range_slope",
        "sim_to_first_mean",
        "sim_to_first_slope",
        "sim_to_first_last_minus_first",
        "set_gyro_mag_rms",
        "set_gyro_mag_diff_rms",
        "set_acc_mag_rms",
        "weight_kg",
        "rpe_label",
    ]
    rows: list[dict[str, object]] = []
    for lag, sub in dataset.groupby("lag_sec"):
        for target in ["vo2_mean", "vo2_peak"]:
            y = pd.to_numeric(sub[target], errors="coerce")
            for feature in feature_candidates:
                if feature not in sub.columns:
                    continue
                x = pd.to_numeric(sub[feature], errors="coerce")
                mask = x.notna() & y.notna()
                if mask.sum() < 10 or x[mask].nunique() < 2:
                    continue
                rho = spearmanr(x[mask], y[mask]).statistic
                rows.append({"lag_sec": lag, "target": target, "feature": feature, "n": int(mask.sum()), "spearman": round(float(rho), 4)})
    return pd.DataFrame(rows).sort_values(["target", "lag_sec", "spearman"], ascending=[True, True, False])


def plot_lag_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    for target in sorted(summary["target"].unique()):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in ["global_mean", "exercise_mean_in_sample", "ridge", "random_forest"]:
            sub = summary[(summary["target"].eq(target)) & (summary["model"].eq(model))].sort_values("lag_sec")
            if sub.empty:
                continue
            ax.plot(sub["lag_sec"], sub["mae"], marker="o", label=model)
        ax.set_xlabel("VO2 lag after set (sec)")
        ax.set_ylabel(f"{target} MAE")
        ax.set_title(f"VO2 Prediction Error by Lag: {target}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"019_{target}_mae_by_lag.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze relation between GT-segmented IMU waveform features and real-time VO2.")
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/019_vo2_gt_waveform_relation"))
    parser.add_argument("--lags-sec", type=float, nargs="+", default=[0, 10, 20, 30, 45, 60])
    parser.add_argument("--min-segment-samples", type=int, default=10)
    parser.add_argument("--min-vo2-points", type=int, default=2)
    parser.add_argument("--resample-points", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, status = build_dataset(args)
    dataset.to_csv(args.output_dir / "019_vo2_set_waveform_dataset.csv", index=False)
    status.to_csv(args.output_dir / "019_vo2_alignment_status.csv", index=False)
    if dataset.empty:
        raise SystemExit("No aligned VO2/IMU set rows found.")
    summary = eval_prediction(dataset, args)
    corr = correlation_table(dataset)
    summary.to_csv(args.output_dir / "019_vo2_prediction_summary.csv", index=False)
    corr.to_csv(args.output_dir / "019_vo2_feature_correlations.csv", index=False)
    plot_lag_summary(summary, args.output_dir)
    best = summary[summary["model"].isin(["ridge", "random_forest"])].sort_values("mae").head(10)
    summary_json = {
        "output_dir": str(args.output_dir),
        "set_lag_rows": int(len(dataset)),
        "subjects": sorted(set(dataset["folder"].astype(str))),
        "lags_sec": args.lags_sec,
        "best_models": best.to_dict(orient="records"),
        "notes": {
            "time_alignment": "IMU pc_time is localized as Asia/Taipei and converted to UTC; VO2 uses Time[utc].",
            "thomas": "thomas0506workout has no overlap because VO2 file is dated 2026-05-05 while IMU session is 2026-05-06.",
            "lag": "VO2 breath-by-breath response can lag movement; results are computed for 0, 10, 20, 30, 45, 60 sec windows after each set.",
            "level": "Set-level analysis is primary because VO2 sampling is breath-by-breath and too sparse/delayed for reliable per-rep targets.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(status.to_string(index=False))
    print()
    print(summary.sort_values(["target", "lag_sec", "mae"]).to_string(index=False))
    print()
    print(corr.groupby(["target", "lag_sec"]).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
