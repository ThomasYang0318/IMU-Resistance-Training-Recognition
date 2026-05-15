from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from evaluate_rep_segmentation_classification import robust_zscore


AXES_9 = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")
SENSOR_GROUPS = {
    "acc": ("ax", "ay", "az"),
    "gyro": ("gx", "gy", "gz"),
    "mag": ("mx", "my", "mz"),
}


@dataclass(frozen=True)
class FeatureInfo:
    feature: str
    sensor_group: str
    family: str
    source: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rep-level IMU feature relevance analysis.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-rep-samples", type=int, default=20)
    parser.add_argument("--top-features", type=int, default=30)
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def zero_crossing_rate(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered = values - float(np.median(values))
    signs = np.sign(centered)
    signs[signs == 0] = 1
    return float(np.mean(signs[1:] != signs[:-1]))


def slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(values))
    y = values.astype(np.float64)
    x_centered = x - float(np.mean(x))
    denom = float(np.sum(x_centered**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x_centered * (y - float(np.mean(y)))) / denom)


def spectral_features(values: np.ndarray) -> dict[str, float]:
    if len(values) < 4:
        return {
            "dominant_freq_ratio": 0.0,
            "spectral_entropy": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }
    centered = values.astype(np.float64) - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 1 or float(np.sum(power[1:])) < 1e-12:
        return {
            "dominant_freq_ratio": 0.0,
            "spectral_entropy": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }
    power = power[1:]
    total = float(np.sum(power))
    prob = power / total
    entropy = -float(np.sum(prob * np.log2(prob + 1e-12)) / math.log2(len(prob)))
    dominant_freq_ratio = float(np.argmax(power) + 1) / float(len(values))
    thirds = np.array_split(power, 3)
    band_ratios = [float(np.sum(band) / total) if len(band) else 0.0 for band in thirds]
    return {
        "dominant_freq_ratio": dominant_freq_ratio,
        "spectral_entropy": entropy,
        "low_band_ratio": band_ratios[0],
        "mid_band_ratio": band_ratios[1],
        "high_band_ratio": band_ratios[2],
    }


def haar_energy_ratios(values: np.ndarray, levels: int = 4) -> dict[str, float]:
    signal = values.astype(np.float64)
    total_energy = float(np.sum(signal**2))
    if total_energy < 1e-12:
        return {f"haar_l{level}_energy_ratio": 0.0 for level in range(1, levels + 1)}
    ratios: dict[str, float] = {}
    current = signal.copy()
    for level in range(1, levels + 1):
        if len(current) < 2:
            ratios[f"haar_l{level}_energy_ratio"] = 0.0
            continue
        if len(current) % 2 == 1:
            current = current[:-1]
        even = current[0::2]
        odd = current[1::2]
        detail = (even - odd) / math.sqrt(2.0)
        approx = (even + odd) / math.sqrt(2.0)
        ratios[f"haar_l{level}_energy_ratio"] = float(np.sum(detail**2) / total_energy)
        current = approx
    return ratios


def series_stats(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    if len(values) == 0:
        return {}
    diff = np.diff(values, prepend=values[:1])
    centered = values - float(np.mean(values))
    std = float(np.std(values))
    centered_std = float(np.std(centered))
    if centered_std < 1e-12:
        skewness = 0.0
        kurtosis = 0.0
    else:
        z = centered / centered_std
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4) - 3.0)
    q75, q25 = np.percentile(values, [75, 25])
    stats = {
        "mean": float(np.mean(values)),
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "iqr": float(q75 - q25),
        "rms": float(np.sqrt(np.mean(values**2))),
        "energy_mean": float(np.mean(values**2)),
        "abs_mean": float(np.mean(np.abs(values))),
        "diff_abs_mean": float(np.mean(np.abs(diff))),
        "diff_rms": float(np.sqrt(np.mean(diff**2))),
        "slope": slope(values),
        "zero_crossing_rate": zero_crossing_rate(values),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    stats.update(spectral_features(values))
    return stats


def add_feature(
    features: dict[str, float],
    feature_info: dict[str, FeatureInfo],
    name: str,
    value: float,
    sensor_group: str,
    family: str,
    source: str,
    description: str,
) -> None:
    features[name] = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
    feature_info.setdefault(name, FeatureInfo(name, sensor_group, family, source, description))


def add_series_features(
    features: dict[str, float],
    feature_info: dict[str, FeatureInfo],
    prefix: str,
    values: np.ndarray,
    sensor_group: str,
    source: str,
) -> None:
    for stat_name, value in series_stats(values).items():
        add_feature(
            features,
            feature_info,
            f"{prefix}__{stat_name}",
            value,
            sensor_group=sensor_group,
            family="time_frequency" if "band" in stat_name or "spectral" in stat_name or "freq" in stat_name else "time",
            source=source,
            description=f"{stat_name} of {source}",
        )
    for wavelet_name, value in haar_energy_ratios(values).items():
        add_feature(
            features,
            feature_info,
            f"{prefix}__{wavelet_name}",
            value,
            sensor_group=sensor_group,
            family="wavelet",
            source=source,
            description=f"{wavelet_name} of {source}",
        )


def group_matrix(segment: pd.DataFrame, axes: tuple[str, ...]) -> np.ndarray:
    available = [axis for axis in axes if axis in segment.columns]
    if not available:
        return np.zeros((len(segment), 0), dtype=np.float64)
    return segment.loc[:, available].to_numpy(dtype=np.float64)


def magnitude(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(values.shape[0], dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def add_pca_group_features(
    features: dict[str, float],
    feature_info: dict[str, FeatureInfo],
    prefix: str,
    x: np.ndarray,
    sensor_group: str,
) -> None:
    if x.shape[0] < 3 or x.shape[1] < 2:
        for idx in range(1, 4):
            add_feature(features, feature_info, f"{prefix}__pca{idx}_var_ratio", 0.0, sensor_group, "pca", prefix, "PCA variance ratio")
        add_feature(features, feature_info, f"{prefix}__dominant_axis_var_share", 0.0, sensor_group, "pca", prefix, "Dominant axis variance share")
        return
    z = np.apply_along_axis(robust_zscore, 0, x)
    variances = np.var(z, axis=0)
    total_var = float(np.sum(variances))
    dominant_share = float(np.max(variances) / total_var) if total_var > 1e-12 else 0.0
    _, s, _ = np.linalg.svd(z, full_matrices=False)
    explained = (s**2) / max(float(np.sum(s**2)), 1e-12)
    for idx in range(3):
        value = float(explained[idx]) if idx < len(explained) else 0.0
        add_feature(features, feature_info, f"{prefix}__pca{idx + 1}_var_ratio", value, sensor_group, "pca", prefix, "PCA variance ratio")
    add_feature(features, feature_info, f"{prefix}__dominant_axis_var_share", dominant_share, sensor_group, "pca", prefix, "Dominant axis variance share")


def dominant_axis(segment: pd.DataFrame) -> str:
    available = [axis for axis in AXES_9 if axis in segment.columns]
    if not available:
        return ""
    stds = segment.loc[:, available].std(axis=0).fillna(0.0)
    return str(stds.idxmax())


def extract_rep_features(
    segment: pd.DataFrame,
    feature_info: dict[str, FeatureInfo],
) -> tuple[dict[str, float], str]:
    features: dict[str, float] = {}

    for axis in AXES_9:
        if axis not in segment.columns:
            continue
        axis_group = "acc" if axis.startswith("a") else "gyro" if axis.startswith("g") else "mag"
        values = segment[axis].to_numpy(dtype=np.float64)
        add_series_features(features, feature_info, f"axis_{axis}", values, axis_group, axis)

    for group_name, axes in SENSOR_GROUPS.items():
        x = group_matrix(segment, axes)
        mag = magnitude(x)
        add_series_features(features, feature_info, f"{group_name}_norm", mag, group_name, f"{group_name}_norm")
        add_pca_group_features(features, feature_info, f"{group_name}_pca", x, group_name)

    x9 = group_matrix(segment, AXES_9)
    all_norm = magnitude(x9)
    add_series_features(features, feature_info, "all9_norm", all_norm, "all9", "all9_norm")
    add_pca_group_features(features, feature_info, "all9_pca", x9, "all9")

    available = [axis for axis in AXES_9 if axis in segment.columns]
    for idx, left in enumerate(available):
        left_values = segment[left].to_numpy(dtype=np.float64)
        left_group = "acc" if left.startswith("a") else "gyro" if left.startswith("g") else "mag"
        for right in available[idx + 1 :]:
            right_values = segment[right].to_numpy(dtype=np.float64)
            right_group = "acc" if right.startswith("a") else "gyro" if right.startswith("g") else "mag"
            if float(np.std(left_values)) < 1e-12 or float(np.std(right_values)) < 1e-12:
                corr = 0.0
            else:
                corr = float(np.corrcoef(left_values, right_values)[0, 1])
            pair_group = left_group if left_group == right_group else f"{left_group}_{right_group}"
            add_feature(
                features,
                feature_info,
                f"corr__{left}__{right}",
                corr,
                sensor_group=pair_group,
                family="axis_correlation",
                source=f"{left},{right}",
                description=f"Correlation between {left} and {right}",
            )

    add_feature(
        features,
        feature_info,
        "rep_duration_samples",
        float(len(segment)),
        sensor_group="temporal",
        family="temporal",
        source="rep_duration",
        description="Ground-truth rep duration in samples",
    )
    return features, dominant_axis(segment)


def build_feature_table(truth: pd.DataFrame, min_rep_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_info: dict[str, FeatureInfo] = {}
    rows: list[dict[str, object]] = []
    df_cache: dict[str, pd.DataFrame] = {}
    for rep in truth.itertuples(index=False):
        start = int(rep.true_start)
        end = int(rep.true_end)
        if end - start < min_rep_samples:
            continue
        file_path = str(rep.file)
        if file_path not in df_cache:
            df_cache[file_path] = pd.read_csv(file_path)
        segment = df_cache[file_path].iloc[start:end].reset_index(drop=True)
        features, dom_axis = extract_rep_features(segment, feature_info)
        rows.append(
            {
                "file": file_path,
                "subject": str(rep.subject),
                "exercise": str(rep.exercise),
                "set_id": str(rep.set_id),
                "rep_id": str(rep.rep_id),
                "true_start": start,
                "true_end": end,
                "dominant_axis": dom_axis,
                **features,
            }
        )

    feature_table = pd.DataFrame(rows)
    meta_table = pd.DataFrame([info.__dict__ for info in feature_info.values()]).sort_values("feature")
    return feature_table, meta_table


def numeric_feature_columns(feature_table: pd.DataFrame, feature_meta: pd.DataFrame) -> list[str]:
    feature_set = set(feature_meta["feature"].tolist())
    return [col for col in feature_table.columns if col in feature_set and pd.api.types.is_numeric_dtype(feature_table[col])]


def scaled_matrix(feature_table: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = feature_table.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(x)


def random_forest(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
    )


def rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def relevance_scores(
    feature_table: pd.DataFrame,
    feature_meta: pd.DataFrame,
    feature_cols: list[str],
    folds: int,
    seed: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_scaled = scaled_matrix(feature_table, feature_cols)
    y = feature_table["exercise"].astype(str).to_numpy()
    groups = feature_table["subject"].astype(str).to_numpy()

    f_scores, _ = f_classif(x_scaled, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    mi_scores = mutual_info_classif(x_scaled, y, random_state=seed)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)

    rf = random_forest(seed, n_estimators)
    rf.fit(x_scaled, y)
    rf_scores = rf.feature_importances_

    n_splits = min(folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, object]] = []
    fold_importance_rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(x_scaled, y, groups), start=1):
        scaler = StandardScaler().fit(x_scaled[train_idx])
        x_train = scaler.transform(x_scaled[train_idx])
        x_val = scaler.transform(x_scaled[val_idx])
        fold_rf = random_forest(seed + fold_idx, max(120, n_estimators // 2))
        fold_rf.fit(x_train, y[train_idx])
        pred = fold_rf.predict(x_val)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": ",".join(sorted(set(groups[train_idx]))),
                "val_subjects": ",".join(sorted(set(groups[val_idx]))),
                "val_samples": int(len(val_idx)),
                "accuracy": round(float(accuracy_score(y[val_idx], pred)), 4),
            }
        )
        fold_importances = fold_rf.feature_importances_
        ranks = pd.Series(fold_importances, index=feature_cols).rank(method="average", ascending=False)
        for feature, importance, rank in zip(feature_cols, fold_importances, ranks.to_numpy(), strict=True):
            fold_importance_rows.append(
                {
                    "fold": fold_idx,
                    "feature": feature,
                    "fold_rf_importance": float(importance),
                    "fold_rank": float(rank),
                    "is_top_20": bool(rank <= 20),
                }
            )

    fold_importance = pd.DataFrame(fold_importance_rows)
    stability = (
        fold_importance.groupby("feature", sort=True)
        .agg(
            fold_rf_importance_mean=("fold_rf_importance", "mean"),
            fold_rf_importance_std=("fold_rf_importance", "std"),
            fold_rank_mean=("fold_rank", "mean"),
            fold_rank_std=("fold_rank", "std"),
            top20_fold_count=("is_top_20", "sum"),
        )
        .reset_index()
    )
    stability["fold_rf_importance_std"] = stability["fold_rf_importance_std"].fillna(0.0)
    stability["fold_rank_std"] = stability["fold_rank_std"].fillna(0.0)

    scores = pd.DataFrame(
        {
            "feature": feature_cols,
            "anova_f": f_scores,
            "mutual_info": mi_scores,
            "rf_importance": rf_scores,
        }
    )
    scores = scores.merge(stability, on="feature", how="left")
    for col in ["anova_f", "mutual_info", "rf_importance", "fold_rf_importance_mean"]:
        max_value = float(scores[col].max())
        scores[f"{col}_norm"] = scores[col] / max_value if max_value > 1e-12 else 0.0
    scores["top20_fold_ratio"] = scores["top20_fold_count"] / float(n_splits)
    scores["composite_score"] = (
        scores["anova_f_norm"]
        + scores["mutual_info_norm"]
        + scores["rf_importance_norm"]
        + scores["fold_rf_importance_mean_norm"]
        + scores["top20_fold_ratio"]
    ) / 5.0
    scores["anova_rank"] = rank_desc(scores["anova_f"])
    scores["mutual_info_rank"] = rank_desc(scores["mutual_info"])
    scores["rf_rank"] = rank_desc(scores["rf_importance"])
    scores["composite_rank"] = rank_desc(scores["composite_score"])
    scores = scores.merge(feature_meta, on="feature", how="left")
    scores = scores.sort_values("composite_score", ascending=False)
    return scores, pd.DataFrame(fold_rows), fold_importance


def subject_wise_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
    n_estimators: int,
) -> tuple[float, float]:
    n_splits = min(folds, len(np.unique(groups)))
    accuracies: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups), start=1):
        scaler = StandardScaler().fit(x[train_idx])
        clf = random_forest(seed + fold_idx, max(120, n_estimators // 2))
        clf.fit(scaler.transform(x[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(x[val_idx]))
        accuracies.append(float(accuracy_score(y[val_idx], pred)))
    return float(np.mean(accuracies)), float(np.std(accuracies))


def ablation_groups(feature_meta: pd.DataFrame, top_scores: pd.DataFrame) -> dict[str, list[str]]:
    by_feature = feature_meta.set_index("feature")
    groups: dict[str, list[str]] = {
        "acc_only": by_feature[by_feature["sensor_group"].isin(["acc"])].index.tolist(),
        "gyro_only": by_feature[by_feature["sensor_group"].isin(["gyro"])].index.tolist(),
        "mag_only": by_feature[by_feature["sensor_group"].isin(["mag"])].index.tolist(),
        "acc_gyro": by_feature[by_feature["sensor_group"].isin(["acc", "gyro", "acc_gyro"])].index.tolist(),
        "magnitudes_only": by_feature[by_feature["source"].str.contains("_norm", regex=False, na=False)].index.tolist(),
        "correlations_only": by_feature[by_feature["family"].eq("axis_correlation")].index.tolist(),
        "wavelet_only": by_feature[by_feature["family"].eq("wavelet")].index.tolist(),
        "pca_only": by_feature[by_feature["family"].eq("pca")].index.tolist(),
        "all_9_axis_features": by_feature.index.tolist(),
        "top20_stable": top_scores.head(20)["feature"].tolist(),
        "top40_stable": top_scores.head(40)["feature"].tolist(),
    }
    return {name: sorted(set(features)) for name, features in groups.items() if features}


def run_ablation(
    feature_table: pd.DataFrame,
    feature_cols: list[str],
    feature_meta: pd.DataFrame,
    scores: pd.DataFrame,
    folds: int,
    seed: int,
    n_estimators: int,
) -> pd.DataFrame:
    y = feature_table["exercise"].astype(str).to_numpy()
    groups = feature_table["subject"].astype(str).to_numpy()
    all_x = feature_table.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows: list[dict[str, object]] = []
    for group_name, cols in ablation_groups(feature_meta, scores).items():
        valid_cols = [col for col in cols if col in all_x.columns]
        if not valid_cols:
            continue
        x = all_x.loc[:, valid_cols].to_numpy(dtype=np.float64)
        mean_acc, std_acc = subject_wise_accuracy(x, y, groups, folds, seed, n_estimators)
        rows.append(
            {
                "feature_set": group_name,
                "num_features": len(valid_cols),
                "mean_accuracy": round(mean_acc, 4),
                "std_accuracy": round(std_acc, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)


def one_vs_rest_effects(feature_table: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subjects = sorted(feature_table["subject"].astype(str).unique())
    for exercise in sorted(feature_table["exercise"].astype(str).unique()):
        mask = feature_table["exercise"].astype(str).eq(exercise)
        for feature in feature_cols:
            in_values = feature_table.loc[mask, feature].to_numpy(dtype=np.float64)
            out_values = feature_table.loc[~mask, feature].to_numpy(dtype=np.float64)
            pooled = math.sqrt((float(np.var(in_values)) + float(np.var(out_values))) / 2.0)
            cohen_d = 0.0 if pooled < 1e-12 else float((np.mean(in_values) - np.mean(out_values)) / pooled)
            global_sign = 1 if cohen_d >= 0 else -1
            subject_signs: list[int] = []
            subject_diffs: list[float] = []
            for subject in subjects:
                subject_mask = feature_table["subject"].astype(str).eq(subject)
                subject_in = feature_table.loc[subject_mask & mask, feature].to_numpy(dtype=np.float64)
                subject_out = feature_table.loc[subject_mask & ~mask, feature].to_numpy(dtype=np.float64)
                if len(subject_in) == 0 or len(subject_out) == 0:
                    continue
                diff = float(np.mean(subject_in) - np.mean(subject_out))
                subject_diffs.append(diff)
                subject_signs.append(1 if diff >= 0 else -1)
            stable_subject_ratio = float(np.mean(np.asarray(subject_signs) == global_sign)) if subject_signs else 0.0
            rows.append(
                {
                    "exercise": exercise,
                    "feature": feature,
                    "cohen_d": round(cohen_d, 4),
                    "abs_cohen_d": round(abs(cohen_d), 4),
                    "stable_subject_ratio": round(stable_subject_ratio, 4),
                    "stable_effect_score": round(abs(cohen_d) * stable_subject_ratio, 4),
                    "mean_subject_diff": round(float(np.mean(subject_diffs)), 4) if subject_diffs else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["exercise", "stable_effect_score"], ascending=[True, False])


def dominant_axis_distribution(feature_table: pd.DataFrame) -> pd.DataFrame:
    counts = (
        feature_table.groupby(["exercise", "dominant_axis"], sort=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("exercise")["count"].transform("sum")
    counts["ratio"] = (counts["count"] / totals).round(4)
    return counts


def plot_top_features(scores: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = scores.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(top["feature"], top["composite_score"], color="#4c78a8")
    ax.set_xlabel("Composite relevance score")
    ax.set_title("Top Rep-Level IMU Features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stability(scores: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = scores.sort_values("fold_rf_importance_mean", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(top["feature"], top["fold_rf_importance_mean"], xerr=top["fold_rf_importance_std"], color="#59a14f")
    ax.set_xlabel("Mean RF importance across subject-wise folds")
    ax.set_title("Cross-Subject Feature Stability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, output_path: Path) -> None:
    data = ablation.sort_values("mean_accuracy")
    fig, ax = plt.subplots(figsize=(9, max(4.8, len(data) * 0.38)))
    ax.barh(data["feature_set"], data["mean_accuracy"], xerr=data["std_accuracy"], color="#f58518")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Subject-wise GroupKFold accuracy")
    ax.set_title("Sensor / Feature Group Ablation")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(data["mean_accuracy"]):
        ax.text(min(float(value) + 0.01, 0.98), y, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_family(scores: pd.DataFrame, output_path: Path) -> None:
    data = (
        scores.groupby("family", sort=True)
        .agg(mean_composite=("composite_score", "mean"), top20_count=("composite_rank", lambda x: int((x <= 20).sum())))
        .reset_index()
        .sort_values("mean_composite")
    )
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(data) * 0.45)))
    ax.barh(data["family"], data["mean_composite"], color="#b279a2")
    ax.set_xlabel("Mean composite relevance")
    ax.set_title("Feature Family Relevance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_exercise_heatmap(effects: pd.DataFrame, feature_meta: pd.DataFrame, output_path: Path) -> None:
    top_features = (
        effects.groupby("exercise", sort=True)
        .head(3)["feature"]
        .drop_duplicates()
        .tolist()
    )
    subset = effects[effects["feature"].isin(top_features)].copy()
    pivot = subset.pivot(index="exercise", columns="feature", values="stable_effect_score").fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.65), max(5, len(pivot.index) * 0.5)))
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Exercise-Specific Stable Feature Effects")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > matrix.max() * 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="abs(Cohen's d) x subject stability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dominant_axis(dist: pd.DataFrame, output_path: Path) -> None:
    pivot = dist.pivot(index="exercise", columns="dominant_axis", values="ratio").fillna(0.0)
    ordered_cols = [axis for axis in AXES_9 if axis in pivot.columns]
    pivot = pivot.loc[:, ordered_cols]
    fig, ax = plt.subplots(figsize=(9, max(4.8, len(pivot.index) * 0.45)))
    bottom = np.zeros(len(pivot), dtype=np.float64)
    colors = plt.cm.tab20(np.linspace(0, 1, len(pivot.columns)))
    for color, axis in zip(colors, pivot.columns, strict=True):
        values = pivot[axis].to_numpy(dtype=float)
        ax.barh(pivot.index, values, left=bottom, label=axis, color=color)
        bottom += values
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Dominant-axis ratio by rep")
    ax.set_title("Dominant IMU Axis Distribution by Exercise")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_embedding(feature_table: pd.DataFrame, scores: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top_features = scores.head(top_n)["feature"].tolist()
    x = feature_table.loc[:, top_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    x_scaled = StandardScaler().fit_transform(x)
    coords = PCA(n_components=2, random_state=0).fit_transform(x_scaled)
    exercises = sorted(feature_table["exercise"].astype(str).unique())
    color_map = dict(zip(exercises, plt.cm.tab10(np.linspace(0, 1, len(exercises))), strict=True))
    fig, ax = plt.subplots(figsize=(8, 6))
    for exercise in exercises:
        mask = feature_table["exercise"].astype(str).eq(exercise).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.65, color=color_map[exercise], label=exercise)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Rep-Level Feature Embedding using Top {top_n} Stable Features")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Rep-Level Feature Classifier Confusion Matrix")
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            value = cm[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def cross_val_confusion(
    feature_table: pd.DataFrame,
    features: list[str],
    folds: int,
    seed: int,
    n_estimators: int,
) -> pd.DataFrame:
    labels = sorted(feature_table["exercise"].astype(str).unique())
    y = feature_table["exercise"].astype(str).to_numpy()
    groups = feature_table["subject"].astype(str).to_numpy()
    x = feature_table.loc[:, features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    n_splits = min(folds, len(np.unique(groups)))
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    for fold_idx, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups), start=1):
        scaler = StandardScaler().fit(x[train_idx])
        clf = random_forest(seed + 100 + fold_idx, max(120, n_estimators // 2))
        clf.fit(scaler.transform(x[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(x[val_idx]))
        y_true_all.extend(y[val_idx])
        y_pred_all.extend(pred)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize="true")
    return pd.DataFrame(cm, index=labels, columns=labels)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth = read_required_csv(args.run_dir / "rep_segmentation_truth_matches.csv")
    feature_table, feature_meta = build_feature_table(truth, args.min_rep_samples)
    feature_cols = numeric_feature_columns(feature_table, feature_meta)

    feature_table.to_csv(args.output_dir / "rep_level_feature_table.csv", index=False)
    feature_meta.to_csv(args.output_dir / "rep_level_feature_metadata.csv", index=False)

    scores, folds, fold_importance = relevance_scores(
        feature_table,
        feature_meta,
        feature_cols,
        folds=args.folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    scores.to_csv(args.output_dir / "rep_feature_relevance_scores.csv", index=False)
    folds.to_csv(args.output_dir / "rep_feature_subjectwise_folds.csv", index=False)
    fold_importance.to_csv(args.output_dir / "rep_feature_fold_importance.csv", index=False)

    ablation = run_ablation(
        feature_table,
        feature_cols,
        feature_meta,
        scores,
        folds=args.folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    ablation.to_csv(args.output_dir / "sensor_group_ablation_accuracy.csv", index=False)

    effects = one_vs_rest_effects(feature_table, feature_cols)
    effects = effects.merge(feature_meta, on="feature", how="left")
    effects.to_csv(args.output_dir / "exercise_feature_effects.csv", index=False)
    effects.groupby("exercise", sort=True).head(10).to_csv(args.output_dir / "top_features_by_exercise.csv", index=False)

    dom_dist = dominant_axis_distribution(feature_table)
    dom_dist.to_csv(args.output_dir / "dominant_axis_by_exercise.csv", index=False)

    plot_top_features(scores, args.output_dir / "top_rep_features_overall.png", args.top_features)
    plot_stability(scores, args.output_dir / "feature_stability_across_subjects.png", args.top_features)
    plot_ablation(ablation, args.output_dir / "sensor_group_ablation_accuracy.png")
    plot_feature_family(scores, args.output_dir / "feature_family_importance.png")
    plot_exercise_heatmap(effects, feature_meta, args.output_dir / "feature_importance_by_exercise.png")
    plot_dominant_axis(dom_dist, args.output_dir / "dominant_axis_by_exercise.png")
    plot_embedding(feature_table, scores, args.output_dir / "exercise_feature_embedding_pca.png", top_n=min(args.top_features, len(feature_cols)))

    top20 = scores.head(20)["feature"].tolist()
    cm_df = cross_val_confusion(feature_table, top20, args.folds, args.seed, args.n_estimators)
    cm_df.to_csv(args.output_dir / "top20_feature_confusion_matrix.csv")
    plot_confusion(cm_df.to_numpy(dtype=float), cm_df.index.tolist(), args.output_dir / "top20_feature_confusion_matrix.png")

    summary = {
        "run_dir": str(args.run_dir),
        "num_reps": int(len(feature_table)),
        "num_subjects": int(feature_table["subject"].nunique()),
        "num_exercises": int(feature_table["exercise"].nunique()),
        "num_features": int(len(feature_cols)),
        "best_ablation_set": str(ablation.iloc[0]["feature_set"]) if not ablation.empty else None,
        "best_ablation_accuracy": float(ablation.iloc[0]["mean_accuracy"]) if not ablation.empty else None,
        "top_10_features": scores.head(10)["feature"].tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(scores.head(20)[["feature", "sensor_group", "family", "composite_score", "top20_fold_ratio"]].to_string(index=False))
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
