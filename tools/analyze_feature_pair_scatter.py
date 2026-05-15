from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeaturePair:
    pair_name: str
    feature_x: str
    feature_y: str
    pair_type: str
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-feature scatter and 8-class separability analysis.")
    parser.add_argument("--feature-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--ranked-pairs", type=int, default=12)
    parser.add_argument("--top-features", type=int, default=60)
    parser.add_argument("--grid-pairs", type=int, default=12)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def feature_info(meta: pd.DataFrame, feature: str) -> dict[str, object]:
    row = meta[meta["feature"].eq(feature)]
    if row.empty:
        return {"sensor_group": "", "family": "", "source": "", "description": ""}
    return row.iloc[0].to_dict()


def first_feature(scores: pd.DataFrame, meta: pd.DataFrame, query: pd.Series) -> str | None:
    allowed = set(meta.loc[query, "feature"].tolist())
    rows = scores[scores["feature"].isin(allowed)]
    if rows.empty:
        return None
    return str(rows.iloc[0]["feature"])


def top_features(scores: pd.DataFrame, meta: pd.DataFrame, query: pd.Series, n: int) -> list[str]:
    allowed = set(meta.loc[query, "feature"].tolist())
    return scores[scores["feature"].isin(allowed)]["feature"].head(n).astype(str).tolist()


def add_pair(
    pairs: list[FeaturePair],
    seen: set[tuple[str, str]],
    pair_name: str,
    feature_x: str | None,
    feature_y: str | None,
    pair_type: str,
    rationale: str,
) -> None:
    if not feature_x or not feature_y or feature_x == feature_y:
        return
    key = tuple(sorted((feature_x, feature_y)))
    if key in seen:
        return
    seen.add(key)
    pairs.append(FeaturePair(pair_name, feature_x, feature_y, pair_type, rationale))


def select_method_pairs(scores: pd.DataFrame, meta: pd.DataFrame) -> list[FeaturePair]:
    pairs: list[FeaturePair] = []
    seen: set[tuple[str, str]] = set()

    is_axis_time = meta["family"].eq("time") & meta["source"].isin(["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"])
    is_norm_time = meta["family"].eq("time") & meta["source"].isin(["acc_norm", "gyro_norm", "mag_norm", "all9_norm"])

    group_queries: dict[str, pd.Series] = {
        "acc_axis_time": meta["sensor_group"].eq("acc") & is_axis_time,
        "acc_norm_time": meta["source"].eq("acc_norm") & meta["family"].eq("time"),
        "gyro_axis_time": meta["sensor_group"].eq("gyro") & is_axis_time,
        "gyro_norm_time": meta["source"].eq("gyro_norm") & meta["family"].eq("time"),
        "mag_axis_time": meta["sensor_group"].eq("mag") & is_axis_time,
        "axis_correlation": meta["family"].eq("axis_correlation"),
        "fft_spectral": meta["family"].eq("time_frequency"),
        "wavelet": meta["family"].eq("wavelet"),
        "pca_variance": meta["family"].eq("pca"),
        "norm_time": is_norm_time,
    }
    for group_name, query in group_queries.items():
        features = top_features(scores, meta, query, 2)
        add_pair(
            pairs,
            seen,
            f"{group_name}_top2",
            features[0] if len(features) > 0 else None,
            features[1] if len(features) > 1 else None,
            "method_top2",
            f"Top two ranked features inside {group_name}.",
        )

    acc_best = first_feature(scores, meta, meta["sensor_group"].eq("acc") & meta["family"].eq("time"))
    gyro_best = first_feature(scores, meta, meta["sensor_group"].eq("gyro") & meta["family"].eq("time"))
    corr_best = first_feature(scores, meta, meta["family"].eq("axis_correlation"))
    wavelet_best = first_feature(scores, meta, meta["family"].eq("wavelet"))
    spectral_best = first_feature(scores, meta, meta["family"].eq("time_frequency"))
    pca_best = first_feature(scores, meta, meta["family"].eq("pca"))
    duration = "rep_duration_samples" if "rep_duration_samples" in set(meta["feature"]) else None

    add_pair(pairs, seen, "best_acc_vs_best_gyro", acc_best, gyro_best, "mixed_sensor", "Best accelerometer feature vs best gyroscope feature.")
    add_pair(pairs, seen, "best_acc_vs_best_corr", acc_best, corr_best, "mixed_family", "Best accelerometer feature vs best axis-correlation feature.")
    add_pair(pairs, seen, "best_acc_vs_best_wavelet", acc_best, wavelet_best, "mixed_family", "Best accelerometer feature vs best wavelet feature.")
    add_pair(pairs, seen, "best_acc_vs_best_spectral", acc_best, spectral_best, "mixed_family", "Best accelerometer feature vs best spectral feature.")
    add_pair(pairs, seen, "best_acc_vs_best_pca", acc_best, pca_best, "mixed_family", "Best accelerometer feature vs best PCA feature.")
    add_pair(pairs, seen, "duration_vs_best_acc", duration, acc_best, "duration", "Rep duration vs best accelerometer feature.")

    return pairs


def select_ranked_pairs(
    table: pd.DataFrame,
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    ranked_pairs: int,
    top_features_count: int,
) -> list[FeaturePair]:
    score_map = scores.set_index("feature")["composite_score"].to_dict()
    meta_index = meta.set_index("feature")
    candidates = scores.head(top_features_count)["feature"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    for left, right in combinations(candidates, 2):
        if left not in table.columns or right not in table.columns:
            continue
        left_meta = meta_index.loc[left]
        right_meta = meta_index.loc[right]
        same_source = str(left_meta["source"]) == str(right_meta["source"])
        same_family = str(left_meta["family"]) == str(right_meta["family"])
        corr = abs(float(np.corrcoef(table[left].to_numpy(dtype=float), table[right].to_numpy(dtype=float))[0, 1]))
        if not np.isfinite(corr):
            corr = 1.0
        if corr > 0.92:
            continue
        diversity_bonus = 0.08 if not same_source else 0.0
        diversity_bonus += 0.04 if not same_family else 0.0
        rows.append(
            {
                "feature_x": left,
                "feature_y": right,
                "score": float(score_map.get(left, 0.0) + score_map.get(right, 0.0) + diversity_bonus),
                "abs_corr": corr,
            }
        )
    if not rows:
        return []
    ranked = pd.DataFrame(rows).sort_values(["score", "abs_corr"], ascending=[False, True]).head(ranked_pairs)
    return [
        FeaturePair(
            pair_name=f"ranked_pair_{idx:02d}",
            feature_x=str(row.feature_x),
            feature_y=str(row.feature_y),
            pair_type="ranked_diverse",
            rationale="High composite relevance with low pairwise redundancy.",
        )
        for idx, row in enumerate(ranked.itertuples(index=False), start=1)
    ]


def select_exercise_effect_pairs(feature_run_dir: Path, pairs: list[FeaturePair], seen: set[tuple[str, str]]) -> None:
    effects_path = feature_run_dir / "top_features_by_exercise.csv"
    if not effects_path.exists():
        return
    effects = pd.read_csv(effects_path)
    for exercise, group in effects.groupby("exercise", sort=True):
        features = group["feature"].astype(str).drop_duplicates().head(2).tolist()
        if len(features) < 2:
            continue
        add_pair(
            pairs,
            seen,
            f"{safe_name(str(exercise))}_best_effect_pair",
            features[0],
            features[1],
            "exercise_effect",
            f"Top two one-vs-rest stable-effect features for {exercise}.",
        )


def select_pairs(
    feature_run_dir: Path,
    table: pd.DataFrame,
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    ranked_pairs: int,
    top_features_count: int,
) -> list[FeaturePair]:
    selected = select_method_pairs(scores, meta)
    seen = {tuple(sorted((pair.feature_x, pair.feature_y))) for pair in selected}
    for pair in select_ranked_pairs(table, scores, meta, ranked_pairs, top_features_count):
        add_pair(selected, seen, pair.pair_name, pair.feature_x, pair.feature_y, pair.pair_type, pair.rationale)
    select_exercise_effect_pairs(feature_run_dir, selected, seen)
    valid = []
    for pair in selected:
        if pair.feature_x in table.columns and pair.feature_y in table.columns:
            valid.append(pair)
    return valid


def random_forest(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
    )


def evaluate_pair(
    table: pd.DataFrame,
    pair: FeaturePair,
    labels: list[str],
    folds: int,
    seed: int,
    n_estimators: int,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    x = table.loc[:, [pair.feature_x, pair.feature_y]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = table["exercise"].astype(str).to_numpy()
    groups = table["subject"].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))

    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups), start=1):
        scaler = StandardScaler().fit(x[train_idx])
        clf = random_forest(seed + fold_idx, n_estimators)
        clf.fit(scaler.transform(x[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(x[val_idx]))
        y_true_all.extend(y[val_idx].tolist())
        y_pred_all.extend(pred.tolist())
        fold_rows.append(
            {
                "pair_name": pair.pair_name,
                "fold": fold_idx,
                "train_subjects": ",".join(sorted(set(groups[train_idx]))),
                "val_subjects": ",".join(sorted(set(groups[val_idx]))),
                "val_samples": int(len(val_idx)),
                "accuracy": round(float(accuracy_score(y[val_idx], pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y[val_idx], pred)), 4),
            }
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        labels=labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "pair_name": pair.pair_name,
            "exercise": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize="true")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    metrics = {
        "pair_name": pair.pair_name,
        "pair_type": pair.pair_type,
        "feature_x": pair.feature_x,
        "feature_y": pair.feature_y,
        "rationale": pair.rationale,
        "accuracy": round(float(accuracy_score(y_true_all, y_pred_all)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_all, y_pred_all)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "min_class_f1": round(float(np.min(f1)), 4),
        "mean_recall": round(float(np.mean(recall)), 4),
    }
    return metrics, fold_rows, per_class, cm_df


def plot_pair_scatter(table: pd.DataFrame, pair: FeaturePair, metrics: dict[str, object], output_path: Path) -> None:
    x_raw = table[[pair.feature_x, pair.feature_y]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x_raw)
    labels = sorted(table["exercise"].astype(str).unique())
    colors = dict(zip(labels, plt.cm.tab10(np.linspace(0, 1, len(labels))), strict=True))

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in labels:
        mask = table["exercise"].astype(str).eq(label).to_numpy()
        ax.scatter(x_scaled[mask, 0], x_scaled[mask, 1], s=12, alpha=0.58, color=colors[label], label=label, linewidths=0)
        centroid = np.median(x_scaled[mask], axis=0)
        ax.scatter(centroid[0], centroid[1], s=95, marker="X", color=colors[label], edgecolor="black", linewidth=0.8)
    ax.set_xlabel(f"{pair.feature_x} (z-score)")
    ax.set_ylabel(f"{pair.feature_y} (z-score)")
    ax.set_title(f"{pair.pair_name}: acc={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}")
    ax.grid(alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scatter_grid(table: pd.DataFrame, pairs: list[FeaturePair], metrics_df: pd.DataFrame, output_path: Path) -> None:
    grid_pairs = pairs
    if not grid_pairs:
        return
    labels = sorted(table["exercise"].astype(str).unique())
    colors = dict(zip(labels, plt.cm.tab10(np.linspace(0, 1, len(labels))), strict=True))
    n_cols = 3
    n_rows = int(np.ceil(len(grid_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.2, n_rows * 4.2), squeeze=False)
    metric_map = metrics_df.set_index("pair_name").to_dict("index")
    for ax, pair in zip(axes.ravel(), grid_pairs, strict=False):
        x_raw = table[[pair.feature_x, pair.feature_y]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        x_scaled = StandardScaler().fit_transform(x_raw)
        for label in labels:
            mask = table["exercise"].astype(str).eq(label).to_numpy()
            ax.scatter(x_scaled[mask, 0], x_scaled[mask, 1], s=5, alpha=0.45, color=colors[label], linewidths=0)
        metric = metric_map[pair.pair_name]
        ax.set_title(f"{pair.pair_name}\nacc={metric['accuracy']:.3f}, F1={metric['macro_f1']:.3f}", fontsize=9)
        ax.set_xlabel(pair.feature_x, fontsize=7)
        ax.set_ylabel(pair.feature_y, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.16)
    for ax in axes.ravel()[len(grid_pairs) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color=colors[label], label=label)
        for label in labels
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(cm: pd.DataFrame, pair_name: str, output_path: Path) -> None:
    labels = cm.index.tolist()
    values = cm.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{pair_name} Confusion Matrix")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pair_accuracy(metrics: pd.DataFrame, output_path: Path) -> None:
    data = metrics.sort_values("macro_f1").tail(30)
    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.32)))
    ax.barh(data["pair_name"], data["macro_f1"], color="#4c78a8", label="macro-F1")
    ax.scatter(data["accuracy"], data["pair_name"], color="#f58518", s=28, label="accuracy", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Subject-wise out-of-fold score")
    ax.set_title("Two-Feature Pair Separability")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pair_dotplot(per_class: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    ordered_pairs = metrics.sort_values("macro_f1", ascending=False)["pair_name"].tolist()
    exercises = sorted(per_class["exercise"].unique())
    pair_to_y = {pair: idx for idx, pair in enumerate(ordered_pairs)}
    exercise_to_x = {exercise: idx for idx, exercise in enumerate(exercises)}
    rows = per_class[per_class["pair_name"].isin(ordered_pairs)].copy()
    x = rows["exercise"].map(exercise_to_x).to_numpy(dtype=float)
    y = rows["pair_name"].map(pair_to_y).to_numpy(dtype=float)
    f1 = rows["f1"].to_numpy(dtype=float)
    sizes = 35 + 260 * f1

    fig, ax = plt.subplots(figsize=(10, max(7, len(ordered_pairs) * 0.33)))
    scatter = ax.scatter(x, y, c=f1, s=sizes, cmap="viridis", vmin=0.0, vmax=1.0, edgecolor="black", linewidth=0.25)
    ax.set_xticks(np.arange(len(exercises)))
    ax.set_xticklabels(exercises, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ordered_pairs)))
    ax.set_yticklabels(ordered_pairs)
    ax.invert_yaxis()
    ax.set_title("Per-Exercise F1 by Two-Feature Pair")
    ax.set_xlabel("Exercise")
    ax.set_ylabel("Feature pair")
    ax.grid(alpha=0.18)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_pair_table(pairs: list[FeaturePair], meta: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        x_info = feature_info(meta, pair.feature_x)
        y_info = feature_info(meta, pair.feature_y)
        rows.append(
            {
                "pair_name": pair.pair_name,
                "pair_type": pair.pair_type,
                "feature_x": pair.feature_x,
                "feature_y": pair.feature_y,
                "x_sensor_group": x_info["sensor_group"],
                "x_family": x_info["family"],
                "x_source": x_info["source"],
                "y_sensor_group": y_info["sensor_group"],
                "y_family": y_info["family"],
                "y_source": y_info["source"],
                "rationale": pair.rationale,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir = args.output_dir / "scatter_pairs"
    confusion_dir = args.output_dir / "confusion_matrices"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    table = read_csv(args.feature_run_dir / "rep_level_feature_table.csv")
    meta = read_csv(args.feature_run_dir / "rep_level_feature_metadata.csv")
    scores = read_csv(args.feature_run_dir / "rep_feature_relevance_scores.csv").sort_values("composite_score", ascending=False)

    pairs = select_pairs(args.feature_run_dir, table, scores, meta, args.ranked_pairs, args.top_features)
    write_pair_table(pairs, meta, args.output_dir / "selected_feature_pairs.csv")

    labels = sorted(table["exercise"].astype(str).unique())
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    per_class_frames: list[pd.DataFrame] = []
    cm_long_rows: list[dict[str, object]] = []

    cm_by_pair: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        metrics, pair_fold_rows, per_class, cm = evaluate_pair(
            table,
            pair,
            labels,
            folds=args.folds,
            seed=args.seed,
            n_estimators=args.n_estimators,
        )
        metric_rows.append(metrics)
        fold_rows.extend(pair_fold_rows)
        per_class_frames.append(per_class)
        cm_by_pair[pair.pair_name] = cm
        for true_label in labels:
            for pred_label in labels:
                cm_long_rows.append(
                    {
                        "pair_name": pair.pair_name,
                        "true_exercise": true_label,
                        "pred_exercise": pred_label,
                        "value": float(cm.loc[true_label, pred_label]),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    fold_df = pd.DataFrame(fold_rows)
    per_class_df = pd.concat(per_class_frames, ignore_index=True).merge(
        metrics_df[["pair_name", "pair_type", "feature_x", "feature_y", "macro_f1", "accuracy"]],
        on="pair_name",
        how="left",
    )
    cm_long_df = pd.DataFrame(cm_long_rows)

    metrics_df.to_csv(args.output_dir / "feature_pair_metrics.csv", index=False)
    fold_df.to_csv(args.output_dir / "feature_pair_fold_metrics.csv", index=False)
    per_class_df.to_csv(args.output_dir / "feature_pair_per_exercise_metrics.csv", index=False)
    cm_long_df.to_csv(args.output_dir / "feature_pair_confusion_matrix_long.csv", index=False)

    ordered_pairs = [pair for pair_name in metrics_df["pair_name"] for pair in pairs if pair.pair_name == pair_name]
    for pair in ordered_pairs:
        pair_metrics = metrics_df[metrics_df["pair_name"].eq(pair.pair_name)].iloc[0].to_dict()
        plot_pair_scatter(table, pair, pair_metrics, scatter_dir / f"{safe_name(pair.pair_name)}.png")
        cm = cm_by_pair[pair.pair_name]
        cm.to_csv(confusion_dir / f"{safe_name(pair.pair_name)}.csv")
        plot_confusion(cm, pair.pair_name, confusion_dir / f"{safe_name(pair.pair_name)}.png")

    plot_pair_accuracy(metrics_df, args.output_dir / "feature_pair_overall_scores.png")
    plot_pair_dotplot(per_class_df, metrics_df, args.output_dir / "feature_pair_per_exercise_f1_dotplot.png")
    top_grid_pairs = ordered_pairs[: args.grid_pairs]
    plot_scatter_grid(table, top_grid_pairs, metrics_df, args.output_dir / "top_feature_pair_scatter_grid.png")

    best = metrics_df.iloc[0].to_dict()
    summary = {
        "feature_run_dir": str(args.feature_run_dir),
        "num_reps": int(len(table)),
        "num_subjects": int(table["subject"].nunique()),
        "num_exercises": int(table["exercise"].nunique()),
        "num_pairs": int(len(metrics_df)),
        "best_pair_name": str(best["pair_name"]),
        "best_feature_x": str(best["feature_x"]),
        "best_feature_y": str(best["feature_y"]),
        "best_accuracy": float(best["accuracy"]),
        "best_macro_f1": float(best["macro_f1"]),
        "scatter_dir": str(scatter_dir),
        "confusion_dir": str(confusion_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
