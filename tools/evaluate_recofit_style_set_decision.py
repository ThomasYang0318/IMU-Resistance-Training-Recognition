from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


DEFAULT_LABELS = [
    "db_bench_press",
    "db_biceps_curl",
    "db_rdl",
    "db_shoulder_press",
    "db_squat",
    "db_triceps_curl",
    "db_weighted_crunch",
    "one_arm_db_row",
]

ZH_LABELS = {
    "db_bench_press": "啞鈴臥推",
    "db_biceps_curl": "啞鈴二頭彎舉",
    "db_rdl": "啞鈴羅馬尼亞硬舉",
    "db_shoulder_press": "啞鈴肩推",
    "db_squat": "啞鈴深蹲",
    "db_triceps_curl": "啞鈴三頭伸展",
    "db_weighted_crunch": "負重捲腹",
    "one_arm_db_row": "單臂啞鈴划船",
}


def cjk_font() -> font_manager.FontProperties | None:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/PrivateFrameworks/FontServices.framework/Resources/Reserved/PingFangUI.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return font_manager.FontProperties(fname=str(path))
    return None


def majority_vote(values: pd.Series, labels: list[str]) -> str:
    counts = Counter(values.astype(str).tolist())
    return max(labels, key=lambda label: (counts.get(label, 0), -labels.index(label)))


def label_order(df: pd.DataFrame, prediction_columns: list[str] | None = None) -> list[str]:
    prediction_columns = prediction_columns or ["prediction"]
    seen = set(df["label"].astype(str).unique())
    for column in prediction_columns:
        if column in df:
            seen |= set(df[column].astype(str).unique())
    labels = [label for label in DEFAULT_LABELS if label in seen]
    labels.extend(sorted(seen - set(labels)))
    return labels


def add_active_run_ids(df: pd.DataFrame, gap_seconds: float) -> pd.DataFrame:
    ordered = df.sort_values(["file", "end_seconds", "window_index"]).copy()
    prev_file = ordered["file"].shift()
    prev_label = ordered["label"].shift()
    prev_time = ordered["end_seconds"].shift()
    new_run = (
        ordered["file"].ne(prev_file)
        | ordered["label"].ne(prev_label)
        | ((ordered["end_seconds"] - prev_time) > gap_seconds)
    )
    ordered["active_run_id"] = new_run.cumsum().astype(int) - 1
    return ordered


def apply_set_majority(df: pd.DataFrame, labels: list[str]) -> pd.Series:
    out = pd.Series(index=df.index, dtype=object)
    for _, run in df.groupby("active_run_id", sort=False):
        chosen = majority_vote(run["prediction"], labels)
        out.loc[run.index] = chosen
    return out.astype(str)


def apply_warmup(df: pd.DataFrame, labels: list[str], warmup_windows: int) -> pd.Series:
    out = pd.Series(index=df.index, dtype=object)
    for _, run in df.groupby("active_run_id", sort=False):
        warmup = run.head(max(1, warmup_windows))
        chosen = majority_vote(warmup["prediction"], labels)
        out.loc[run.index] = chosen
    return out.astype(str)


def apply_row_vote(df: pd.DataFrame, labels: list[str], prediction_columns: list[str]) -> pd.Series:
    return df[prediction_columns].apply(lambda row: majority_vote(row, labels), axis=1).astype(str)


def apply_pooled_set_majority(df: pd.DataFrame, labels: list[str], prediction_columns: list[str]) -> pd.Series:
    out = pd.Series(index=df.index, dtype=object)
    for _, run in df.groupby("active_run_id", sort=False):
        votes = pd.concat([run[column] for column in prediction_columns], ignore_index=True)
        chosen = majority_vote(votes, labels)
        out.loc[run.index] = chosen
    return out.astype(str)


def apply_pooled_warmup(
    df: pd.DataFrame,
    labels: list[str],
    prediction_columns: list[str],
    warmup_windows: int,
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=object)
    for _, run in df.groupby("active_run_id", sort=False):
        warmup = run.head(max(1, warmup_windows))
        votes = pd.concat([warmup[column] for column in prediction_columns], ignore_index=True)
        chosen = majority_vote(votes, labels)
        out.loc[run.index] = chosen
    return out.astype(str)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "min_recall": float(np.diag(prop).min()),
        "per_class_recall": {label: float(prop[idx, idx]) for idx, label in enumerate(labels)},
    }


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
    stem: str,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    prop = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    prop_df = pd.DataFrame(prop, index=labels, columns=labels)
    prop_df.to_csv(output_dir / f"{stem}_confusion_matrix_row_proportion.csv", float_format="%.6f")
    (prop_df * 100.0).to_csv(output_dir / f"{stem}_confusion_matrix_row_percent.csv", float_format="%.2f")
    prop_df.rename(index=ZH_LABELS, columns=ZH_LABELS).to_csv(
        output_dir / f"{stem}_confusion_matrix_row_proportion_zh.csv",
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    font_prop = cjk_font()
    tick_labels = [ZH_LABELS.get(label, label) for label in labels]
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    image = ax.imshow(prop, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontproperties=font_prop)
    ax.set_yticklabels(tick_labels, fontproperties=font_prop)
    ax.set_xlabel("預測類別", fontproperties=font_prop)
    ax.set_ylabel("真實類別", fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = prop[row, col]
            color = "white" if value >= 0.55 else "#222222"
            ax.text(col, row, f"{value * 100:.1f}%", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("列比例", rotation=270, labelpad=14, fontproperties=font_prop)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_confusion_matrix_row_proportion.png", dpi=220)
    fig.savefig(output_dir / f"{stem}_confusion_matrix_row_proportion.pdf")
    plt.close(fig)


def evaluate_prediction_file(
    prediction_path: Path,
    output_dir: Path,
    name: str,
    warmup_windows: list[int],
    gap_seconds: float,
) -> dict[str, object]:
    df = pd.read_csv(prediction_path)
    required = {"file", "window_index", "end_seconds", "label", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{prediction_path} missing required columns: {sorted(missing)}")
    labels = label_order(df)
    df = add_active_run_ids(df, gap_seconds)
    y_true = df["label"].astype(str).to_numpy()

    mode_predictions: dict[str, pd.Series] = {"window": df["prediction"].astype(str)}
    mode_predictions["set_majority"] = apply_set_majority(df, labels)
    for warmup in warmup_windows:
        mode_predictions[f"warmup_{warmup}"] = apply_warmup(df, labels, warmup)

    summary: dict[str, object] = {
        "prediction_file": str(prediction_path),
        "labels": labels,
        "active_runs": int(df["active_run_id"].nunique()),
        "windows": int(len(df)),
        "modes": {},
    }
    for mode, pred in mode_predictions.items():
        y_pred = pred.to_numpy(dtype=object)
        summary["modes"][mode] = summarize(y_true, y_pred, labels)
        save_confusion_outputs(
            y_true,
            y_pred,
            labels,
            output_dir,
            f"{name}_{mode}",
            f"{name}: {mode.replace('_', ' ')}",
        )
    return summary


def evaluate_pooled_vote(
    named_prediction_paths: dict[str, Path],
    output_dir: Path,
    name: str,
    warmup_windows: list[int],
    gap_seconds: float,
) -> dict[str, object]:
    keys = ["file", "window_index", "end_seconds", "label"]
    merged: pd.DataFrame | None = None
    prediction_columns: list[str] = []
    for pred_name, path in named_prediction_paths.items():
        frame = pd.read_csv(path)
        required = set(keys) | {"prediction"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        pred_column = f"{pred_name}_prediction"
        prediction_columns.append(pred_column)
        frame = frame[keys + ["prediction"]].rename(columns={"prediction": pred_column})
        merged = frame if merged is None else merged.merge(frame, on=keys, how="inner")
    if merged is None:
        raise ValueError("No prediction files provided for pooled vote.")

    labels = label_order(merged, prediction_columns)
    merged = add_active_run_ids(merged, gap_seconds)
    y_true = merged["label"].astype(str).to_numpy()
    mode_predictions: dict[str, pd.Series] = {
        "window_vote": apply_row_vote(merged, labels, prediction_columns),
        "set_majority": apply_pooled_set_majority(merged, labels, prediction_columns),
    }
    for warmup in warmup_windows:
        mode_predictions[f"warmup_{warmup}"] = apply_pooled_warmup(merged, labels, prediction_columns, warmup)

    summary: dict[str, object] = {
        "prediction_files": {key: str(value) for key, value in named_prediction_paths.items()},
        "labels": labels,
        "active_runs": int(merged["active_run_id"].nunique()),
        "windows": int(len(merged)),
        "vote_columns": prediction_columns,
        "modes": {},
    }
    for mode, pred in mode_predictions.items():
        y_pred = pred.to_numpy(dtype=object)
        summary["modes"][mode] = summarize(y_true, y_pred, labels)
        save_confusion_outputs(
            y_true,
            y_pred,
            labels,
            output_dir,
            f"{name}_{mode}",
            f"{name}: {mode.replace('_', ' ')}",
        )
    return summary


def parse_named_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply RecoFit-style set-level temporal decisions to active-window predictions.")
    parser.add_argument(
        "--prediction",
        action="append",
        default=[
            "base_hgb=artifacts/realtime_action_active_only_8class_models_20260518/window_predictions_hist_gradient_boosting.csv",
            "posture_hgb=artifacts/realtime_action_active_only_8class_posture_hgb_20260519/window_predictions_hist_gradient_boosting.csv",
        ],
        help="Named prediction CSV, e.g. name=path/to/window_predictions.csv. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/recofit_style_set_decision_20260519"))
    parser.add_argument("--warmup-windows", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10, 11])
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {}
    named_predictions: dict[str, Path] = {}
    for raw_prediction in args.prediction:
        name, prediction_path = parse_named_prediction(raw_prediction)
        named_predictions[name] = prediction_path
        run_summary[name] = evaluate_prediction_file(
            prediction_path=prediction_path,
            output_dir=args.output_dir,
            name=name,
            warmup_windows=args.warmup_windows,
            gap_seconds=args.gap_seconds,
        )
    if len(named_predictions) >= 2:
        vote_inputs = (
            {key: named_predictions[key] for key in ("base_hgb", "posture_hgb")}
            if {"base_hgb", "posture_hgb"}.issubset(named_predictions)
            else named_predictions
        )
        run_summary["pooled_vote"] = evaluate_pooled_vote(
            named_prediction_paths=vote_inputs,
            output_dir=args.output_dir,
            name="pooled_vote",
            warmup_windows=args.warmup_windows,
            gap_seconds=args.gap_seconds,
        )
    (args.output_dir / "summary_recofit_style_set_decision.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
