from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ds_ms_tcn import DSMSTCN, DSMSTCNConfig, MSTCN
from evaluate_rep_segmentation_classification import (
    PhaseSegment,
    RepSegment,
    phase_metric_rows,
    phase_metric_rows_by_phase,
    plot_exercise_accuracy_table,
    plot_phase_metrics,
    plot_phase_metrics_by_phase,
    plot_segmentation_metrics,
    plot_segmentation_metrics_by_exercise,
    plot_segmentation_metrics_by_subject,
    segmentation_metric_rows,
    segmentation_metric_rows_by_exercise,
    segmentation_metric_rows_by_subject,
    write_csv,
)


IMU_9AXIS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")
ACTIVE_PHASES = {"concentric", "eccentric"}
REST_ACTIONS = {"big_rest", "rest", "none", "nan", ""}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SessionData:
    path: Path
    subject: str
    features: np.ndarray
    macro_labels: np.ndarray
    micro_labels: np.ndarray
    active_mask: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    reps: np.ndarray
    sets: np.ndarray


@dataclass(frozen=True)
class WindowSpec:
    session_idx: int
    start: int
    end: int
    valid_len: int
    subject: str
    has_active: bool


@dataclass
class PredictionBundle:
    macro_pred: dict[int, np.ndarray]
    micro_pred: dict[int, np.ndarray]
    macro_stage_pred: dict[int, list[np.ndarray]]
    valid_mask: dict[int, np.ndarray]


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        sessions: Sequence[SessionData],
        specs: Sequence[WindowSpec],
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.sessions = sessions
        self.specs = list(specs)
        self.seq_len = seq_len
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.specs[index]
        session = self.sessions[spec.session_idx]
        x = np.zeros((self.seq_len, session.features.shape[1]), dtype=np.float32)
        macro = np.full((self.seq_len,), IGNORE_INDEX, dtype=np.int64)
        micro = np.full((self.seq_len,), IGNORE_INDEX, dtype=np.int64)
        mask = np.zeros((self.seq_len,), dtype=np.bool_)

        length = max(0, min(spec.valid_len, self.seq_len, len(session.features) - spec.start))
        if length:
            x[:length] = session.features[spec.start : spec.start + length]
            macro[:length] = session.macro_labels[spec.start : spec.start + length]
            micro[:length] = session.micro_labels[spec.start : spec.start + length]
            mask[:length] = macro[:length] != IGNORE_INDEX
        x = (x - self.mean) / self.std
        return {
            "x": torch.from_numpy(x),
            "macro": torch.from_numpy(macro),
            "micro": torch.from_numpy(micro),
            "mask": torch.from_numpy(mask),
            "spec_index": torch.tensor(index, dtype=torch.long),
        }


def clean_label_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def whole_session_files(data_dirs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for data_dir in data_dirs:
        files.extend(sorted(data_dir.rglob("*whole_session*.csv")))
    return sorted(set(files))


def infer_time_seconds(sensor_ts: pd.Series) -> np.ndarray:
    values = pd.to_numeric(sensor_ts, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.arange(len(values), dtype=np.float64) / 100.0
    if not valid.all():
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    median_delta = float(np.median(diffs)) if len(diffs) else 1.0
    if median_delta > 1000.0:
        scale = 1_000_000.0
    elif median_delta > 10.0:
        scale = 1000.0
    else:
        scale = 1.0
    out = (values - values[0]) / scale
    out = np.maximum.accumulate(out)
    return out


def resample_session(df: pd.DataFrame, sample_rate_hz: int, imu_columns: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray]:
    time_seconds = infer_time_seconds(df["sensor_ts"])
    duration = float(time_seconds[-1]) if len(time_seconds) else 0.0
    if duration <= 0.0 or len(df) < 2:
        out = df.copy().reset_index(drop=True)
        return out, np.arange(len(out), dtype=np.int64)

    target_t = np.arange(0.0, duration + 1e-9, 1.0 / float(sample_rate_hz), dtype=np.float64)
    if len(target_t) < 2:
        out = df.copy().reset_index(drop=True)
        return out, np.arange(len(out), dtype=np.int64)

    resampled: dict[str, np.ndarray] = {}
    for col in imu_columns:
        resampled[col] = np.interp(target_t, time_seconds, df[col].to_numpy(dtype=np.float64)).astype(np.float32)

    right = np.searchsorted(time_seconds, target_t, side="left")
    right = np.clip(right, 0, len(time_seconds) - 1)
    left = np.clip(right - 1, 0, len(time_seconds) - 1)
    choose_left = np.abs(target_t - time_seconds[left]) <= np.abs(time_seconds[right] - target_t)
    nearest = np.where(choose_left, left, right).astype(np.int64)

    for col in ("action_type", "phase", "rep", "set", "subject_id"):
        resampled[col] = df[col].iloc[nearest].to_numpy()
    return pd.DataFrame(resampled), nearest


def build_label_maps(files: Sequence[Path]) -> tuple[list[str], list[str]]:
    actions: set[str] = set()
    for path in files:
        df = pd.read_csv(path, usecols=lambda col: col in {"action_type", "phase"})
        action_values = clean_label_series(df["action_type"]).str.lower()
        phase_values = clean_label_series(df["phase"]).str.lower()
        active = phase_values.isin(ACTIVE_PHASES)
        for action in action_values[active].unique().tolist():
            if action not in REST_ACTIONS:
                actions.add(str(action))
    action_names = sorted(actions)
    micro_names = ["other"]
    for action in action_names:
        for phase in ("concentric", "eccentric"):
            micro_names.append(f"{action}_{phase}")
    return action_names, micro_names


def encode_labels(
    df: pd.DataFrame,
    domain: str,
    action_names: Sequence[str],
    micro_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_to_idx = {label: idx for idx, label in enumerate(action_names)}
    macro_offset = 0 if domain == "exercise-only" else 1
    micro_to_idx = {label: idx for idx, label in enumerate(micro_names)}

    actions = clean_label_series(df["action_type"]).str.lower().to_numpy(dtype=object)
    phases = clean_label_series(df["phase"]).str.lower().to_numpy(dtype=object)
    reps = clean_label_series(df["rep"]).to_numpy(dtype=object)
    sets = clean_label_series(df["set"]).to_numpy(dtype=object)
    active = np.asarray([(phase in ACTIVE_PHASES and action in action_to_idx) for action, phase in zip(actions, phases)], dtype=bool)

    macro = np.full(len(df), IGNORE_INDEX if domain == "exercise-only" else 0, dtype=np.int64)
    micro = np.full(len(df), IGNORE_INDEX if domain == "exercise-only" else 0, dtype=np.int64)
    for idx, is_active in enumerate(active):
        if not is_active:
            continue
        action = str(actions[idx])
        phase = str(phases[idx])
        macro[idx] = action_to_idx[action] + macro_offset
        micro[idx] = micro_to_idx[f"{action}_{phase}"]
    return macro, micro, active, actions, phases, reps, sets


def read_sessions(
    data_dirs: Sequence[Path],
    domain: str,
    sample_rate_hz: int,
    max_files: int | None,
) -> tuple[list[SessionData], list[str], list[str], list[str]]:
    files = whole_session_files(data_dirs)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No whole-session CSV files found under {data_dirs}")

    action_names, micro_names = build_label_maps(files)
    macro_names = list(action_names) if domain == "exercise-only" else ["other", *action_names]
    sessions: list[SessionData] = []
    required = set(IMU_9AXIS) | {"sensor_ts", "action_type", "phase", "rep", "set", "subject_id"}

    for path in files:
        df = pd.read_csv(path, usecols=lambda col: col in required)
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] skip {path} missing columns: {sorted(missing)}")
            continue
        df = df.dropna(subset=list(IMU_9AXIS) + ["sensor_ts"]).reset_index(drop=True)
        if df.empty:
            continue
        df, _nearest = resample_session(df, sample_rate_hz=sample_rate_hz, imu_columns=IMU_9AXIS)
        macro, micro, active, actions, phases, reps, sets = encode_labels(df, domain, action_names, micro_names)
        if domain == "exercise-only" and not active.any():
            continue
        subject = str(clean_label_series(df["subject_id"]).iloc[0])
        sessions.append(
            SessionData(
                path=path,
                subject=subject,
                features=df.loc[:, IMU_9AXIS].to_numpy(dtype=np.float32),
                macro_labels=macro,
                micro_labels=micro,
                active_mask=active,
                actions=actions,
                phases=phases,
                reps=reps,
                sets=sets,
            )
        )
    if not sessions:
        raise RuntimeError("No usable sessions after resampling and label encoding.")
    return sessions, list(action_names), macro_names, micro_names


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def make_windows(
    sessions: Sequence[SessionData],
    domain: str,
    seq_len: int,
    stride: int,
    other_window_ratio: float,
    seed: int,
    max_windows: int | None,
) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    rng = random.Random(seed)
    if domain == "exercise-only":
        for session_idx, session in enumerate(sessions):
            for start, end in contiguous_true_runs(session.active_mask):
                block_len = end - start
                if block_len <= 0:
                    continue
                if block_len <= seq_len:
                    specs.append(WindowSpec(session_idx, start, end, block_len, session.subject, True))
                    continue
                block_starts = list(range(start, end - seq_len + 1, stride))
                if block_starts[-1] != end - seq_len:
                    block_starts.append(end - seq_len)
                for win_start in block_starts:
                    specs.append(WindowSpec(session_idx, win_start, win_start + seq_len, seq_len, session.subject, True))
    else:
        active_specs: list[WindowSpec] = []
        other_specs: list[WindowSpec] = []
        for session_idx, session in enumerate(sessions):
            length = len(session.features)
            starts = list(range(0, max(1, length), stride))
            if starts and starts[-1] + seq_len < length:
                starts.append(max(0, length - seq_len))
            seen: set[int] = set()
            for win_start in starts:
                win_start = min(win_start, max(0, length - 1))
                if win_start in seen:
                    continue
                seen.add(win_start)
                win_end = min(length, win_start + seq_len)
                valid_len = win_end - win_start
                has_active = bool(session.active_mask[win_start:win_end].any())
                spec = WindowSpec(session_idx, win_start, win_end, valid_len, session.subject, has_active)
                if has_active:
                    active_specs.append(spec)
                else:
                    other_specs.append(spec)
        max_other = int(round(len(active_specs) * max(0.0, other_window_ratio)))
        if len(other_specs) > max_other:
            other_specs = rng.sample(other_specs, max_other)
        specs = active_specs + other_specs
    specs.sort(key=lambda item: (item.subject, item.session_idx, item.start))
    if max_windows is not None and len(specs) > max_windows:
        specs = rng.sample(specs, max_windows)
        specs.sort(key=lambda item: (item.subject, item.session_idx, item.start))
    return specs


def compute_feature_stats(sessions: Sequence[SessionData], train_subjects: set[str], domain: str) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    for session in sessions:
        if session.subject not in train_subjects:
            continue
        if domain == "exercise-only":
            values = session.features[session.active_mask]
        else:
            values = session.features
        if len(values):
            chunks.append(values.astype(np.float32))
    if not chunks:
        raise RuntimeError("No training samples available for feature normalization.")
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def count_labels(sessions: Sequence[SessionData], specs: Sequence[WindowSpec], label_kind: str, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for spec in specs:
        labels = sessions[spec.session_idx].macro_labels if label_kind == "macro" else sessions[spec.session_idx].micro_labels
        values = labels[spec.start : spec.start + spec.valid_len]
        values = values[values != IGNORE_INDEX]
        if len(values):
            counts += np.bincount(values.astype(np.int64), minlength=num_classes)
    return counts


def class_weights(counts: np.ndarray) -> torch.Tensor:
    weights = np.ones_like(counts, dtype=np.float32)
    positive = counts > 0
    if positive.any():
        median = float(np.median(counts[positive]))
        weights[positive] = np.sqrt(median / np.maximum(counts[positive], 1.0)).astype(np.float32)
        weights = np.clip(weights, 0.25, 4.0)
        weights = weights / max(float(weights[positive].mean()), 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def temporal_smoothing_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] < 2:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=1)
    diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1)
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    loss = torch.clamp(diff.pow(2), max=16.0)
    return loss.masked_select(valid).mean()


def model_loss(
    micro_logits: torch.Tensor | None,
    macro_outputs: Sequence[torch.Tensor],
    micro_target: torch.Tensor,
    macro_target: torch.Tensor,
    mask: torch.Tensor,
    micro_weight: torch.Tensor | None,
    macro_weight: torch.Tensor | None,
    smooth_weight: float,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=macro_target.device)
    if micro_logits is not None:
        total = total + F.cross_entropy(micro_logits, micro_target, weight=micro_weight, ignore_index=IGNORE_INDEX)
        total = total + smooth_weight * temporal_smoothing_loss(micro_logits, mask)
    for logits in macro_outputs:
        total = total + F.cross_entropy(logits, macro_target, weight=macro_weight, ignore_index=IGNORE_INDEX)
        total = total + smooth_weight * temporal_smoothing_loss(logits, mask)
    return total


def train_one_fold(
    model_kind: str,
    sessions: Sequence[SessionData],
    train_specs: Sequence[WindowSpec],
    val_specs: Sequence[WindowSpec],
    mean: np.ndarray,
    std: np.ndarray,
    cfg: DSMSTCNConfig,
    args: argparse.Namespace,
    fold_idx: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    if model_kind == "ds_ms_tcn":
        model: torch.nn.Module = DSMSTCN(cfg)
    elif model_kind == "ms_tcn":
        model = MSTCN(cfg)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    model.to(device)

    train_ds = SequenceWindowDataset(sessions, train_specs, args.seq_len_samples, mean, std)
    val_ds = SequenceWindowDataset(sessions, val_specs, args.seq_len_samples, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    macro_counts = count_labels(sessions, train_specs, "macro", cfg.macro_classes)
    micro_counts = count_labels(sessions, train_specs, "micro", cfg.micro_classes)
    macro_weight = class_weights(macro_counts).to(device)
    micro_weight = class_weights(micro_counts).to(device) if model_kind == "ds_ms_tcn" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            macro = batch["macro"].to(device)
            micro = batch["micro"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            micro_logits, macro_outputs = model(x)
            loss = model_loss(
                micro_logits,
                macro_outputs,
                micro,
                macro,
                mask,
                micro_weight,
                macro_weight,
                args.smooth_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * x.size(0)
            train_count += x.size(0)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                macro = batch["macro"].to(device)
                micro = batch["micro"].to(device)
                mask = batch["mask"].to(device)
                micro_logits, macro_outputs = model(x)
                loss = model_loss(
                    micro_logits,
                    macro_outputs,
                    micro,
                    macro,
                    mask,
                    micro_weight,
                    macro_weight,
                    args.smooth_loss_weight,
                )
                val_loss += float(loss.detach().cpu()) * x.size(0)
                val_count += x.size(0)
        row = {
            "fold": fold_idx,
            "epoch": epoch,
            "train_loss": round(train_loss / max(train_count, 1), 6),
            "val_loss": round(val_loss / max(val_count, 1), 6),
        }
        history.append(row)
        print(f"[{model_kind}] fold={fold_idx} epoch={epoch} train_loss={row['train_loss']} val_loss={row['val_loss']}")
        if row["val_loss"] < best_val_loss:
            best_val_loss = float(row["val_loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"fold_{fold_idx:02d}_training_history.csv", history)
    torch.save(best_state or model.state_dict(), output_dir / f"fold_{fold_idx:02d}_best.pt")
    return model, {
        "fold": fold_idx,
        "best_val_loss": round(best_val_loss, 6),
        "train_windows": len(train_specs),
        "val_windows": len(val_specs),
    }


def aggregate_predictions(
    model: torch.nn.Module,
    sessions: Sequence[SessionData],
    val_specs: Sequence[WindowSpec],
    mean: np.ndarray,
    std: np.ndarray,
    cfg: DSMSTCNConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> PredictionBundle:
    dataset = SequenceWindowDataset(sessions, val_specs, args.seq_len_samples, mean, std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    session_ids = sorted({spec.session_idx for spec in val_specs})
    macro_sum = {idx: np.zeros((len(sessions[idx].features), cfg.macro_classes), dtype=np.float32) for idx in session_ids}
    macro_counts = {idx: np.zeros(len(sessions[idx].features), dtype=np.float32) for idx in session_ids}
    micro_sum = {idx: np.zeros((len(sessions[idx].features), cfg.micro_classes), dtype=np.float32) for idx in session_ids}
    micro_counts = {idx: np.zeros(len(sessions[idx].features), dtype=np.float32) for idx in session_ids}
    stage_sum = {
        idx: [np.zeros((len(sessions[idx].features), cfg.macro_classes), dtype=np.float32) for _ in range(cfg.macro_stages)]
        for idx in session_ids
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            spec_indices = batch["spec_index"].cpu().numpy().astype(int)
            micro_logits, macro_outputs = model(x)
            macro_np = F.softmax(macro_outputs[-1], dim=1).detach().cpu().numpy()
            micro_np = F.softmax(micro_logits, dim=1).detach().cpu().numpy() if micro_logits is not None else None
            stage_np = [F.softmax(logits, dim=1).detach().cpu().numpy() for logits in macro_outputs]
            for batch_idx, spec_index in enumerate(spec_indices):
                spec = val_specs[int(spec_index)]
                length = min(spec.valid_len, args.seq_len_samples)
                if length <= 0:
                    continue
                start = spec.start
                end = min(start + length, len(sessions[spec.session_idx].features))
                length = end - start
                if length <= 0:
                    continue
                macro_sum[spec.session_idx][start:end] += macro_np[batch_idx, :, :length].T
                macro_counts[spec.session_idx][start:end] += 1.0
                for stage_idx, stage_logits in enumerate(stage_np):
                    stage_sum[spec.session_idx][stage_idx][start:end] += stage_logits[batch_idx, :, :length].T
                if micro_np is not None:
                    micro_sum[spec.session_idx][start:end] += micro_np[batch_idx, :, :length].T
                    micro_counts[spec.session_idx][start:end] += 1.0

    macro_pred: dict[int, np.ndarray] = {}
    micro_pred: dict[int, np.ndarray] = {}
    macro_stage_pred: dict[int, list[np.ndarray]] = {}
    valid_mask: dict[int, np.ndarray] = {}
    for session_idx in session_ids:
        macro_valid = macro_counts[session_idx] > 0
        macro_pred[session_idx] = np.full(len(sessions[session_idx].features), IGNORE_INDEX, dtype=np.int64)
        macro_pred[session_idx][macro_valid] = np.argmax(macro_sum[session_idx][macro_valid], axis=1)
        valid_mask[session_idx] = macro_valid
        macro_stage_pred[session_idx] = []
        for stage_idx in range(cfg.macro_stages):
            stage_pred = np.full(len(sessions[session_idx].features), IGNORE_INDEX, dtype=np.int64)
            stage_pred[macro_valid] = np.argmax(stage_sum[session_idx][stage_idx][macro_valid], axis=1)
            macro_stage_pred[session_idx].append(stage_pred)
        micro_valid = micro_counts[session_idx] > 0
        micro_pred[session_idx] = np.full(len(sessions[session_idx].features), IGNORE_INDEX, dtype=np.int64)
        if micro_valid.any():
            micro_pred[session_idx][micro_valid] = np.argmax(micro_sum[session_idx][micro_valid], axis=1)
    return PredictionBundle(macro_pred, micro_pred, macro_stage_pred, valid_mask)


def true_rep_segments(session: SessionData, min_samples: int) -> list[RepSegment]:
    segments: list[RepSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str, str] | None = None
    for idx, active in enumerate(session.active_mask.tolist()):
        key = (session.subject, str(session.actions[idx]), str(session.sets[idx]), str(session.reps[idx]))
        if active and start is None:
            start = idx
            last_key = key
        elif active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                segments.append(RepSegment(session.path, last_key[0], last_key[1], last_key[2], last_key[3], start, idx, "label"))
            start = idx
            last_key = key
        elif (not active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                segments.append(RepSegment(session.path, last_key[0], last_key[1], last_key[2], last_key[3], start, idx, "label"))
            start = None
            last_key = None
    if start is not None and last_key is not None and len(session.active_mask) - start >= min_samples:
        segments.append(RepSegment(session.path, last_key[0], last_key[1], last_key[2], last_key[3], start, len(session.active_mask), "label"))
    return segments


def true_phase_segments(session: SessionData, min_samples: int) -> list[PhaseSegment]:
    segments: list[PhaseSegment] = []
    start: int | None = None
    last_key: tuple[str, str, str, str, str] | None = None
    for idx, active in enumerate(session.active_mask.tolist()):
        key = (
            session.subject,
            str(session.actions[idx]),
            str(session.sets[idx]),
            str(session.reps[idx]),
            str(session.phases[idx]),
        )
        if active and start is None:
            start = idx
            last_key = key
        elif active and key != last_key:
            if start is not None and last_key is not None and idx - start >= min_samples:
                segments.append(PhaseSegment(session.path, last_key[0], last_key[1], last_key[2], last_key[3], last_key[4], start, idx, "label"))
            start = idx
            last_key = key
        elif (not active) and start is not None:
            if last_key is not None and idx - start >= min_samples:
                segments.append(PhaseSegment(session.path, last_key[0], last_key[1], last_key[2], last_key[3], last_key[4], start, idx, "label"))
            start = None
            last_key = None
    if start is not None and last_key is not None and len(session.active_mask) - start >= min_samples:
        segments.append(PhaseSegment(session.path, last_key[0], last_key[1], last_key[2], last_key[3], last_key[4], start, len(session.active_mask), "label"))
    return segments


def parse_micro_label(label: str) -> tuple[str, str] | None:
    if label == "other":
        return None
    for suffix in ("_concentric", "_eccentric"):
        if label.endswith(suffix):
            return label[: -len(suffix)], suffix[1:]
    return None


def remove_short_runs(labels: np.ndarray, min_len: int, ignore_value: int = IGNORE_INDEX) -> np.ndarray:
    if min_len <= 1 or len(labels) == 0:
        return labels.copy()
    out = labels.copy()
    runs: list[tuple[int, int, int]] = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            runs.append((start, idx, int(labels[start])))
            start = idx
    for run_idx, (start, end, value) in enumerate(runs):
        if value == ignore_value or end - start >= min_len:
            continue
        left = runs[run_idx - 1][2] if run_idx > 0 else ignore_value
        right = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else ignore_value
        if left == right and left != ignore_value:
            out[start:end] = left
        elif left != ignore_value and (run_idx == len(runs) - 1 or runs[run_idx - 1][1] - runs[run_idx - 1][0] >= runs[run_idx + 1][1] - runs[run_idx + 1][0]):
            out[start:end] = left
        elif right != ignore_value:
            out[start:end] = right
    return out


def predicted_phase_segments_from_micro(
    session: SessionData,
    micro_pred: np.ndarray,
    micro_names: Sequence[str],
    min_samples: int,
    source: str,
) -> list[PhaseSegment]:
    labels = remove_short_runs(micro_pred, min_len=min_samples)
    segments: list[PhaseSegment] = []
    start: int | None = None
    current: tuple[str, str] | None = None
    rep_idx = 0
    set_id = "pred"
    for idx, label_idx in enumerate(labels.tolist()):
        parsed = parse_micro_label(micro_names[label_idx]) if 0 <= label_idx < len(micro_names) else None
        if parsed is not None and start is None:
            start = idx
            current = parsed
        elif parsed != current and start is not None:
            exercise, phase = current if current is not None else ("other", "other")
            if idx - start >= min_samples:
                segments.append(PhaseSegment(session.path, session.subject, exercise, set_id, str(rep_idx), phase, start, idx, source))
                rep_idx += 1
            start = idx if parsed is not None else None
            current = parsed
        elif parsed is not None and start is None:
            start = idx
            current = parsed
    if start is not None and current is not None and len(labels) - start >= min_samples:
        exercise, phase = current
        segments.append(PhaseSegment(session.path, session.subject, exercise, set_id, str(rep_idx), phase, start, len(labels), source))
    return segments


def phase_order_map(phase_truth: Sequence[PhaseSegment]) -> dict[str, tuple[str, str]]:
    votes: dict[str, dict[tuple[str, str], int]] = {}
    grouped: dict[tuple[Path, str, str, str, str], list[PhaseSegment]] = {}
    for segment in phase_truth:
        grouped.setdefault((segment.file_path, segment.subject, segment.exercise, segment.set_id, segment.rep_id), []).append(segment)
    for (_, _, exercise, _, _), segments in grouped.items():
        order: list[str] = []
        for segment in sorted(segments, key=lambda item: item.start):
            if segment.phase not in order:
                order.append(segment.phase)
        if len(order) >= 2:
            pair = (order[0], order[1])
            votes.setdefault(exercise, {})
            votes[exercise][pair] = votes[exercise].get(pair, 0) + 1
    out: dict[str, tuple[str, str]] = {}
    for exercise, counts in votes.items():
        out[exercise] = max(counts.items(), key=lambda item: item[1])[0]
    return out


def predicted_reps_from_phases(
    phase_segments: Sequence[PhaseSegment],
    orders: dict[str, tuple[str, str]],
    min_samples: int,
    source: str,
) -> list[RepSegment]:
    by_file_exercise: dict[tuple[Path, str], list[PhaseSegment]] = {}
    for segment in phase_segments:
        by_file_exercise.setdefault((segment.file_path, segment.exercise), []).append(segment)
    reps: list[RepSegment] = []
    for (file_path, exercise), segments in by_file_exercise.items():
        first_phase, second_phase = orders.get(exercise, ("eccentric", "concentric"))
        open_start: int | None = None
        subject = segments[0].subject if segments else ""
        rep_idx = 0
        for segment in sorted(segments, key=lambda item: item.start):
            if open_start is None:
                open_start = segment.start
            if segment.phase == first_phase:
                open_start = segment.start
            if segment.phase == second_phase and open_start is not None:
                end = segment.end
                if end - open_start >= min_samples:
                    reps.append(RepSegment(file_path, subject, exercise, "pred", str(rep_idx), open_start, end, source))
                    rep_idx += 1
                open_start = None
    return reps


def macro_segments_from_labels(
    session: SessionData,
    labels: np.ndarray,
    label_names: Sequence[str],
    include_other: bool,
    min_samples: int,
    source: str,
) -> list[RepSegment]:
    segments: list[RepSegment] = []
    start: int | None = None
    current_label: str | None = None
    seg_idx = 0
    for idx, label_idx in enumerate(labels.tolist()):
        label = label_names[label_idx] if 0 <= label_idx < len(label_names) else None
        if label == "other" and not include_other:
            label = None
        if label is not None and start is None:
            start = idx
            current_label = label
        elif label != current_label and start is not None:
            if current_label is not None and idx - start >= min_samples:
                segments.append(RepSegment(session.path, session.subject, current_label, "macro", str(seg_idx), start, idx, source))
                seg_idx += 1
            start = idx if label is not None else None
            current_label = label
    if start is not None and current_label is not None and len(labels) - start >= min_samples:
        segments.append(RepSegment(session.path, session.subject, current_label, "macro", str(seg_idx), start, len(labels), source))
    return segments


def class_aware_iou(a: RepSegment, b: RepSegment) -> float:
    if a.exercise != b.exercise:
        return 0.0
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / float(union) if union > 0 else 0.0


def class_aware_segment_metrics(
    predicted: Sequence[RepSegment],
    truth: Sequence[RepSegment],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pred_by_file: dict[Path, list[RepSegment]] = {}
    truth_by_file: dict[Path, list[RepSegment]] = {}
    for segment in predicted:
        pred_by_file.setdefault(segment.file_path, []).append(segment)
    for segment in truth:
        truth_by_file.setdefault(segment.file_path, []).append(segment)
    for threshold in thresholds:
        pairs: list[tuple[float, int, int]] = []
        pred_offset = 0
        truth_offset = 0
        for file_path in sorted(set(pred_by_file) | set(truth_by_file)):
            file_pred = pred_by_file.get(file_path, [])
            file_truth = truth_by_file.get(file_path, [])
            for pred_idx, pred_segment in enumerate(file_pred):
                for true_idx, true_segment in enumerate(file_truth):
                    iou = class_aware_iou(pred_segment, true_segment)
                    if iou >= threshold:
                        pairs.append((iou, pred_offset + pred_idx, truth_offset + true_idx))
            pred_offset += len(file_pred)
            truth_offset += len(file_truth)
        pairs.sort(reverse=True)
        matched_pred: set[int] = set()
        matched_truth: set[int] = set()
        matched_ious: list[float] = []
        for iou, pred_idx, true_idx in pairs:
            if pred_idx in matched_pred or true_idx in matched_truth:
                continue
            matched_pred.add(pred_idx)
            matched_truth.add(true_idx)
            matched_ious.append(iou)
        tp = len(matched_ious)
        fp = len(predicted) - tp
        fn = len(truth) - tp
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "iou_threshold": threshold,
                "true_segments": len(truth),
                "predicted_segments": len(predicted),
                "matched_segments": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
            }
        )
    return rows


def sample_metrics(
    sessions: Sequence[SessionData],
    bundle: PredictionBundle,
    label_names: Sequence[str],
    label_kind: str,
    output_dir: Path,
) -> dict[str, object]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for session_idx, pred in (bundle.macro_pred if label_kind == "macro" else bundle.micro_pred).items():
        session = sessions[session_idx]
        truth = session.macro_labels if label_kind == "macro" else session.micro_labels
        valid = (truth != IGNORE_INDEX) & (pred != IGNORE_INDEX) & bundle.valid_mask[session_idx]
        y_true.extend(truth[valid].astype(int).tolist())
        y_pred.extend(pred[valid].astype(int).tolist())
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "samples": 0}
    labels = list(range(len(label_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(label_names), output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    write_csv(
        output_dir / f"{label_kind}_confusion_matrix.csv",
        [
            {"true_label": label_names[i], "pred_label": label_names[j], "count": int(cm[i, j])}
            for i in labels
            for j in labels
        ],
    )
    (output_dir / f"{label_kind}_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(8, len(label_names) * 0.8), max(7, len(label_names) * 0.65)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True, xticks_rotation=45)
    ax.set_title(f"{label_kind.title()} Sample-wise Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label_kind}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(label_names) * 0.8), max(7, len(label_names) * 0.65)))
    display = ConfusionMatrixDisplay(confusion_matrix=norm, display_labels=label_names)
    display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True, xticks_rotation=45)
    ax.set_title(f"{label_kind.title()} Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label_kind}_confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)
    return {
        "accuracy": round(float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "samples": len(y_true),
    }


def plot_method_metrics(rows: Sequence[dict[str, object]], output_dir: Path, filename: str = "ds_ms_tcn_method_comparison.png") -> None:
    if not rows:
        return
    methods = [str(row["method"]) for row in rows]
    metrics = ["macro_sample_f1", "macro_segment_f1_iou_0.50", "rep_f1_iou_0.50", "rep_f1_iou_0.90"]
    labels = ["Macro sample F1", "Macro segment F1@0.50", "Rep F1@0.50", "Rep F1@0.90"]
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.7), 5.5))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
        ax.bar(x + (idx - 1.5) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("DS-MS-TCN 9-axis Method Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_timeline_examples(
    sessions: Sequence[SessionData],
    bundle: PredictionBundle,
    macro_names: Sequence[str],
    micro_names: Sequence[str],
    output_dir: Path,
    max_examples: int,
    max_samples: int,
) -> None:
    if max_examples <= 0:
        return
    out_dir = output_dir / "timeline_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for session_idx in sorted(bundle.macro_pred):
        session = sessions[session_idx]
        active_runs = contiguous_true_runs(session.active_mask & bundle.valid_mask[session_idx])
        if not active_runs:
            continue
        start, end = active_runs[0]
        pad = min(250, start)
        start = max(0, start - pad)
        end = min(len(session.features), max(end + pad, start + max_samples))
        end = min(end, start + max_samples)
        x = np.arange(start, end)
        rows = [session.macro_labels[start:end], bundle.macro_stage_pred[session_idx][0][start:end], bundle.macro_pred[session_idx][start:end]]
        row_names = ["macro truth", "stage 1", "final macro"]
        if (bundle.micro_pred[session_idx][start:end] != IGNORE_INDEX).any():
            rows.extend([session.micro_labels[start:end], bundle.micro_pred[session_idx][start:end]])
            row_names.extend(["micro truth", "micro pred"])
        fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.7 * len(rows)), sharex=True)
        if len(rows) == 1:
            axes = [axes]
        for ax, values, title in zip(axes, rows, row_names):
            ax.step(x, values, where="post", linewidth=1.2)
            ax.set_ylabel(title)
            ax.grid(axis="x", alpha=0.15)
        axes[-1].set_xlabel("Resampled sample index")
        fig.suptitle(f"{session.subject} {session.path.name} stage timeline")
        fig.tight_layout()
        fig.savefig(out_dir / f"{made + 1:03d}_{session.subject}_{session.path.stem}.png", dpi=180)
        plt.close(fig)
        made += 1
        if made >= max_examples:
            return


def plot_waveform_examples(
    sessions: Sequence[SessionData],
    truth_reps: Sequence[RepSegment],
    pred_reps: Sequence[RepSegment],
    output_dir: Path,
    max_examples: int,
    max_samples: int,
) -> None:
    if max_examples <= 0:
        return
    out_dir = output_dir / "waveform_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_by_file: dict[Path, list[RepSegment]] = {}
    pred_by_file: dict[Path, list[RepSegment]] = {}
    for segment in truth_reps:
        truth_by_file.setdefault(segment.file_path, []).append(segment)
    for segment in pred_reps:
        pred_by_file.setdefault(segment.file_path, []).append(segment)
    made = 0
    for session in sessions:
        file_truth = sorted(truth_by_file.get(session.path, []), key=lambda item: item.start)
        if not file_truth:
            continue
        first = file_truth[0]
        start = max(0, first.start - 250)
        end = min(len(session.features), start + max_samples)
        signal = np.linalg.norm(session.features[start:end, :3], axis=1)
        x = np.arange(start, end)
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, sharey=True)
        for ax, title, segments, color_start, color_end in (
            (axes[0], "Ground truth", file_truth, "#1b9e77", "#386cb0"),
            (axes[1], "Prediction", sorted(pred_by_file.get(session.path, []), key=lambda item: item.start), "#d95f02", "#e7298a"),
        ):
            ax.plot(x, signal, color="#4d4d4d", linewidth=1.0)
            for segment in segments:
                if segment.end < start or segment.start > end:
                    continue
                ax.axvline(segment.start, color=color_start, linestyle="-", linewidth=1.0, alpha=0.85)
                ax.axvline(segment.end, color=color_end, linestyle="--", linewidth=1.0, alpha=0.85)
            ax.set_ylabel(title)
            ax.grid(axis="x", alpha=0.12)
        axes[-1].set_xlabel("Resampled sample index")
        fig.suptitle(f"{session.subject} {session.path.name} rep boundary waveform")
        fig.tight_layout()
        fig.savefig(out_dir / f"{made + 1:03d}_{session.subject}_{session.path.stem}.png", dpi=180)
        plt.close(fig)
        made += 1
        if made >= max_examples:
            return


def write_segments(path: Path, segments: Sequence[RepSegment]) -> None:
    write_csv(
        path,
        [
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "exercise": segment.exercise,
                "set_id": segment.set_id,
                "rep_id": segment.rep_id,
                "start": segment.start,
                "end": segment.end,
                "samples": segment.n_samples,
                "source": segment.source,
            }
            for segment in segments
        ],
    )


def write_phase_segments(path: Path, segments: Sequence[PhaseSegment]) -> None:
    write_csv(
        path,
        [
            {
                "file": str(segment.file_path),
                "subject": segment.subject,
                "exercise": segment.exercise,
                "set_id": segment.set_id,
                "rep_id": segment.rep_id,
                "phase": segment.phase,
                "start": segment.start,
                "end": segment.end,
                "samples": segment.n_samples,
                "source": segment.source,
            }
            for segment in segments
        ],
    )


def evaluate_model_outputs(
    model_kind: str,
    domain: str,
    sessions: Sequence[SessionData],
    bundle: PredictionBundle,
    macro_names: Sequence[str],
    micro_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    true_reps = [segment for session in sessions for segment in true_rep_segments(session, args.min_rep_samples)]
    true_phases = [segment for session in sessions for segment in true_phase_segments(session, args.min_phase_samples)]
    orders = phase_order_map(true_phases)

    macro_truth_segments: list[RepSegment] = []
    macro_pred_segments: list[RepSegment] = []
    include_other = domain == "full-session"
    for session_idx, macro_pred in bundle.macro_pred.items():
        session = sessions[session_idx]
        macro_truth_segments.extend(
            macro_segments_from_labels(session, session.macro_labels, macro_names, include_other=False, min_samples=args.min_rep_samples, source="macro_truth")
        )
        macro_pred_segments.extend(
            macro_segments_from_labels(session, macro_pred, macro_names, include_other=False, min_samples=args.min_rep_samples, source=model_kind)
        )

    pred_phases: list[PhaseSegment] = []
    pred_reps: list[RepSegment] = []
    if model_kind == "ds_ms_tcn":
        for session_idx, micro_pred in bundle.micro_pred.items():
            session = sessions[session_idx]
            pred_phases.extend(
                predicted_phase_segments_from_micro(
                    session,
                    micro_pred,
                    micro_names,
                    min_samples=args.min_phase_samples,
                    source=model_kind,
                )
            )
        pred_reps = predicted_reps_from_phases(pred_phases, orders, min_samples=args.min_rep_samples, source=model_kind)
    else:
        pred_reps = list(macro_pred_segments)

    macro_sample = sample_metrics(sessions, bundle, macro_names, "macro", output_dir)
    micro_sample: dict[str, object] | None = None
    if model_kind == "ds_ms_tcn":
        micro_sample = sample_metrics(sessions, bundle, micro_names, "micro", output_dir)

    macro_segment_rows = class_aware_segment_metrics(macro_pred_segments, macro_truth_segments, args.iou_thresholds)
    write_csv(output_dir / "macro_segment_iou_metrics.csv", macro_segment_rows)

    rep_rows = segmentation_metric_rows(pred_reps, true_reps, args.iou_thresholds)
    rep_by_exercise_rows = segmentation_metric_rows_by_exercise(pred_reps, true_reps, args.iou_thresholds)
    rep_by_subject_rows = segmentation_metric_rows_by_subject(pred_reps, true_reps, args.iou_thresholds)
    write_csv(output_dir / "rep_segmentation_metrics.csv", rep_rows)
    write_csv(output_dir / "rep_segmentation_metrics_by_exercise.csv", rep_by_exercise_rows)
    write_csv(output_dir / "rep_segmentation_metrics_by_subject.csv", rep_by_subject_rows)
    write_segments(output_dir / "rep_segmentation_truth_segments.csv", true_reps)
    write_segments(output_dir / "rep_segmentation_pred_segments.csv", pred_reps)
    plot_segmentation_metrics(rep_rows, output_dir)
    plot_segmentation_metrics_by_exercise(rep_by_exercise_rows, output_dir)
    plot_segmentation_metrics_by_subject(rep_by_subject_rows, output_dir)
    plot_exercise_accuracy_table(rep_by_exercise_rows, output_dir, args.iou_thresholds)

    phase_rows: list[dict[str, object]] = []
    phase_by_phase_rows: list[dict[str, object]] = []
    if pred_phases:
        phase_rows = phase_metric_rows(pred_phases, true_phases, args.iou_thresholds)
        phase_by_phase_rows = phase_metric_rows_by_phase(pred_phases, true_phases, args.iou_thresholds)
        write_csv(output_dir / "phase_split_metrics.csv", phase_rows)
        write_csv(output_dir / "phase_split_metrics_by_phase.csv", phase_by_phase_rows)
        write_phase_segments(output_dir / "phase_split_truth_segments.csv", true_phases)
        write_phase_segments(output_dir / "phase_split_pred_segments.csv", pred_phases)
        plot_phase_metrics(phase_rows, output_dir)
        plot_phase_metrics_by_phase(phase_by_phase_rows, output_dir)

    plot_timeline_examples(sessions, bundle, macro_names, micro_names, output_dir, args.plot_examples, args.example_samples)
    plot_waveform_examples(sessions, true_reps, pred_reps, output_dir, args.plot_examples, args.example_samples)

    macro_segment_050 = next((row for row in macro_segment_rows if abs(float(row["iou_threshold"]) - 0.5) < 1e-9), {})
    rep_050 = next((row for row in rep_rows if abs(float(row["iou_threshold"]) - 0.5) < 1e-9), {})
    rep_075 = next((row for row in rep_rows if abs(float(row["iou_threshold"]) - 0.75) < 1e-9), {})
    rep_090 = next((row for row in rep_rows if abs(float(row["iou_threshold"]) - 0.9) < 1e-9), {})
    summary = {
        "method": model_kind,
        "domain": domain,
        "macro_sample": macro_sample,
        "micro_sample": micro_sample,
        "macro_segment_iou_metrics": macro_segment_rows,
        "rep_segmentation_metrics": rep_rows,
        "phase_split_metrics": phase_rows,
        "num_true_reps": len(true_reps),
        "num_predicted_reps": len(pred_reps),
        "macro_sample_f1": macro_sample.get("macro_f1", 0.0),
        "macro_segment_f1_iou_0.50": macro_segment_050.get("f1", 0.0),
        "rep_f1_iou_0.50": rep_050.get("f1", 0.0),
        "rep_f1_iou_0.75": rep_075.get("f1", 0.0),
        "rep_f1_iou_0.90": rep_090.get("f1", 0.0),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_dirs = [Path(path) for path in args.data_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions, action_names, macro_names, micro_names = read_sessions(
        data_dirs,
        domain=args.domain,
        sample_rate_hz=args.sample_rate_hz,
        max_files=args.max_files,
    )
    specs = make_windows(
        sessions,
        domain=args.domain,
        seq_len=args.seq_len_samples,
        stride=args.stride_samples,
        other_window_ratio=args.other_window_ratio,
        seed=args.seed,
        max_windows=args.max_windows,
    )
    if not specs:
        raise RuntimeError("No trainable windows produced.")
    subjects = np.asarray([spec.subject for spec in specs], dtype=object)
    unique_subjects = sorted(set(subjects.tolist()))
    n_splits = min(args.folds, len(unique_subjects))
    if n_splits < 2:
        raise RuntimeError("Need at least two subjects for GroupKFold.")
    splitter = GroupKFold(n_splits=n_splits)
    device = choose_device(args.device)
    print(f"[INFO] device={device} sessions={len(sessions)} windows={len(specs)} subjects={len(unique_subjects)}")

    metadata = {
        "domain": args.domain,
        "data_dirs": [str(path) for path in data_dirs],
        "sample_rate_hz": args.sample_rate_hz,
        "seq_len_samples": args.seq_len_samples,
        "stride_samples": args.stride_samples,
        "imu_columns": list(IMU_9AXIS),
        "action_names": action_names,
        "macro_names": macro_names,
        "micro_names": micro_names,
        "subjects": unique_subjects,
        "num_sessions": len(sessions),
        "num_windows": len(specs),
        "folds": n_splits,
        "epochs": args.epochs,
        "model_kinds": list(args.model_kinds),
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "macro_stages": args.macro_stages,
        "batch_size": args.batch_size,
        "max_files": args.max_files,
        "max_windows": args.max_windows,
        "note": "9-axis adapted DS-MS-TCN; the referenced paper used acc+gyro, so this is not a pure same-input comparison.",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    comparison_rows: list[dict[str, object]] = []
    for model_kind in args.model_kinds:
        model_dir = args.output_dir / model_kind
        model_dir.mkdir(parents=True, exist_ok=True)
        cfg = DSMSTCNConfig(
            input_channels=len(IMU_9AXIS),
            micro_classes=len(micro_names),
            macro_classes=len(macro_names),
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            macro_stages=args.macro_stages,
            dropout=args.dropout,
        )
        fold_rows: list[dict[str, object]] = []
        aggregate_macro: dict[int, np.ndarray] = {}
        aggregate_micro: dict[int, np.ndarray] = {}
        aggregate_stage: dict[int, list[np.ndarray]] = {}
        aggregate_valid: dict[int, np.ndarray] = {}

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(specs)), groups=subjects), start=1):
            train_specs = [specs[int(idx)] for idx in train_idx]
            val_specs = [specs[int(idx)] for idx in val_idx]
            train_subjects = sorted({spec.subject for spec in train_specs})
            val_subjects = sorted({spec.subject for spec in val_specs})
            overlap = sorted(set(train_subjects) & set(val_subjects))
            if overlap:
                raise RuntimeError(f"Subject leakage in fold {fold_idx}: {overlap}")
            mean, std = compute_feature_stats(sessions, set(train_subjects), args.domain)
            model, fold_summary = train_one_fold(
                model_kind,
                sessions,
                train_specs,
                val_specs,
                mean,
                std,
                cfg,
                args,
                fold_idx,
                model_dir,
                device,
            )
            bundle = aggregate_predictions(model, sessions, val_specs, mean, std, cfg, args, device)
            aggregate_macro.update(bundle.macro_pred)
            aggregate_micro.update(bundle.micro_pred)
            aggregate_stage.update(bundle.macro_stage_pred)
            aggregate_valid.update(bundle.valid_mask)
            fold_rows.append(
                {
                    **fold_summary,
                    "train_subjects": ",".join(train_subjects),
                    "val_subjects": ",".join(val_subjects),
                }
            )

        write_csv(model_dir / "fold_manifest.csv", fold_rows)
        bundle = PredictionBundle(aggregate_macro, aggregate_micro, aggregate_stage, aggregate_valid)
        summary = evaluate_model_outputs(model_kind, args.domain, sessions, bundle, macro_names, micro_names, model_dir, args)
        comparison_rows.append(
            {
                "domain": args.domain,
                "method": model_kind,
                "macro_sample_f1": summary.get("macro_sample_f1", 0.0),
                "macro_segment_f1_iou_0.50": summary.get("macro_segment_f1_iou_0.50", 0.0),
                "rep_f1_iou_0.50": summary.get("rep_f1_iou_0.50", 0.0),
                "rep_f1_iou_0.75": summary.get("rep_f1_iou_0.75", 0.0),
                "rep_f1_iou_0.90": summary.get("rep_f1_iou_0.90", 0.0),
                "num_true_reps": summary.get("num_true_reps", 0),
                "num_predicted_reps": summary.get("num_predicted_reps", 0),
            }
        )

    write_csv(args.output_dir / "ds_ms_tcn_method_comparison.csv", comparison_rows)
    plot_method_metrics(comparison_rows, args.output_dir)
    summary = {
        "metadata": metadata,
        "method_comparison": comparison_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 9-axis DS-MS-TCN / MS-TCN baselines for workout sequence segmentation.")
    parser.add_argument("--data-dirs", nargs="+", default=["datasets/workout"])
    parser.add_argument("--domain", choices=["exercise-only", "full-session"], default="exercise-only")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_rep_classification/012_ds_ms_tcn_9axis_exercise_only"))
    parser.add_argument("--model-kinds", nargs="+", choices=["ds_ms_tcn", "ms_tcn"], default=["ds_ms_tcn", "ms_tcn"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-rate-hz", type=int, default=50)
    parser.add_argument("--seq-len-seconds", type=float, default=40.0)
    parser.add_argument("--stride-fraction", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--macro-stages", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-loss-weight", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--other-window-ratio", type=float, default=1.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    parser.add_argument("--min-rep-samples", type=int, default=10)
    parser.add_argument("--min-phase-samples", type=int, default=5)
    parser.add_argument("--plot-examples", type=int, default=6)
    parser.add_argument("--example-samples", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    args.seq_len_samples = max(1, int(round(args.seq_len_seconds * args.sample_rate_hz)))
    args.stride_samples = max(1, int(round(args.seq_len_samples * args.stride_fraction)))
    return args


if __name__ == "__main__":
    run_training(parse_args())
