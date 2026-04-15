import argparse
import csv
import json
import math
import os
import socket
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
DEFAULT_ACTIONS = [
    "squat",
    "deadlift",
    "bench_press",
    "overhead_press",
    "barbell_row",
    "biceps_curl",
    "lateral_raise",
    "lunge",
]


@dataclass
class SubjectProfile:
    age: float = 25.0
    sex: str = "unknown"  # male / female / unknown
    exp_level: int = 0  # 0: no exp, 1: ~6 months, 2: >=1 year


def parse_experience(raw: str) -> int:
    text = (raw or "").strip().lower()
    if text in {"0", "none", "novice", "未有健身經驗", "無經驗"}:
        return 0
    if text in {"1", "6m", "6_month", "半年", "六個月"}:
        return 1
    if text in {"2", "1y", "1_year_plus", "一年上", "一年以上"}:
        return 2
    return 0


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def ts_to_sec(v, fallback: float) -> float:
    x = safe_float(v, fallback)
    # 粗略時間尺度判斷: ns/us/ms/s
    if x > 1e15:
        return x / 1e9
    if x > 1e12:
        return x / 1e6
    if x > 1e9:
        return x / 1e3
    if x > 1e6:
        return x / 1e3
    return x


class OnePoleLPF:
    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self.prev = None

    def update(self, x: float) -> float:
        if self.prev is None:
            self.prev = x
            return x
        y = self.alpha * x + (1.0 - self.alpha) * self.prev
        self.prev = y
        return y


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xc = x - np.mean(x)
    return float(np.sum(np.diff(np.signbit(xc)).astype(np.int32))) / (len(x) - 1)


def extract_window_features(window: np.ndarray, profile: SubjectProfile) -> np.ndarray:
    feats: List[float] = []

    for i in range(window.shape[1]):
        x = window[:, i]
        feats.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.max(x) - np.min(x)),
                rms(x),
                zero_crossing_rate(x),
            ]
        )

    acc_mag = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)
    mag_mag = np.linalg.norm(window[:, 6:9], axis=1)
    for sig in [acc_mag, gyro_mag, mag_mag]:
        feats.extend(
            [
                float(np.mean(sig)),
                float(np.std(sig)),
                float(np.max(sig) - np.min(sig)),
                rms(sig),
                zero_crossing_rate(sig),
            ]
        )

    sex_map = {"male": 1.0, "female": 0.0, "unknown": 0.5}
    feats.extend(
        [
            (profile.age - 30.0) / 15.0,
            sex_map.get(profile.sex.lower(), 0.5),
            profile.exp_level / 2.0,
        ]
    )
    return np.array(feats, dtype=np.float32).reshape(1, -1)


class ActionClassifier:
    """
    Wrapper:
    - 優先載入 sklearn model + encoder（現有專案可直接用）
    - 若無模型，退化成簡單規則分類（確保 pipeline 可跑）
    """

    def __init__(self, model_path: Optional[str], encoder_path: Optional[str]):
        self.model = None
        self.encoder = None
        self.valid = False
        if model_path and encoder_path and joblib is not None:
            try:
                self.model = joblib.load(model_path)
                self.encoder = joblib.load(encoder_path)
                self.valid = True
            except Exception:
                self.valid = False

    def predict(self, window: np.ndarray, profile: SubjectProfile) -> Tuple[str, float]:
        feats = extract_window_features(window, profile)
        if self.valid:
            pred_id = self.model.predict(feats)[0]
            label = self.encoder.inverse_transform([pred_id])[0]
            prob = 0.0
            if hasattr(self.model, "predict_proba"):
                try:
                    prob = float(np.max(self.model.predict_proba(feats)))
                except Exception:
                    prob = 0.0
            return str(label), prob

        # fallback heuristic
        gyro_std = float(np.std(np.linalg.norm(window[:, 3:6], axis=1)))
        acc_std = float(np.std(np.linalg.norm(window[:, 0:3], axis=1)))
        if gyro_std < 0.7 and acc_std < 0.07:
            return "rest", 0.5
        if gyro_std > 3.5:
            return "biceps_curl", 0.35
        if acc_std > 0.2:
            return "squat", 0.3
        return "barbell_row", 0.25


class RepSegmenter:
    """
    線上 rep segmentation：
    1) 用最近窗口找主導軸（gyro variance 最大）
    2) 自適應門檻 + 零交越
    3) 每 2 個半週期視為 1 rep，輸出向心/離心時間
    """

    def __init__(self, fs: float = 50.0):
        self.fs = float(fs)
        self.axis_hist: Deque[np.ndarray] = deque(maxlen=int(fs * 1.2))
        self.signal_lpf = OnePoleLPF(alpha=0.22)

        self.prev_sign = 0
        self.prev_ts = None
        self.last_cross_ts = None
        self.last_peak = 0.0
        self.curr_peak = 0.0
        self.crosses: List[Tuple[float, int, float]] = []  # (ts, sign_before_cross, peak)

        self.min_half_sec = 0.15
        self.deadband_scale = 0.8

    def _dominant_signal(self, sample: Dict[str, float]) -> float:
        gyro = np.array([sample["gx"], sample["gy"], sample["gz"]], dtype=np.float32)
        self.axis_hist.append(gyro)
        if len(self.axis_hist) < 8:
            return float(gyro[0])
        arr = np.stack(self.axis_hist, axis=0)
        axis = int(np.argmax(np.var(arr, axis=0)))
        return float(gyro[axis])

    def update(self, ts_sec: float, sample: Dict[str, float], positive_is_concentric=True):
        raw_sig = self._dominant_signal(sample)
        sig = self.signal_lpf.update(raw_sig)
        self.curr_peak = max(self.curr_peak, abs(sig))

        if len(self.axis_hist) < 8:
            self.prev_sign = 1 if sig > 0 else -1
            self.prev_ts = ts_sec
            return None

        hist = np.stack(self.axis_hist, axis=0)
        energy = np.linalg.norm(hist, axis=1)
        mad = np.median(np.abs(energy - np.median(energy))) + 1e-6
        th = max(0.6, self.deadband_scale * 1.4826 * mad)

        sign = 0
        if sig > th:
            sign = 1
        elif sig < -th:
            sign = -1

        if sign == 0:
            self.prev_ts = ts_sec
            return None

        if self.prev_sign == 0:
            self.prev_sign = sign
            self.prev_ts = ts_sec
            self.curr_peak = abs(sig)
            return None

        if sign != self.prev_sign:
            if self.last_cross_ts is None or (ts_sec - self.last_cross_ts) >= self.min_half_sec:
                self.crosses.append((ts_sec, self.prev_sign, max(self.last_peak, self.curr_peak)))
                self.last_cross_ts = ts_sec
                self.last_peak = self.curr_peak
                self.curr_peak = 0.0

            self.prev_sign = sign

            if len(self.crosses) >= 3:
                c0, c1, c2 = self.crosses[-3], self.crosses[-2], self.crosses[-1]
                if c0[1] == c2[1] and c0[1] != c1[1]:
                    d1 = c1[0] - c0[0]
                    d2 = c2[0] - c1[0]

                    if d1 > self.min_half_sec and d2 > self.min_half_sec:
                        if positive_is_concentric:
                            t_con = d1 if c0[1] > 0 else d2
                            t_ecc = d2 if c0[1] > 0 else d1
                        else:
                            t_con = d1 if c0[1] < 0 else d2
                            t_ecc = d2 if c0[1] < 0 else d1

                        amp_sym = 1.0 - min(1.0, abs(c0[2] - c1[2]) / (max(c0[2], c1[2]) + 1e-6))
                        tempo_sym = 1.0 - min(1.0, abs(t_con - t_ecc) / (t_con + t_ecc + 1e-6))
                        stability = float(np.clip(0.55 * tempo_sym + 0.45 * amp_sym, 0.0, 1.0))

                        # consume one full rep (avoid double count on overlap)
                        self.crosses = [c2]
                        return {
                            "t_concentric_sec": float(t_con),
                            "t_eccentric_sec": float(t_ecc),
                            "stability": stability,
                        }

        self.prev_ts = ts_sec
        return None


class ResistanceRealtimeSystem:
    def __init__(
        self,
        fs: float,
        window_sec: float,
        hop_sec: float,
        model_path: Optional[str],
        encoder_path: Optional[str],
        subject: SubjectProfile,
    ):
        self.fs = float(fs)
        self.window_size = int(fs * window_sec)
        self.hop_size = max(1, int(fs * hop_sec))
        self.subject = subject

        self.classifier = ActionClassifier(model_path, encoder_path)
        self.segmenter = RepSegmenter(fs=fs)

        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.sample_counter = 0
        self.pred_hist: Deque[str] = deque(maxlen=5)
        self.current_action = "rest"

        self.rep_counter = defaultdict(int)
        self.rep_tempo_history = defaultdict(lambda: deque(maxlen=12))

    def _smooth_label(self, label: str) -> str:
        self.pred_hist.append(label)
        return Counter(self.pred_hist).most_common(1)[0][0]

    def update(self, row: Dict[str, str]) -> Optional[Dict]:
        ts_raw = row.get("sensor_ts", row.get("timestamp_ms", row.get("timestamp", time.time())))
        ts = ts_to_sec(ts_raw, time.time())
        sample = {k: safe_float(row.get(k, 0.0)) for k in IMU_COLS}
        vec = np.array([sample[k] for k in IMU_COLS], dtype=np.float32)

        self.buffer.append(vec)
        self.sample_counter += 1

        classified_this_step = False
        if len(self.buffer) >= self.window_size and self.sample_counter % self.hop_size == 0:
            window = np.array(self.buffer, dtype=np.float32)
            raw_label, conf = self.classifier.predict(window, self.subject)
            self.current_action = self._smooth_label(raw_label)
            classified_this_step = True
        else:
            conf = 0.0

        # 無動作不做 rep segmentation / 計數
        if self.current_action in {"rest", "no_action"}:
            if classified_this_step:
                return {
                    "action_type": "no_action",
                    "rep": 0,
                    "t_concentric_sec": 0.0,
                    "t_eccentric_sec": 0.0,
                    "stability_score": 0.0,
                    "classifier_conf": round(conf, 3),
                }
            return None

        phase_out = self.segmenter.update(ts, sample, positive_is_concentric=True)
        if phase_out is None:
            return None

        self.rep_counter[self.current_action] += 1
        rep_idx = self.rep_counter[self.current_action]
        total_t = phase_out["t_concentric_sec"] + phase_out["t_eccentric_sec"]
        self.rep_tempo_history[self.current_action].append(total_t)

        hist = np.array(self.rep_tempo_history[self.current_action], dtype=np.float32)
        cv = float(np.std(hist) / (np.mean(hist) + 1e-6)) if len(hist) >= 3 else 0.0
        stability_final = float(np.clip(0.7 * phase_out["stability"] + 0.3 * (1.0 - min(cv, 1.0)), 0.0, 1.0))

        return {
            "action_type": self.current_action,
            "rep": rep_idx,
            "t_concentric_sec": round(phase_out["t_concentric_sec"], 3),
            "t_eccentric_sec": round(phase_out["t_eccentric_sec"], 3),
            "stability_score": round(100.0 * stability_final, 1),
            "classifier_conf": round(conf, 3),
        }


def parse_row_from_text(line: str) -> Dict[str, str]:
    # 支援「完整 CSV header + row」或「只有 row」
    parts = [x.strip() for x in line.strip().split(",")]
    if len(parts) < 13:
        raise ValueError("CSV columns too short")

    default_cols = [
        "pc_time",
        "serial_num",
        "sensor_ts",
        "host_ts",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "mx",
        "my",
        "mz",
        "action_type",
        "phase",
        "rep",
        "set",
        "rpe",
        "weight_kg",
        "subject_id",
        "ppg_a",
        "ppg_b",
        "ppg_c",
        "ppg_d",
        "ppg_e",
    ]
    if len(parts) >= len(default_cols):
        return {k: parts[i] for i, k in enumerate(default_cols)}
    # 最少至少有 IMU 欄位
    return {k: parts[i] for i, k in enumerate(default_cols[: len(parts)])}


def run_udp(args):
    subject = SubjectProfile(age=args.age, sex=args.sex, exp_level=args.exp_level)
    system = ResistanceRealtimeSystem(
        fs=args.fs,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        subject=subject,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_ip, args.udp_port))
    sock.settimeout(1.0)
    print(f"[INFO] UDP listening on {args.udp_ip}:{args.udp_port}")

    try:
        while True:
            try:
                data, _ = sock.recvfrom(8192)
            except socket.timeout:
                continue
            line = data.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                row = parse_row_from_text(line)
                out = system.update(row)
                if out:
                    print(json.dumps(out, ensure_ascii=False))
            except Exception as e:
                print(f"[WARN] parse/update fail: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def run_csv(args):
    subject = SubjectProfile(age=args.age, sex=args.sex, exp_level=args.exp_level)
    system = ResistanceRealtimeSystem(
        fs=args.fs,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        subject=subject,
    )

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = system.update(row)
            if out:
                print(json.dumps(out, ensure_ascii=False))


def build_argparser():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    default_model_path = os.path.join(project_root, "models", "action_rf_model.pkl")
    default_encoder_path = os.path.join(project_root, "models", "action_label_encoder.pkl")

    p = argparse.ArgumentParser(description="Realtime resistance-training classifier + rep/phase counter")
    p.add_argument("--mode", choices=["udp", "csv"], default="udp")
    p.add_argument("--udp-ip", default="0.0.0.0")
    p.add_argument("--udp-port", type=int, default=10000)
    p.add_argument("--csv-path", default="")

    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--window-sec", type=float, default=1.28)
    p.add_argument("--hop-sec", type=float, default=0.2)

    p.add_argument("--model-path", default=default_model_path)
    p.add_argument("--encoder-path", default=default_encoder_path)

    p.add_argument("--age", type=float, default=25.0)
    p.add_argument("--sex", choices=["male", "female", "unknown"], default="unknown")
    p.add_argument("--exp-level", type=int, choices=[0, 1, 2], default=0)
    return p


def main():
    args = build_argparser().parse_args()
    if args.mode == "udp":
        run_udp(args)
    else:
        if not args.csv_path:
            raise ValueError("--csv-path is required in csv mode")
        run_csv(args)


if __name__ == "__main__":
    main()
