import argparse
import csv
import json
import math
import socket
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
MODEL_FEATURE_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


@dataclass
class SubjectProfile:
    age: float = 25.0
    sex: str = "unknown"
    exp_level: int = 0


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def ts_to_sec(v, fallback: float) -> float:
    x = safe_float(v, fallback)
    # Heuristic by magnitude:
    # ns: ~1e18, us: ~1e15, ms: ~1e12
    if x > 1e17:
        return x / 1e9
    if x > 1e14:
        return x / 1e6
    if x > 1e11:
        return x / 1e3
    if x > 1e6:
        return x / 1e3
    return x


def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def rms(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def signal_energy(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(x * x for x in xs) / len(xs)


def zero_crossing_rate(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    prev = xs[0] - m
    crosses = 0
    for i in range(1, n):
        cur = xs[i] - m
        if (prev < 0 <= cur) or (prev > 0 >= cur):
            crosses += 1
        prev = cur
    return crosses / (n - 1)


def safe_corr(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    sa = std(a)
    sb = std(b)
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    ma = mean(a)
    mb = mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    return cov / (sa * sb)


def extract_model_features(window: List[Dict[str, float]]) -> List[float]:
    feats: List[float] = []

    cols = {k: [row[k] for row in window] for k in MODEL_FEATURE_COLS}
    for col in MODEL_FEATURE_COLS:
        x = cols[col]
        mn = min(x)
        mx = max(x)
        feats.extend(
            [
                mean(x),
                std(x),
                mn,
                mx,
                mx - mn,
                rms(x),
                signal_energy(x),
                zero_crossing_rate(x),
            ]
        )

    ax, ay, az = cols["ax"], cols["ay"], cols["az"]
    gx, gy, gz = cols["gx"], cols["gy"], cols["gz"]

    acc_mag = [math.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]) for i in range(len(window))]
    gyro_mag = [math.sqrt(gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i]) for i in range(len(window))]

    for mag in (acc_mag, gyro_mag):
        mn = min(mag)
        mx = max(mag)
        feats.extend(
            [
                mean(mag),
                std(mag),
                mn,
                mx,
                mx - mn,
                rms(mag),
                signal_energy(mag),
                zero_crossing_rate(mag),
            ]
        )

    feats.extend(
        [
            safe_corr(ax, ay),
            safe_corr(ax, az),
            safe_corr(ay, az),
            safe_corr(gx, gy),
            safe_corr(gx, gz),
            safe_corr(gy, gz),
        ]
    )
    return feats


def argmax(xs: List[float]) -> int:
    best_idx = 0
    best_val = xs[0]
    for i in range(1, len(xs)):
        if xs[i] > best_val:
            best_val = xs[i]
            best_idx = i
    return best_idx


class PortableRFClassifier:
    def __init__(self, model_json_path: str):
        self.valid = False
        self.classes: List[str] = []
        self.trees: List[List[dict]] = []
        self.n_features = 0

        if not model_json_path:
            return
        try:
            with open(model_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("model_type") != "portable_rf_v1":
                return
            self.classes = [str(x) for x in payload.get("classes", [])]
            self.trees = payload.get("trees", [])
            self.n_features = int(payload.get("n_features", 0))
            self.valid = bool(self.classes and self.trees and self.n_features > 0)
        except Exception:
            self.valid = False

    def _predict_tree_counts(self, nodes: List[dict], feats: List[float]) -> List[float]:
        idx = 0
        while True:
            node = nodes[idx]
            left = int(node["left"])
            right = int(node["right"])
            if left == -1 and right == -1:
                values = [float(v) for v in node["value"]]
                return values
            fidx = int(node["feature"])
            th = float(node["threshold"])
            idx = left if feats[fidx] <= th else right

    def predict(self, feats: List[float]) -> Tuple[str, float]:
        if not self.valid or len(feats) != self.n_features:
            return "", 0.0

        votes = [0.0] * len(self.classes)
        for tree_nodes in self.trees:
            counts = self._predict_tree_counts(tree_nodes, feats)
            pred_i = argmax(counts)
            votes[pred_i] += 1.0

        total = sum(votes) if votes else 1.0
        best_i = argmax(votes)
        conf = votes[best_i] / total if total > 0 else 0.0
        return self.classes[best_i], conf


class OnePoleLPF:
    def __init__(self, alpha: float = 0.22):
        self.alpha = float(alpha)
        self.prev: Optional[float] = None

    def update(self, x: float) -> float:
        if self.prev is None:
            self.prev = x
            return x
        y = self.alpha * x + (1.0 - self.alpha) * self.prev
        self.prev = y
        return y


class ActionClassifier:
    def __init__(self, model_json_path: str):
        self.portable = PortableRFClassifier(model_json_path)

    def _fallback_predict(self, window: List[Dict[str, float]]) -> Tuple[str, float]:
        acc_mag = []
        gyro_mag = []
        for row in window:
            ax, ay, az = row["ax"], row["ay"], row["az"]
            gx, gy, gz = row["gx"], row["gy"], row["gz"]
            acc_mag.append(math.sqrt(ax * ax + ay * ay + az * az))
            gyro_mag.append(math.sqrt(gx * gx + gy * gy + gz * gz))

        acc_std = std(acc_mag)
        gyro_std = std(gyro_mag)

        if gyro_std < 0.7 and acc_std < 0.07:
            return "rest", 0.55
        if gyro_std > 3.5:
            return "db_biceps_curl", 0.35
        if acc_std > 0.2:
            return "db_squat", 0.30
        return "one_arm_db_row", 0.25

    def predict(self, window: List[Dict[str, float]]) -> Tuple[str, float]:
        if self.portable.valid:
            feats = extract_model_features(window)
            label, conf = self.portable.predict(feats)
            if label:
                return label, conf
        return self._fallback_predict(window)


class RepSegmenter:
    def __init__(self, fs: float = 50.0):
        self.axis_hist: Deque[Tuple[float, float, float]] = deque(maxlen=max(8, int(fs * 1.2)))
        self.signal_lpf = OnePoleLPF(alpha=0.22)
        self.prev_sign = 0
        self.last_cross_ts: Optional[float] = None
        self.last_peak = 0.0
        self.curr_peak = 0.0
        self.crosses: List[Tuple[float, int, float]] = []
        self.min_half_sec = 0.15
        self.deadband_scale = 0.8

    def _dominant_signal(self, sample: Dict[str, float]) -> float:
        gyro = (sample["gx"], sample["gy"], sample["gz"])
        self.axis_hist.append(gyro)
        if len(self.axis_hist) < 8:
            return gyro[0]

        vals = list(self.axis_hist)
        vars_ = []
        for axis in range(3):
            seq = [v[axis] for v in vals]
            vars_.append(std(seq) ** 2)
        axis = argmax(vars_)
        return gyro[axis]

    def update(self, ts_sec: float, sample: Dict[str, float], positive_is_concentric=True):
        raw_sig = self._dominant_signal(sample)
        sig = self.signal_lpf.update(raw_sig)
        self.curr_peak = max(self.curr_peak, abs(sig))

        if len(self.axis_hist) < 8:
            self.prev_sign = 1 if sig > 0 else -1
            return None

        energy = []
        for gx, gy, gz in self.axis_hist:
            energy.append(math.sqrt(gx * gx + gy * gy + gz * gz))
        med = sorted(energy)[len(energy) // 2]
        abs_dev = [abs(e - med) for e in energy]
        mad = sorted(abs_dev)[len(abs_dev) // 2] + 1e-6
        th = max(0.6, self.deadband_scale * 1.4826 * mad)

        sign = 0
        if sig > th:
            sign = 1
        elif sig < -th:
            sign = -1

        if sign == 0:
            return None

        if self.prev_sign == 0:
            self.prev_sign = sign
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
                        stability = max(0.0, min(1.0, 0.55 * tempo_sym + 0.45 * amp_sym))

                        self.crosses = [c2]
                        return {
                            "t_concentric_sec": float(t_con),
                            "t_eccentric_sec": float(t_ecc),
                            "stability": float(stability),
                        }
        return None


class ResistanceRealtimeSystem:
    def __init__(self, fs: float, window_sec: float, hop_sec: float, model_json_path: str, subject: SubjectProfile):
        self.fs = float(fs)
        self.window_size = int(fs * window_sec)
        self.hop_size = max(1, int(fs * hop_sec))
        self.subject = subject

        self.classifier = ActionClassifier(model_json_path)
        self.segmenter = RepSegmenter(fs=fs)

        self.buffer: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.sample_counter = 0
        self.pred_hist: Deque[str] = deque(maxlen=5)
        self.current_action = "rest"
        self.last_classifier_conf = 0.0

        self.rep_counter = defaultdict(int)
        self.rep_tempo_history = defaultdict(lambda: deque(maxlen=12))

    def _smooth_label(self, label: str) -> str:
        self.pred_hist.append(label)
        return Counter(self.pred_hist).most_common(1)[0][0]

    def update(self, row: Dict[str, str]) -> Optional[Dict]:
        ts_raw = row.get("sensor_ts", row.get("timestamp_ms", row.get("timestamp", time.time())))
        ts = ts_to_sec(ts_raw, time.time())
        sample = {k: safe_float(row.get(k, 0.0)) for k in IMU_COLS}

        self.buffer.append(sample)
        self.sample_counter += 1

        classified_this_step = False
        if len(self.buffer) >= self.window_size and self.sample_counter % self.hop_size == 0:
            window = list(self.buffer)
            raw_label, conf = self.classifier.predict(window)
            self.current_action = self._smooth_label(raw_label)
            self.last_classifier_conf = conf
            classified_this_step = True
        else:
            conf = self.last_classifier_conf

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

        hist = list(self.rep_tempo_history[self.current_action])
        cv = (std(hist) / (mean(hist) + 1e-6)) if len(hist) >= 3 else 0.0
        stability_final = max(0.0, min(1.0, 0.7 * phase_out["stability"] + 0.3 * (1.0 - min(cv, 1.0))))

        return {
            "action_type": self.current_action,
            "rep": rep_idx,
            "t_concentric_sec": round(phase_out["t_concentric_sec"], 3),
            "t_eccentric_sec": round(phase_out["t_eccentric_sec"], 3),
            "stability_score": round(100.0 * stability_final, 1),
            "classifier_conf": round(conf, 3),
        }


def parse_row_from_text(line: str) -> Dict[str, str]:
    parts = [x.strip() for x in line.strip().split(",")]
    if len(parts) < 13:
        raise ValueError("CSV columns too short")

    default_cols = [
        "pc_time", "serial_num", "sensor_ts", "host_ts",
        "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
        "action_type", "phase", "rep", "set", "rpe", "weight_kg", "subject_id",
        "ppg_a", "ppg_b", "ppg_c", "ppg_d", "ppg_e",
    ]
    if len(parts) >= len(default_cols):
        return {k: parts[i] for i, k in enumerate(default_cols)}
    return {k: parts[i] for i, k in enumerate(default_cols[: len(parts)])}


def run_udp(args):
    subject = SubjectProfile(age=args.age, sex=args.sex, exp_level=args.exp_level)
    system = ResistanceRealtimeSystem(
        fs=args.fs,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        model_json_path=args.model_json,
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
        model_json_path=args.model_json,
        subject=subject,
    )

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = system.update(row)
            if out:
                print(json.dumps(out, ensure_ascii=False))


def build_argparser():
    p = argparse.ArgumentParser(description="Realtime resistance pipeline (luckfox stdlib runtime)")
    p.add_argument("--mode", choices=["udp", "csv"], default="udp")
    p.add_argument("--udp-ip", default="0.0.0.0")
    p.add_argument("--udp-port", type=int, default=10000)
    p.add_argument("--csv-path", default="")
    p.add_argument("--model-json", default="action_rf_model_portable.json")

    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--hop-sec", type=float, default=1.0)

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
