from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

import numpy as np


@dataclass
class RealtimeResult:
    timestamp_ms: int
    state: str              # "REST" or "ACTIVE"
    reps: int
    period_sec: Optional[float]
    confidence: float
    peak_value: Optional[float]


class EMAFilter:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.y = None

    def update(self, x: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


class RealtimeULift:
    """
    Minimal real-time version inspired by uLift:
    1) detect workout / rest
    2) count reps during active periods
    3) keep API simple
    """

    def __init__(
        self,
        fs: float = 100.0,
        buffer_sec: float = 4.0,
        gate_sec: float = 2.0,
        smooth_alpha: float = 0.2,
        active_std_th: float = 0.12,
        active_energy_th: float = 0.18,
        min_rep_interval_sec: float = 0.45,
        max_rep_interval_sec: float = 4.0,
        peak_th_ratio: float = 0.55,
    ):
        self.fs = fs
        self.buffer_n = int(buffer_sec * fs)
        self.gate_n = int(gate_sec * fs)

        self.mag_buf = deque(maxlen=self.buffer_n)
        self.ts_buf = deque(maxlen=self.buffer_n)

        self.smoother = EMAFilter(alpha=smooth_alpha)

        self.active_std_th = active_std_th
        self.active_energy_th = active_energy_th

        self.min_rep_interval = min_rep_interval_sec
        self.max_rep_interval = max_rep_interval_sec
        self.peak_th_ratio = peak_th_ratio

        self.state = "REST"
        self.reps = 0

        self.last_peak_ts: Optional[float] = None
        self.last_period_sec: Optional[float] = None
        self.last_peak_value: Optional[float] = None

        # For hysteresis
        self.active_score_hist = deque(maxlen=max(5, int(0.5 * fs / 10)))

    def _magnitude(self, ax: float, ay: float, az: float) -> float:
        return math.sqrt(ax * ax + ay * ay + az * az)

    def _preprocess_window(self, x: np.ndarray) -> np.ndarray:
        # remove DC
        x = x - np.mean(x)

        # robust normalize
        mad = np.median(np.abs(x - np.median(x))) + 1e-8
        x = x / (1.4826 * mad + 1e-8)
        return x

    def _detect_state(self) -> Tuple[str, float]:
        if len(self.mag_buf) < self.gate_n:
            return "REST", 0.0

        x = np.array(list(self.mag_buf)[-self.gate_n:], dtype=float)
        x = self._preprocess_window(x)

        std = float(np.std(x))
        energy = float(np.mean(np.abs(x)))

        score = 0.5 * min(std / self.active_std_th, 2.0) + \
                0.5 * min(energy / self.active_energy_th, 2.0)

        self.active_score_hist.append(score)
        smoothed_score = float(np.mean(self.active_score_hist))

        # hysteresis
        if self.state == "REST":
            new_state = "ACTIVE" if smoothed_score > 1.15 else "REST"
        else:
            new_state = "ACTIVE" if smoothed_score > 0.85 else "REST"

        conf = max(0.0, min(smoothed_score / 1.5, 1.0))
        return new_state, conf

    def _estimate_period_autocorr(self, x: np.ndarray) -> Optional[float]:
        """
        Estimate dominant repetition period from autocorrelation.
        """
        if len(x) < int(1.5 * self.fs):
            return None

        x = self._preprocess_window(x)
        ac = np.correlate(x, x, mode="full")
        ac = ac[len(ac)//2:]

        min_lag = int(self.min_rep_interval * self.fs)
        max_lag = min(int(self.max_rep_interval * self.fs), len(ac) - 1)

        if max_lag <= min_lag:
            return None

        search = ac[min_lag:max_lag]
        if len(search) == 0:
            return None

        lag = int(np.argmax(search)) + min_lag
        if lag <= 0:
            return None

        return lag / self.fs

    def _count_rep(self, now_ts_sec: float) -> None:
        """
        Peak-based counting on preprocessed magnitude.
        Uses dynamic threshold + refractory period.
        """
        if len(self.mag_buf) < max(int(1.2 * self.fs), 5):
            return

        x = np.array(self.mag_buf, dtype=float)
        x = self._preprocess_window(x)

        # dynamic threshold from recent amplitude
        recent = x[-int(min(len(x), 2 * self.fs)):]
        amp = np.max(recent) - np.min(recent)
        peak_th = np.min(recent) + self.peak_th_ratio * amp

        # local peak at previous sample
        if len(recent) < 3:
            return

        a, b, c = recent[-3], recent[-2], recent[-1]
        is_peak = (b > a) and (b >= c) and (b > peak_th)

        if not is_peak:
            return

        peak_ts_sec = now_ts_sec - 1.0 / self.fs

        if self.last_peak_ts is None:
            self.reps += 1
            self.last_peak_ts = peak_ts_sec
            self.last_peak_value = float(b)
            return

        dt = peak_ts_sec - self.last_peak_ts
        if self.min_rep_interval <= dt <= self.max_rep_interval:
            self.reps += 1
            self.last_period_sec = dt
            self.last_peak_ts = peak_ts_sec
            self.last_peak_value = float(b)

    def update(self, timestamp_ms: int, ax: float, ay: float, az: float) -> RealtimeResult:
        mag = self._magnitude(ax, ay, az)
        mag_s = self.smoother.update(mag)

        self.ts_buf.append(timestamp_ms)
        self.mag_buf.append(mag_s)

        new_state, conf = self._detect_state()

        # state transition
        if self.state == "REST" and new_state == "ACTIVE":
            self.last_peak_ts = None
            self.last_period_sec = None
            self.last_peak_value = None

        if new_state == "ACTIVE":
            self._count_rep(timestamp_ms / 1000.0)

            # refresh period estimate from autocorr if enough data
            x = np.array(self.mag_buf, dtype=float)
            est_period = self._estimate_period_autocorr(x)
            if est_period is not None:
                self.last_period_sec = est_period
        else:
            self.last_peak_ts = None
            self.last_peak_value = None

        self.state = new_state

        return RealtimeResult(
            timestamp_ms=timestamp_ms,
            state=self.state,
            reps=self.reps,
            period_sec=self.last_period_sec,
            confidence=conf,
            peak_value=self.last_peak_value,
        )