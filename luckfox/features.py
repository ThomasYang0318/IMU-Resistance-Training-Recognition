from __future__ import annotations

import bisect
import math
from statistics import median
from typing import Iterable, Sequence


IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")
ACTIVE_PHASES = {"concentric", "eccentric"}
OTHER_LABEL = "Other"


def clean_label(value: object) -> str:
    return str(value or "").strip().lower()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def infer_time_seconds(raw_times: Sequence[float]) -> list[float]:
    if not raw_times:
        return []
    diffs = [b - a for a, b in zip(raw_times, raw_times[1:]) if b > a and math.isfinite(b - a)]
    med = median(diffs) if diffs else 1.0
    if med > 1000.0:
        scale = 1_000_000.0
    elif med > 10.0:
        scale = 1000.0
    else:
        scale = 1.0
    first = raw_times[0]
    out: list[float] = []
    last = 0.0
    for value in raw_times:
        current = (value - first) / scale
        if current < last:
            current = last
        out.append(current)
        last = current
    return out


def mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def std(values: Sequence[float], mu: float | None = None) -> float:
    if not values:
        return 0.0
    center = mean(values) if mu is None else mu
    return math.sqrt(sum((value - center) * (value - center) for value in values) / float(len(values)))


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / float(len(values))) if values else 0.0


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = fraction * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def basic_stats(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0] * 10
    ordered = sorted(values)
    mu = mean(values)
    diffs = [b - a for a, b in zip(values, values[1:])]
    return [
        mu,
        std(values, mu),
        ordered[0],
        ordered[-1],
        ordered[-1] - ordered[0],
        rms(values),
        percentile(ordered, 0.5),
        mean([abs(value) for value in values]),
        values[-1] - values[0],
        std(diffs) if diffs else 0.0,
    ]


def magnitude(block: Sequence[Sequence[float]], start: int) -> list[float]:
    return [
        math.sqrt(row[start] * row[start] + row[start + 1] * row[start + 1] + row[start + 2] * row[start + 2])
        for row in block
    ]


def nan_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    x_mu = mean(xs)
    y_mu = mean(ys)
    x_std = std(xs, x_mu)
    y_std = std(ys, y_mu)
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    cov = sum((x - x_mu) * (y - y_mu) for x, y in zip(xs, ys)) / float(len(xs))
    return cov / (x_std * y_std)


def vector_mean(block: Sequence[Sequence[float]], start: int) -> tuple[float, float, float]:
    if not block:
        return 0.0, 0.0, 0.0
    n = float(len(block))
    return (
        sum(row[start] for row in block) / n,
        sum(row[start + 1] for row in block) / n,
        sum(row[start + 2] for row in block) / n,
    )


def projection_stats(block: Sequence[Sequence[float]], start: int, gravity_unit: tuple[float, float, float]) -> list[float]:
    gx, gy, gz = gravity_unit
    vertical: list[float] = []
    horizontal: list[float] = []
    for row in block:
        vx = row[start]
        vy = row[start + 1]
        vz = row[start + 2]
        dot = vx * gx + vy * gy + vz * gz
        hx = vx - dot * gx
        hy = vy - dot * gy
        hz = vz - dot * gz
        vertical.append(dot)
        horizontal.append(math.sqrt(hx * hx + hy * hy + hz * hz))
    ratio = rms(vertical) / max(rms(horizontal), 1e-6)
    return [*basic_stats(vertical), *basic_stats(horizontal), ratio]


def scale_features(block: Sequence[Sequence[float]], duration: float, include_posture: bool = True) -> list[float]:
    features: list[float] = []
    for axis in range(len(IMU_COLUMNS)):
        features.extend(basic_stats([row[axis] for row in block]))

    for start in (0, 3, 6):
        features.extend(basic_stats(magnitude(block, start)))

    for start in (0, 3, 6):
        xs = [row[start] for row in block]
        ys = [row[start + 1] for row in block]
        zs = [row[start + 2] for row in block]
        features.extend([nan_corr(xs, ys), nan_corr(xs, zs), nan_corr(ys, zs)])

    for start in (0, 3, 6):
        jerk_mag: list[float] = []
        for prev, current in zip(block, block[1:]):
            dx = current[start] - prev[start]
            dy = current[start + 1] - prev[start + 1]
            dz = current[start + 2] - prev[start + 2]
            jerk_mag.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        features.extend(basic_stats(jerk_mag))

    ax, ay, az = vector_mean(block, 0)
    gravity_norm = math.sqrt(ax * ax + ay * ay + az * az)
    gravity_unit = (ax / max(gravity_norm, 1e-6), ay / max(gravity_norm, 1e-6), az / max(gravity_norm, 1e-6))
    dynamic_mag: list[float] = []
    for row in block:
        dx = row[0] - ax
        dy = row[1] - ay
        dz = row[2] - az
        dynamic_mag.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    features.extend(basic_stats(dynamic_mag))
    features.extend([gravity_unit[0], gravity_unit[1], gravity_unit[2], gravity_norm, duration, float(len(block))])

    if include_posture:
        gx, gy, gz = gravity_unit
        pitch = math.atan2(-gx, math.sqrt(gy * gy + gz * gz))
        roll = math.atan2(gy, gz)
        inclination = math.acos(max(0.0, min(1.0, abs(gz))))
        features.extend([pitch, roll, inclination])
        for start in (0, 3, 6):
            features.extend(projection_stats(block, start, gravity_unit))

    return features


def extract_features(
    times: Sequence[float],
    samples: Sequence[Sequence[float]],
    end_time: float,
    scales_seconds: Sequence[float] = (0.75, 2.0, 4.0),
    include_posture: bool = True,
) -> list[float] | None:
    rows: list[float] = []
    for scale in scales_seconds:
        start_time = end_time - float(scale)
        start_idx = bisect.bisect_left(times, start_time)
        end_idx = bisect.bisect_right(times, end_time)
        if end_idx <= start_idx:
            return None
        block = samples[start_idx:end_idx]
        duration = max(0.0, times[end_idx - 1] - times[start_idx])
        rows.extend(scale_features(block, duration, include_posture=include_posture))
    return rows


def endpoint_label(
    times: Sequence[float],
    labels: Sequence[str],
    active: Sequence[bool],
    end_time: float,
    endpoint_seconds: float,
    exercise_labels: Sequence[str],
    min_active_fraction: float,
) -> tuple[str, float]:
    start_idx = bisect.bisect_left(times, end_time - endpoint_seconds)
    end_idx = bisect.bisect_right(times, end_time)
    if end_idx <= start_idx:
        return OTHER_LABEL, 0.0
    active_slice = active[start_idx:end_idx]
    active_fraction = sum(1 for value in active_slice if value) / float(len(active_slice))
    if active_fraction < min_active_fraction:
        return OTHER_LABEL, active_fraction
    counts = {label: 0 for label in exercise_labels}
    for label, is_active in zip(labels[start_idx:end_idx], active_slice):
        if is_active and label in counts:
            counts[label] += 1
    if not counts or max(counts.values()) <= 0:
        return OTHER_LABEL, active_fraction
    return max(exercise_labels, key=lambda label: (counts.get(label, 0), -exercise_labels.index(label))), active_fraction


def active_labels_from_rows(actions: Iterable[str], phases: Iterable[str], exercise_labels: Sequence[str]) -> tuple[list[str], list[bool]]:
    exercise_set = set(exercise_labels)
    labels: list[str] = []
    active: list[bool] = []
    for action_raw, phase_raw in zip(actions, phases):
        action = clean_label(action_raw)
        phase = clean_label(phase_raw)
        is_active = action in exercise_set and phase in ACTIVE_PHASES
        active.append(is_active)
        labels.append(action if is_active else OTHER_LABEL)
    return labels, active
