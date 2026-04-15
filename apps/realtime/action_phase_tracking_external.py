import math
import socket
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# =========================================================
# Data Structures
# =========================================================

@dataclass
class IMUSample:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float

    @staticmethod
    def from_csv_line(line: str):
        """
        expected format:
        timestamp,ax,ay,az,gx,gy,gz,mx,my,mz
        """
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) != 10:
            raise ValueError(f"expected 10 fields, got {len(parts)}: {line}")

        vals = [float(x) for x in parts]
        return IMUSample(
            timestamp=vals[0],
            ax=vals[1],
            ay=vals[2],
            az=vals[3],
            gx=vals[4],
            gy=vals[5],
            gz=vals[6],
            mx=vals[7],
            my=vals[8],
            mz=vals[9],
        )


@dataclass
class PhaseResult:
    phase: str
    rep_done: bool = False
    rep_count_increment: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineOutput:
    timestamp: float
    raw_action: str
    smoothed_action: str
    phase: str
    count: int
    debug: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Motion Feature Extractor
# =========================================================

class MotionFeatureExtractor:
    def extract_instant_features(
        self,
        current: IMUSample,
        prev: Optional[IMUSample]
    ) -> Dict[str, float]:
        if prev is None:
            dt = 0.01
            dax = day = daz = 0.0
            dgx = dgy = dgz = 0.0
            dmx = dmy = dmz = 0.0
        else:
            dt = max(current.timestamp - prev.timestamp, 1e-6)

            dax = (current.ax - prev.ax) / dt
            day = (current.ay - prev.ay) / dt
            daz = (current.az - prev.az) / dt

            dgx = (current.gx - prev.gx) / dt
            dgy = (current.gy - prev.gy) / dt
            dgz = (current.gz - prev.gz) / dt

            dmx = (current.mx - prev.mx) / dt
            dmy = (current.my - prev.my) / dt
            dmz = (current.mz - prev.mz) / dt

        acc_mag = math.sqrt(current.ax**2 + current.ay**2 + current.az**2)
        gyro_mag = math.sqrt(current.gx**2 + current.gy**2 + current.gz**2)
        mag_mag = math.sqrt(current.mx**2 + current.my**2 + current.mz**2)

        return {
            "ax": current.ax, "ay": current.ay, "az": current.az,
            "gx": current.gx, "gy": current.gy, "gz": current.gz,
            "mx": current.mx, "my": current.my, "mz": current.mz,

            "dax": dax, "day": day, "daz": daz,
            "dgx": dgx, "dgy": dgy, "dgz": dgz,
            "dmx": dmx, "dmy": dmy, "dmz": dmz,

            "dt": dt,
            "acc_mag": acc_mag,
            "gyro_mag": gyro_mag,
            "mag_mag": mag_mag,
        }


# =========================================================
# Action Smoother
# =========================================================

class ExternalActionSmoother:
    def __init__(self, hold_count: int = 5):
        self.hold_count = hold_count
        self.current_action = "unknown"
        self.candidate_action = None
        self.candidate_count = 0

    def update(self, raw_action: str) -> str:
        if raw_action == self.current_action:
            self.candidate_action = None
            self.candidate_count = 0
            return self.current_action

        if self.candidate_action != raw_action:
            self.candidate_action = raw_action
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= self.hold_count:
            self.current_action = raw_action
            self.candidate_action = None
            self.candidate_count = 0

        return self.current_action


# =========================================================
# Simple Rule-Based Action Classifier
# =========================================================

class SimpleActionClassifier:
    """
    先用簡單規則做粗分類：
    - idle
    - db_squat
    - bicep_curl
    - shoulder_press
    - unknown

    這不是最終模型，只是讓流程先打通。
    """

    def __init__(self):
        self.idle_gyro_thresh = 8.0
        self.idle_daz_thresh = 0.2

        self.curl_gyro_y_thresh = 30.0
        self.press_gyro_x_thresh = 30.0
        self.squat_daz_thresh = 1.0

    def predict(self, inst_features: Dict[str, float]) -> str:
        gx = inst_features["gx"]
        gy = inst_features["gy"]
        gz = inst_features["gz"]
        daz = inst_features["daz"]
        gyro_mag = inst_features["gyro_mag"]

        # 幾乎沒動
        if gyro_mag < self.idle_gyro_thresh and abs(daz) < self.idle_daz_thresh:
            return "idle"

        # 二頭彎舉：Y軸角速度主導
        if abs(gy) > self.curl_gyro_y_thresh and abs(gy) > abs(gx):
            return "bicep_curl"

        # 肩推：X軸角速度主導
        if abs(gx) > self.press_gyro_x_thresh and abs(gx) > abs(gy):
            return "shoulder_press"

        # 深蹲：Z向加速度變化顯著
        if abs(daz) > self.squat_daz_thresh:
            return "db_squat"

        if gyro_mag > self.idle_gyro_thresh:
            return "unknown"

        return "idle"


# =========================================================
# Base Phase Tracker
# =========================================================

class BasePhaseTracker:
    def __init__(self, name: str):
        self.name = name
        self.phase = "idle"
        self.rep_count = 0

    def reset_phase_only(self):
        pass

    def update(self, inst_features: Dict[str, float]) -> PhaseResult:
        raise NotImplementedError


# =========================================================
# Squat Phase Tracker
# =========================================================

class SquatPhaseTracker(BasePhaseTracker):
    """
    standing -> descending -> bottom -> ascending -> standing
    rep +1 when ascending -> standing
    """

    def __init__(self):
        super().__init__("db_squat")
        self.phase = "standing"

        self.down_thresh = -0.8
        self.up_thresh = 0.8
        self.near_stop_thresh = 0.25
        self.bottom_acc_thresh = -1.2
        self.top_acc_thresh = 0.7

    def reset_phase_only(self):
        self.phase = "standing"

    def update(self, inst_features: Dict[str, float]) -> PhaseResult:
        az = inst_features["az"]
        daz = inst_features["daz"]
        rep_done = False

        if self.phase == "standing":
            if daz < self.down_thresh:
                self.phase = "descending"

        elif self.phase == "descending":
            if abs(daz) < self.near_stop_thresh and az < self.bottom_acc_thresh:
                self.phase = "bottom"
            elif daz > self.up_thresh:
                self.phase = "ascending"

        elif self.phase == "bottom":
            if daz > self.up_thresh:
                self.phase = "ascending"

        elif self.phase == "ascending":
            if abs(daz) < self.near_stop_thresh and az > self.top_acc_thresh:
                self.phase = "standing"
                self.rep_count += 1
                rep_done = True

        return PhaseResult(
            phase=self.phase,
            rep_done=rep_done,
            rep_count_increment=1 if rep_done else 0,
            extra={"az": az, "daz": daz},
        )


# =========================================================
# Curl Phase Tracker
# =========================================================

class CurlPhaseTracker(BasePhaseTracker):
    """
    bottom -> lifting -> top -> lowering -> bottom
    """

    def __init__(self):
        super().__init__("bicep_curl")
        self.phase = "bottom"

        self.lift_thresh = 25.0
        self.lower_thresh = -25.0
        self.near_stop_thresh = 8.0

    def reset_phase_only(self):
        self.phase = "bottom"

    def update(self, inst_features: Dict[str, float]) -> PhaseResult:
        gy = inst_features["gy"]
        dgy = inst_features["dgy"]
        rep_done = False

        if self.phase == "bottom":
            if gy > self.lift_thresh:
                self.phase = "lifting"

        elif self.phase == "lifting":
            if abs(gy) < self.near_stop_thresh and dgy < -5.0:
                self.phase = "top"

        elif self.phase == "top":
            if gy < self.lower_thresh:
                self.phase = "lowering"

        elif self.phase == "lowering":
            if abs(gy) < self.near_stop_thresh and dgy > 5.0:
                self.phase = "bottom"
                self.rep_count += 1
                rep_done = True

        return PhaseResult(
            phase=self.phase,
            rep_done=rep_done,
            rep_count_increment=1 if rep_done else 0,
            extra={"gy": gy, "dgy": dgy},
        )


# =========================================================
# Shoulder Press Phase Tracker
# =========================================================

class ShoulderPressPhaseTracker(BasePhaseTracker):
    """
    down -> pressing_up -> top -> lowering -> down
    """

    def __init__(self):
        super().__init__("shoulder_press")
        self.phase = "down"

        self.up_thresh = 20.0
        self.down_thresh = -20.0
        self.near_stop_thresh = 7.0

    def reset_phase_only(self):
        self.phase = "down"

    def update(self, inst_features: Dict[str, float]) -> PhaseResult:
        gx = inst_features["gx"]
        dgx = inst_features["dgx"]
        rep_done = False

        if self.phase == "down":
            if gx > self.up_thresh:
                self.phase = "pressing_up"

        elif self.phase == "pressing_up":
            if abs(gx) < self.near_stop_thresh and dgx < -5.0:
                self.phase = "top"

        elif self.phase == "top":
            if gx < self.down_thresh:
                self.phase = "lowering"

        elif self.phase == "lowering":
            if abs(gx) < self.near_stop_thresh and dgx > 5.0:
                self.phase = "down"
                self.rep_count += 1
                rep_done = True

        return PhaseResult(
            phase=self.phase,
            rep_done=rep_done,
            rep_count_increment=1 if rep_done else 0,
            extra={"gx": gx, "dgx": dgx},
        )


# =========================================================
# Idle Tracker
# =========================================================

class IdleTracker(BasePhaseTracker):
    def __init__(self):
        super().__init__("idle")
        self.phase = "idle"

    def reset_phase_only(self):
        self.phase = "idle"

    def update(self, inst_features: Dict[str, float]) -> PhaseResult:
        return PhaseResult(
            phase="idle",
            rep_done=False,
            rep_count_increment=0,
            extra={}
        )


# =========================================================
# Unknown Tracker
# =========================================================

class UnknownTracker(BasePhaseTracker):
    def __init__(self):
        super().__init__("unknown")
        self.phase = "unknown"

    def reset_phase_only(self):
        self.phase = "unknown"

    def update(self, inst_features: Dict[str, float]) -> PhaseResult:
        return PhaseResult(
            phase="unknown",
            rep_done=False,
            rep_count_increment=0,
            extra={}
        )


# =========================================================
# Tracker Manager
# =========================================================

class TrackerManager:
    def __init__(self):
        self.trackers = {
            "db_squat": SquatPhaseTracker(),
            "bicep_curl": CurlPhaseTracker(),
            "shoulder_press": ShoulderPressPhaseTracker(),
            "idle": IdleTracker(),
            "unknown": UnknownTracker(),
        }
        self.active_action = "unknown"

    def switch_action(self, new_action: str):
        self.active_action = new_action

    def update(self, action: str, inst_features: Dict[str, float]) -> PhaseResult:
        tracker = self.trackers.get(action, self.trackers["unknown"])
        return tracker.update(inst_features)

    def get_count(self, action: str) -> int:
        tracker = self.trackers.get(action)
        return 0 if tracker is None else tracker.rep_count


# =========================================================
# Main Pipeline
# =========================================================

class ActionPhasePipelineAuto:
    def __init__(self, action_hold_count: int = 5):
        self.prev_sample: Optional[IMUSample] = None
        self.feature_extractor = MotionFeatureExtractor()
        self.action_classifier = SimpleActionClassifier()
        self.smoother = ExternalActionSmoother(hold_count=action_hold_count)
        self.tracker_manager = TrackerManager()

        self.last_smoothed_action = "unknown"
        self.global_count = 0

    def process(self, sample: IMUSample) -> PipelineOutput:
        inst_features = self.feature_extractor.extract_instant_features(
            current=sample,
            prev=self.prev_sample
        )

        raw_action = self.action_classifier.predict(inst_features)
        smoothed_action = self.smoother.update(raw_action)

        if smoothed_action != self.last_smoothed_action:
            self.tracker_manager.switch_action(smoothed_action)
            self.last_smoothed_action = smoothed_action

        phase_result = self.tracker_manager.update(smoothed_action, inst_features)

        if phase_result.rep_done:
            self.global_count += phase_result.rep_count_increment

        self.prev_sample = sample

        return PipelineOutput(
            timestamp=sample.timestamp,
            raw_action=raw_action,
            smoothed_action=smoothed_action,
            phase=phase_result.phase,
            count=self.tracker_manager.get_count(smoothed_action),
            debug={
                "global_count": self.global_count,
                "inst_features": inst_features,
                "phase_extra": phase_result.extra,
            }
        )


# =========================================================
# Pretty Print
# =========================================================

def print_output(out: PipelineOutput):
    inst = out.debug["inst_features"]
    print(
        f"[{out.timestamp:.3f}] "
        f"raw={out.raw_action:<15} | "
        f"smooth={out.smoothed_action:<15} | "
        f"phase={out.phase:<12} | "
        f"count={out.count:<3} | "
        f"acc_mag={inst['acc_mag']:.3f} | "
        f"gyro_mag={inst['gyro_mag']:.3f} | "
        f"mag_mag={inst['mag_mag']:.3f}"
    )


# =========================================================
# UDP Realtime Receiver
# =========================================================

def run_udp_realtime_auto(
    host: str = "0.0.0.0",
    port: int = 10000,
    action_hold_count: int = 3,
):
    pipeline = ActionPhasePipelineAuto(action_hold_count=action_hold_count)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))

    print(f"[INFO] listening UDP on {host}:{port}")
    print("[INFO] expected packet format:")
    print("       timestamp,ax,ay,az,gx,gy,gz,mx,my,mz")
    print("[INFO] press Ctrl+C to stop")

    try:
        while True:
            data, addr = sock.recvfrom(4096)
            line = data.decode(errors="ignore").strip()

            if not line:
                continue

            try:
                sample = IMUSample.from_csv_line(line)
                out = pipeline.process(sample)
                print_output(out)

            except Exception as e:
                print(f"[WARN] invalid packet from {addr}: {line}")
                print(f"       reason: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
    finally:
        sock.close()
        print("[INFO] socket closed")


if __name__ == "__main__":
    run_udp_realtime_auto(
        host="0.0.0.0",
        port=10000,
        action_hold_count=3,
    )