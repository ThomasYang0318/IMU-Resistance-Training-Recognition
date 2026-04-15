import socket
import csv
import math
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from collections import deque

import numpy as np

from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl


# =========================
# Config
# =========================

# UDP port，IMU sender 要送到這個 port
PORT = 10000

# 是否儲存資料（做 dataset / debug 很重要）
SAVE_CSV = True

# CSV 檔名（時間戳避免覆蓋）
CSV_FILENAME = f"imu9axis_6dof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 軌跡最多保留點數，避免記憶體爆掉，畫面上也不會太亂
MAX_TRAJ_POINTS = 300

# 是否使用磁力計修正 yaw（避免漂移）
USE_MAG_YAW = True

# 若你的 accel 單位是 raw counts，不是 m/s^2，可先把這裡改成 False
ACC_INPUT_IS_MS2 = False

# 若 ACC_INPUT_IS_MS2=False，下面用來把 accel raw 轉成 m/s^2
# raw → g 的比例（±2g 常見設定）
# 常見：±2g 量測範圍時，16384 LSB/g
ACC_LSB_PER_G = 16384.0

# 若 gyro 本來就是 deg/s，保持 True
GYRO_INPUT_IS_DEG_S = True

# 靜止判定門檻
# | |a|-g | 小於這個 → 近似靜止
STATIONARY_ACC_ERR_MS2 = 0.20     # | |a|-g | < 0.35
# gyro 小於這個 → 近似靜止
STATIONARY_GYRO_DPS = 1.2         # |gyro| < 2 deg/s

# 軌跡積分控制
VEL_DAMPING = 0.985               # 速度阻尼，避免無限制飄移
POS_DAMPING = 0.999               # 位置微阻尼
ACC_DEADBAND_MS2 = 0.10           # 去掉小震動

# dt 限制（避免時間錯誤）
MAX_DT = 0.05
MIN_DT = 0.001

# 顯示比例，避免飛出去
BODY_AXIS_LEN = 0.18
WORLD_BOX = 0.6                   # 畫面 world 範圍 +-WORLD_BOX


# =========================
# Shared data
# =========================
@dataclass
class IMUSample:
    recv_time: str                # 接收時間（PC）
    timestamp: int                # sensor timestamp（毫秒）

    # 原始資料（已轉單位）
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float

    # 姿態
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # 接收頻率
    fps: float = 0.0

    q: np.ndarray = None

    # norm（用來判斷靜止/品質）
    a_norm: float = 0.0
    g_norm: float = 0.0
    m_norm: float = 0.0

    # 世界座標加速度（重力補償後）
    ax_w: float = 0.0
    ay_w: float = 0.0
    az_w: float = 0.0

    # 世界座標速度
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # 位置（短期軌跡）
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # 狀態
    stationary: bool = False
    bias_ready: bool = False


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.traj = deque(maxlen=MAX_TRAJ_POINTS)


shared = SharedState()


# =========================
# Math utils
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_angle_deg(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def q_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def q_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def q_to_rotmat(q):
    q = q_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)


def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr = math.cos(r / 2)
    sr = math.sin(r / 2)
    cp = math.cos(p / 2)
    sp = math.sin(p / 2)
    cy = math.cos(y / 2)
    sy = math.sin(y / 2)

    q = np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr
    ], dtype=float)
    return q_normalize(q)


def quaternion_to_axis_angle(q):
    q = q_normalize(q)
    w = clamp(q[0], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1e-12, 1.0 - w*w))

    if s < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = np.array([q[1]/s, q[2]/s, q[3]/s], dtype=float)

    return axis, math.degrees(angle)


def body_to_world(q, v_body):
    R = q_to_rotmat(q)
    return R @ np.asarray(v_body, dtype=float)


def accel_to_ms2(ax, ay, az):
    if ACC_INPUT_IS_MS2:
        return float(ax), float(ay), float(az)
    scale = 9.80665 / ACC_LSB_PER_G
    return ax * scale, ay * scale, az * scale


def gyro_to_deg_s(gx, gy, gz):
    scale = 16.4
    return gx / scale, gy / scale, gz / scale


def apply_deadband(v, th):
    return 0.0 if abs(v) < th else v


# =========================
# Orientation + motion estimator
# =========================
class MotionEstimator:
    """
    目標：
    1. 先讓姿態穩
    2. 產出短時相對位移，適合動作辨識/視覺化
    3. 不是長時間高精度導航
    """
    def __init__(self, alpha_rp=0.985, alpha_yaw=0.995):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.alpha_rp = alpha_rp
        self.alpha_yaw = alpha_yaw
        self.initialized = False

        # Gyro LPF
        self.gx_f = None
        self.gy_f = None
        self.gz_f = None

        # Bias calibration
        self.bias_samples = []
        self.bias_ready = False
        self.gx_bias = 0.0
        self.gy_bias = 0.0
        self.gz_bias = 0.0

        # Motion states
        self.v = np.zeros(3, dtype=float)
        self.p = np.zeros(3, dtype=float)

    def update(self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
        dt = clamp(dt, MIN_DT, MAX_DT)

        # unit normalize / convert
        ax_ms2, ay_ms2, az_ms2 = accel_to_ms2(ax, ay, az)
        gx_dps, gy_dps, gz_dps = gyro_to_deg_s(gx, gy, gz)

        # gyro low-pass
        beta = 0.75
        if self.gx_f is None:
            self.gx_f, self.gy_f, self.gz_f = gx_dps, gy_dps, gz_dps
        else:
            self.gx_f = beta * self.gx_f + (1 - beta) * gx_dps
            self.gy_f = beta * self.gy_f + (1 - beta) * gy_dps
            self.gz_f = beta * self.gz_f + (1 - beta) * gz_dps

        gx_dps = self.gx_f
        gy_dps = self.gy_f
        gz_dps = self.gz_f

        a_vec = np.array([ax_ms2, ay_ms2, az_ms2], dtype=float)
        a_norm = float(np.linalg.norm(a_vec))
        g_norm = float(np.linalg.norm([gx_dps, gy_dps, gz_dps]))
        m_norm = float(np.linalg.norm([mx, my, mz]))

        # bias collect: 開始時請保持靜止
        if not self.bias_ready:
            is_quiet = abs(a_norm - 9.80665) < 1.2 and g_norm < 8.0
            if is_quiet:
                self.bias_samples.append((gx_dps, gy_dps, gz_dps))
            if len(self.bias_samples) >= 200:
                arr = np.array(self.bias_samples, dtype=float)
                self.gx_bias, self.gy_bias, self.gz_bias = arr.mean(axis=0)
                self.bias_ready = True

        if self.bias_ready:
            gx_dps -= self.gx_bias
            gy_dps -= self.gy_bias
            gz_dps -= self.gz_bias

        # stationary detection
        stationary = (
            abs(a_norm - 9.80665) < STATIONARY_ACC_ERR_MS2 and
            np.linalg.norm([gx_dps, gy_dps, gz_dps]) < STATIONARY_GYRO_DPS
        )

        # 姿態初始化
        if not self.initialized:
            roll_acc = math.degrees(math.atan2(ay_ms2, az_ms2))
            pitch_acc = math.degrees(math.atan2(-ax_ms2, math.sqrt(ay_ms2*ay_ms2 + az_ms2*az_ms2) + 1e-12))
            yaw_mag = self._yaw_from_mag(mx, my, mz, roll_acc, pitch_acc) if USE_MAG_YAW else 0.0

            self.roll = roll_acc
            self.pitch = pitch_acc
            self.yaw = yaw_mag
            self.initialized = True
        else:
            # gyro integrate
            self.roll += gx_dps * dt
            self.pitch += gy_dps * dt
            self.yaw += gz_dps * dt

            # accel correction
            roll_acc = math.degrees(math.atan2(ay_ms2, az_ms2))
            pitch_acc = math.degrees(math.atan2(-ax_ms2, math.sqrt(ay_ms2*ay_ms2 + az_ms2*az_ms2) + 1e-12))

            self.roll = self.alpha_rp * self.roll + (1.0 - self.alpha_rp) * roll_acc
            self.pitch = self.alpha_rp * self.pitch + (1.0 - self.alpha_rp) * pitch_acc

            # yaw correction
            if USE_MAG_YAW:
                yaw_mag = self._yaw_from_mag(mx, my, mz, self.roll, self.pitch)
                yaw_err = wrap_angle_deg(yaw_mag - self.yaw)
                self.yaw += (1.0 - self.alpha_yaw) * yaw_err

        self.yaw = wrap_angle_deg(self.yaw)
        q = euler_to_quaternion(self.roll, self.pitch, self.yaw)

        # --- gravity estimate in body frame ---
        if not hasattr(self, "g_body_est"):
            self.g_body_est = np.array([ax_ms2, ay_ms2, az_ms2], dtype=float)

        a_body_vec = np.array([ax_ms2, ay_ms2, az_ms2], dtype=float)

        # 只有接近靜止才更新重力估計
        if stationary:
            g_beta = 0.92
            self.g_body_est = g_beta * self.g_body_est + (1.0 - g_beta) * a_body_vec

        # 線性加速度
        a_body_lin = a_body_vec - self.g_body_est
        a_world_lin = body_to_world(q, a_body_lin)

        # 分軸 deadband：Z 軸放比較小，避免上抬被吃掉
        xy_deadband = ACC_DEADBAND_MS2
        z_deadband = 0.02
        # deadband
        a_world_lin = np.array([
            apply_deadband(a_world_lin[0], ACC_DEADBAND_MS2),
            apply_deadband(a_world_lin[1], ACC_DEADBAND_MS2),
            apply_deadband(a_world_lin[2], ACC_DEADBAND_MS2),
        ], dtype=float)

        # 若靜止，直接把速度夾回 0
        if stationary:
            # 靜止時直接清掉線性加速度與速度，位置不再更新
            a_world_lin[:] = 0.0
            self.v[:] = 0.0
        else:
            self.v += a_world_lin * dt
            self.v *= VEL_DAMPING
            self.p += self.v * dt
            self.p *= POS_DAMPING

        # 限制畫面不要飄太遠
        self.p = np.clip(self.p, -WORLD_BOX, WORLD_BOX)

        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "q": q,
            "a_norm": a_norm,
            "g_norm": g_norm,
            "m_norm": m_norm,
            "a_world": a_world_lin,
            "v": self.v.copy(),
            "p": self.p.copy(),
            "stationary": stationary,
            "bias_ready": self.bias_ready,
            "ax_ms2": ax_ms2,
            "ay_ms2": ay_ms2,
            "az_ms2": az_ms2,
            "gx_dps": gx_dps,
            "gy_dps": gy_dps,
            "gz_dps": gz_dps,
        }

    def _yaw_from_mag(self, mx, my, mz, roll_deg, pitch_deg):
        r = math.radians(roll_deg)
        p = math.radians(pitch_deg)

        mx2 = mx * math.cos(p) + mz * math.sin(p)
        my2 = (
            mx * math.sin(r) * math.sin(p)
            + my * math.cos(r)
            - mz * math.sin(r) * math.cos(p)
        )

        yaw = math.degrees(math.atan2(-my2, mx2))
        return yaw


# =========================
# UDP receiver thread
# =========================
def udp_receiver():
    estimator = MotionEstimator()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(("0.0.0.0", PORT))
        sock.settimeout(0.5)
    except OSError as e:
        print(f"[ERROR] bind failed on UDP {PORT}: {e}")
        return

    print(f"Listening UDP on port {PORT}...")

    csv_file = None
    csv_writer = None

    if SAVE_CSV:
        csv_file = open(CSV_FILENAME, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "recv_time", "timestamp",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "mx", "my", "mz",
            "roll", "pitch", "yaw",
            "ax_w", "ay_w", "az_w",
            "vx", "vy", "vz",
            "x", "y", "z",
            "a_norm", "g_norm", "m_norm",
            "stationary", "bias_ready"
        ])
        print(f"Saving CSV to {CSV_FILENAME}")

    last_ts = None
    last_wall = time.time()
    fps = 0.0

    try:
        while shared.running:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout: 
                continue

            msg = data.decode(errors="ignore").strip()
            parts = msg.split(",")

            if len(parts) != 10:
                print(f"[WARN] invalid packet from {addr}: {msg}")
                continue

            try:
                timestamp = int(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                gx = float(parts[4])
                gy = float(parts[5])
                gz = float(parts[6])
                mx = float(parts[7])
                my = float(parts[8])
                mz = float(parts[9])
            except ValueError:
                print(f"[WARN] parse error from {addr}: {msg}")
                continue

            if last_ts is None:
                dt = 0.01
            else:
                dt = (timestamp - last_ts) / 1000.0
                if dt <= 0 or dt > 0.2:
                    dt = 0.01
            last_ts = timestamp

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_wall)
            last_wall = now

            est = estimator.update(ax, ay, az, gx, gy, gz, mx, my, mz, dt)

            sample = IMUSample(
                recv_time=datetime.now().isoformat(timespec="milliseconds"),
                timestamp=timestamp,

                ax=est["ax_ms2"],
                ay=est["ay_ms2"],
                az=est["az_ms2"],

                gx=est["gx_dps"],
                gy=est["gy_dps"],
                gz=est["gz_dps"],

                mx=mx, my=my, mz=mz,

                roll=est["roll"],
                pitch=est["pitch"],
                yaw=est["yaw"],
                fps=fps,
                q=est["q"],

                a_norm=est["a_norm"],
                g_norm=est["g_norm"],
                m_norm=est["m_norm"],

                ax_w=float(est["a_world"][0]),
                ay_w=float(est["a_world"][1]),
                az_w=float(est["a_world"][2]),

                vx=float(est["v"][0]),
                vy=float(est["v"][1]),
                vz=float(est["v"][2]),

                x=float(est["p"][0]),
                y=float(est["p"][1]),
                z=float(est["p"][2]),

                stationary=bool(est["stationary"]),
                bias_ready=bool(est["bias_ready"]),
            )

            with shared.lock:
                shared.latest = sample
                shared.traj.append([sample.x, sample.y, sample.z])

            if csv_writer:
                csv_writer.writerow([
                    sample.recv_time, sample.timestamp,
                    sample.ax, sample.ay, sample.az,
                    sample.gx, sample.gy, sample.gz,
                    sample.mx, sample.my, sample.mz,
                    sample.roll, sample.pitch, sample.yaw,
                    sample.ax_w, sample.ay_w, sample.az_w,
                    sample.vx, sample.vy, sample.vz,
                    sample.x, sample.y, sample.z,
                    sample.a_norm, sample.g_norm, sample.m_norm,
                    int(sample.stationary), int(sample.bias_ready)
                ])

    finally:
        print("Closing UDP socket")
        sock.close()
        if csv_file:
            csv_file.close()


# =========================
# 3D helpers
# =========================
def create_cube(size=(0.12, 0.08, 0.03)):
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

    vertices = np.array([
        [-sx, -sy, -sz],
        [ sx, -sy, -sz],
        [ sx,  sy, -sz],
        [-sx,  sy, -sz],
        [-sx, -sy,  sz],
        [ sx, -sy,  sz],
        [ sx,  sy,  sz],
        [-sx,  sy,  sz],
    ], dtype=float)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4],  # left
    ], dtype=np.int32)

    colors = np.array([
        [0.55, 0.55, 0.55, 1.0],
        [0.55, 0.55, 0.55, 1.0],
        [1.00, 0.25, 0.25, 1.0],
        [1.00, 0.25, 0.25, 1.0],
        [0.25, 0.80, 1.00, 1.0],
        [0.25, 0.80, 1.00, 1.0],
        [0.25, 0.90, 0.35, 1.0],
        [0.25, 0.90, 0.35, 1.0],
        [1.00, 1.00, 0.25, 1.0],
        [1.00, 1.00, 0.25, 1.0],
        [0.75, 0.35, 1.00, 1.0],
        [0.75, 0.35, 1.00, 1.0],
    ], dtype=float)

    md = gl.MeshData(vertexes=vertices, faces=faces, faceColors=colors)
    cube = gl.GLMeshItem(
        meshdata=md,
        smooth=False,
        drawFaces=True,
        drawEdges=True,
        edgeColor=(1, 1, 1, 1)
    )
    return cube


def make_line_item(color=(1, 1, 1, 1), width=2):
    return gl.GLLinePlotItem(
        pos=np.zeros((2, 3), dtype=float),
        color=color,
        width=width,
        antialias=True,
        mode="lines"
    )


# =========================
# Main window
# =========================
class IMUViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU 9-Axis UDP 6DOF Viewer")
        self.resize(1500, 850)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # -----------------
        # Left panel
        # -----------------
        left_panel = QtWidgets.QFrame()
        left_panel.setMinimumWidth(430)
        left_panel.setStyleSheet("""
            QFrame {
                background: #1e1e1e;
                border-radius: 12px;
            }
            QLabel {
                color: #e8e8e8;
                font-family: Menlo, Consolas, monospace;
                font-size: 15px;
            }
        """)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)

        title = QtWidgets.QLabel("IMU 6DOF Monitor")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        left_layout.addWidget(title)

        self.lb_status = QtWidgets.QLabel("status: waiting for UDP packets...")
        self.lb_time = QtWidgets.QLabel("recv_time:")
        self.lb_ts = QtWidgets.QLabel("timestamp:")
        self.lb_fps = QtWidgets.QLabel("fps:")

        self.lb_acc = QtWidgets.QLabel("A_body = (0, 0, 0)")
        self.lb_gyr = QtWidgets.QLabel("G_body = (0, 0, 0)")
        self.lb_mag = QtWidgets.QLabel("M_body = (0, 0, 0)")

        self.lb_anorm = QtWidgets.QLabel("|a| = 0.00")
        self.lb_gnorm = QtWidgets.QLabel("|g| = 0.00")
        self.lb_mnorm = QtWidgets.QLabel("|m| = 0.00")

        self.lb_roll = QtWidgets.QLabel("roll : 0.00 deg")
        self.lb_pitch = QtWidgets.QLabel("pitch: 0.00 deg")
        self.lb_yaw = QtWidgets.QLabel("yaw  : 0.00 deg")

        self.lb_aw = QtWidgets.QLabel("A_world = (0, 0, 0)")
        self.lb_vel = QtWidgets.QLabel("V = (0, 0, 0)")
        self.lb_pos = QtWidgets.QLabel("P = (0, 0, 0)")

        self.lb_stationary = QtWidgets.QLabel("stationary : False")
        self.lb_bias = QtWidgets.QLabel("gyro bias  : calibrating...")

        labels = [
            self.lb_status, self.lb_time, self.lb_ts, self.lb_fps,
            self.lb_acc, self.lb_gyr, self.lb_mag,
            self.lb_anorm, self.lb_gnorm, self.lb_mnorm,
            self.lb_roll, self.lb_pitch, self.lb_yaw,
            self.lb_aw, self.lb_vel, self.lb_pos,
            self.lb_stationary, self.lb_bias
        ]
        for w in labels:
            left_layout.addWidget(w)

        left_layout.addStretch()

        note = QtWidgets.QLabel(
            "UDP format:\n"
            "timestamp,ax,ay,az,gx,gy,gz,mx,my,mz\n\n"
            "This viewer estimates short-term relative trajectory.\n"
            "Suitable for motion visualization / action features,\n"
            "not for long-term absolute navigation."
        )
        note.setStyleSheet("color: #b8b8b8; font-size: 13px;")
        left_layout.addWidget(note)

        layout.addWidget(left_panel, 0)

        # -----------------
        # Right: 3D view
        # -----------------
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=1.6, elevation=18, azimuth=42)
        self.view.setBackgroundColor((18, 18, 18))

        # world grid
        grid = gl.GLGridItem()
        grid.setSize(x=1.2, y=1.2)
        grid.setSpacing(x=0.1, y=0.1)
        self.view.addItem(grid)

        # world axis
        world_axis = gl.GLAxisItem()
        world_axis.setSize(0.35, 0.35, 0.35)
        self.view.addItem(world_axis)

        # IMU body
        self.cube = create_cube()
        self.view.addItem(self.cube)

        # IMU body axes
        self.axis_x = make_line_item(color=(1, 0, 0, 1), width=3)
        self.axis_y = make_line_item(color=(0, 1, 0, 1), width=3)
        self.axis_z = make_line_item(color=(0.2, 0.6, 1, 1), width=3)
        self.view.addItem(self.axis_x)
        self.view.addItem(self.axis_y)
        self.view.addItem(self.axis_z)

        # trajectory
        self.traj_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=float),
            color=(1.0, 0.9, 0.2, 1.0),
            width=2,
            antialias=True,
            mode="line_strip"
        )
        self.view.addItem(self.traj_line)

        layout.addWidget(self.view, 1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(30)

    def refresh_ui(self):
        with shared.lock:
            sample = shared.latest
            traj = np.array(shared.traj, dtype=float) if len(shared.traj) > 0 else None

        if sample is None:
            return

        self.lb_status.setText("status     : receiving")
        self.lb_time.setText(f"recv_time  : {sample.recv_time}")
        self.lb_ts.setText(f"timestamp  : {sample.timestamp}")
        self.lb_fps.setText(f"fps        : {sample.fps:7.2f}")

        self.lb_acc.setText(f"A_body     : ({sample.ax:8.3f}, {sample.ay:8.3f}, {sample.az:8.3f}) m/s²")
        self.lb_gyr.setText(f"G_body     : ({sample.gx:8.3f}, {sample.gy:8.3f}, {sample.gz:8.3f}) deg/s")
        self.lb_mag.setText(f"M_body     : ({sample.mx:8.3f}, {sample.my:8.3f}, {sample.mz:8.3f})")

        self.lb_anorm.setText(f"|a|        : {sample.a_norm:8.3f} m/s²")
        self.lb_gnorm.setText(f"|g|        : {sample.g_norm:8.3f} deg/s")
        self.lb_mnorm.setText(f"|m|        : {sample.m_norm:8.3f}")

        self.lb_roll.setText(f"roll       : {sample.roll:8.3f} deg")
        self.lb_pitch.setText(f"pitch      : {sample.pitch:8.3f} deg")
        self.lb_yaw.setText(f"yaw        : {sample.yaw:8.3f} deg")

        self.lb_aw.setText(f"A_world    : ({sample.ax_w:8.3f}, {sample.ay_w:8.3f}, {sample.az_w:8.3f}) m/s²")
        self.lb_vel.setText(f"V          : ({sample.vx:8.3f}, {sample.vy:8.3f}, {sample.vz:8.3f}) m/s")
        self.lb_pos.setText(f"P          : ({sample.x:8.3f}, {sample.y:8.3f}, {sample.z:8.3f}) m")

        self.lb_stationary.setText(f"stationary : {sample.stationary}")
        self.lb_bias.setText(f"gyro bias  : {'ready' if sample.bias_ready else 'calibrating... keep still at startup'}")

        self.update_body(sample)

        if traj is not None and len(traj) >= 2:
            self.traj_line.setData(pos=traj)

    def update_body(self, sample):
        q = sample.q
        if q is None:
            return

        p = np.array([sample.x, sample.y, sample.z], dtype=float)
        R = q_to_rotmat(q)

        # cube
        self.cube.resetTransform()
        axis, angle_deg = quaternion_to_axis_angle(q)
        self.cube.rotate(angle_deg, axis[0], axis[1], axis[2])
        self.cube.translate(p[0], p[1], p[2])

        # body axes: 以目前位置為原點
        ex = p + R @ np.array([BODY_AXIS_LEN, 0.0, 0.0], dtype=float)
        ey = p + R @ np.array([0.0, BODY_AXIS_LEN, 0.0], dtype=float)
        ez = p + R @ np.array([0.0, 0.0, BODY_AXIS_LEN], dtype=float)

        self.axis_x.setData(pos=np.vstack([p, ex]))
        self.axis_y.setData(pos=np.vstack([p, ey]))
        self.axis_z.setData(pos=np.vstack([p, ez]))

    def closeEvent(self, event):
        shared.running = False
        event.accept()


# =========================
# Main
# =========================
def main():
    recv_thread = threading.Thread(target=udp_receiver, daemon=False)
    recv_thread.start()

    app = QtWidgets.QApplication([])
    win = IMUViewer()
    win.show()
    app.exec_()

    shared.running = False
    recv_thread.join(timeout=2.0)
    print("Receiver thread stopped")


if __name__ == "__main__":
    main()