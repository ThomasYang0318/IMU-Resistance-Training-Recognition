import socket
import csv
import math
import time
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl


# =========================
# Config
# =========================
PORT = 10000
SAVE_CSV = True
CSV_FILENAME = f"imu9axis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# =========================
# Shared data
# =========================
@dataclass
class IMUSample:
    recv_time: str
    timestamp: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    fps: float = 0.0
    q: np.ndarray = None


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.running = True


shared = SharedState()


# =========================
# Math utils
# =========================
def q_normalize(q):
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


def q_from_axis_angle(axis, angle_rad):
    axis = np.array(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis /= n
    s = math.sin(angle_rad / 2.0)
    return q_normalize(np.array([math.cos(angle_rad / 2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float))


def q_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)


def rotmat_to_euler_zyx(R):
    # yaw(Z), pitch(Y), roll(X)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)

    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


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
    w = max(-1.0, min(1.0, q[0]))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1e-12, 1.0 - w*w))

    if s < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = np.array([q[1]/s, q[2]/s, q[3]/s], dtype=float)

    return axis, math.degrees(angle)


# =========================
# Orientation estimator
# =========================
class ComplementaryOrientation:
    """
    簡化版：
    - roll / pitch: gyro + accel complementary filter
    - yaw: gyro + tilt-compensated magnetometer complementary filter
    """
    def __init__(self, alpha_rp=0.98, alpha_yaw=0.95): #alpha_rp=0.98, alpha_yaw=0.95
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.alpha_rp = alpha_rp
        self.alpha_yaw = alpha_yaw
        self.initialized = False


    def update(self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
        # gyro low-pass
        beta = 0.7
        self.gx_f = beta * getattr(self, "gx_f", gx) + (1-beta) * gx
        self.gy_f = beta * getattr(self, "gy_f", gy) + (1-beta) * gy
        self.gz_f = beta * getattr(self, "gz_f", gz) + (1-beta) * gz

        gx = self.gx_f
        gy = self.gy_f
        gz = self.gz_f

        if dt <= 0:
            dt = 0.01
        dt = min(max(dt, 0.001), 0.05)

        # 假設 gyro 單位是 deg/s
        if not self.initialized:
            roll_acc = math.degrees(math.atan2(ay, az))
            pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az) + 1e-12))
            yaw_mag = self._yaw_from_mag(mx, my, mz, roll_acc, pitch_acc)
            self.roll = roll_acc
            self.pitch = pitch_acc
            self.yaw = yaw_mag
            self.initialized = True
        else:
            # Gyro integration
            self.roll += gx * dt
            self.pitch += gy * dt
            self.yaw += gz * dt

            # Accel correction
            roll_acc = math.degrees(math.atan2(ay, az))
            pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az) + 1e-12))

            self.roll = self.alpha_rp * self.roll + (1 - self.alpha_rp) * roll_acc
            self.pitch = self.alpha_rp * self.pitch + (1 - self.alpha_rp) * pitch_acc

            # Magnetometer correction
            yaw_mag = self._yaw_from_mag(mx, my, mz, self.roll, self.pitch)
            yaw_err = yaw_mag - self.yaw
            while yaw_err > 180:
                yaw_err -= 360
            while yaw_err < -180:
                yaw_err += 360
            # self.yaw += (1 - self.alpha_yaw) * yaw_err

        q = euler_to_quaternion(self.roll, self.pitch, self.yaw)
        return self.roll, self.pitch, self.yaw, q

    def _yaw_from_mag(self, mx, my, mz, roll_deg, pitch_deg):
        r = math.radians(roll_deg)
        p = math.radians(pitch_deg)

        # tilt compensation
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
    estimator = ComplementaryOrientation()

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
            "recv_time",
            "timestamp",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "mx", "my", "mz",
            "roll", "pitch", "yaw"
        ])
        print(f"Saving to {CSV_FILENAME}")

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

            # 依你目前的資料格式，timestamp 看起來像毫秒
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

            roll, pitch, yaw, q = estimator.update(
                ax, ay, az,
                gx, gy, gz,
                mx, my, mz,
                dt
            )

            sample = IMUSample(
                recv_time=datetime.now().isoformat(timespec="milliseconds"),
                timestamp=timestamp,
                ax=ax, ay=ay, az=az,
                gx=gx, gy=gy, gz=gz,
                mx=mx, my=my, mz=mz,
                roll=roll, pitch=pitch, yaw=yaw,
                fps=fps,
                q=q
            )

            with shared.lock:
                shared.latest = sample

            if csv_writer:
                csv_writer.writerow([
                    sample.recv_time,
                    timestamp,
                    ax, ay, az,
                    gx, gy, gz,
                    mx, my, mz,
                    roll, pitch, yaw
                ])

    finally:
        print("Closing UDP socket")
        sock.close()
        if csv_file:
            csv_file.close()


# =========================
# 3D widget helpers
# =========================
def create_cube(size=(2.0, 1.0, 0.3)):
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
        [0.7, 0.7, 0.7, 1.0],
        [0.7, 0.7, 0.7, 1.0],
        [1.0, 0.2, 0.2, 1.0],  # top red
        [1.0, 0.2, 0.2, 1.0],
        [0.2, 0.8, 1.0, 1.0],
        [0.2, 0.8, 1.0, 1.0],
        [0.2, 0.8, 0.3, 1.0],
        [0.2, 0.8, 0.3, 1.0],
        [1.0, 1.0, 0.2, 1.0],
        [1.0, 1.0, 0.2, 1.0],
        [0.8, 0.3, 1.0, 1.0],
        [0.8, 0.3, 1.0, 1.0],
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


# =========================
# Main window
# =========================
class IMUViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU 9-Axis UDP Realtime Viewer")
        self.resize(1400, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # Left panel
        left_panel = QtWidgets.QFrame()
        left_panel.setMinimumWidth(360)
        left_panel.setStyleSheet("""
            QFrame {
                background: #1e1e1e;
                border-radius: 12px;
            }
            QLabel {
                color: #e8e8e8;
                font-family: Menlo, Consolas, monospace;
                font-size: 16px;
            }
        """)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)

        title = QtWidgets.QLabel("Current 9-Axis Data")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        left_layout.addWidget(title)

        self.lb_status = QtWidgets.QLabel("status: waiting for UDP packets...")
        self.lb_time = QtWidgets.QLabel("recv_time:")
        self.lb_ts = QtWidgets.QLabel("timestamp:")
        self.lb_fps = QtWidgets.QLabel("fps:")

        self.lb_acc = QtWidgets.QLabel("A = (0, 0, 0)")
        self.lb_gyr = QtWidgets.QLabel("G = (0, 0, 0)")
        self.lb_mag = QtWidgets.QLabel("M = (0, 0, 0)")

        self.lb_roll = QtWidgets.QLabel("roll : 0.00 deg")
        self.lb_pitch = QtWidgets.QLabel("pitch: 0.00 deg")
        self.lb_yaw = QtWidgets.QLabel("yaw  : 0.00 deg")

        for w in [
            self.lb_status, self.lb_time, self.lb_ts, self.lb_fps,
            self.lb_acc, self.lb_gyr, self.lb_mag,
            self.lb_roll, self.lb_pitch, self.lb_yaw
        ]:
            left_layout.addWidget(w)

        left_layout.addStretch()

        tip = QtWidgets.QLabel(
            "UDP format:\n"
            "timestamp,ax,ay,az,gx,gy,gz,mx,my,mz"
        )
        tip.setStyleSheet("color: #b0b0b0; font-size: 14px;")
        left_layout.addWidget(tip)

        layout.addWidget(left_panel, 0)

        # Right panel: 3D view
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=8, elevation=18, azimuth=45)
        self.view.setBackgroundColor((20, 20, 20))

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        grid.setSize(10, 10)
        self.view.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(3, 3, 3)
        self.view.addItem(axis)

        self.cube = create_cube()
        self.view.addItem(self.cube)

        layout.addWidget(self.view, 1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(30)  # ~33 FPS

    def refresh_ui(self):
        with shared.lock:
            sample = shared.latest

        if sample is None:
            return

        self.lb_status.setText("status: receiving")
        self.lb_time.setText(f"recv_time : {sample.recv_time}")
        self.lb_ts.setText(f"timestamp : {sample.timestamp}")
        self.lb_fps.setText(f"fps       : {sample.fps:7.2f}")

        self.lb_acc.setText(f"A = ({sample.ax:8.2f}, {sample.ay:8.2f}, {sample.az:8.2f})")
        self.lb_gyr.setText(f"G = ({sample.gx:8.2f}, {sample.gy:8.2f}, {sample.gz:8.2f})")
        self.lb_mag.setText(f"M = ({sample.mx:8.2f}, {sample.my:8.2f}, {sample.mz:8.2f})")

        self.lb_roll.setText(f"roll : {sample.roll:8.2f} deg")
        self.lb_pitch.setText(f"pitch: {sample.pitch:8.2f} deg")
        self.lb_yaw.setText(f"yaw  : {sample.yaw:8.2f} deg")

        self.update_cube(sample.q)

    def update_cube(self, q):
        if q is None:
            return

        self.cube.resetTransform()

        axis, angle_deg = quaternion_to_axis_angle(q)
        self.cube.rotate(angle_deg, axis[0], axis[1], axis[2])

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