import socket
import struct
import threading
import math
import time
import numpy as np
import subprocess

from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

# Remove PyBluez dependency: we'll detect address via bluetoothctl/hcitool and use socket.AF_BLUETOOTH when available
try:
    AF_BLUETOOTH = socket.AF_BLUETOOTH
    BTPROTO_RFCOMM = socket.BTPROTO_RFCOMM
except Exception:
    AF_BLUETOOTH = None
    BTPROTO_RFCOMM = None

DEVICE_NAME = "luckfox"
BT_CHANNEL = 1
BT_DISCOVER_DURATION = 8  # seconds


# =========================================================
# 基本設定
# =========================================================
PORT = 5005

# 封包格式：
# uint32 timestamp_ms + 9 * int16
PACKET_FORMAT = "<I9h"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# -------------------------
# LSM9DS0 sensitivity
# -------------------------
ACC_LSB_PER_G = 16384.0              # ±2g
GYRO_DPS_PER_LSB = 0.00875           # ±245 dps
MAG_GAUSS_PER_LSB = 1.0 / 12500.0    # 粗略換算

# 是否使用磁力計修正 yaw
# 先設 False，避免磁力計未校正時讓 yaw 更亂
USE_MAG_YAW = False

# complementary filter 參數
ALPHA_RP = 0.96
ALPHA_YAW = 0.98

# 開始時校正 gyro 偏移
GYRO_CALIBRATION_SECONDS = 3.0

# 當 timestamp 不正常時用的預設 dt
DEFAULT_DT = 1.0 / 50.0


# =========================================================
# 共用資料區
# =========================================================
class SharedData:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.new_data = False

        # 姿態（deg）
        self.rpy = [0.0, 0.0, 0.0]

        # 四元數
        self.quaternion = [1.0, 0.0, 0.0, 0.0]

        # 這些欄位保留，但固定為 0
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.linear_acc_world = [0.0, 0.0, 0.0]
        self.trajectory = []

        self.raw = {
            "timestamp": 0,
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
            "mx": 0.0, "my": 0.0, "mz": 0.0,
            "packet_count": 0,
            "status": "INIT",
            "is_static": True
        }

        self.gyro_bias = [0.0, 0.0, 0.0]


shared = SharedData()


# =========================================================
# 工具函式
# =========================================================
def clamp(value, low, high):
    return max(low, min(high, value))


def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    Euler angles (deg) -> quaternion [w, x, y, z]
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y_, z], dtype=float)


def norm3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)


def calibrate_imu(sock, seconds=3.0):
    """
    保留原名但不再使用 UDP socket，這個函式會在 bt_worker 內被替換呼叫。
    """
    # fallback if called unexpectedly
    return np.array([0.0, 0.0, 0.0], dtype=float)


# =========================================================
# Bluetooth RFCOMM client 接收（替代原本的 udp_worker）
# =========================================================

def find_device_address_by_name(name):
    """Use bluetoothctl devices and hcitool scan as fallback to find MAC for a given device name."""
    try:
        r = subprocess.run(["bluetoothctl", "devices"], capture_output=True, text=True, timeout=5)
        out = r.stdout
        for line in out.splitlines():
            line = line.strip()
            # Expected: Device XX:XX:... Name
            if not line.startswith("Device "):
                continue
            parts = line.split(None, 2)
            if len(parts) >= 3:
                mac = parts[1]
                nm = parts[2]
                if nm.lower() == name.lower() or name.lower() in nm.lower():
                    return mac
    except Exception:
        pass

    # fallback: try hcitool scan (may require sudo)
    try:
        r = subprocess.run(["hcitool", "scan"], capture_output=True, text=True, timeout=10)
        out = r.stdout
        for line in out.splitlines():
            line = line.strip()
            # lines like: \tXX:XX:...\tName
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    mac = parts[1].strip()
                    nm = parts[2].strip()
                    if nm.lower() == name.lower() or name.lower() in nm.lower():
                        return mac
    except Exception:
        pass

    return None


def parse_imu_line(line):
    """
    解析來自板子的 CSV 行，回傳 (sensor_ts_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_uT, my_uT, mz_uT)

    假定格式為：serial,sensor_type,sensor_timestamp,host_monotonic_timestamp,<ppg...>,ax,ay,az,gx,gy,gz,mx,my,mz
    我們取最後 9 個欄位當作 imu 值；sensor_timestamp 嘗試解析為 microseconds 或 milliseconds。
    """
    parts = line.strip().split(',')
    if len(parts) < 9:
        raise ValueError("line too short")

    # 取最後 9 個欄位
    imu_fields = parts[-9:]
    vals = [float(x) for x in imu_fields]
    ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_uT, my_uT, mz_uT = vals

    # sensor timestamp 嘗試從 parts[2] 解析（若存在）
    sensor_ts_ms = 0
    try:
        if len(parts) > 2:
            sensor_ts = parts[2]
            # 有些實作使用微秒 (us)
            sensor_ts_val = float(sensor_ts)
            if sensor_ts_val > 1e12:
                # 可能是 ns? fallback
                sensor_ts_ms = int(sensor_ts_val / 1_000_000)
            elif sensor_ts_val > 1e9:
                # us -> ms
                sensor_ts_ms = int(sensor_ts_val / 1000)
            else:
                # already ms
                sensor_ts_ms = int(sensor_ts_val)
    except Exception:
        sensor_ts_ms = 0

    return sensor_ts_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_uT, my_uT, mz_uT


def calibrate_imu_from_reader(reader, seconds=3.0):
    print("\n[INFO] 開始 IMU 校正（Bluetooth），請保持 IMU 靜止...")
    print(f"[INFO] 校正時間：{seconds:.1f} 秒")

    gx_list = []
    gy_list = []
    gz_list = []

    start = time.time()
    while time.time() - start < seconds:
        try:
            line = reader.readline()
            if not line:
                time.sleep(0.01)
                continue
            try:
                _, _, _, _, gx, gy, gz, _, _, _ = parse_imu_line(line)
                gx_list.append(gx)
                gy_list.append(gy)
                gz_list.append(gz)
            except Exception:
                continue
        except Exception:
            time.sleep(0.01)
            continue

    if len(gx_list) < 10:
        print("[WARN] 校正資料太少，gyro bias 使用 0\n")
        return np.array([0.0, 0.0, 0.0], dtype=float)

    gyro_bias = np.array([np.mean(gx_list), np.mean(gy_list), np.mean(gz_list)], dtype=float)
    print("[INFO] IMU 校正完成")
    print(f"[INFO] gyro_bias (dps) = {gyro_bias}\n")
    return gyro_bias


def bt_worker():
    # 1) discover device address
    addr = find_device_address_by_name(DEVICE_NAME)
    if addr is None:
        print(f"[ERROR] Device named '{DEVICE_NAME}' not found. Run 'bluetoothctl devices' to inspect available devices.")
        return

    sock = None

    # 2) try native RFCOMM socket if supported by Python build
    if AF_BLUETOOTH is not None and BTPROTO_RFCOMM is not None:
        try:
            bt_sock = socket.socket(AF_BLUETOOTH, socket.SOCK_STREAM, BTPROTO_RFCOMM)
            print(f"[INFO] Connecting to {addr} channel {BT_CHANNEL} via socket.AF_BLUETOOTH ...")
            bt_sock.connect((addr, BT_CHANNEL))
            sock = bt_sock
            print("[INFO] RFCOMM connected via socket")
        except Exception as e:
            print(f"[WARN] RFCOMM socket connect failed: {e}")
            try:
                bt_sock.close()
            except Exception:
                pass
            sock = None

    # 3) fallback: try opening /dev/rfcomm0 if exists
    if sock is None:
        try:
            # if /dev/rfcomm0 exists, assume user bound it (e.g., via rfcomm bind)
            rfpath = "/dev/rfcomm0"
            with open(rfpath, "rb") as f:
                pass
            # open a file-like object for reading lines
            try:
                rf = open(rfpath, "r", encoding="utf-8", errors="ignore")
                reader = rf
                print(f"[INFO] Using serial device {rfpath} for RFCOMM data")
            except Exception as e:
                print(f"[ERROR] Cannot open {rfpath}: {e}")
                return

            # perform calibration and read loop using reader
            gyro_bias = calibrate_imu_from_reader(reader, GYRO_CALIBRATION_SECONDS)
            gx_bias, gy_bias, gz_bias = gyro_bias

            with shared.lock:
                shared.gyro_bias = [gx_bias, gy_bias, gz_bias]
                shared.raw['status'] = 'RUN'

            roll = pitch = yaw = 0.0
            last_ts = None
            packet_count = 0

            while shared.running:
                line = reader.readline()
                if not line:
                    time.sleep(0.002)
                    continue
                try:
                    ts_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_uT, my_uT, mz_uT = parse_imu_line(line)
                    if ts_ms == 0:
                        ts_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF

                    gx = gx_dps - gx_bias
                    gy = gy_dps - gy_bias
                    gz = gz_dps - gz_bias

                    if last_ts is None:
                        dt = DEFAULT_DT
                    else:
                        diff_ms = (ts_ms - last_ts) & 0xFFFFFFFF
                        dt = diff_ms / 1000.0
                        if dt <= 0.0 or dt > 0.2:
                            dt = DEFAULT_DT
                    last_ts = ts_ms
                    packet_count += 1

                    roll_acc = math.degrees(math.atan2(ay_g, az_g))
                    pitch_acc = math.degrees(math.atan2(-ax_g, math.sqrt(ay_g * ay_g + az_g * az_g)))

                    yaw_mag = yaw
                    if USE_MAG_YAW:
                        roll_rad = math.radians(roll_acc)
                        pitch_rad = math.radians(pitch_acc)

                        mx2 = mx_uT * math.cos(pitch_rad) + mz_uT * math.sin(pitch_rad)
                        my2 = (
                            mx_uT * math.sin(roll_rad) * math.sin(pitch_rad)
                            + my_uT * math.cos(roll_rad)
                            - mz_uT * math.sin(roll_rad) * math.cos(pitch_rad)
                        )
                        yaw_mag = math.degrees(math.atan2(-my2, mx2))

                    roll = ALPHA_RP * (roll + gx * dt) + (1.0 - ALPHA_RP) * roll_acc
                    pitch = ALPHA_RP * (pitch + gy * dt) + (1.0 - ALPHA_RP) * pitch_acc

                    if USE_MAG_YAW:
                        yaw = ALPHA_YAW * (yaw + gz * dt) + (1.0 - ALPHA_YAW) * yaw_mag
                    else:
                        yaw = yaw + gz * dt

                    q_display = euler_to_quaternion(roll, pitch, yaw)

                    acc_norm_g = norm3(ax_g, ay_g, az_g)
                    gyro_norm_dps = norm3(gx, gy, gz)

                    is_static = (
                        abs(acc_norm_g - 1.0) < 0.05 and
                        gyro_norm_dps < 3.0
                    )

                    with shared.lock:
                        shared.rpy = [roll, pitch, yaw]
                        shared.quaternion = q_display.tolist()
                        shared.position = [0.0, 0.0, 0.0]
                        shared.velocity = [0.0, 0.0, 0.0]
                        shared.linear_acc_world = [0.0, 0.0, 0.0]
                        shared.trajectory = []

                        shared.raw = {
                            'timestamp': ts_ms,
                            'ax': ax_g, 'ay': ay_g, 'az': az_g,
                            'gx': gx, 'gy': gy, 'gz': gz,
                            'mx': mx_uT, 'my': my_uT, 'mz': mz_uT,
                            'packet_count': packet_count,
                            'status': 'RUN',
                            'is_static': is_static
                        }
                        shared.new_data = True

                except Exception as e:
                    print(f"[WARN] Parse/processing error (rfcomm): {e}")
                    continue

            return
        except FileNotFoundError:
            # no /dev/rfcomm0, continue to try socket method
            pass

    # 4) if we have a socket from AF_BLUETOOTH, use it for line reading
    if sock is not None:
        try:
            reader = sock.makefile('r', encoding='utf-8', newline='\n')

            gyro_bias = calibrate_imu_from_reader(reader, GYRO_CALIBRATION_SECONDS)
            gx_bias, gy_bias, gz_bias = gyro_bias

            with shared.lock:
                shared.gyro_bias = [gx_bias, gy_bias, gz_bias]
                shared.raw['status'] = 'RUN'

            roll = pitch = yaw = 0.0
            last_ts = None
            packet_count = 0

            while shared.running:
                line = reader.readline()
                if not line:
                    time.sleep(0.002)
                    continue

                try:
                    ts_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_uT, my_uT, mz_uT = parse_imu_line(line)
                    if ts_ms == 0:
                        ts_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF

                    gx = gx_dps - gx_bias
                    gy = gy_dps - gy_bias
                    gz = gz_dps - gz_bias

                    if last_ts is None:
                        dt = DEFAULT_DT
                    else:
                        diff_ms = (ts_ms - last_ts) & 0xFFFFFFFF
                        dt = diff_ms / 1000.0
                        if dt <= 0.0 or dt > 0.2:
                            dt = DEFAULT_DT
                    last_ts = ts_ms
                    packet_count += 1

                    roll_acc = math.degrees(math.atan2(ay_g, az_g))
                    pitch_acc = math.degrees(math.atan2(-ax_g, math.sqrt(ay_g * ay_g + az_g * az_g)))

                    yaw_mag = yaw
                    if USE_MAG_YAW:
                        roll_rad = math.radians(roll_acc)
                        pitch_rad = math.radians(pitch_acc)

                        mx2 = mx_uT * math.cos(pitch_rad) + mz_uT * math.sin(pitch_rad)
                        my2 = (
                            mx_uT * math.sin(roll_rad) * math.sin(pitch_rad)
                            + my_uT * math.cos(roll_rad)
                            - mz_uT * math.sin(roll_rad) * math.cos(pitch_rad)
                        )
                        yaw_mag = math.degrees(math.atan2(-my2, mx2))

                    roll = ALPHA_RP * (roll + gx * dt) + (1.0 - ALPHA_RP) * roll_acc
                    pitch = ALPHA_RP * (pitch + gy * dt) + (1.0 - ALPHA_RP) * pitch_acc

                    if USE_MAG_YAW:
                        yaw = ALPHA_YAW * (yaw + gz * dt) + (1.0 - ALPHA_YAW) * yaw_mag
                    else:
                        yaw = yaw + gz * dt

                    q_display = euler_to_quaternion(roll, pitch, yaw)

                    acc_norm_g = norm3(ax_g, ay_g, az_g)
                    gyro_norm_dps = norm3(gx, gy, gz)

                    is_static = (
                        abs(acc_norm_g - 1.0) < 0.05 and
                        gyro_norm_dps < 3.0
                    )

                    with shared.lock:
                        shared.rpy = [roll, pitch, yaw]
                        shared.quaternion = q_display.tolist()
                        shared.position = [0.0, 0.0, 0.0]
                        shared.velocity = [0.0, 0.0, 0.0]
                        shared.linear_acc_world = [0.0, 0.0, 0.0]
                        shared.trajectory = []

                        shared.raw = {
                            'timestamp': ts_ms,
                            'ax': ax_g, 'ay': ay_g, 'az': az_g,
                            'gx': gx, 'gy': gy, 'gz': gz,
                            'mx': mx_uT, 'my': my_uT, 'mz': mz_uT,
                            'packet_count': packet_count,
                            'status': 'RUN',
                            'is_static': is_static
                        }
                        shared.new_data = True

                except Exception as e:
                    print(f"[WARN] Parse/processing error (socket): {e}")
                    continue

        finally:
            try:
                sock.close()
            except Exception:
                pass

    print("[INFO] BT worker stopped")


# =========================================================
# 3D Viewer
# =========================================================
class IMUViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU UDP Receiver 3D Rotation Only")
        self.resize(1200, 850)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=6.0, elevation=20, azimuth=35)
        self.setCentralWidget(self.view)

        # 地板
        grid = gl.GLGridItem()
        grid.scale(0.5, 0.5, 0.5)
        self.view.addItem(grid)

        # 世界座標軸
        axis = gl.GLAxisItem()
        axis.setSize(1.0, 1.0, 1.0)
        self.view.addItem(axis)

        # ---------------------------------
        # IMU 盒子
        # ---------------------------------
        verts = np.array([
            [-0.5, -0.20, -0.05],
            [ 0.5, -0.20, -0.05],
            [ 0.5,  0.20, -0.05],
            [-0.5,  0.20, -0.05],
            [-0.5, -0.20,  0.05],
            [ 0.5, -0.20,  0.05],
            [ 0.5,  0.20,  0.05],
            [-0.5,  0.20,  0.05],
        ], dtype=float)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 3, 7], [0, 7, 4],
        ], dtype=int)

        colors = np.array([[0.2, 0.7, 0.9, 0.85]] * len(faces), dtype=float)

        self.box = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors,
            drawEdges=True,
            edgeColor=(1, 1, 1, 1),
            smooth=False,
        )
        self.view.addItem(self.box)

        # timer 更新畫面
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(16)

    def update_view(self):
        with shared.lock:
            if not shared.new_data:
                return

            roll, pitch, yaw = shared.rpy
            q = shared.quaternion.copy()
            raw = shared.raw.copy()
            bias = shared.gyro_bias.copy()
            shared.new_data = False

        self.setWindowTitle(
            f"pkts={raw['packet_count']} | "
            f"ts={raw['timestamp']} | "
            f"static={raw['is_static']} | "
            f"roll={roll:+.1f} | pitch={pitch:+.1f} | yaw={yaw:+.1f} deg | "
            f"gyro=({raw['gx']:+.2f},{raw['gy']:+.2f},{raw['gz']:+.2f}) dps"
        )

        # quaternion -> axis-angle
        w, x, y_, z = q
        w = clamp(w, -1.0, 1.0)

        angle_rad = 2.0 * math.acos(w)
        s = math.sqrt(max(1e-10, 1.0 - w * w))

        axis_x = x / s
        axis_y = y_ / s
        axis_z = z / s

        # 重設變換
        self.box.resetTransform()

        # 只旋轉，不平移
        self.box.rotate(math.degrees(angle_rad), axis_x, axis_y, axis_z)

    def closeEvent(self, event):
        shared.running = False
        event.accept()


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    worker = threading.Thread(target=bt_worker, daemon=True)
    worker.start()

    app = QtWidgets.QApplication([])
    win = IMUViewer()
    win.show()
    app.exec_()