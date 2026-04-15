import socket
import csv
import math
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from collections import deque

import numpy as np

# UI 與 3D 繪圖庫
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

# =========================
# 設定參數 (Config)
# =========================
PORT = 10000                      # UDP 監聽連接埠
SAVE_CSV = True                   # 是否存檔
CSV_FILENAME = f"imu_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

MAX_TRAJ_POINTS = 300             # 軌跡保留點數
USE_MAG_YAW = False               # 如果你的 IMU 沒磁力計，請設為 False

# 單位轉換參數 (根據你的感測器手冊調整)
ACC_LSB_PER_G = 16384.0           # ±2g 模式下的比例
GYRO_LSB_PER_DPS = 16.4           # ±2000dps 模式下的比例

# 運動濾波參數
VEL_DAMPING = 0.985               # 速度衰減 (避免飄移)
ACC_DEADBAND_MS2 = 0.15           # 加速度死區 (過濾細微雜訊)
WORLD_BOX = 1.0                   # 畫面顯示範圍限制

# =========================
# 數據結構與共享狀態
# =========================
@dataclass
class IMUSample:
    recv_time: str
    timestamp: int
    ax: float; ay: float; az: float
    gx: float; gy: float; gz: float
    mx: float; my: float; mz: float
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0
    fps: float = 0.0
    q: np.ndarray = None
    x: float = 0.0; y: float = 0.0; z: float = 0.0
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
# 數學運算工具 (Math Utils)
# =========================
def q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

def euler_to_quaternion(r_deg, p_deg, y_deg):
    r, p, y = np.radians([r_deg, p_deg, y_deg])
    cr, cp, cy = np.cos([r, p, y] / 2)
    sr, sp, sy = np.sin([r, p, y] / 2)
    q = np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        sy*cp*sr + cy*sp*cr,
        sy*cp*cr - cy*sp*sr
    ])
    return q_normalize(q)

def q_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x**2+y**2)]
    ])

def quaternion_to_axis_angle(q):
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1e-12, 1.0 - w*w))
    axis = q[1:4]/s if s > 1e-6 else np.array([1.0, 0.0, 0.0])
    return axis, math.degrees(angle)

# =========================
# 姿態與運動估算器 (Estimator)
# =========================
class MotionEstimator:
    def __init__(self, alpha=0.98):
        self.roll = 0.0; self.pitch = 0.0; self.yaw = 0.0
        self.alpha = alpha
        self.v = np.zeros(3); self.p = np.zeros(3)
        self.bias = np.zeros(3); self.bias_samples = []
        self.bias_ready = False

    def update(self, raw_data, dt):
        ts, ax, ay, az, gx, gy, gz, mx, my, mz = raw_data
        
        # 1. 單位轉換
        a_ms2 = np.array([ax, ay, az]) * (9.806 / ACC_LSB_PER_G)
        g_dps = np.array([gx, gy, gz]) / GYRO_LSB_PER_DPS
        
        # 2. 靜止校準
        if not self.bias_ready:
            self.bias_samples.append(g_dps)
            if len(self.bias_samples) >= 100:
                self.bias = np.mean(self.bias_samples, axis=0)
                self.bias_ready = True
            return None # 校準中不更新
        
        g_dps -= self.bias
        
        # 3. 互補濾波姿態計算
        r_acc = math.degrees(math.atan2(ay, az))
        p_acc = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        
        self.roll = self.alpha * (self.roll + g_dps[0]*dt) + (1-self.alpha) * r_acc
        self.pitch = self.alpha * (self.pitch + g_dps[1]*dt) + (1-self.alpha) * p_acc
        self.yaw += g_dps[2] * dt # 簡單積分，未加入磁力計修正
        
        q = euler_to_quaternion(self.roll, self.pitch, self.yaw)
        
        # 4. 線性位移估算 (簡化版)
        stationary = np.linalg.norm(g_dps) < 1.5 and abs(np.linalg.norm(a_ms2)-9.8) < 0.25
        if stationary:
            self.v *= 0.0
        else:
            # 移除重力分量 (假設 Z 軸向上)
            a_world = q_to_rotmat(q) @ a_ms2 - np.array([0, 0, 9.8])
            for i in range(3):
                if abs(a_world[i]) > ACC_DEADBAND_MS2:
                    self.v[i] += a_world[i] * dt
            self.v *= VEL_DAMPING
            self.p += self.v * dt
            self.p = np.clip(self.p, -WORLD_BOX, WORLD_BOX)

        return {"q": q, "p": self.p, "rpy": (self.roll, self.pitch, self.yaw), "stat": stationary}

# =========================
# UDP 接收執行緒
# =========================
def udp_receiver():
    estimator = MotionEstimator()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(0.5)
    
    last_ts = None
    print(f"UDP 監聽中... Port: {PORT}")

    while shared.running:
        try:
            data, addr = sock.recvfrom(2048)
            parts = [float(x) for x in data.decode().strip().split(',')]
            if len(parts) < 10: continue
            
            curr_ts = parts[0]
            dt = (curr_ts - last_ts)/1000.0 if last_ts else 0.01
            last_ts = curr_ts
            
            res = estimator.update(parts, dt)
            if res:
                sample = IMUSample(
                    recv_time=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    timestamp=int(curr_ts),
                    ax=parts[1], ay=parts[2], az=parts[3],
                    gx=parts[4], gy=parts[5], gz=parts[6],
                    mx=parts[7], my=parts[8], mz=parts[9],
                    roll=res["rpy"][0], pitch=res["rpy"][1], yaw=res["rpy"][2],
                    q=res["q"], x=res["p"][0], y=res["p"][1], z=res["p"][2],
                    stationary=res["stat"], bias_ready=estimator.bias_ready
                )
                with shared.lock:
                    shared.latest = sample
                    shared.traj.append([sample.x, sample.y, sample.z])
        except: continue
    sock.close()

# =========================
# UI 介面
# =========================
class IMUViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Luckfox Pico 9-Axis 6DOF Monitor")
        self.resize(1200, 800)
        
        # 佈局設定
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        
        # 左側面板 (數據)
        self.label = QtWidgets.QLabel("等待數據...")
        self.label.setStyleSheet("font-family: monospace; font-size: 14px; background: #222; color: #eee; padding: 10px;")
        layout.addWidget(self.label, 1)
        
        # 右側 3D 視窗
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((30, 30, 30))
        layout.addWidget(self.view, 3)
        
        # 3D 物件
        self.view.addItem(gl.GLGridItem())
        # 建立開發板模型 (長方體)
        verts = np.array([[-0.2,-0.1,-0.02],[0.2,-0.1,-0.02],[0.2,0.1,-0.02],[-0.2,0.1,-0.02],
                          [-0.2,-0.1,0.02],[0.2,-0.1,0.02],[0.2,0.1,0.02],[-0.2,0.1,0.02]])
        faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]])
        self.mesh = gl.GLMeshItem(vertexes=verts, faces=faces, color=(0, 0.7, 0.9, 0.8), drawEdges=True)
        self.view.addItem(self.mesh)
        
        self.traj = gl.GLLinePlotItem(color=(1,1,0,1), width=2)
        self.view.addItem(self.traj)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)

    def update_ui(self):
        with shared.lock:
            s = shared.latest
            t_data = np.array(shared.traj)
            
        if not s: return
        
        # 更新文字
        self.label.setText(f"Time: {s.recv_time}\n"
                           f"Roll:  {s.roll:6.1f}°\nPitch: {s.pitch:6.1f}°\nYaw:   {s.yaw:6.1f}°\n\n"
                           f"Pos X: {s.x:6.3f}m\nPos Y: {s.y:6.3f}m\nPos Z: {s.z:6.3f}m\n\n"
                           f"Status: {'靜止' if s.stationary else '移動中'}\n"
                           f"Bias:   {'已就緒' if s.bias_ready else '校準中...'}")
        
        # 更新 3D
        self.mesh.resetTransform()
        axis, angle = quaternion_to_axis_angle(s.q)
        self.mesh.rotate(angle, axis[0], axis[1], axis[2])
        self.mesh.translate(s.x, s.y, s.z)
        
        if len(t_data) > 1:
            self.traj.setData(pos=t_data)

if __name__ == "__main__":
    t = threading.Thread(target=udp_receiver, daemon=True)
    t.start()
    app = QtWidgets.QApplication([])
    win = IMUViewer()
    win.show()
    app.exec_()
    shared.running = False