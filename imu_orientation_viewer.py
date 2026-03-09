import socket
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore


UDP_IP = "0.0.0.0"
UDP_PORT = 10000

WINDOW_SEC = 5
FS_EST = 100
MAXLEN = WINDOW_SEC * FS_EST

# ----- LSM9DS0 scale assumptions -----
ACC_LSB_PER_G = 16384.0       # ±2g
GYRO_LSB_PER_DPS = 8.75       # mdps/LSB -> 0.00875 dps/LSB => 1 dps = 114.2857 LSB
GYRO_DPS_PER_LSB = 0.00875
MAG_LSB_SCALE = 1.0           # 先直接用 raw，Madgwick 只需要方向


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def quat_multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)


def quat_to_euler_deg(q):
    w, x, y, z = q

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.degrees(np.sign(sinp) * (np.pi / 2))
    else:
        pitch = np.degrees(np.arcsin(sinp))

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


class MadgwickAHRS:
    def __init__(self, beta=0.08):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def update(self, gx, gy, gz, ax, ay, az, mx, my, mz, dt):
        q1, q2, q3, q4 = self.q  # w, x, y, z

        # normalize accel
        a = np.array([ax, ay, az], dtype=float)
        if np.linalg.norm(a) < 1e-12:
            return self.q
        a = normalize(a)
        ax, ay, az = a

        # normalize mag
        m = np.array([mx, my, mz], dtype=float)
        if np.linalg.norm(m) < 1e-12:
            return self.q
        m = normalize(m)
        mx, my, mz = m

        # reference direction of earth's magnetic field
        q = self.q
        h = quat_multiply(q, quat_multiply(np.array([0.0, mx, my, mz]), np.array([q[0], -q[1], -q[2], -q[3]])))
        bx = np.sqrt(h[1]*h[1] + h[2]*h[2])
        bz = h[3]

        # objective function
        f = np.array([
            2*(q2*q4 - q1*q3) - ax,
            2*(q1*q2 + q3*q4) - ay,
            2*(0.5 - q2*q2 - q3*q3) - az,
            2*bx*(0.5 - q3*q3 - q4*q4) + 2*bz*(q2*q4 - q1*q3) - mx,
            2*bx*(q2*q3 - q1*q4) + 2*bz*(q1*q2 + q3*q4) - my,
            2*bx*(q1*q3 + q2*q4) + 2*bz*(0.5 - q2*q2 - q3*q3) - mz
        ], dtype=float)

        # Jacobian
        J = np.array([
            [-2*q3,               2*q4,              -2*q1,               2*q2],
            [ 2*q2,               2*q1,               2*q4,               2*q3],
            [ 0.0,               -4*q2,              -4*q3,               0.0],
            [-2*bz*q3,            2*bz*q4,           -4*bx*q3-2*bz*q1,   -4*bx*q4+2*bz*q2],
            [-2*bx*q4+2*bz*q2,    2*bx*q3+2*bz*q1,    2*bx*q2+2*bz*q4,   -2*bx*q1+2*bz*q3],
            [ 2*bx*q3,            2*bx*q4-4*bz*q2,    2*bx*q1-4*bz*q3,    2*bx*q2]
        ], dtype=float)

        step = J.T @ f
        n = np.linalg.norm(step)
        if n > 1e-12:
            step = step / n

        # quaternion derivative from gyro
        q_dot_omega = 0.5 * quat_multiply(self.q, np.array([0.0, gx, gy, gz], dtype=float))
        q_dot = q_dot_omega - self.beta * step

        self.q = self.q + q_dot * dt
        self.q = normalize(self.q)
        return self.q


class OrientationViewer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.ts_buf = deque(maxlen=MAXLEN)
        self.ax_buf = deque(maxlen=MAXLEN)
        self.ay_buf = deque(maxlen=MAXLEN)
        self.az_buf = deque(maxlen=MAXLEN)
        self.gx_buf = deque(maxlen=MAXLEN)
        self.gy_buf = deque(maxlen=MAXLEN)
        self.gz_buf = deque(maxlen=MAXLEN)
        self.mx_buf = deque(maxlen=MAXLEN)
        self.my_buf = deque(maxlen=MAXLEN)
        self.mz_buf = deque(maxlen=MAXLEN)

        self.rate_count = 0
        self.rate_t0 = time.time()
        self.current_hz = 0.0

        self.prev_ts = None
        self.ahrs = MadgwickAHRS(beta=0.08)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.latest_raw = [0.0] * 9

        self.app = QtWidgets.QApplication([])

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("IMU Orientation Viewer")
        self.win.resize(1500, 900)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QGridLayout()
        central.setLayout(layout)

        # ----- left: plots -----
        self.plot_widget = pg.GraphicsLayoutWidget()

        self.p1 = self.plot_widget.addPlot(title="Accelerometer")
        self.p1.showGrid(x=True, y=True)
        self.p1.addLegend()
        self.c_ax = self.p1.plot(pen='r', name='ax')
        self.c_ay = self.p1.plot(pen='g', name='ay')
        self.c_az = self.p1.plot(pen='b', name='az')

        self.plot_widget.nextRow()
        self.p2 = self.plot_widget.addPlot(title="Gyroscope")
        self.p2.showGrid(x=True, y=True)
        self.p2.addLegend()
        self.c_gx = self.p2.plot(pen='r', name='gx')
        self.c_gy = self.p2.plot(pen='g', name='gy')
        self.c_gz = self.p2.plot(pen='b', name='gz')

        self.plot_widget.nextRow()
        self.p3 = self.plot_widget.addPlot(title="Magnetometer")
        self.p3.showGrid(x=True, y=True)
        self.p3.addLegend()
        self.c_mx = self.p3.plot(pen='r', name='mx')
        self.c_my = self.p3.plot(pen='g', name='my')
        self.c_mz = self.p3.plot(pen='b', name='mz')

        # ----- right: 3D -----
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=7)
        self.gl_view.opts['elevation'] = 20
        self.gl_view.opts['azimuth'] = 40

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        self.gl_view.addItem(grid)

        self.world_x = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
            color=(1, 0, 0, 1), width=3, antialias=True
        )
        self.world_y = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 2, 0]], dtype=float),
            color=(0, 1, 0, 1), width=3, antialias=True
        )
        self.world_z = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 2]], dtype=float),
            color=(0, 0, 1, 1), width=3, antialias=True
        )
        self.gl_view.addItem(self.world_x)
        self.gl_view.addItem(self.world_y)
        self.gl_view.addItem(self.world_z)

        # IMU body mesh
        verts = np.array([
            [-0.8, -0.4, -0.1],
            [ 0.8, -0.4, -0.1],
            [ 0.8,  0.4, -0.1],
            [-0.8,  0.4, -0.1],
            [-0.8, -0.4,  0.1],
            [ 0.8, -0.4,  0.1],
            [ 0.8,  0.4,  0.1],
            [-0.8,  0.4,  0.1],
        ], dtype=float)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 3, 7], [0, 7, 4],
        ])

        face_colors = np.tile(np.array([[0.7, 0.75, 0.9, 0.8]]), (faces.shape[0], 1))
        mesh_data = gl.MeshData(vertexes=verts, faces=faces, faceColors=face_colors)
        self.imu_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawEdges=True,
            edgeColor=(1, 1, 1, 1)
        )
        self.gl_view.addItem(self.imu_mesh)

        # body axes
        self.body_x = gl.GLLinePlotItem(color=(1, 0.4, 0.4, 1), width=4, antialias=True)
        self.body_y = gl.GLLinePlotItem(color=(0.4, 1, 0.4, 1), width=4, antialias=True)
        self.body_z = gl.GLLinePlotItem(color=(0.4, 0.6, 1, 1), width=4, antialias=True)
        self.gl_view.addItem(self.body_x)
        self.gl_view.addItem(self.body_y)
        self.gl_view.addItem(self.body_z)

        self.info = QtWidgets.QLabel("Waiting for data...")
        self.info.setStyleSheet("font-size: 14px; padding: 8px;")

        layout.addWidget(self.plot_widget, 0, 0)
        layout.addWidget(self.gl_view, 0, 1)
        layout.addWidget(self.info, 1, 0, 1, 2)

        self.win.show()

        print(f"Listening on {UDP_IP}:{UDP_PORT}")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(20)

    def handle_udp(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break

            line = data.decode(errors="ignore").strip()
            print("RECV:", line)
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 10:
                continue

            try:
                ts = int(parts[0])
                ax, ay, az = map(float, parts[1:4])
                gx, gy, gz = map(float, parts[4:7])
                mx, my, mz = map(float, parts[7:10])
            except Exception:
                continue

            self.latest_raw = [ax, ay, az, gx, gy, gz, mx, my, mz]

            self.ts_buf.append(ts)
            self.ax_buf.append(ax)
            self.ay_buf.append(ay)
            self.az_buf.append(az)
            self.gx_buf.append(gx)
            self.gy_buf.append(gy)
            self.gz_buf.append(gz)
            self.mx_buf.append(mx)
            self.my_buf.append(my)
            self.mz_buf.append(mz)

            # rate
            self.rate_count += 1
            if self.rate_count >= 100:
                now = time.time()
                self.current_hz = 100.0 / (now - self.rate_t0)
                self.rate_t0 = now
                self.rate_count = 0

            # dt from sender timestamp
            if self.prev_ts is None:
                dt = 1.0 / FS_EST
            else:
                dt = (ts - self.prev_ts) / 1000.0
                if dt <= 0 or dt > 0.1:
                    dt = 1.0 / FS_EST
            self.prev_ts = ts

            # convert raw -> physical
            ax_g = ax / ACC_LSB_PER_G
            ay_g = ay / ACC_LSB_PER_G
            az_g = az / ACC_LSB_PER_G

            gx_rads = np.radians(gx * GYRO_DPS_PER_LSB)
            gy_rads = np.radians(gy * GYRO_DPS_PER_LSB)
            gz_rads = np.radians(gz * GYRO_DPS_PER_LSB)

            mx_u = mx * MAG_LSB_SCALE
            my_u = my * MAG_LSB_SCALE
            mz_u = mz * MAG_LSB_SCALE

            self.q = self.ahrs.update(
                gx_rads, gy_rads, gz_rads,
                ax_g, ay_g, az_g,
                mx_u, my_u, mz_u,
                dt
            )
            self.roll, self.pitch, self.yaw = quat_to_euler_deg(self.q)

    def update_plots(self):
        if len(self.ts_buf) < 2:
            return

        x = np.arange(len(self.ts_buf)) / FS_EST

        self.c_ax.setData(x, np.array(self.ax_buf, dtype=float))
        self.c_ay.setData(x, np.array(self.ay_buf, dtype=float))
        self.c_az.setData(x, np.array(self.az_buf, dtype=float))

        self.c_gx.setData(x, np.array(self.gx_buf, dtype=float))
        self.c_gy.setData(x, np.array(self.gy_buf, dtype=float))
        self.c_gz.setData(x, np.array(self.gz_buf, dtype=float))

        self.c_mx.setData(x, np.array(self.mx_buf, dtype=float))
        self.c_my.setData(x, np.array(self.my_buf, dtype=float))
        self.c_mz.setData(x, np.array(self.mz_buf, dtype=float))

    def update_3d(self):
        R = quat_to_rotmat(self.q)

        # body axes in world frame
        axis_len = 2.0
        origin = np.array([0.0, 0.0, 0.0])
        ex = R @ np.array([axis_len, 0.0, 0.0])
        ey = R @ np.array([0.0, axis_len, 0.0])
        ez = R @ np.array([0.0, 0.0, axis_len])

        self.body_x.setData(pos=np.array([origin, ex]))
        self.body_y.setData(pos=np.array([origin, ey]))
        self.body_z.setData(pos=np.array([origin, ez]))

        # rotate cube
        self.imu_mesh.resetTransform()

        # convert rotation matrix -> axis angle
        trace = np.trace(R)
        angle = np.degrees(np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0)))

        if abs(angle) > 1e-6:
            rx = R[2, 1] - R[1, 2]
            ry = R[0, 2] - R[2, 0]
            rz = R[1, 0] - R[0, 1]
            axis = np.array([rx, ry, rz], dtype=float)
            axis = normalize(axis)
            self.imu_mesh.rotate(angle, axis[0], axis[1], axis[2])

    def update_label(self):
        ax, ay, az, gx, gy, gz, mx, my, mz = self.latest_raw
        self.info.setText(
            f"Rate: {self.current_hz:.1f} Hz    |    "
            f"Roll: {self.roll:.1f}°    Pitch: {self.pitch:.1f}°    Yaw: {self.yaw:.1f}°    |    "
            f"A=({ax:.0f}, {ay:.0f}, {az:.0f})    "
            f"G=({gx:.0f}, {gy:.0f}, {gz:.0f})    "
            f"M=({mx:.0f}, {my:.0f}, {mz:.0f})"
        )

    def update_all(self):
        self.handle_udp()
        self.update_plots()
        self.update_3d()
        self.update_label()

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    viewer = OrientationViewer()
    viewer.run()