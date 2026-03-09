import socket
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

UDP_IP = "0.0.0.0"
UDP_PORT = 10000

WINDOW_SEC = 5
FS_EST = 100
MAXLEN = WINDOW_SEC * FS_EST


class IMU9AxisMonitor:
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

        self.acc_mag_buf = deque(maxlen=MAXLEN)
        self.gyr_mag_buf = deque(maxlen=MAXLEN)
        self.mag_mag_buf = deque(maxlen=MAXLEN)

        self.rate_count = 0
        self.rate_t0 = time.time()
        self.current_hz = 0.0

        self.latest = [0.0] * 9

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="9-axis IMU Monitor")
        self.win.resize(1400, 900)

        self.p1 = self.win.addPlot(title="Accelerometer")
        self.p1.showGrid(x=True, y=True)
        self.p1.addLegend()
        self.c_ax = self.p1.plot(pen='r', name='ax')
        self.c_ay = self.p1.plot(pen='g', name='ay')
        self.c_az = self.p1.plot(pen='b', name='az')
        self.c_am = self.p1.plot(pen='y', name='|a|')

        self.win.nextRow()
        self.p2 = self.win.addPlot(title="Gyroscope")
        self.p2.showGrid(x=True, y=True)
        self.p2.addLegend()
        self.c_gx = self.p2.plot(pen='r', name='gx')
        self.c_gy = self.p2.plot(pen='g', name='gy')
        self.c_gz = self.p2.plot(pen='b', name='gz')
        self.c_gm = self.p2.plot(pen='y', name='|g|')

        self.win.nextRow()
        self.p3 = self.win.addPlot(title="Magnetometer")
        self.p3.showGrid(x=True, y=True)
        self.p3.addLegend()
        self.c_mx = self.p3.plot(pen='r', name='mx')
        self.c_my = self.p3.plot(pen='g', name='my')
        self.c_mz = self.p3.plot(pen='b', name='mz')
        self.c_mm = self.p3.plot(pen='y', name='|m|')

        self.win.nextRow()
        self.label = self.win.addLabel("Waiting for IMU data...")

        print(f"Listening on {UDP_IP}:{UDP_PORT}")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(20)

    def handle_udp(self):
        recv_count = 0

        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break

            recv_count += 1
            line = data.decode(errors="ignore").strip()
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 10:
                print("SKIP short:", repr(line))
                continue

            try:
                ts = int(parts[0])
                ax, ay, az = map(float, parts[1:4])
                gx, gy, gz = map(float, parts[4:7])
                mx, my, mz = map(float, parts[7:10])
            except Exception as e:
                print("SKIP parse:", e, repr(line))
                continue

            acc_mag = float(np.sqrt(ax*ax + ay*ay + az*az))
            gyr_mag = float(np.sqrt(gx*gx + gy*gy + gz*gz))
            mag_mag = float(np.sqrt(mx*mx + my*my + mz*mz))

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

            self.acc_mag_buf.append(acc_mag)
            self.gyr_mag_buf.append(gyr_mag)
            self.mag_mag_buf.append(mag_mag)

            self.latest = [ax, ay, az, gx, gy, gz, mx, my, mz]

            self.rate_count += 1
            now = time.time()
            if now - self.rate_t0 >= 1.0:
                self.current_hz = self.rate_count / (now - self.rate_t0)
                self.rate_count = 0
                self.rate_t0 = now
                print(f"[RATE] {self.current_hz:.1f} Hz, buf={len(self.ts_buf)}")

        return recv_count

    def update_plots(self):
        if len(self.ts_buf) < 2:
            return

        t0 = self.ts_buf[0]
        x = np.array([(t - t0) / 1000.0 for t in self.ts_buf], dtype=float)

        self.c_ax.setData(x, np.array(self.ax_buf, dtype=float))
        self.c_ay.setData(x, np.array(self.ay_buf, dtype=float))
        self.c_az.setData(x, np.array(self.az_buf, dtype=float))
        self.c_am.setData(x, np.array(self.acc_mag_buf, dtype=float))

        self.c_gx.setData(x, np.array(self.gx_buf, dtype=float))
        self.c_gy.setData(x, np.array(self.gy_buf, dtype=float))
        self.c_gz.setData(x, np.array(self.gz_buf, dtype=float))
        self.c_gm.setData(x, np.array(self.gyr_mag_buf, dtype=float))

        self.c_mx.setData(x, np.array(self.mx_buf, dtype=float))
        self.c_my.setData(x, np.array(self.my_buf, dtype=float))
        self.c_mz.setData(x, np.array(self.mz_buf, dtype=float))
        self.c_mm.setData(x, np.array(self.mag_mag_buf, dtype=float))

        ax, ay, az, gx, gy, gz, mx, my, mz = self.latest
        self.label.setText(
            f"Rate={self.current_hz:.1f} Hz | "
            f"A=({ax:.1f}, {ay:.1f}, {az:.1f}) | "
            f"G=({gx:.1f}, {gy:.1f}, {gz:.1f}) | "
            f"M=({mx:.1f}, {my:.1f}, {mz:.1f}) | "
            f"Samples={len(self.ts_buf)}"
        )

    def update_all(self):
        self.handle_udp()
        self.update_plots()

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    monitor = IMU9AxisMonitor()
    monitor.run()