import socket
import csv
import time
import select
import sys
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore


UDP_IP = "0.0.0.0"
UDP_PORT = 10000

WINDOW_SEC = 5          # 顯示最近幾秒
FS_EST = 100            # 預估頻率
MAXLEN = WINDOW_SEC * FS_EST


class IMUMonitor:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.ts_buf = deque(maxlen=MAXLEN)
        self.ax_buf = deque(maxlen=MAXLEN)
        self.ay_buf = deque(maxlen=MAXLEN)
        self.az_buf = deque(maxlen=MAXLEN)

        self.recording = False
        self.csv_file = None
        self.csv_writer = None

        self.rate_count = 0
        self.rate_t0 = time.time()
        self.current_hz = 0.0

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="IMU Visual Monitor")
        self.win.resize(1200, 700)

        self.plot = self.win.addPlot(title="Real-time IMU (ax, ay, az)")
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()

        self.curve_ax = self.plot.plot(pen='r', name='ax')
        self.curve_ay = self.plot.plot(pen='g', name='ay')
        self.curve_az = self.plot.plot(pen='b', name='az')

        self.label = pg.LabelItem(justify="left")
        self.win.addItem(self.label)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)  # 約 50 FPS 畫面更新

        print(f"Listening on {UDP_IP}:{UDP_PORT}")
        print("Commands:")
        print("  r + Enter  -> start/stop recording")
        print("  q + Enter  -> quit")

    def start_recording(self):
        filename = f"imu_{int(time.time())}.csv"
        self.csv_file = open(filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "ax", "ay", "az"])
        self.recording = True
        print(f"[RECORDING START] {filename}")

    def stop_recording(self):
        self.recording = False
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print("[RECORDING STOP]")

    def handle_keyboard(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        for r in readable:
            if r == sys.stdin:
                cmd = sys.stdin.readline().strip()
                if cmd == "r":
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif cmd == "q":
                    self.close()

    def handle_udp(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                break

            line = data.decode(errors="ignore").strip()
            parts = line.split(",")

            if len(parts) < 4:
                continue

            try:
                ts = int(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
            except ValueError:
                continue

            self.ts_buf.append(ts)
            self.ax_buf.append(ax)
            self.ay_buf.append(ay)
            self.az_buf.append(az)

            self.rate_count += 1
            now = time.time()
            if now - self.rate_t0 >= 1.0:
                self.current_hz = self.rate_count / (now - self.rate_t0)
                self.rate_count = 0
                self.rate_t0 = now

            if self.recording and self.csv_writer:
                self.csv_writer.writerow([ts, ax, ay, az])

    def update_plot(self):
        if len(self.ts_buf) < 2:
            return

        t0 = self.ts_buf[0]
        x = [(t - t0) / 1000.0 for t in self.ts_buf]

        self.curve_ax.setData(x, list(self.ax_buf))
        self.curve_ay.setData(x, list(self.ay_buf))
        self.curve_az.setData(x, list(self.az_buf))

        latest_ax = self.ax_buf[-1]
        latest_ay = self.ay_buf[-1]
        latest_az = self.az_buf[-1]

        self.label.setText(
            f"Samples: {len(self.ts_buf)}   |   "
            f"Rate: {self.current_hz:.1f} Hz   |   "
            f"Latest: ax={latest_ax:.1f}, ay={latest_ay:.1f}, az={latest_az:.1f}   |   "
            f"Recording: {'ON' if self.recording else 'OFF'}"
        )

    def update(self):
        self.handle_udp()
        self.handle_keyboard()
        self.update_plot()

    def close(self):
        print("Shutting down...")
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
        self.sock.close()
        QtWidgets.QApplication.quit()

    def run(self):
        self.app.aboutToQuit.connect(self.close)
        self.app.exec()


if __name__ == "__main__":
    monitor = IMUMonitor()
    monitor.run()