import os
import time

UART_DEV = "/dev/ttyS1"
BAUD = 115200

# 開啟 UART
fd = os.open(UART_DEV, os.O_WRONLY)

print("UART sender started")

while True:
    ts = int(time.time() * 1000)

    # 先用假資料測試
    ax, ay, az = 1.1, 2.2, 3.3
    gx, gy, gz = 4.4, 5.5, 6.6

    line = f"{ts},{ax},{ay},{az},{gx},{gy},{gz}\n"

    os.write(fd, line.encode())

    time.sleep(0.01)  # 100 Hz