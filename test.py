import socket
import math
import time

HOST = "0.0.0.0"
PORT = 10000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Listening on UDP {PORT}...")

last_t = None

# velocity
vx = vy = vz = 0.0

# position (relative)
px = py = pz = 0.0

# 參數，可慢慢調
ACC_DEADBAND = 0.03     # 加速度太小就當雜訊
VEL_DAMPING   = 0.90    # 速度阻尼，避免越積越飄
VEL_ZERO_TH   = 0.02    # 太小速度歸零

def deadband(x, th):
    return 0.0 if abs(x) < th else x

while True:
    data, addr = sock.recvfrom(2048)
    line = data.decode(errors="ignore").strip()

    parts = line.split(",")
    if len(parts) < 4:
        continue

    try:
        # 假設格式: timestamp, ax, ay, az, ...
        t_ms = float(parts[0])
        ax = float(parts[1])
        ay = float(parts[2])
        az = float(parts[3])
    except ValueError:
        continue

    t = t_ms / 1000.0

    if last_t is None:
        last_t = t
        continue

    dt = t - last_t
    last_t = t

    if dt <= 0 or dt > 0.2:
        continue

    # 1) 小雜訊去掉
    ax = deadband(ax, ACC_DEADBAND)
    ay = deadband(ay, ACC_DEADBAND)
    az = deadband(az, ACC_DEADBAND)

    # 2) 積分得到速度
    vx += ax * dt
    vy += ay * dt
    vz += az * dt

    # 3) 加阻尼，減少漂移
    vx *= VEL_DAMPING
    vy *= VEL_DAMPING
    vz *= VEL_DAMPING

    # 4) 很小速度直接當 0
    vx = 0.0 if abs(vx) < VEL_ZERO_TH else vx
    vy = 0.0 if abs(vy) < VEL_ZERO_TH else vy
    vz = 0.0 if abs(vz) < VEL_ZERO_TH else vz

    # 5) 再積分得到相對位移
    px += vx * dt
    py += vy * dt
    pz += vz * dt

    # 6) 找主要移動軸
    axis_vals = {"X": abs(px), "Y": abs(py), "Z": abs(pz)}
    main_axis = max(axis_vals, key=axis_vals.get)

    print(
        f"pos=({px:+.3f}, {py:+.3f}, {pz:+.3f})   "
        f"vel=({vx:+.3f}, {vy:+.3f}, {vz:+.3f})   "
        f"main={main_axis}"
    )