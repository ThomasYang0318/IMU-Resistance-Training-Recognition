import socket
import time
import math
import random

UDP_IP = "127.0.0.1"
UDP_PORT = 10000
HZ = 50.0
DT = 1.0 / HZ

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_sample(ax, ay, az, gx, gy, gz):
    msg = f"{ax:.4f},{ay:.4f},{az:.4f},{gx:.4f},{gy:.4f},{gz:.4f}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

def simulate_action(t, action_name):
    """
    回傳假六軸資料
    只是為了測流程，不代表真實生理/動作模型
    """
    noise_a = 0.03
    noise_g = 2.0

    if action_name == "db_biceps_curl":
        ax = 0.15 * math.sin(2 * math.pi * 0.8 * t) + random.uniform(-noise_a, noise_a)
        ay = 0.05 * math.cos(2 * math.pi * 0.8 * t) + random.uniform(-noise_a, noise_a)
        az = 1.00 + 0.25 * math.sin(2 * math.pi * 0.8 * t) + random.uniform(-noise_a, noise_a)
        gx = 40 * math.sin(2 * math.pi * 0.8 * t) + random.uniform(-noise_g, noise_g)
        gy = 10 * math.cos(2 * math.pi * 0.8 * t) + random.uniform(-noise_g, noise_g)
        gz = 5 * math.sin(2 * math.pi * 0.8 * t) + random.uniform(-noise_g, noise_g)

    elif action_name == "db_squat":
        ax = 0.08 * math.sin(2 * math.pi * 0.5 * t) + random.uniform(-noise_a, noise_a)
        ay = 0.02 * math.sin(2 * math.pi * 0.5 * t) + random.uniform(-noise_a, noise_a)
        az = 1.00 + 0.35 * math.sin(2 * math.pi * 0.5 * t) + random.uniform(-noise_a, noise_a)
        gx = 15 * math.sin(2 * math.pi * 0.5 * t) + random.uniform(-noise_g, noise_g)
        gy = 8 * math.cos(2 * math.pi * 0.5 * t) + random.uniform(-noise_g, noise_g)
        gz = 4 * math.sin(2 * math.pi * 0.5 * t) + random.uniform(-noise_g, noise_g)

    elif action_name == "db_shoulder_press":
        ax = 0.05 * math.sin(2 * math.pi * 0.7 * t) + random.uniform(-noise_a, noise_a)
        ay = 0.12 * math.cos(2 * math.pi * 0.7 * t) + random.uniform(-noise_a, noise_a)
        az = 1.00 + 0.30 * math.sin(2 * math.pi * 0.7 * t) + random.uniform(-noise_a, noise_a)
        gx = 18 * math.sin(2 * math.pi * 0.7 * t) + random.uniform(-noise_g, noise_g)
        gy = 35 * math.cos(2 * math.pi * 0.7 * t) + random.uniform(-noise_g, noise_g)
        gz = 6 * math.sin(2 * math.pi * 0.7 * t) + random.uniform(-noise_g, noise_g)

    else:
        # 靜止
        ax = random.uniform(-0.01, 0.01)
        ay = random.uniform(-0.01, 0.01)
        az = 1.00 + random.uniform(-0.01, 0.01)
        gx = random.uniform(-1, 1)
        gy = random.uniform(-1, 1)
        gz = random.uniform(-1, 1)

    return ax, ay, az, gx, gy, gz

def main():
    actions = [
        ("db_biceps_curl", 8),
        ("db_squat", 8),
        ("db_shoulder_press", 8),
    ]

    print(f"Sending fake IMU to {UDP_IP}:{UDP_PORT}")

    t = 0.0
    try:
        while True:
            for action_name, duration_sec in actions:
                print(f"\nNow simulating: {action_name} for {duration_sec}s")
                start = time.time()

                while time.time() - start < duration_sec:
                    ax, ay, az, gx, gy, gz = simulate_action(t, action_name)
                    send_sample(ax, ay, az, gx, gy, gz)
                    time.sleep(DT)
                    t += DT

                # 中間插 2 秒靜止
                print("Now simulating: rest for 2s")
                rest_start = time.time()
                while time.time() - rest_start < 2:
                    ax, ay, az, gx, gy, gz = simulate_action(t, "rest")
                    send_sample(ax, ay, az, gx, gy, gz)
                    time.sleep(DT)
                    t += DT

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()