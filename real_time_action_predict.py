import socket
import json
import time
import joblib
import numpy as np
from collections import deque, Counter


# =========================
# 設定區
# =========================
MODEL_PATH = "action_rf_model.pkl"
ENCODER_PATH = "action_label_encoder.pkl"
META_PATH = "action_model_meta.json"

UDP_IP = "0.0.0.0"
UDP_PORT = 10000

# 顯示與穩定化
PREDICT_EVERY_N_SAMPLES = 10     # 每收到幾筆做一次預測
SMOOTHING_VOTES = 5              # 最近幾次預測做 majority vote

# 若要印原始資料可改 True
DEBUG_PRINT_RAW = False


# =========================
# 與 train_action_model.py 保持一致
# =========================
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def zero_crossing_rate(x: np.ndarray) -> float:
    xc = x - np.mean(x)
    return float(np.sum(np.diff(np.signbit(xc)).astype(np.int32))) / max(len(x) - 1, 1)


def signal_energy(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def extract_features_from_array(window_arr: np.ndarray) -> np.ndarray:
    """
    window_arr shape = (window_size, 6)
    columns = ax, ay, az, gx, gy, gz
    """
    feats = []

    for i in range(window_arr.shape[1]):
        x = window_arr[:, i].astype(np.float32)

        feats.extend([
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.max(x) - np.min(x)),
            rms(x),
            signal_energy(x),
            zero_crossing_rate(x),
        ])

    ax = window_arr[:, 0].astype(np.float32)
    ay = window_arr[:, 1].astype(np.float32)
    az = window_arr[:, 2].astype(np.float32)
    gx = window_arr[:, 3].astype(np.float32)
    gy = window_arr[:, 4].astype(np.float32)
    gz = window_arr[:, 5].astype(np.float32)

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    for mag in [acc_mag, gyro_mag]:
        feats.extend([
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.min(mag)),
            float(np.max(mag)),
            float(np.max(mag) - np.min(mag)),
            rms(mag),
            signal_energy(mag),
            zero_crossing_rate(mag),
        ])

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats.extend([
        safe_corr(ax, ay),
        safe_corr(ax, az),
        safe_corr(ay, az),
        safe_corr(gx, gy),
        safe_corr(gx, gz),
        safe_corr(gy, gz),
    ])

    return np.array(feats, dtype=np.float32).reshape(1, -1)


# =========================
# 資料解析
# 支援:
# 1) CSV: "ax,ay,az,gx,gy,gz"
# 2) JSON: {"ax":...,"ay":...,"az":...,"gx":...,"gy":...,"gz":...}
# =========================
def parse_packet(data: bytes):
    text = data.decode("utf-8", errors="ignore").strip()

    if DEBUG_PRINT_RAW:
        print("RAW:", text)

    # JSON 格式
    if text.startswith("{") and text.endswith("}"):
        obj = json.loads(text)
        return [
            float(obj["ax"]),
            float(obj["ay"]),
            float(obj["az"]),
            float(obj["gx"]),
            float(obj["gy"]),
            float(obj["gz"]),
        ]

    # CSV 格式
    parts = text.split(",")
    if len(parts) >= 6:
        return [float(parts[i]) for i in range(6)]

    raise ValueError(f"無法解析封包: {text}")


# =========================
# 平滑輸出
# =========================
def majority_vote(labels):
    if not labels:
        return None
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


# =========================
# 主程式
# =========================
def main():
    # 載入模型
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    window_size = int(meta["window_size"])
    step_size = int(meta["step_size"])
    valid_actions = meta.get("valid_actions", {})

    print("Model loaded.")
    print(f"Window size: {window_size}")
    print(f"Step size: {step_size}")
    print(f"Listening UDP on {UDP_IP}:{UDP_PORT}")
    print("Press Ctrl+C to stop.\n")

    # buffer: 最近一個 window 的原始 IMU
    buffer = deque(maxlen=window_size)
    # recent predictions: 做平滑
    pred_history = deque(maxlen=SMOOTHING_VOTES)

    # 開 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    sample_count = 0
    last_predict_time = time.time()

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            try:
                sample = parse_packet(data)
            except Exception as e:
                print(f"[WARN] parse failed: {e}")
                continue

            buffer.append(sample)
            sample_count += 1

            if len(buffer) < window_size:
                continue

            if sample_count % PREDICT_EVERY_N_SAMPLES != 0:
                continue

            window_arr = np.array(buffer, dtype=np.float32)
            feats = extract_features_from_array(window_arr)

            pred_id = model.predict(feats)[0]
            pred_label = encoder.inverse_transform([pred_id])[0]
            pred_history.append(pred_label)

            smooth_label = majority_vote(pred_history)

            now = time.time()
            dt = now - last_predict_time
            last_predict_time = now

            # 若 meta 內有中文映射就一起顯示
            zh_label = valid_actions.get(smooth_label, smooth_label)

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"pred={smooth_label} ({zh_label}) | "
                f"raw={pred_label} | "
                f"buffer={len(buffer)} | "
                f"dt={dt:.3f}s"
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        sock.close()
        print("Socket closed.")


if __name__ == "__main__":
    main()