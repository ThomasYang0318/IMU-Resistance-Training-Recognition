import asyncio
import struct
import csv
import threading
from datetime import datetime
from bleak import BleakScanner
import tkinter as tk
from tkinter import ttk

COMPANY_ID = 0x1234

CSV_FILENAME = f"imu_ble_labeled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
CSV_HEADERS = [
    "pc_time",
    "device_name",
    "timestamp_ms",
    "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps",
    "mx_uT", "my_uT", "mz_uT",
    "action_type",
    "phase",
    "rep",
    "set",
    "weight_kg",
    "subject_id",
]


# =========================
# Global label state
# =========================
LABEL_DEFAULTS = {
    "subject_id": "",
    "weight_kg": 0.0,
    "action_type": "none",   # 例如 squat / curl / lateral_raise
    "phase": "none",         # up / down / rest / none
    "rep": 0,
    "set": 0,
}
label_state = LABEL_DEFAULTS.copy()

label_lock = threading.Lock()
csv_lock = threading.Lock()


# =========================
# CSV init
# =========================
csv_file = open(CSV_FILENAME, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(CSV_HEADERS)
csv_file.flush()


def get_label_snapshot():
    with label_lock:
        return label_state.copy()


def update_label_state(**kwargs):
    with label_lock:
        label_state.update(kwargs)


def increment_label_state(key, step=1):
    with label_lock:
        label_state[key] += step


def reset_label_state():
    with label_lock:
        label_state.clear()
        label_state.update(LABEL_DEFAULTS)


# =========================
# BLE payload parse
# =========================
def parse_payload(data: bytes):
    # 4 bytes timestamp + 9 * int16 = 22 bytes
    if len(data) < 22:
        return None

    vals = struct.unpack("<Ihhhhhhhhh", data[:22])
    timestamp_ms, ax_i, ay_i, az_i, gx_i, gy_i, gz_i, mx_i, my_i, mz_i = vals

    ax = ax_i / 1000.0
    ay = ay_i / 1000.0
    az = az_i / 1000.0

    gx = float(gx_i)
    gy = float(gy_i)
    gz = float(gz_i)

    mx = mx_i / 10.0
    my = my_i / 10.0
    mz = mz_i / 10.0

    return timestamp_ms, ax, ay, az, gx, gy, gz, mx, my, mz


# =========================
# CSV write
# =========================
def write_csv_row(device_name, parsed):
    timestamp_ms, ax, ay, az, gx, gy, gz, mx, my, mz = parsed
    labels = get_label_snapshot()

    row_data = {
        "pc_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "device_name": device_name,
        "timestamp_ms": timestamp_ms,
        "ax_g": ax,
        "ay_g": ay,
        "az_g": az,
        "gx_dps": gx,
        "gy_dps": gy,
        "gz_dps": gz,
        "mx_uT": mx,
        "my_uT": my,
        "mz_uT": mz,
        "action_type": labels["action_type"],
        "phase": labels["phase"],
        "rep": labels["rep"],
        "set": labels["set"],
        "weight_kg": labels["weight_kg"],
        "subject_id": labels["subject_id"],
    }
    row = [row_data[col] for col in CSV_HEADERS]

    with csv_lock:
        csv_writer.writerow(row)
        csv_file.flush()


# =========================
# BLE callback
# =========================
def callback(device, adv):
    md = adv.manufacturer_data
    if COMPANY_ID not in md:
        return

    parsed = parse_payload(md[COMPANY_ID])
    if parsed is None:
        return

    timestamp_ms, ax, ay, az, gx, gy, gz, mx, my, mz = parsed

    labels = get_label_snapshot()

    print(
        f"{device.name or 'Unknown':<12} "
        f"t={timestamp_ms:>10} ms | "
        f"acc=({ax:+.3f},{ay:+.3f},{az:+.3f}) g | "
        f"gyro=({gx:+.1f},{gy:+.1f},{gz:+.1f}) dps | "
        f"mag=({mx:+.1f},{my:+.1f},{mz:+.1f}) uT | "
        f"action={labels['action_type']} phase={labels['phase']} "
        f"rep={labels['rep']} set={labels['set']} "
        f"weight={labels['weight_kg']} subject_id={labels['subject_id']}"
    )

    write_csv_row(device.name or "Unknown", parsed)


# =========================
# BLE scanner thread
# =========================
async def ble_main():
    scanner = BleakScanner(detection_callback=callback)
    await scanner.start()
    print("Scanning IMU 9-axis BLE advertisements...")
    print(f"CSV saving to: {CSV_FILENAME}")
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await scanner.stop()


def run_ble_loop():
    asyncio.run(ble_main())


# =========================
# GUI
# =========================
class LabelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU BLE Label Tool")
        self.root.geometry("1200x800")

        # ===== Current state display =====
        self.status_var = tk.StringVar()
        self.update_status()

        status_frame = ttk.LabelFrame(root, text="目前標記狀態")
        status_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(status_frame, textvariable=self.status_var, justify="left").pack(anchor="w", padx=10, pady=10)

        # ===== Subject ID =====
        subject_frame = ttk.LabelFrame(root, text="受試者 ID")
        subject_frame.pack(fill="x", padx=10, pady=5)

        self.subject_id_var = tk.StringVar(value="")
        ttk.Entry(subject_frame, textvariable=self.subject_id_var, width=20).pack(side="left", padx=5, pady=5)
        ttk.Button(subject_frame, text="套用受試者 ID", command=self.apply_subject_id).pack(side="left", padx=5, pady=5)

        # ===== Weight =====
        weight_frame = ttk.LabelFrame(root, text="重量設定 (kg)")
        weight_frame.pack(fill="x", padx=10, pady=5)

        self.weight_var = tk.StringVar(value="0.0")
        ttk.Entry(weight_frame, textvariable=self.weight_var, width=12).pack(side="left", padx=5, pady=5)
        ttk.Button(weight_frame, text="套用重量", command=self.apply_weight).pack(side="left", padx=5, pady=5)

        # ===== Action type =====
        action_frame = ttk.LabelFrame(root, text="動作型態")
        action_frame.pack(fill="x", padx=10, pady=5)

        action_buttons = [
            ("無", "none"),
            ("啞鈴臥推", "db_bench_press"),
            ("單手啞鈴划船", "one_arm_db_row"),
            ("啞鈴肩推", "db_shoulder_press"),
            ("啞鈴二頭彎舉", "db_biceps_curl"),
            ("啞鈴三頭彎舉", "db_triceps_curl"),
            ("啞鈴深蹲", "db_squat"),
            ("啞鈴羅馬尼亞硬舉", "db_rdl"),
            ("啞鈴負重卷腹", "db_weighted_crunch"),
        ]

        for text, val in action_buttons:
            ttk.Button(action_frame, text=text, command=lambda v=val: self.set_action(v)).pack(side="left", padx=4, pady=4)

        # ===== Phase =====
        phase_frame = ttk.LabelFrame(root, text="動作階段")
        phase_frame.pack(fill="x", padx=10, pady=5)

        phase_buttons = [
            ("None", "none"),
            ("Up", "up"),
            ("Down", "down"),
            ("Rest", "rest"),
        ]

        for text, val in phase_buttons:
            ttk.Button(phase_frame, text=text, command=lambda v=val: self.set_phase(v)).pack(side="left", padx=4, pady=4)

        # ===== Rep / Set =====
        count_frame = ttk.LabelFrame(root, text="次數 / 組數")
        count_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(count_frame, text="Rep +1", command=self.inc_rep).pack(side="left", padx=5, pady=5)
        ttk.Button(count_frame, text="Rep Reset", command=self.reset_rep).pack(side="left", padx=5, pady=5)

        ttk.Button(count_frame, text="Set +1", command=self.inc_set).pack(side="left", padx=5, pady=5)
        ttk.Button(count_frame, text="Set Reset", command=self.reset_set).pack(side="left", padx=5, pady=5)

        # ===== Quick control =====
        quick_frame = ttk.LabelFrame(root, text="快速控制")
        quick_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(quick_frame, text="清空標記", command=self.clear_labels).pack(side="left", padx=5, pady=5)

        # ===== keyboard shortcuts info =====
        info_frame = ttk.LabelFrame(root, text="快捷鍵")
        info_frame.pack(fill="x", padx=10, pady=10)

        info_text = (
            "u = phase -> up\n"
            "d = phase -> down\n"
            "h = phase -> rest\n"
            "r = rep +1\n"
            "e = set +1\n"
            "c = clear labels\n"
        )
        ttk.Label(info_frame, text=info_text, justify="left").pack(anchor="w", padx=10, pady=10)

        # key bindings
        root.bind("u", lambda e: self.set_phase("up"))
        root.bind("d", lambda e: self.set_phase("down"))
        root.bind("h", lambda e: self.set_phase("rest"))
        root.bind("r", lambda e: self.inc_rep())
        root.bind("e", lambda e: self.inc_set())
        root.bind("c", lambda e: self.clear_labels())

    def update_status(self):
        labels = get_label_snapshot()
        text = (
            f"subject_id  : {labels['subject_id']}\n"
            f"weight_kg   : {labels['weight_kg']}\n"
            f"action_type : {labels['action_type']}\n"
            f"phase       : {labels['phase']}\n"
            f"rep         : {labels['rep']}\n"
            f"set         : {labels['set']}\n"
        )
        self.status_var.set(text)

    def _refresh_with_state(self, **kwargs):
        update_label_state(**kwargs)
        self.update_status()

    def set_action(self, value):
        self._refresh_with_state(action_type=value)

    def set_phase(self, value):
        self._refresh_with_state(phase=value)

    def inc_rep(self):
        increment_label_state("rep")
        self.update_status()

    def reset_rep(self):
        self._refresh_with_state(rep=0)

    def inc_set(self):
        increment_label_state("set")
        self.update_status()

    def reset_set(self):
        self._refresh_with_state(set=0)

    def apply_weight(self):
        try:
            w = float(self.weight_var.get())
            self._refresh_with_state(weight_kg=w)
        except ValueError:
            print("[WARN] weight 必須是數字")

    def apply_subject_id(self):
        self._refresh_with_state(subject_id=self.subject_id_var.get().strip())

    def clear_labels(self):
        reset_label_state()
        self.weight_var.set("0.0")
        self.subject_id_var.set("")
        self.update_status()


# =========================
# Main
# =========================
def main():
    ble_thread = threading.Thread(target=run_ble_loop, daemon=True)
    ble_thread.start()

    root = tk.Tk()
    gui = LabelGUI(root)

    def on_close():
        print("Closing...")
        try:
            csv_file.close()
        except:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
