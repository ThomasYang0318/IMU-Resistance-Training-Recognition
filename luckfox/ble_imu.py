import time
import struct
import subprocess

# smbus2 用來透過 I2C 與 IMU 溝通
from smbus2 import SMBus

# =========================
# BLE 基本設定
# =========================

HCI_DEV = "hci0"         # Linux 藍牙介面名稱
COMPANY_ID = 0x1234      # 自訂 Manufacturer Data 的 company ID

# =========================
# I2C / LSM9DS0 設定
# =========================

I2C_BUS = 2              # Luckfox 上使用的 I2C bus 編號

# LSM9DS0 的兩個 I2C 位址
ADDR_G = 0x6B            # Gyroscope 位址
ADDR_XM = 0x1D           # Accelerometer + Magnetometer 位址

# -------- Gyro registers --------
WHO_AM_I_G = 0x0F
CTRL_REG1_G = 0x20
CTRL_REG4_G = 0x23
OUT_X_L_G = 0x28         # gyro X low byte 起始位置

# -------- Accel/Mag registers --------
WHO_AM_I_XM = 0x0F
CTRL_REG1_XM = 0x20
CTRL_REG2_XM = 0x21
CTRL_REG5_XM = 0x24
CTRL_REG6_XM = 0x25
CTRL_REG7_XM = 0x26

OUT_X_L_A = 0x28         # accel X low byte 起始位置
OUT_X_L_M = 0x08         # mag X low byte 起始位置

# =========================
# BLE HCI helpers
# =========================

def run_cmd(args):
    """
    執行 shell 指令。
    這裡主要拿來呼叫 hcitool，直接控制 BLE 廣播。
    """
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print("CMD FAIL:", " ".join(args))
        print(r.stderr)
    return r


def hci_cmd(*vals):
    """
    將整數形式的 HCI 參數轉成 hex 字串，
    再組成 hcitool 指令送出去。

    例如：
    hci_cmd(0x08, 0x000A, 0x01)
    會變成類似：
    hcitool -i hci0 cmd 08 000a 01
    """
    args = ["hcitool", "-i", HCI_DEV, "cmd"]
    args += [f"{v:02x}" for v in vals]
    return run_cmd(args)


def adv_disable():
    """
    關閉 BLE advertising。
    HCI OGF/OCF:
    0x08 / 0x000A = LE Set Advertising Enable
    0x00 = disable
    """
    hci_cmd(0x08, 0x000A, 0x00)


def adv_enable():
    """
    開啟 BLE advertising。
    0x01 = enable
    """
    hci_cmd(0x08, 0x000A, 0x01)


def adv_setup():
    """
    設定 BLE 廣播參數。

    這裡設定成：
    - interval 約 100 ms
    - ADV_NONCONN_IND：不可連線，只做 broadcast
    - 使用所有 advertising channels
    """
    hci_cmd(
        0x08, 0x0006,   # LE Set Advertising Parameters
        0xA0, 0x00,     # Min interval = 0x00A0
        0xA0, 0x00,     # Max interval = 0x00A0
        0x03,           # ADV_NONCONN_IND
        0x00,           # own address type
        0x00,           # peer address type
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # peer address
        0x07,           # all advertising channels
        0x00            # filter policy
    )


def set_adv_payload(payload22: bytes):
    """
    設定 BLE 廣播 payload。

    你現在新版 payload 是：
    - timestamp_ms: 4 bytes
    - imu9: 18 bytes
    總共 22 bytes

    這 22 bytes 會被放進 manufacturer data 裡。
    """
    if len(payload22) != 22:
        raise ValueError("payload must be 22 bytes")

    # company ID 拆成低位 / 高位
    cid_lo = COMPANY_ID & 0xFF
    cid_hi = (COMPANY_ID >> 8) & 0xFF

    adv = bytearray()

    # -------------------------
    # AD Structure 1: Flags
    # [Length=2][Type=0x01][Data=0x06]
    # -------------------------
    adv += bytes([0x02, 0x01, 0x06])

    # -------------------------
    # AD Structure 2: Manufacturer Specific Data
    #
    # 內容為：
    # [Length][Type=0xFF][Company ID 2 bytes][payload22]
    #
    # 1(type) + 2(company id) + 22(payload) = 25 = 0x19
    # -------------------------
    adv += bytes([0x19, 0xFF, cid_lo, cid_hi])
    adv += payload22

    adv_len = len(adv)

    # BLE advertising data 最多 31 bytes，不足補 0
    while len(adv) < 31:
        adv.append(0x00)

    # LE Set Advertising Data
    cmd = [0x08, 0x0008, adv_len] + list(adv)
    hci_cmd(*cmd)

# =========================
# LSM9DS0 helpers
# =========================

def write_reg(bus, addr, reg, val):
    """
    寫一個 byte 到指定裝置的指定 register。
    """
    bus.write_byte_data(addr, reg, val)


def read_block(bus, addr, reg, length):
    """
    從指定 register 起始位置連續讀取多個 byte。

    reg | 0x80 是啟用 auto-increment，
    讓 IMU 可以自動往後讀多個 register。
    """
    return bus.read_i2c_block_data(addr, reg | 0x80, length)


def to_int16(lo, hi):
    """
    將兩個 byte (low, high) 組成一個 signed int16。
    IMU 常用 little-endian，所以 low 在前、high 在後。
    """
    v = (hi << 8) | lo
    if v & 0x8000:
        v -= 65536
    return v


def init_lsm9ds0(bus):
    """
    初始化 LSM9DS0：
    - 開啟 gyro
    - 開啟 accel
    - 開啟 magnetometer
    - 設定量測範圍與模式
    """

    # -------- Gyro --------
    # CTRL_REG1_G = 0x0F
    # PD=1 (開機), X/Y/Z enable
    write_reg(bus, ADDR_G, CTRL_REG1_G, 0x0F)

    # CTRL_REG4_G = 0x00
    # Full scale = ±245 dps
    write_reg(bus, ADDR_G, CTRL_REG4_G, 0x00)

    # -------- Accel --------
    # CTRL_REG1_XM = 0x67
    # 100Hz + XYZ enable
    write_reg(bus, ADDR_XM, CTRL_REG1_XM, 0x67)

    # CTRL_REG2_XM = 0x00
    # Accel full scale = ±2g
    write_reg(bus, ADDR_XM, CTRL_REG2_XM, 0x00)

    # -------- Magnetometer --------
    # CTRL_REG5_XM = 0x94
    # high-resolution + mag output rate 設定
    write_reg(bus, ADDR_XM, CTRL_REG5_XM, 0x94)

    # CTRL_REG6_XM = 0x00
    # Mag full scale = ±2 gauss
    write_reg(bus, ADDR_XM, CTRL_REG6_XM, 0x00)

    # CTRL_REG7_XM = 0x00
    # continuous-conversion mode
    write_reg(bus, ADDR_XM, CTRL_REG7_XM, 0x00)

    time.sleep(0.1)  # 給感測器一點時間穩定


def read_accel_g(bus):
    """
    讀取加速度資料，並轉成 g 單位。
    回傳: ax, ay, az
    """
    data = read_block(bus, ADDR_XM, OUT_X_L_A, 6)

    ax_raw = to_int16(data[0], data[1])
    ay_raw = to_int16(data[2], data[3])
    az_raw = to_int16(data[4], data[5])

    # ±2g 模式下，LSM9DS0 accel sensitivity 約為 0.061 mg/LSB
    # => 0.000061 g/LSB
    scale_g_per_lsb = 0.000061

    ax = ax_raw * scale_g_per_lsb
    ay = ay_raw * scale_g_per_lsb
    az = az_raw * scale_g_per_lsb
    return ax, ay, az


def read_gyro_dps(bus):
    """
    讀取陀螺儀資料，並轉成 deg/s。
    回傳: gx, gy, gz
    """
    data = read_block(bus, ADDR_G, OUT_X_L_G, 6)

    gx_raw = to_int16(data[0], data[1])
    gy_raw = to_int16(data[2], data[3])
    gz_raw = to_int16(data[4], data[5])

    # ±245 dps 模式下，sensitivity 約 8.75 mdps/LSB
    # => 0.00875 dps/LSB
    scale_dps_per_lsb = 0.00875

    gx = gx_raw * scale_dps_per_lsb
    gy = gy_raw * scale_dps_per_lsb
    gz = gz_raw * scale_dps_per_lsb
    return gx, gy, gz


def read_mag_uT(bus):
    """
    讀取磁力計資料，並轉成近似 uT。

    注意：
    這裡的 scale 目前是近似值，之後若你要做 heading / 姿態融合，
    最好再依 datasheet 與實測校正。
    """
    data = read_block(bus, ADDR_XM, OUT_X_L_M, 6)

    mx_raw = to_int16(data[0], data[1])
    my_raw = to_int16(data[2], data[3])
    mz_raw = to_int16(data[4], data[5])

    scale_uT_per_lsb = 0.008  # 近似值

    mx = mx_raw * scale_uT_per_lsb
    my = my_raw * scale_uT_per_lsb
    mz = mz_raw * scale_uT_per_lsb
    return mx, my, mz


def read_real_imu9(bus):
    """
    一次讀出 9 軸資料。
    回傳順序固定為：
    ax, ay, az, gx, gy, gz, mx, my, mz
    """
    ax, ay, az = read_accel_g(bus)
    gx, gy, gz = read_gyro_dps(bus)
    mx, my, mz = read_mag_uT(bus)
    return ax, ay, az, gx, gy, gz, mx, my, mz

# =========================
# 打包前的量化
# =========================

def clamp_i16(v: int) -> int:
    """
    限制數值在 int16 可表示範圍內，避免 struct.pack 時 overflow。
    """
    return max(-32768, min(32767, v))


def q_acc_g(x):
    """
    將加速度從 g 量化成 int16。
    這裡用 x * 1000，代表 1 LSB = 0.001 g
    """
    return clamp_i16(int(x * 1000.0))


def q_gyro_dps(x):
    """
    將角速度量化成 int16。
    這裡直接取整數，代表 1 LSB = 1 deg/s
    """
    return clamp_i16(int(x))


def q_mag_uT(x):
    """
    將磁場從 uT 量化成 int16。
    這裡用 x * 10，代表 1 LSB = 0.1 uT
    """
    return clamp_i16(int(x * 10.0))


def pack_imu9_with_time(timestamp_ms, ax, ay, az, gx, gy, gz, mx, my, mz):
    """
    將時間戳 + 9軸資料打包成 binary。

    格式：
    <Ihhhhhhhhh
    <   : little-endian
    I   : uint32 timestamp_ms
    h   : int16

    總長度：
    4 + 9*2 = 22 bytes
    """
    return struct.pack(
        "<Ihhhhhhhhh",
        timestamp_ms,
        q_acc_g(ax), q_acc_g(ay), q_acc_g(az),
        q_gyro_dps(gx), q_gyro_dps(gy), q_gyro_dps(gz),
        q_mag_uT(mx), q_mag_uT(my), q_mag_uT(mz)
    )

# =========================
# Main
# =========================

def main():
    """
    主程式流程：

    1. 開啟 I2C bus
    2. 初始化 IMU
    3. 設定 BLE advertising
    4. 持續讀取 9軸資料
    5. 加入 timestamp_ms
    6. 打包成 22 bytes
    7. 更新 BLE 廣播內容
    """

    bus = SMBus(I2C_BUS)
    init_lsm9ds0(bus)

    # 先關掉再設定，避免舊 advertising 狀態干擾
    adv_disable()
    adv_setup()

    try:
        while True:
            # 讀取 9 軸資料
            ax, ay, az, gx, gy, gz, mx, my, mz = read_real_imu9(bus)

            # 使用 monotonic clock 產生毫秒時間戳
            # 好處是時間單調遞增，不會受系統校時影響
            timestamp_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF

            # 打包成 22-byte payload
            payload = pack_imu9_with_time(
                timestamp_ms,
                ax, ay, az,
                gx, gy, gz,
                mx, my, mz
            )

            # 更新 BLE 廣播資料
            set_adv_payload(payload)

            # 廣播只需要 enable 一次即可；
            # 但保留這行在某些環境下也能正常工作
            adv_enable()

            # 印出目前資料，方便你在 terminal debug
            print(
                f"t={timestamp_ms} ms | "
                f"acc=({ax:+.3f},{ay:+.3f},{az:+.3f}) g | "
                f"gyro=({gx:+.2f},{gy:+.2f},{gz:+.2f}) dps | "
                f"mag=({mx:+.2f},{my:+.2f},{mz:+.2f}) uT"
            )

            # 每 0.2 秒更新一次
            time.sleep(0.2)

    except KeyboardInterrupt:
        # Ctrl+C 時關閉廣播
        adv_disable()

    finally:
        # 不管如何都把 I2C bus 關掉
        bus.close()


if __name__ == "__main__":
    main()