import time
import struct
import socket
from smbus2 import SMBus

# =========================
# 網路與硬體設定
# =========================
TARGET_IP = "192.168.50.235"   # 改成你的接收端電腦 IP
TARGET_PORT = 5005             # 要和接收端一致
I2C_BUS = 2                    # Luckfox I2C Bus

# LSM9DS0 I2C 地址
ADDR_G  = 0x6B
ADDR_XM = 0x1D

# =========================
# LSM9DS0 暫存器定義
# =========================
# Gyro Registers
CTRL_REG1_G = 0x20
CTRL_REG4_G = 0x23
OUT_X_L_G   = 0x28

# Accel/Mag Registers
CTRL_REG1_XM = 0x20
CTRL_REG2_XM = 0x21
CTRL_REG5_XM = 0x24
CTRL_REG6_XM = 0x25
CTRL_REG7_XM = 0x26
OUT_X_L_A    = 0x28
OUT_X_L_M    = 0x08

PACKET_FORMAT = "<I9h"
SEND_HZ = 50.0
SEND_INTERVAL = 1.0 / SEND_HZ


class LSM9DS0UDP:
    def __init__(self, bus_id, remote_ip, remote_port):
        self.bus = SMBus(bus_id)
        self.addr = (remote_ip, remote_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.init_sensor()

    def write_reg(self, addr, reg, data):
        self.bus.write_byte_data(addr, reg, data)

    def read_block(self, addr, reg, length):
        # 啟動 auto-increment 多位元組讀取
        return self.bus.read_i2c_block_data(addr, reg | 0x80, length)

    def init_sensor(self):
        """
        初始化 LSM9DS0
        參考 SparkFun LSM9DS0 設定概念
        """

        # =========================
        # Gyro
        # CTRL_REG1_G = 0x0F
        # 95Hz ODR, 正常模式, XYZ enable
        # =========================
        self.write_reg(ADDR_G, CTRL_REG1_G, 0x0F)

        # CTRL_REG4_G = 0x00
        # ±245 dps
        self.write_reg(ADDR_G, CTRL_REG4_G, 0x00)

        # =========================
        # Accelerometer
        # CTRL_REG1_XM = 0x67
        # 100Hz ODR, XYZ enable
        # =========================
        self.write_reg(ADDR_XM, CTRL_REG1_XM, 0x67)

        # CTRL_REG2_XM = 0x00
        # ±2g, BW 預設
        self.write_reg(ADDR_XM, CTRL_REG2_XM, 0x00)

        # =========================
        # Magnetometer
        # CTRL_REG5_XM = 0x94
        # High resolution, 50Hz ODR
        # =========================
        self.write_reg(ADDR_XM, CTRL_REG5_XM, 0x94)

        # CTRL_REG6_XM = 0x00
        # ±2 gauss
        self.write_reg(ADDR_XM, CTRL_REG6_XM, 0x00)

        # CTRL_REG7_XM = 0x00
        # Continuous-conversion mode
        self.write_reg(ADDR_XM, CTRL_REG7_XM, 0x00)

        time.sleep(0.1)

    @staticmethod
    def to_int16(low, high):
        val = (high << 8) | low
        return val if val < 32768 else val - 65536

    def read_raw(self):
        # 各讀 6 bytes = X/Y/Z，每軸 16-bit
        g_data = self.read_block(ADDR_G, OUT_X_L_G, 6)
        a_data = self.read_block(ADDR_XM, OUT_X_L_A, 6)
        m_data = self.read_block(ADDR_XM, OUT_X_L_M, 6)

        ax = self.to_int16(a_data[0], a_data[1])
        ay = self.to_int16(a_data[2], a_data[3])
        az = self.to_int16(a_data[4], a_data[5])

        gx = self.to_int16(g_data[0], g_data[1])
        gy = self.to_int16(g_data[2], g_data[3])
        gz = self.to_int16(g_data[4], g_data[5])

        mx = self.to_int16(m_data[0], m_data[1])
        my = self.to_int16(m_data[2], m_data[3])
        mz = self.to_int16(m_data[4], m_data[5])

        return ax, ay, az, gx, gy, gz, mx, my, mz

    def run(self):
        print(f"[INFO] UDP stream start -> {self.addr}")
        print(f"[INFO] Packet format: {PACKET_FORMAT}, size={struct.calcsize(PACKET_FORMAT)} bytes")
        print(f"[INFO] Send rate: {SEND_HZ:.1f} Hz")

        next_time = time.perf_counter()

        try:
            while True:
                timestamp_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
                raw = self.read_raw()

                packet = struct.pack(PACKET_FORMAT, timestamp_ms, *raw)
                self.sock.sendto(packet, self.addr)

                # 終端機簡單顯示
                ax, ay, az, gx, gy, gz, mx, my, mz = raw
                print(
                    f"t={timestamp_ms:10d} | "
                    f"acc=({ax:6d},{ay:6d},{az:6d}) | "
                    f"gyro=({gx:6d},{gy:6d},{gz:6d}) | "
                    f"mag=({mx:6d},{my:6d},{mz:6d})"
                )

                next_time += SEND_INTERVAL
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # 落後太多時重設節奏，避免越來越飄
                    next_time = time.perf_counter()

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            self.bus.close()
            self.sock.close()
            print("[INFO] Resources closed.")


if __name__ == "__main__":
    imu = LSM9DS0UDP(I2C_BUS, TARGET_IP, TARGET_PORT)
    imu.run()