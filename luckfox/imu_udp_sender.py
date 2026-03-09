import socket
import time
import struct
import fcntl
import math

# ===============================
# 設定
# ===============================

DEST_IP = "192.168.0.125"   # Mac IP
DEST_PORT = 10000

I2C_DEV = "/dev/i2c-2"

ADDR_XM = 0x1d   # accel/mag
ADDR_G  = 0x6b   # gyro

FS = 100
DT = 1.0 / FS

# ===============================
# I2C ioctl constants
# ===============================

I2C_SLAVE = 0x0703


# ===============================
# I2C helper
# ===============================

def i2c_write(fd, addr, reg, val):
    fcntl.ioctl(fd, I2C_SLAVE, addr)
    fd.write(bytes([reg, val]))

def i2c_read(fd, addr, reg, length):
    fcntl.ioctl(fd, I2C_SLAVE, addr)
    fd.write(bytes([reg]))
    return fd.read(length)


# ===============================
# LSM9DS0 init
# ===============================

def init_lsm9ds0(fd):

    # gyro
    i2c_write(fd, ADDR_G, 0x20, 0x0F)

    # accel
    i2c_write(fd, ADDR_XM, 0x20, 0x67)

    print("LSM9DS0 initialized")


# ===============================
# read accel
# ===============================

def read_accel(fd):

    data = i2c_read(fd, ADDR_XM, 0x28 | 0x80, 6)

    ax = struct.unpack('<h', data[0:2])[0]
    ay = struct.unpack('<h', data[2:4])[0]
    az = struct.unpack('<h', data[4:6])[0]

    return ax, ay, az


# ===============================
# main
# ===============================

def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    fd = open(I2C_DEV, "rb+", buffering=0)

    init_lsm9ds0(fd)

    last = time.time()
    count = 0

    print(f"UDP sender → {DEST_IP}:{DEST_PORT} @ {FS}Hz")

    while True:

        now = time.time()

        if now - last >= DT:

            last = now

            ts = int(time.time() * 1000)

            ax, ay, az = read_accel(fd)

            msg = f"{ts},{ax},{ay},{az}"

            sock.sendto(msg.encode(), (DEST_IP, DEST_PORT))

            count += 1

            if count % 100 == 0:
                print(f"sent={count}, last={msg}")


if __name__ == "__main__":
    main()