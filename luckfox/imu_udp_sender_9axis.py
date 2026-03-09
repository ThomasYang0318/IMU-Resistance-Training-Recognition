import socket
import time
import struct
import fcntl

DEST_IP = "192.168.0.125"   # 改成你的 Mac IP
DEST_PORT = 10000

I2C_DEV = "/dev/i2c-2"

ADDR_XM = 0x1d   # accel/mag
ADDR_G  = 0x6b   # gyro

FS = 100
DT = 1.0 / FS

I2C_SLAVE = 0x0703


def i2c_write(fd, addr, reg, val):
    fcntl.ioctl(fd, I2C_SLAVE, addr)
    fd.write(bytes([reg, val]))


def i2c_read(fd, addr, reg, length):
    fcntl.ioctl(fd, I2C_SLAVE, addr)
    fd.write(bytes([reg]))
    return fd.read(length)


def read_vec3(fd, addr, start_reg):
    data = i2c_read(fd, addr, start_reg | 0x80, 6)
    x = struct.unpack('<h', data[0:2])[0]
    y = struct.unpack('<h', data[2:4])[0]
    z = struct.unpack('<h', data[4:6])[0]
    return x, y, z


def init_lsm9ds0(fd):
    # ---------- Gyro ----------
    # CTRL_REG1_G (0x20)
    # ODR=95Hz, Cutoff=12.5, normal mode, XYZ enable
    i2c_write(fd, ADDR_G, 0x20, 0x0F)

    # CTRL_REG4_G (0x23)
    # full-scale = 245 dps
    i2c_write(fd, ADDR_G, 0x23, 0x00)

    # ---------- Accel ----------
    # CTRL_REG1_XM (0x20)
    # 100 Hz accel, all axes enabled
    i2c_write(fd, ADDR_XM, 0x20, 0x67)

    # CTRL_REG2_XM (0x21)
    # accel scale = ±2g
    i2c_write(fd, ADDR_XM, 0x21, 0x00)

    # ---------- Magnetometer ----------
    # CTRL_REG5_XM (0x24)
    # temp disable, mag high resolution, ODR = 50 Hz
    i2c_write(fd, ADDR_XM, 0x24, 0xF0)

    # CTRL_REG6_XM (0x25)
    # mag full scale = ±2 gauss
    i2c_write(fd, ADDR_XM, 0x25, 0x00)

    # CTRL_REG7_XM (0x26)
    # continuous-conversion mode
    i2c_write(fd, ADDR_XM, 0x26, 0x00)

    print("LSM9DS0 9-axis initialized")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fd = open(I2C_DEV, "rb+", buffering=0)

    init_lsm9ds0(fd)

    last_send = time.time()
    count = 0
    stat_t0 = time.time()

    print(f"9-axis UDP sender → {DEST_IP}:{DEST_PORT} @ {FS}Hz")

    while True:
        now = time.time()
        if now - last_send < DT:
            continue

        last_send = now
        ts = int(time.time() * 1000)

        ax, ay, az = read_vec3(fd, ADDR_XM, 0x28)  # accel
        gx, gy, gz = read_vec3(fd, ADDR_G,  0x28)  # gyro
        mx, my, mz = read_vec3(fd, ADDR_XM, 0x08)  # mag

        msg = f"{ts},{ax},{ay},{az},{gx},{gy},{gz},{mx},{my},{mz}"

        try:
            sock.sendto(msg.encode(), (DEST_IP, DEST_PORT))
        except OSError as e:
            print(f"send failed: {e}")
            time.sleep(1)
            continue

        count += 1
        if time.time() - stat_t0 >= 1.0:
            print(f"rate≈{count/(time.time()-stat_t0):.1f} Hz, last={msg}")
            stat_t0 = time.time()
            count = 0


if __name__ == "__main__":
    main()