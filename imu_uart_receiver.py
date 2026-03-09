import serial

PORT = "/dev/cu.usbserial-xxxx"   # 改成你的 USB-TTL
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

print("UART receiver started")

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if line:
        print(line)