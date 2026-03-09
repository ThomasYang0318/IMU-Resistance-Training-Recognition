import socket
import csv
import time
import select
import sys

UDP_IP = "0.0.0.0"
UDP_PORT = 10000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT}")
print("Commands:")
print("  r  → start/stop recording")
print("  q  → quit")

recording = False
csv_file = None
csv_writer = None

# rate monitor
count = 0
t0 = time.time()

try:

    while True:

        readable, _, _ = select.select([sock, sys.stdin], [], [], 0.1)

        for r in readable:

            # -------------------------
            # UDP data
            # -------------------------
            if r == sock:

                data, addr = sock.recvfrom(1024)
                line = data.decode().strip()

                print(line)

                count += 1

                # print Hz every second
                now = time.time()
                if now - t0 > 1:
                    print(f"[IMU RATE] {count} Hz")
                    count = 0
                    t0 = now

                if recording and csv_writer:
                    csv_writer.writerow(line.split(","))

            # -------------------------
            # keyboard input
            # -------------------------
            elif r == sys.stdin:

                cmd = sys.stdin.readline().strip()

                if cmd == "r":

                    recording = not recording

                    if recording:

                        filename = f"imu_{int(time.time())}.csv"
                        csv_file = open(filename, "w", newline="")
                        csv_writer = csv.writer(csv_file)

                        csv_writer.writerow(["timestamp", "ax", "ay", "az"])

                        print(f"[RECORDING START] → {filename}")

                    else:

                        print("[RECORDING STOP]")

                        if csv_file:
                            csv_file.flush()
                            csv_file.close()
                            csv_file = None
                            csv_writer = None

                elif cmd == "q":

                    raise KeyboardInterrupt

except KeyboardInterrupt:

    print("\nShutting down...")

finally:

    if csv_file:
        csv_file.flush()
        csv_file.close()

    sock.close()

    print("Socket closed")