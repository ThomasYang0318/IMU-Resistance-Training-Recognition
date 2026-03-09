import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 10000))
print("listening...")
while True:
    data, addr = sock.recvfrom(1024)
    print(addr, data.decode(errors="ignore"))