import socket
import struct
import numpy as np
import cv2

TEMP_MIN_C = 20.0
TEMP_MAX_C = 150.0
TEMP_RANGE = TEMP_MAX_C - TEMP_MIN_C

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 5005))
s.settimeout(10)
s.listen(1)

try:
    conn, addr = s.accept()
    print(f"[✅ Conectado desde] {addr}")
except socket.timeout:
    print("⏰ Tiempo de espera agotado. Nadie se conectó.")
    s.close()
    exit()

while True:
    header = conn.recv(4)
    if not header:
        break

    length = struct.unpack(">I", header)[0]

    if length != 160 * 120:
        print(f"[⚠️ Warning] Expected 19200 bytes but got {length}")
        continue

    buffer = b''
    while len(buffer) < length:
        chunk = conn.recv(length - len(buffer))
        if not chunk:
            break
        buffer += chunk

    if len(buffer) != length:
        print("[⚠️ Incomplete frame received]")
        continue

    frame = np.frombuffer(buffer, dtype=np.uint8).reshape((120, 160))

    # Convert to temperature map
    temperature_matrix = (frame.astype(np.float32) / 255.0) * TEMP_RANGE + TEMP_MIN_C

    # Apply colormap
    color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    color_frame = cv2.resize(color_frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Display
    cv2.imshow("Thermal View (Live)", color_frame)
    print(f"Max Temp Received: {np.max(temperature_matrix):.2f} °C")

    if cv2.waitKey(1) == ord('q'):
        break

conn.close()