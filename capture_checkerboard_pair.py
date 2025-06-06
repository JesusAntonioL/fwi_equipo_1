import cv2
import socket
import struct
import numpy as np
import os

# === Config ===
SAVE_DIR = "calibration_pairs"
os.makedirs(SAVE_DIR, exist_ok=True)
thermal_host = "0.0.0.0"
thermal_port = 5005
img_count = 0

# === Optic Camera Setup ===
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("❌ Cannot open optic camera.")
    exit()

# === Thermal Socket Setup ===
thermal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
thermal_socket.bind((thermal_host, thermal_port))
thermal_socket.listen(1)
thermal_socket.settimeout(10)
try:
    conn, addr = thermal_socket.accept()
    print(f"✅ Thermal camera connected from {addr}")
except socket.timeout:
    print("⏰ Timeout waiting for thermal camera.")
    exit()

print("📷 Press SPACE to capture image pair. Press Q to quit.")

while True:
    # --- Get optic frame ---
    ret, optic = cap.read()
    if not ret:
        break

    # --- Get thermal frame over TCP ---
    try:
        header = conn.recv(4)
        if not header:
            break
        length = struct.unpack(">I", header)[0]
        buffer = b''
        while len(buffer) < length:
            chunk = conn.recv(length - len(buffer))
            if not chunk:
                break
            buffer += chunk
        thermal_raw = np.frombuffer(buffer, dtype=np.uint8).reshape((120, 160))
        thermal_colored = cv2.applyColorMap(thermal_raw, cv2.COLORMAP_INFERNO)
    except Exception as e:
        print(f"Thermal read error: {e}")
        break

    # --- Display both images ---
    thermal_upscaled = cv2.resize(thermal_colored, (640, 480))
    preview = np.hstack((cv2.resize(optic, (640, 480)), thermal_upscaled))
    cv2.imshow("Optic | Thermal", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE key
        optic_path = os.path.join(SAVE_DIR, f"optic_{img_count:02d}.png")
        thermal_path = os.path.join(SAVE_DIR, f"thermal_{img_count:02d}.png")
        cv2.imwrite(optic_path, optic)
        cv2.imwrite(thermal_path, thermal_colored)
        print(f"[💾 Saved] {optic_path}, {thermal_path}")
        img_count += 1

cap.release()
conn.close()
cv2.destroyAllWindows()