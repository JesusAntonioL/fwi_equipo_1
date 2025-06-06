import cv2
import socket
import struct
import numpy as np
import os

# === Config ===
OUTPUT_DIR = "synced_data_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEMP_MIN_C = 20.0
TEMP_MAX_C = 150.0
TEMP_RANGE = TEMP_MAX_C - TEMP_MIN_C
SAVE_LIMIT = 100  # Max number of frame pairs to save

# === Optic Camera Setup ===
video_path = 1  # Use 1 for webcam
cap = cv2.VideoCapture(video_path)

# Force resolution (only works if webcam supports it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("❌ Cannot open optic video source.")
    exit()

# === Thermal Camera Setup ===
thermal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
thermal_socket.bind(("0.0.0.0", 5005))
thermal_socket.settimeout(10)
thermal_socket.listen(1)
try:
    conn, addr = thermal_socket.accept()
    print(f"[✅ Thermal connected from] {addr}")
except socket.timeout:
    print("⏰ Timeout waiting for thermal camera.")
    thermal_socket.close()
    exit()

# === Sync Loop ===
count = 0
while count < SAVE_LIMIT:
    # ---- Optic Frame ----
    ret, optic_frame = cap.read()
    if not ret:
        print("🔚 End of optic video stream.")
        break

    # ---- Thermal Frame ----
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

        thermal_raw = np.frombuffer(buffer, dtype=np.uint8).reshape((120, 160))  # Save this one
        thermal_colored = cv2.applyColorMap(thermal_raw, cv2.COLORMAP_INFERNO)

    except Exception as e:
        print(f"❌ Thermal read error: {e}")
        break

    # ---- Save Synced Frames ----
    optic_path = os.path.join(OUTPUT_DIR, f"optic_{count:04d}.png")
    thermal_img_path = os.path.join(OUTPUT_DIR, f"thermal_{count:04d}.png")
    thermal_raw_path = os.path.join(OUTPUT_DIR, f"thermal_raw_{count:04d}.npy")

    cv2.imwrite(optic_path, optic_frame)
    cv2.imwrite(thermal_img_path, thermal_colored)
    np.save(thermal_raw_path, thermal_raw)

    print(f"[💾 Saved pair {count}]")

    # ---- Display (Optional) ----
    preview_thermal = cv2.resize(thermal_colored, (optic_frame.shape[1], optic_frame.shape[0]))
    combined = np.hstack((optic_frame, preview_thermal))
    cv2.imshow("Optic + Thermal Preview", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

# === Cleanup ===
cap.release()
conn.close()
cv2.destroyAllWindows()
print("✅ Finished saving synchronized frame pairs.")