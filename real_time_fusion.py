import cv2
import socket
import struct
import numpy as np

from ultralytics import YOLO
import torch

# === Defines GPU or CPU in App ===
device = torch.device(0 if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")

print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# === Config ===
H_PATH = "output/homography_matrix.npy"
SAVE_FRAMES = False
FUSE_ALPHA = 0.25
FUSE_BETA = 0.75

# === Load homography matrix ===
H = np.load(H_PATH)
print("✅ Loaded homography matrix.")

MODEL_PATH = "Models/fire_l.pt"
model = YOLO(MODEL_PATH)

# === Thermal TCP setup ===
thermal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
thermal_socket.bind(("0.0.0.0", 5005))
thermal_socket.listen(1)
thermal_socket.settimeout(10)
try:
    conn, addr = thermal_socket.accept()
    print(f"[🟢 Thermal camera connected from {addr}]")
except socket.timeout:
    print("❌ Thermal connection timeout.")
    exit()

# === Optic camera setup ===
cap = cv2.VideoCapture("/dev/video2") # Use 1 for Video Capture Camera in Windows and /dev/video2 for Video Capture Camera in Ubuntu
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("❌ Cannot open optic camera.")
    exit()

frame_idx = 0
print("📡 Press 'q' to quit, 's' to save frame.")

while True:
    # --- Get optic frame ---
    ret, optic = cap.read()
    if not ret:
        print("⚠️ Failed to grab optic frame.")
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
        print(f"⚠️ Thermal error: {e}")
        break
    
    results = model.predict(optic, conf=0.5, device=device)
    annotated_frame = results[0].plot()

    # --- Warp thermal to match optic using homography ---
    warped_thermal = cv2.warpPerspective(thermal_colored, H, (optic.shape[1], optic.shape[0]))

    # --- Fuse ---
    fused = cv2.addWeighted(annotated_frame, FUSE_ALPHA, warped_thermal, FUSE_BETA, 0)

    # --- Display ---
    cv2.imshow("Optic", optic)
    cv2.imshow("Thermal Warped", warped_thermal)
    cv2.imshow("Fused", fused)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') or (SAVE_FRAMES and frame_idx < 100):
        cv2.imwrite(f"fused_output/optic_{frame_idx:03d}.png", optic)
        cv2.imwrite(f"fused_output/thermal_{frame_idx:03d}.png", warped_thermal)
        cv2.imwrite(f"fused_output/fused_{frame_idx:03d}.png", fused)
        print(f"[💾 Saved frame {frame_idx}]")
        frame_idx += 1

# === Cleanup ===
cap.release()
conn.close()
cv2.destroyAllWindows()