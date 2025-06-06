import cv2
import socket
import struct
import numpy as np
from ultralytics import YOLO
import torch

# === Device Setup ===
device = torch.device(0 if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Using CPU only.")

# === Config ===
SAVE_FRAMES = False
FUSE_ALPHA = 0.25
FUSE_BETA = 0.75
MODEL_PATH = "Models/fire_l.pt"
model = YOLO(MODEL_PATH)

# === Load Calibration Parameters ===
K1 = np.load("stereo_cal/intrinsics_optic.npy")
D1 = np.load("stereo_cal/distortion_optic.npy")
K2 = np.load("stereo_cal/intrinsics_thermal.npy")
D2 = np.load("stereo_cal/distortion_thermal.npy")
R1 = np.load("stereo_cal/stereo_rect_R1.npy")
R2 = np.load("stereo_cal/stereo_rect_R2.npy")
P1 = np.load("stereo_cal/stereo_rect_P1.npy")
P2 = np.load("stereo_cal/stereo_rect_P2.npy")

# === Camera & Socket Setup ===
# Optic camera
cap = cv2.VideoCapture("/dev/video2")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("❌ Cannot open optic camera.")
    exit()

# Thermal TCP socket
thermal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
thermal_socket.bind(("0.0.0.0", 5005))
thermal_socket.listen(1)
thermal_socket.settimeout(10)
try:
    conn, addr = thermal_socket.accept()
    print(f"🟢 Thermal camera connected from {addr}")
except socket.timeout:
    print("❌ Thermal connection timeout.")
    exit()

# === Setup Sizes and Rectification Maps ===
ret, sample_optic = cap.read()
if not ret:
    print("❌ Failed to grab sample optic frame.")
    exit()

optic_size = (sample_optic.shape[1], sample_optic.shape[0])  # (width, height)
thermal_size = (160, 120)  # FLIR Lepton size

# Build undistort/rectify maps
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, optic_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, thermal_size, cv2.CV_32FC1)

print("✅ Stereo rectification maps initialized.")
print("📡 Press 'q' to quit, 's' to save frame.")

# === Main Loop ===
frame_idx = 0
while True:
    # === Get Optic Frame ===
    ret, optic = cap.read()
    if not ret:
        print("⚠️ Failed to grab optic frame.")
        break

    # === Get Thermal Frame over TCP ===
    try:
        header = conn.recv(4)
        if not header:
            print("⚠️ No header received.")
            break
        length = struct.unpack(">I", header)[0]

        buffer = b''
        while len(buffer) < length:
            chunk = conn.recv(length - len(buffer))
            if not chunk:
                print("⚠️ Lost connection to thermal camera.")
                break
            buffer += chunk

        if len(buffer) != 160 * 120:
            print(f"⚠️ Received {len(buffer)} bytes, expected 19200.")
            continue

        thermal_raw = np.frombuffer(buffer, dtype=np.uint8).reshape((120, 160))

    except Exception as e:
        print(f"⚠️ Thermal error: {e}")
        break

    # === Rectification ===
    rect_optic = cv2.remap(optic, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    rect_thermal = cv2.remap(thermal_raw, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    # === Convert thermal to color after rectification ===
    thermal_color = cv2.applyColorMap(rect_thermal, cv2.COLORMAP_INFERNO)
    thermal_resized = cv2.resize(thermal_color, (rect_optic.shape[1], rect_optic.shape[0]))

    # === Object Detection ===
    results = model.predict(rect_optic, conf=0.5, device=device)
    annotated_optic = results[0].plot()

    # === Fusion ===
    fused = cv2.addWeighted(annotated_optic, FUSE_ALPHA, thermal_resized, FUSE_BETA, 0)

    # === Display ===
    cv2.imshow("Optic Rectified", rect_optic)
    cv2.imshow("Thermal Rectified", thermal_resized)
    cv2.imshow("Fused View", fused)

    # === Save or Quit ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') or (SAVE_FRAMES and frame_idx < 100):
        cv2.imwrite(f"fused_output/optic_{frame_idx:03d}.png", rect_optic)
        cv2.imwrite(f"fused_output/thermal_{frame_idx:03d}.png", thermal_resized)
        cv2.imwrite(f"fused_output/fused_{frame_idx:03d}.png", fused)
        print(f"[💾 Saved frame {frame_idx}]")
        frame_idx += 1

# === Cleanup ===
cap.release()
conn.close()
cv2.destroyAllWindows()