import cv2
import socket
import struct
import numpy as np
import os
from ultralytics import YOLO
import torch
from pymavlink import mavutil
import subprocess
import time
import threading
import sys

cv2.setNumThreads(1) # Allows to control the number for internal parallelization within the library.
python_executable = sys.executable  # Utilizes the actual venv

# === Define GPU or CPU ===
device = torch.device(0 if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# === Mode Selection ===
USE_LIVE_STREAM = False #If True this will enable real time data obtaining for processing for optic and thermal imaging
SHOW_FUSION = True #If True will show fusioned images with display, set False for faster process of data.
ENABLE_RECEIVE_GPS = False #If True will receive data from RTK antennas, if set False will use lat and long pre-saved.

# === Data Configuration ===
DATA_FOLDER = "synced_data_2" #Folder for data when USE_LIVE_STREAM is False
DESIRED_FPS = 30 #Desired FPS when USE_LIVE_STREAM is False
FRAME_DELAY = 1.0 / DESIRED_FPS

# === Config ===
H_PATH = "output/homography_matrix.npy" #Path for homography matrix
FUSE_ALPHA = 0.25 #Percentage of showing the optic image
FUSE_BETA = 0.75 #Percentage of showing the thermal image
MODEL_PATH = "Models/fire_l.pt"
model = YOLO(MODEL_PATH) # Loading YOLO model
confidence = 0.5 #confidence required

# === Load Homography ===
H = np.load(H_PATH)
H_inv = np.linalg.inv(H)
print(H_inv)
print("✅ Loaded homography matrix.")

# === GPS Connection Protocol through Mavlink ===
if ENABLE_RECEIVE_GPS:
    master = mavutil.mavlink_connection('COM5', baud=115200) # Check COM port from ardupilot when the antenna is connected
    master.wait_heartbeat()
    print("Connected to system:", master.target_system)
else:
    # === Latitude and Longitude of CB1 Escamilla ===
    lat = 25.65617721063847
    lon = -100.28706376985032

# === Function to obtain and parse the lat, lon and alt data from the GPS RTK system ===
def obtain_coord():
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt
        print(f"Latitude: {lat}, Longitude: {lon}, Altura: {alt}")
        return lat, lon

# === Threaded Thermal Receiver ===
class ThermalThread(threading.Thread):
    def __init__(self, port=5005):
        super().__init__()
        self.daemon = True
        self.frame = None
        self.raw = None
        self.lock = threading.Lock()
        self.running = True
        self.port = port

    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("0.0.0.0", self.port))
        server_socket.listen(1)
        server_socket.settimeout(10)
        try:
            conn, addr = server_socket.accept()
            print(f"[Thermal camera connected from {addr}]")
        except socket.timeout:
            print("Thermal connection timeout.")
            return

        while self.running:
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
                raw = np.frombuffer(buffer, dtype=np.uint8).reshape((120, 160))
                colored = cv2.applyColorMap(raw, cv2.COLORMAP_INFERNO)
                with self.lock:
                    self.frame = colored
                    self.raw = raw
            except:
                break
        conn.close()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.raw.copy() if self.raw is not None else None

    def stop(self):
        self.running = False

# === Threaded Optic Camera ===
class OpticThread(threading.Thread):
    def __init__(self, device=1):
        super().__init__()
        self.daemon = True
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            print("Cannot open optic camera.")
            exit()
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

# === Start threads ===
if USE_LIVE_STREAM:
    thermal_thread = ThermalThread()
    optic_thread = OpticThread()
    thermal_thread.start()
    optic_thread.start()
else:
    image_index = 0
    def files_exist(idx):
        return all([
            os.path.exists(f"{DATA_FOLDER}/optic_{idx:04d}.png"),
            os.path.exists(f"{DATA_FOLDER}/thermal_{idx:04d}.png"),
            os.path.exists(f"{DATA_FOLDER}/thermal_raw_{idx:04d}.npy"),
        ])

print("📡 Press 'q' to quit, 's' to save frame.")
frame_idx = 0

# === Main Loop for Image Processing ===
try:
    while True:
        if USE_LIVE_STREAM:
            optic = optic_thread.get_frame()
            thermal_colored, thermal_raw = thermal_thread.get_frame()
            if optic is None or thermal_colored is None:
                continue
        else:
            if not files_exist(image_index):
                print("📁 No more data to process.")
                break
            optic = cv2.imread(f"{DATA_FOLDER}/optic_{image_index:04d}.png")
            thermal_colored = cv2.imread(f"{DATA_FOLDER}/thermal_{image_index:04d}.png")
            thermal_raw = np.load(f"{DATA_FOLDER}/thermal_raw_{image_index:04d}.npy")
            image_index += 1
            time.sleep(FRAME_DELAY)
            pass
        
        # === Generate optic image prediction ===
        results = model.predict(optic, conf=confidence, device=device)
        annotated_frame = results[0].plot()
        maxVal = 0

        for det in results[0].boxes:
            cls_id = int(det.cls[0])
            cls_name = model.names[cls_id]
            
            # === Processing of hottest point in the bbox if the class is fire ===
            if cls_name.lower() == "fire":
                bbox = det.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                bbox_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                bbox_pts_thermal = cv2.perspectiveTransform(np.array([bbox_pts]), H_inv)[0]
                xt, yt, wt, ht = cv2.boundingRect(bbox_pts_thermal.astype(np.int32))
                xt, yt = max(0, xt), max(0, yt)
                wt = min(thermal_raw.shape[1] - xt, wt)
                ht = min(thermal_raw.shape[0] - yt, ht)

                roi = thermal_raw[yt:yt+ht, xt:xt+wt]
                if roi.size > 0:
                    maxVal = np.max(roi)
                    # === Process for generating the themal transformation and generating the pointer for the hottest point and visulation of temperature ===
                    if SHOW_FUSION:
                        maxLoc = np.unravel_index(np.argmax(roi), roi.shape)
                        thermal_pt = (xt + maxLoc[1], yt + maxLoc[0])
                        optic_pt = cv2.perspectiveTransform(np.array([[thermal_pt]], dtype=np.float32), H)[0][0]
                        optic_pt = tuple(map(int, optic_pt))

                        cv2.drawMarker(annotated_frame, optic_pt, (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
                        cv2.putText(annotated_frame, f"{maxVal:.1f}C", (optic_pt[0]+10, optic_pt[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        print("🔥 Max Temp in Detection:", maxVal, "C")
                    else:
                        print("🔥 Max Temp in Detection:", maxVal, "C")

        # === Show fused image ===
        if SHOW_FUSION:
            warped_thermal = cv2.warpPerspective(thermal_colored, H, (optic.shape[1], optic.shape[0]))
            fused = cv2.addWeighted(annotated_frame, FUSE_ALPHA, warped_thermal, FUSE_BETA, 0)
            cv2.imshow("Fused", fused)

        # === Updates the frame ===
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            frame_idx += 1
            
        # === Value of risky temperature, runs the prediction model ===
        if maxVal >= 70:
            subprocess.run([python_executable, 'G:/My Drive/expansion_incendios/fwi-map2.py', str(lat), str(lon), str(maxVal)])
            break

finally:
    if USE_LIVE_STREAM:
        optic_thread.stop()
        thermal_thread.stop()
    cv2.destroyAllWindows()