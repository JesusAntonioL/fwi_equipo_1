import cv2
import numpy as np
import os
import glob

# === Config ===
PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 20.0
N_CORNERS = PATTERN_SIZE[0] * PATTERN_SIZE[1]
DATA_DIR = "calibration_pairs"

# === Prepare object points ===
objp = np.zeros((N_CORNERS, 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
optic_points = []
thermal_points = []

# === Load manual points ===
manual_optic = sorted(glob.glob(os.path.join(DATA_DIR, "optic_annotated_*.npy")))
manual_thermal = sorted(glob.glob(os.path.join(DATA_DIR, "thermal_annotated_*.npy")))

assert len(manual_optic) == len(manual_thermal), "Mismatched number of manual annotation files."

for optic_file, thermal_file in zip(manual_optic, manual_thermal):
    opt_pts = np.load(optic_file)
    thm_pts = np.load(thermal_file)

    assert opt_pts.shape[0] == N_CORNERS and thm_pts.shape[0] == N_CORNERS, f"Incorrect number of points in {optic_file} / {thermal_file}"

    objpoints.append(objp)
    optic_points.append(opt_pts)
    thermal_points.append(thm_pts)

    print(f"📦 Loaded {optic_file} and {thermal_file}")

# === Image size from original images ===
optic_img = cv2.imread(os.path.join(DATA_DIR, "optic_00.png"))
optic_size = optic_img.shape[1::-1]

thermal_img = cv2.imread(os.path.join(DATA_DIR, "thermal_00.png"))
thermal_size = thermal_img.shape[1::-1]

# === Intrinsic calibration ===
_, K1, D1, _, _ = cv2.calibrateCamera(objpoints, optic_points, optic_size, None, None)
_, K2, D2, _, _ = cv2.calibrateCamera(objpoints, thermal_points, thermal_size, None, None)

# === Stereo calibration ===
flags = cv2.CALIB_FIX_INTRINSIC
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, optic_points, thermal_points,
    K1, D1, K2, D2,
    optic_size,
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
    flags=flags
)

print("\n✅ Stereo Calibration Done!")
print(f"Rotation Matrix (R):\n{R}")
print(f"Translation Vector (T):\n{T}")

# === Save calibration ===
np.savez("stereo_calibration_manual.npz",
         K1=K1, D1=D1,
         K2=K2, D2=D2,
         R=R, T=T, E=E, F=F,
         image_size=optic_size)

print("💾 Calibration saved to stereo_calibration_manual.npz")