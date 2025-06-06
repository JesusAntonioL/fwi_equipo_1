import cv2
import numpy as np
import os
import glob

# === Config ===
PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 20.0
N_CORNERS = PATTERN_SIZE[0] * PATTERN_SIZE[1]
DATA_DIR = "stereo_calibration_pairs"

# === Prepare object points ===
objp = np.zeros((N_CORNERS, 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
optic_points = []
thermal_points = []

# === Load manual points ===
manual_optic = sorted(glob.glob(os.path.join(DATA_DIR, "optic_*_annotated.npy")))
manual_thermal = sorted(glob.glob(os.path.join(DATA_DIR, "thermal_*_annotated.npy")))

assert len(manual_optic) == len(manual_thermal), "Mismatched number of annotation files."

for optic_file, thermal_file in zip(manual_optic, manual_thermal):
    opt_pts = np.load(optic_file)
    thm_pts = np.load(thermal_file)

    assert opt_pts.shape[0] == N_CORNERS and thm_pts.shape[0] == N_CORNERS, \
        f"Invalid corner count in: {optic_file} / {thermal_file}"

    objpoints.append(objp)
    optic_points.append(opt_pts.astype(np.float32))
    thermal_points.append(thm_pts.astype(np.float32))

    print(f"📦 Loaded {optic_file} and {thermal_file}")

# === Image size from original images ===
optic_img = cv2.imread(os.path.join(DATA_DIR, "optic_01.png"))
optic_size = optic_img.shape[1::-1]

thermal_img = cv2.imread(os.path.join(DATA_DIR, "thermal_01.png"), cv2.IMREAD_GRAYSCALE)
thermal_size = thermal_img.shape[1::-1]

# === Calibrate both cameras ===
print("\n🔍 Calibrating optic camera...")
_, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, optic_points, optic_size, None, None)

print("🔍 Calibrating thermal camera...")
_, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, thermal_points, thermal_size, None, None)

# === Compute reprojection errors ===
def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, D):
    errors = []
    for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
        proj_points, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        proj_points = proj_points.reshape(-1, 2)
        error = cv2.norm(imgp, proj_points, cv2.NORM_L2) / len(objp)
        errors.append(error)
    return errors

optic_errors = compute_reprojection_error(objpoints, optic_points, rvecs1, tvecs1, K1, D1)
thermal_errors = compute_reprojection_error(objpoints, thermal_points, rvecs2, tvecs2, K2, D2)

# === Print results ===
print("\n📊 Reprojection Errors:")
for i, (eo, et) in enumerate(zip(optic_errors, thermal_errors)):
    print(f"Pair {i:02d}: Optic = {eo:.3f}px, Thermal = {et:.3f}px")

avg_o = np.mean(optic_errors)
avg_t = np.mean(thermal_errors)

print(f"\n✅ Average Optic Error:   {avg_o:.3f}px")
print(f"✅ Average Thermal Error: {avg_t:.3f}px")

# === Optional: flag worst samples ===
threshold = 1.0
bad_pairs = [i for i, (eo, et) in enumerate(zip(optic_errors, thermal_errors)) if eo > threshold or et > threshold]
if bad_pairs:
    print(f"\n⚠️  High-error pairs (>{threshold}px): {bad_pairs}")
else:
    print("\n🎯 All image pairs within acceptable reprojection error range.")

# === Stereo Calibration ===
print("\n🔁 Performing stereo calibration...")

flags = cv2.CALIB_FIX_INTRINSIC  # We already calibrated both cameras
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    optic_points,
    thermal_points,
    K1, D1,
    K2, D2,
    optic_size,
    criteria=criteria,
    flags=flags
)

print("\n📸 Stereo calibration complete.")
print(f" - Reprojection error: {retval:.4f}")
print(f" - Translation vector:\n{T}")
print(f" - Rotation matrix:\n{R}")

# === Save results ===
np.save("stereo_cal/stereo_R.npy", R)
np.save("stereo_cal/stereo_T.npy", T)
np.save("stereo_cal/stereo_E.npy", E)
np.save("stereo_cal/stereo_F.npy", F)
print("\n💾 Stereo calibration matrices saved.")

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2,
    optic_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
)

# Save rectification transforms if needed
np.save("stereo_cal/stereo_rect_R1.npy", R1)
np.save("stereo_cal/stereo_rect_R2.npy", R2)
np.save("stereo_cal/stereo_rect_P1.npy", P1)
np.save("stereo_cal/stereo_rect_P2.npy", P2)
np.save("stereo_cal/stereo_Q.npy", Q)
print("💾 Stereo rectification matrices saved.")

# === Save monocular calibration results ===
np.save("stereo_cal/intrinsics_optic.npy", K1)
np.save("stereo_cal/distortion_optic.npy", D1)
np.save("stereo_cal/intrinsics_thermal.npy", K2)
np.save("stereo_cal/distortion_thermal.npy", D2)
print("💾 Monocular calibration matrices saved.")
