import cv2
import numpy as np
import os

# === Paths ===
optic_path = "calibration_pairs/optic_01.png"
thermal_path = "calibration_pairs/thermal_02.png"

optic_points_display = []
thermal_points_display = []
optic_points = []
thermal_points = []

# === Output Directory ===
os.makedirs("output", exist_ok=True)

optic_img = cv2.imread(optic_path)
thermal_img = cv2.imread(thermal_path)

if optic_img is None or thermal_img is None:
    print("❌ Could not load input images.")
    exit()

# === GUI scale factor for thermal (display only) ===
THERMAL_SCALE = 8  # 3x scale (160x120 → 480x360 for example)

thermal_display = cv2.resize(thermal_img, (thermal_img.shape[1]*THERMAL_SCALE, thermal_img.shape[0]*THERMAL_SCALE))

# === Storage ===
optic_points = []
thermal_points_scaled = []  # Scaled click points
thermal_points = []         # Actual thermal coordinates

# === Mouse callback ===
def select_point(event, x, y, flags, param):
    name, points_scaled, img, scale, true_points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        true_x, true_y = int(x / scale), int(y / scale)
        true_points.append((true_x, true_y))
        points_scaled.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(len(true_points)), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        print(f"[{name}] Point {len(true_points)}: ({true_x}, {true_y})")

# === Clones ===
optic_copy = optic_img.copy()
thermal_copy = thermal_display.copy()

cv2.namedWindow("Optic")
cv2.namedWindow("Thermal")

cv2.setMouseCallback("Optic", select_point, param=("Optic", optic_points_display, optic_copy, 1.0, optic_points))
cv2.setMouseCallback("Thermal", select_point, param=("Thermal", thermal_points_display, thermal_copy, THERMAL_SCALE, thermal_points))

print("📍 Click 4 matching points in EACH image, in the same order.")

while True:
    cv2.imshow("Optic", optic_copy)
    cv2.imshow("Thermal", thermal_copy)

    if len(optic_points) == 4 and len(thermal_points) == 4:
        print("✅ 4 points selected in both images. Computing homography...")

        pts_optic = np.array(optic_points, dtype=np.float32)
        pts_thermal = np.array(thermal_points, dtype=np.float32)

        H, _ = cv2.findHomography(pts_thermal, pts_optic)
        warped_thermal = cv2.warpPerspective(thermal_img, H, (optic_img.shape[1], optic_img.shape[0]))
        fused = cv2.addWeighted(optic_img, 0.25, warped_thermal, 0.75, 0)

        cv2.imshow("Warped Thermal", warped_thermal)
        cv2.imshow("Fused", fused)

        cv2.imwrite("output/warped_thermal.png", warped_thermal)
        cv2.imwrite("output/fused_manual.png", fused)
        np.save("output/homography_matrix.npy", H)

        print("💾 Saved to /output")
        print("Press any key to exit.")
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
