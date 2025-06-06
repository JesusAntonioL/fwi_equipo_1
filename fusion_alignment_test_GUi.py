import cv2
import os
import numpy as np

# === Config ===
DATA_DIR = "synced_data_1"
IMG_INDEX = 22
ALPHA = 0.25
BETA = 0.75

# === Load Images ===
optic_path = os.path.join(DATA_DIR, f"optic_{IMG_INDEX:04d}.png")
thermal_path = os.path.join(DATA_DIR, f"thermal_{IMG_INDEX:04d}.png")

optic_img = cv2.imread(optic_path)        # Expected: 1920x1080
thermal_img = cv2.imread(thermal_path)    # Expected: 160x120

if optic_img is None or thermal_img is None:
    print("❌ Could not load images. Check paths or index.")
    exit()

# === Fusion Function with Offset and Resizing ===
def fuse_images_with_offset(dx, dy, scale_w, scale_h, crop_x_percent, crop_y_percent):
    # --- Resize thermal ---
    new_width = int(thermal_img.shape[1] * scale_w)
    new_height = int(thermal_img.shape[0] * scale_h)
    resized_thermal = cv2.resize(thermal_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # --- Crop optic image based on % margins ---
    h, w, _ = optic_img.shape
    crop_x = int(w * crop_x_percent / 100.0)
    crop_y = int(h * crop_y_percent / 100.0)
    cropped_optic = optic_img[crop_y:h-crop_y, crop_x:w-crop_x]

    # Resize cropped optic to original shape (for display/fusion)
    optic_resized = cv2.resize(cropped_optic, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- Shifted thermal for fusion ---
    shifted_thermal = np.zeros_like(optic_resized)
    h, w, _ = optic_resized.shape

    x1, y1 = max(0, dx), max(0, dy)
    x2, y2 = min(w, dx + new_width), min(h, dy + new_height)
    tx1, ty1 = max(0, -dx), max(0, -dy)
    tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)

    if x1 < x2 and y1 < y2:
        shifted_thermal[y1:y2, x1:x2] = resized_thermal[ty1:ty2, tx1:tx2]

    fused = cv2.addWeighted(optic_resized, ALPHA, shifted_thermal, BETA, 0)
    return optic_resized, resized_thermal, fused


def update(val):
    dx = cv2.getTrackbarPos("dx", "Fusion Tuner") - 500
    dy = cv2.getTrackbarPos("dy", "Fusion Tuner") - 500
    scale_w = cv2.getTrackbarPos("scale_w", "Fusion Tuner") / 100.0
    scale_h = cv2.getTrackbarPos("scale_h", "Fusion Tuner") / 100.0
    crop_x = cv2.getTrackbarPos("crop_x", "Fusion Tuner") / 100.0
    crop_y = cv2.getTrackbarPos("crop_y", "Fusion Tuner") / 100.0

    optic_adj, thermal_adj, fused = fuse_images_with_offset(dx, dy, scale_w, scale_h, crop_x, crop_y)

    cv2.imshow("Optic (Cropped)", optic_adj)
    cv2.imshow("Thermal (Scaled)", thermal_adj)
    cv2.imshow("Fused", fused)


# === Create GUI Window ===
cv2.namedWindow("Fusion Tuner", cv2.WINDOW_NORMAL)

# === Create Trackbars ===
cv2.createTrackbar("dx", "Fusion Tuner", 500, 1000, update)
cv2.createTrackbar("dy", "Fusion Tuner", 500, 1000, update)
cv2.createTrackbar("scale_w", "Fusion Tuner", 1200, 3000, update)
cv2.createTrackbar("scale_h", "Fusion Tuner", 900, 3000, update)
cv2.createTrackbar("crop_x", "Fusion Tuner", 0, 30, update)
cv2.createTrackbar("crop_y", "Fusion Tuner", 0, 30, update)

# === Wait for the window and trackbars to be ready ===
cv2.imshow("Fusion Tuner", np.zeros((100, 500, 3), dtype=np.uint8))  # Dummy display
cv2.waitKey(100)  # Give OpenCV time to render GUI

# === Now it's safe to call update
update(0)




print("🎛 Adjust dx, dy, scale_w, scale_h with sliders. Press 's' to save fused image, 'q' to quit.")

# === Main Loop ===
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        dx = cv2.getTrackbarPos("dx", "Fusion Tuner") - 500
        dy = cv2.getTrackbarPos("dy", "Fusion Tuner") - 500
        scale_w = cv2.getTrackbarPos("scale_w", "Fusion Tuner") / 100.0
        scale_h = cv2.getTrackbarPos("scale_h", "Fusion Tuner") / 100.0
        crop_x = cv2.getTrackbarPos("crop_x", "Fusion Tuner") / 100.0
        crop_y = cv2.getTrackbarPos("crop_y", "Fusion Tuner") / 100.0

        _, _, fused = fuse_images_with_offset(dx, dy, scale_w, scale_h, crop_x, crop_y)

        os.makedirs("fused_output", exist_ok=True)
        out_path = f"fused_output/fused_{IMG_INDEX:04d}_dx{dx}_dy{dy}_sw{scale_w:.2f}_sh{scale_h:.2f}_cx{crop_x:.2f}_cy{crop_y:.2f}.png"
        cv2.imwrite(out_path, fused)
        print(f"[💾 Saved] {out_path}")


cv2.destroyAllWindows()