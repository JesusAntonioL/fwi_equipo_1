import cv2
import numpy as np
import os

# === Config ===
PATTERN_SIZE = (9, 6)  # Internal corners (columns, rows)
DATA_DIR = "stereo_calibration_pairs"
WINDOW_NAME = "Click the checkerboard corners (left to right, top to bottom)"
SCALE = 8  # Use 1 if you want full-res images, >1 to downscale
image_type = "thermal"  # Change to "thermal" to annotate thermal images

# === Helper ===
def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < num_points:
            scaled_x = int(x * SCALE)
            scaled_y = int(y * SCALE)
            points.append([scaled_x, scaled_y])
            print(f"📍 Point {len(points)}: ({scaled_x}, {scaled_y})")

# === Start manual annotation ===
image_paths = sorted([
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
    if f.startswith(f"{image_type}_") and f.endswith(".png")
    and "annotated" not in f
])

num_points = PATTERN_SIZE[0] * PATTERN_SIZE[1]

for path in image_paths:
    print(f"\n🖼️ Annotating: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if image_type == "thermal" else cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Failed to load {path}")
        continue

    if SCALE != 1:
        img = cv2.resize(img, (img.shape[1] * SCALE, img.shape[0] * SCALE))

    points = []

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, onclick)

    while True:
        temp = img.copy()
        for p in points:
            cv2.circle(temp, (int(p[0] / SCALE), int(p[1] / SCALE)), 5, (0, 0, 255), -1)
        cv2.imshow(WINDOW_NAME, temp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            print("🔄 Reset points.")
        elif key == 13 or key == 10:  # Enter
            if len(points) == num_points:
                break
            else:
                print(f"⚠️ Need {num_points} points, currently have {len(points)}.")
                break
        elif key == ord('q'):
            print("🚪 Exiting annotation.")
            exit()

    if len(points) == num_points:
        points = np.array(points, dtype=np.float32)
        save_path = path.replace(".png", f"_annotated.npy")
        np.save(save_path, points)
        print(f"💾 Saved annotation: {save_path}")

cv2.destroyAllWindows()
