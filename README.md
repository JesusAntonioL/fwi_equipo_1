# 🔥 FWI-APP: Fire Warning and Image Fusion System | Equipo 1

This repository contains the complete codebase and supporting resources for a wildfire detection and image fusion system. The system uses **thermal and optic camera calibration**, **YOLO-based fire detection**, **GPS logging**, and **real-time fusion**.  

---

## 📁 Project Structure

### 📂 `calibration_pairs/`
Contains calibration image pairs used for stereo calibration between optic and thermal cameras.

### 📂 `expansion_sim/`
Includes simulation tools or scripts for expanding and testing the system logic in simulated environments.

### 📂 `Models/`
Holds trained models or YOLO weights used for object detection.

### 📂 `output/`
Stores output data such as predictions, logs, or result images.

### 📂 `raspberrypi_video_nonGUI/`
Scripts for headless video processing on Raspberry Pi (without GUI).

### 📂 `runs/`
Folder automatically created by YOLO or training processes to store experiment results.

### 📂 `STEREO_FILES/`
Stereo image sets or outputs for calibration and alignment.

### 📂 `synced_data_2/`
Synchronized data between thermal and optic cameras, possibly used for evaluation or training.

---

## 📄 Key Files

### 🔧 `.gitignore`
Specifies which files and folders Git should ignore (like temp files, cache, etc).

### 📜 `allfileshas.txt`
Possibly a hash record or list of all relevant files used in this project.

---

### 🎥 Media Files
- `Forest_fire_video_1.mp4` – Example fire footage for testing detection algorithms.
- `thermal-aerial-fire.mp4` – Aerial view of a fire scene using thermal imaging.
- `Wildfire_northamerica.mp4` – Wildfire footage from North America.
- `wildfiredetection-72d0f-firebase-adminsdk-...` – Possibly related to Firebase SDK or logging.

---

### 🧪 Calibration & Image Processing
- `capture_checkerboard_pair.py` – Captures checkerboard images for stereo calibration.
- `select_checkerbox_points.py` – Selects checkerboard corners for calibration.
- `stereo_calibration.py` – Calibrates the stereo setup between optic and thermal cameras.

---

### Made by:
- Jesús Antonio López Malacón | A01742257
- Carlos Martínez García | A01351950
- Dulce Naranjo Sarmiento | A00832765
- Samantha López Vizcarra | A01742394
- Juan Alberto Moreno Cantú | A00833357  

