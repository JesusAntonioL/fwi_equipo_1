# 🔥 FWI-APP: Fire Warning and Image Fusion System | Equipo 1

This repository contains the complete codebase and supporting resources for a wildfire detection and image fusion system. The system uses **thermal and optic camera calibration**, **YOLO-based fire detection**, **GPS logging**, and **real-time fusion**.  

It is vital to create a virtual environment and run the requirements.txt to include all the required dependencies.

---

## 📁 Project Structure

### 📂 `calibration_pairs/`
Contains calibration image pairs used for stereo calibration between optic and thermal cameras.

### 📂 `expansion_incendios/`
Includes simulation tools or scripts for developing the ROI, calculating the weights utilizing WOE and displaying the results in 2D and 3D.

**This folder should be implemented in Google Drive Desktop to run the entire code.**

### 📂 `Models/`
Holds trained models or YOLO weights used for object fire and smoke detection.

### 📂 `output/`
Stores output data such as predictions, logs, or result images.

### 📂 `raspberrypi_video_nonGUI/`
Scripts for headless video processing on Raspberry Pi 4 (without GUI) for the Lepton FLIR 3.1R.

### 📂 `runs/`
Folder automatically created by YOLO or training processes to store experiment results.

### 📂 `STEREO_FILES/`
Stereo image sets or outputs for calibration and alignment, this wasn't part of the main proposal, but was tried to improve the fusions of both thermal and optic images due to the misalignment of the images by distance variation.

### 📂 `synced_data_2/`
Synchronized data between thermal and optic cameras, used for evaluation and run withouth the need of a camera.

---

## 📄 Key Files

### 🎥 Media Files
- `Forest_fire_video_1.mp4` – Example fire footage for testing detection algorithms.
- `thermal-aerial-fire.mp4` – Aerial view of a fire scene using thermal imaging.
- `Wildfire_northamerica.mp4` – Wildfire footage from North America.

---

### 🧪 Calibration & Image Processing
- `capture_checkerboard_pair.py` – Captures checkerboard images for calibration.
- `select_checkerbox_points.py` – Selects checkerboard corners for calibration.
- `extract_gps.py` – Utilized to extract gps data from the RTK antenna.
- `read_optic.py` – Utilized to read the optic camera for testing purposes.
- `read_termal.py` – Utilized to read the optic thermal for testing purposes.
- `real_time_fusion.py` – Utilized to test the real time fusion algorithm.
- `main.py` – Main code which reads and processes both thermal and optic images, uses the algorithm to find the hottest spot and runs the ROI and visualization of it.

---

### 📂 expansion_incendios
- `fwi-map.py` – Collects coordinate and date data and calls corresponding API data to generate an interactive map that, based on a formula, determines the fire risk in a specific area, runs the visualization of the calculation based on the .html.
- `woeandiv-calculation.py` – Use the .CSV file created by woeadata2 to perform a Weight of Evidence and Information Value analysis and determine IV values ​​for each variable of interest for the risk formula (NDVI, land slope angle, and land surface temperature).
- `woe-data2.py` – Uses data from GEE datasets to generate a .CSV file containing NDVI, slope angle, and ground temperature data for areas where significant fires have occurred.
- `Mapa3D.html` – Visulizes the generated heatmap in the 3D visualizator.
- `fwi-heatmap.html` – Visulizes the generated heatmap in the 2D visualizator.
---

### Made by:
- Jesús Antonio López Malacón | A01742257
- Carlos Martínez García | A01351950
- Dulce Naranjo Sarmiento | A00832765
- Samantha López Vizcarra | A01742394
- Juan Alberto Moreno Cantú | A00833357  

In work with GTI (Green Tech Innovation).