import cv2
import os
import numpy as np
from PyQt6.QtWidgets import QCheckBox, QMessageBox, QLineEdit, QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSlider, QSizePolicy, QDialog, QComboBox, QDialogButtonBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
from email.message import EmailMessage
import ssl
import smtplib
import torch

# Initialize Firebase
cred = credentials.Certificate('wildfiredetection-72d0f-firebase-adminsdk-4dohx-bc2cdd8cf4.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://wildfiredetection-72d0f-default-rtdb.firebaseio.com/'
})

# Get a reference to the database
ref = db.reference('reports')
contacts_ref = db.reference('contacts')  # Reference where phone numbers are stored

# Define main App PATH
mainPath = "C:/Users/chuy2/OneDrive/Escritorio/TEC/Semestre 8/Diseño e implementación de sistemas mecatrónicos/gti-cv-app"


# Defines GPU or CPU in App
device = torch.device(0 if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load all YOLOv8 models
        self.models = {
            "Nano": YOLO(mainPath + "/Models/fire_n.pt"),
            "Small": YOLO(mainPath + "/Models/fire_s.pt"),
            "Medium": YOLO(mainPath + "/Models/fire_m.pt"),
            "Large": YOLO(mainPath + "/Models/fire_l.pt"),
        }

        # Set default model
        self.model = self.models['Nano']
        #print(self.model.device.type, device)

        # Initialize video and slider properties
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_counter = 0
        self.total_frames = 0
        self.video_length_slider_pressed = False  # To track when the user is dragging the slider
        self.prev_gray = None  # To store the previous frame for optical flow

        # Add toggle for frame enhancement
        self.enhancement_enabled = False

        # Add toggle for airflow detection
        self.airflow_enabled = False
        self.fireflow_enabled = False
        self.maxThermalImg_enabled = False
        # In your ObjectDetectionApp class or main window setup:
        


    def initUI(self):
        self.setWindowTitle("Detección de Incendios y Humo")
        self.adjustSize()

        # Main layout
        main_layout = QHBoxLayout()

        # Video area
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Seleccionar video")
        self.video_label.setFixedSize(800, 450)
        # Set the alignment to center
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Add border using setStyleSheet
        self.video_label.setStyleSheet("border: 2px solid black;")

        video_layout.addWidget(self.video_label)

        # Slider for video progress
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setValue(0)
        self.video_slider.setTickInterval(1)
        self.video_slider.sliderPressed.connect(self.pause_slider_update)
        self.video_slider.sliderReleased.connect(self.seek_video)

        video_layout.addWidget(self.video_slider)  # Add the slider below the video

        # Controls layout on the right
        controls_layout = QVBoxLayout()

        # Enhancement toggle
        self.enhancement_checkbox = QCheckBox("Activar Estandarización")
        self.enhancement_checkbox.setChecked(False)
        self.enhancement_checkbox.stateChanged.connect(self.toggle_enhancement)
        controls_layout.addWidget(self.enhancement_checkbox)

        # Airflow detection toggle
        self.airflow_checkbox = QCheckBox("Detección de Flujo de Aire")
        self.airflow_checkbox.setChecked(False)
        self.airflow_checkbox.stateChanged.connect(self.toggle_airflow)
        controls_layout.addWidget(self.airflow_checkbox)

        # Fire detection toggle
        self.fireflow_checkbox = QCheckBox("Detección de Flujo de Fuego")
        self.fireflow_checkbox.setChecked(False)
        self.fireflow_checkbox.stateChanged.connect(self.toggle_fireflow)
        controls_layout.addWidget(self.fireflow_checkbox)

        # Fire Thermal Detection
        self.maxThermalImg_checkbox = QCheckBox("Filtro Térmico")
        self.maxThermalImg_checkbox.setChecked(False)
        self.maxThermalImg_checkbox.stateChanged.connect(self.toggle_maxThermalImg)
        controls_layout.addWidget(self.maxThermalImg_checkbox)

        # Add Model Upload Button
        self.upload_model_button = QPushButton("Agregar Modelo")
        self.upload_model_button.clicked.connect(self.add_custom_model)
        controls_layout.addWidget(self.upload_model_button)

        # Model Selector button
        self.model_button = QPushButton("Seleccionar Modelo")
        self.model_button.clicked.connect(self.open_model_selector)
        controls_layout.addWidget(self.model_button)

        # Model Selector button
        self.model_button = QPushButton("Seleccionar Cámara")
        self.model_button.clicked.connect(self.open_camera_selector)
        controls_layout.addWidget(self.model_button)

        # Upload video or image button
        self.upload_button = QPushButton("Seleccionar Video")
        self.upload_button.clicked.connect(self.open_file)
        controls_layout.addWidget(self.upload_button)

        # Threshold sliders
        self.iou_slider = self.create_slider("Threshold IOU", controls_layout, self.update_iou_threshold, 0, 100, 10, 50)
        self.confidence_slider = self.create_slider("Threshold Confidence", controls_layout, self.update_confidence_threshold, 0, 100, 10, 20)

        # FPS Slider
        self.fps_slider = self.create_slider("FPS", controls_layout, self.update_fps, 1, 60, 10, 30)

        # Start and Stop buttons
        start_stop_layout = QHBoxLayout()
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.start_video)
        start_stop_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Pausa")
        self.stop_button.clicked.connect(self.stop_video)
        start_stop_layout.addWidget(self.stop_button)
        controls_layout.addLayout(start_stop_layout)

        # **Add Fire Report Button**
        self.report_button = QPushButton("Generar Reporte de Incendio")
        self.report_button.clicked.connect(self.open_fire_report_form)
        controls_layout.addWidget(self.report_button)

        self.add_contact_button = QPushButton("Add Contact")
        self.add_contact_button.clicked.connect(self.open_contact_form)
        controls_layout.addWidget(self.add_contact_button)

        # Add video and controls to the main layout
        main_layout.addLayout(video_layout)
        main_layout.addLayout(controls_layout)

        # Set the main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set default thresholds
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.2

    def create_button(self, text):
        """Helper function to create styled buttons"""
        button = QPushButton(text)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button.setFont(QFont("Arial", 12))
        return button

    def create_slider(self, label_text, layout, on_change, min_value, max_value, intervals, default_Value):
        """Helper function to create a slider with a numerical value on the right"""
        slider_layout = QVBoxLayout()  # Vertical layout for label and slider-value pair

        # Create label for slider description
        label = QLabel(label_text)

        # Create a horizontal layout for the slider and value label
        slider_and_value_layout = QHBoxLayout()

        # Create the slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_Value)  # Default value
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(intervals)

        # Create a label to show the current slider value
        value_label = QLabel(str(default_Value))  # Start with default value (50)

        # Connect slider value change to update the value label and on_change function
        slider.valueChanged.connect(on_change)
        slider.valueChanged.connect(lambda: value_label.setText(str(slider.value())))

        # Add the slider and value label to the horizontal layout
        slider_and_value_layout.addWidget(slider)      # Slider on the left
        slider_and_value_layout.addWidget(value_label)  # Value label on the right

        # Add the main label and the horizontal slider-value pair to the vertical layout
        slider_layout.addWidget(label)  # Label above the slider
        slider_layout.addLayout(slider_and_value_layout)  # Slider and value label side by side

        # Add the complete layout to the provided layout
        layout.addLayout(slider_layout)

        return slider

    def update_iou_threshold(self):
        self.iou_threshold = self.iou_slider.value() / 100.0

    def update_confidence_threshold(self):
        self.confidence_threshold = self.confidence_slider.value() / 100.0

    def open_model_selector(self):
        dialog = ModelSelectorDialog(self)
        if dialog.exec():
            selected_model = dialog.get_selected_model()
            if selected_model in self.models:
                self.model = self.models[selected_model]

    #GTI TOLE
    def open_camera_selector(self):
        dialog = CameraSelectorDialog(self)
        if dialog.exec():
            selected_camera = dialog.get_selected_camera()
            print(f"Cámara seleccionada: {selected_camera}")

            # Accesar a la camara
            self.cap = cv2.VideoCapture(int(selected_camera))

    def toggle_enhancement(self):
        self.enhancement_enabled = self.enhancement_checkbox.isChecked()

    def toggle_airflow(self):
        self.airflow_enabled = self.airflow_checkbox.isChecked()

    def toggle_fireflow(self):
        self.fireflow_enabled = self.fireflow_checkbox.isChecked()

    def toggle_maxThermalImg(self):
        self.maxThermalImg_enabled = self.maxThermalImg_checkbox.isChecked()

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image or Video", filter="Video Files (*.mp4 *.mov);;Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            if file_name.endswith(('.mp4', '.mov')):
                self.load_video(file_name)
            else:
                self.run_object_detection(file_name)

    def load_video(self, file_name):
        self.cap = cv2.VideoCapture(file_name)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_slider.setMaximum(self.total_frames)  # Set slider range based on total frames

    def start_video(self):
        if self.cap is not None:
            self.timer.start(1000 // self.fps_slider.value())  # Set timer interval based on FPS slider

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()

    def pause_slider_update(self):
        self.video_length_slider_pressed = True

    def seek_video(self):
        self.video_length_slider_pressed = False
        slider_value = self.video_slider.value()
        self.frame_counter = slider_value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, slider_value)
        self.start_video()

    def update_fps(self):
        """Update FPS based on the FPS slider"""
        fps = self.fps_slider.value()
        self.timer.setInterval(1000 // fps)


    def increase_orange_intensity(self, frame):
        try:
            # Convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define a more focused range for detecting fire-like orange (in HSV space)
            lower_orange = np.array([5, 50, 50])     # Lower bound (reddish tones)
            upper_orange = np.array([25, 255, 255])  # Upper bound (yellowish-orange tones)

            # Create a mask for orange areas
            mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # Intensify the orange color
            hsv[:, :, 1] = np.where(mask > 0, np.clip(hsv[:, :, 1] * 2.5, 0, 255), hsv[:, :, 1])  # Increase saturation
            hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] * 1.8, 0, 255), hsv[:, :, 2])  # Increase brightness

            # Convert back to BGR (regular color format)
            enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            return enhanced_frame
        except Exception as e:
            print(f"Failed to enhance frame: {e}")
            return frame  # Return the original frame if there's an issue
        
    def extract_smoke_frame(self, frame, results):
        smoke_class_index = 0  # Replace with the actual index for "smoke" in your model's class list
        
        # Create a mask for the detected smoke
        smoke_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for result in results[0].boxes:
            if result.cls == smoke_class_index:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                # Fill the mask for the detected smoke
                smoke_mask[y1:y2, x1:x2] = 255  # Set to 255 (white) where smoke is detected

        # Bitwise AND to isolate smoke areas
        smoke_frame = cv2.bitwise_and(frame, frame, mask=smoke_mask)

        return smoke_frame

    def calc_optical_flow_in_bbox(self, prev_frame, curr_frame, x1, y1, x2, y2):
        """Calculate the optical flow within a bounding box."""
        if prev_frame is None:
            return None

        # Extract region of interest (ROI) for the bounding box from both frames
        prev_roi = prev_frame[y1:y2, x1:x2]
        curr_roi = curr_frame[y1:y2, x1:x2]

        # Calculate optical flow within the bounding box
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow

    def draw_propagation_vectors(self, frame, flow, origin):
        """Draw propagation vectors for the fire inside the bounding box with improved spacing and minimal clustering."""
        if flow is None:
            return

        step = 20  # Step size for potential arrow positions
        scale_factor = 10  # Scale factor for better visualization
        min_distance = 15  # Minimum distance between arrow start points
        x0, y0 = origin  # Starting coordinates for the arrows

        # Get dimensions of the flow matrix
        h, w = flow.shape[:2]

        # Track drawn arrow start points to ensure spacing
        drawn_points = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get the flow vector (fx, fy)
                fx, fy = flow[y, x]

                # Only consider vectors with significant flow magnitude
                if np.linalg.norm((fx, fy)) > 1:  # Adjust threshold as needed
                    # Scale the flow vectors
                    fx *= scale_factor
                    fy *= scale_factor

                    # Start point for the arrow
                    start_point = (x0 + x, y0 + y)

                    # Ensure sufficient distance from previously drawn arrows
                    if all(np.linalg.norm((start_point[0] - px, start_point[1] - py)) >= min_distance for px, py in drawn_points):
                        # End point for the arrow
                        end_point = (int(start_point[0] + fx), int(start_point[1] + fy))

                        # Ensure the end point is within frame bounds
                        end_point = (
                            max(0, min(frame.shape[1] - 1, end_point[0])),
                            max(0, min(frame.shape[0] - 1, end_point[1]))
                        )

                        # Draw the arrow with improved visualization
                        cv2.arrowedLine(
                            frame,
                            start_point,
                            end_point,
                            (255, 0, 0),  # Bright red color
                            thickness=2,  # Increased thickness
                            tipLength=0.4  # Larger tip length
                        )

                        # Add the current start point to the list of drawn points
                        drawn_points.append(start_point)


    def add_custom_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar Modelo Personalizado", 
            "", 
            "YOLO Model Files (*.pt);;All Files (*)"
        )
        
        if file_name:
            try:
                # Load the user-provided model
                custom_model = YOLO(file_name)
                
                # Add the model to the models dictionary with a unique key
                model_name = os.path.basename(file_name).split('.')[0]
                self.models[model_name] = custom_model
                
                # Inform the user
                QMessageBox.information(
                    self, 
                    "Modelo Agregado", 
                    f"El modelo '{model_name}' se agregó correctamente."
                )
            except Exception as e:
                # Handle errors in loading the model
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"No se pudo cargar el modelo: {e}"
                )

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to grayscale for optical flow
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # YOLOv8 detection on the current frame
                results = self.model.predict(frame, conf=self.confidence_threshold, iou=self.iou_threshold, device= device)
                result_frame = results[0].plot()

                # Enhance frame colors if enabled
                if self.enhancement_enabled:
                    result_frame = self.increase_orange_intensity(result_frame)

                # Mask Limits for Thermal
                lower_limit = np.array([120, 120, 120])
                upper_limit = np.array([180, 180, 180])

                if self.maxThermalImg_enabled:
                    # Crear la máscara basada en los límites de intensidad
                    mask = cv2.inRange(frame, lower_limit, upper_limit)

                    # Encontrar contornos en la máscara
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Dibujar los contornos en el cuadro original
                    cv2.drawContours(result_frame, contours, -1, (0, 0, 255), 2)  # Contornos en rojo (BGR: (0, 0, 255))

                    # Aplicar la máscara al cuadro original (opcional)
                    #result_frame = cv2.bitwise_and(result_frame, result_frame, mask=mask)

                smoke_class_index = 0  # Replace with the actual index for "smoke" in your model's class list
                smoke_bboxes = [result.xyxy.cpu().numpy() for result in results[0].boxes if result.cls == smoke_class_index]

                fire_class_index = 1  # Assuming 'fire' is class index 1 in the model
                fire_bboxes = [result.xyxy.cpu().numpy() for result in results[0].boxes if result.cls == fire_class_index]

                if self.airflow_enabled and smoke_bboxes:
                    # Extract smoke frame only if smoke is detected
                    new_frame = self.extract_smoke_frame(result_frame, results)
                    flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 1, 25, 2, 5, 1.2, 0)

                    # Draw airflow arrows only within the smoke bounding box
                    for bbox in smoke_bboxes:
                        # bbox should now be in the format [x1, y1, x2, y2]
                        # Make sure to convert it to a regular list or array if it's still a tensor
                        bbox = bbox.flatten()  # Flatten if it's still multidimensional
                        x1, y1, x2, y2 = map(int, bbox)

                        # Extract the region of interest (ROI) for airflow calculation
                        roi_flow = flow[y1:y2, x1:x2]

                        # Draw arrows only in this ROI
                        self.draw_airflow_arrows(new_frame, roi_flow, (y1, x1))
                        
                        alpha = 0.5  # Control transparency; 0.0 = frame1 only, 1.0 = frame2 only
                        result_frame = cv2.addWeighted(new_frame, alpha, result_frame, 1 - alpha, 0)

                if self.fireflow_enabled and fire_bboxes:
                    for bbox in fire_bboxes:
                        bbox = bbox.flatten()  # Flatten the bbox coordinates
                        x1, y1, x2, y2 = map(int, bbox)

                        # Region of interest (ROI) for the fire bounding box
                        roi_flow = self.calc_optical_flow_in_bbox(self.prev_gray, gray, x1, y1, x2, y2)

                        # Draw propagation arrows inside the bounding box
                        self.draw_propagation_vectors(result_frame, roi_flow, (x1, y1))

                # Update previous frame for optical flow calculation
                self.prev_gray = gray

                # Show the processed frame
                self.display_results(result_frame)

                # Update slider only if the user is not manually dragging it
                if not self.video_length_slider_pressed:
                    self.frame_counter += 1
                    self.video_slider.setValue(self.frame_counter)
            else:
                self.stop_video()  # Stop video if it reaches the end

   

    def draw_airflow_arrows(self, frame, flow, origin, target_arrows=100):
        # Get the dimensions of the flow
        h, w = flow.shape[:2]
        y0, x0 = origin  # Origin for placing arrows

        # Dynamically calculate step size to ensure target_arrows are drawn
        total_area = h * w
        step = max(10, int((total_area / target_arrows) ** 0.5))  # At least 10 pixels apart

        scale_factor = 10  # Adjust scale factor for arrow size

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get the flow vector at the point (x, y)
                fx, fy = flow[y, x]

                # Check if flow is significant enough to draw an arrow
                if np.linalg.norm((fx, fy)) > 2:  # Adjust the threshold as needed
                    # Scale the flow vectors
                    fx *= scale_factor
                    fy *= scale_factor

                    # Define the start and end points for the arrow
                    start_point = (x0 + x, y0 + y)
                    end_point = (int(start_point[0] + fx), int(start_point[1] + fy))

                    # Ensure the end point is within frame bounds
                    end_point = (
                        max(0, min(frame.shape[1] - 1, end_point[0])), 
                        max(0, min(frame.shape[0] - 1, end_point[1]))
                    )

                    # Draw the arrow on the frame with improved visibility
                    cv2.arrowedLine(
                        frame,
                        start_point,
                        end_point,
                        (0, 255, 0),  # Bright green color
                        thickness=2,
                        tipLength=0.3
                    )



    def display_results(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        q_img = QImage(rgb_image.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def open_fire_report_form(self):
        form = FireReportForm(self)
        form.exec()
    
    def open_contact_form(self):
        form = ContactForm(self)
        form.exec()



class ModelSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Model")

        # Create layout
        layout = QVBoxLayout()

        self.model_combobox = QComboBox(self)
        self.model_combobox.addItems(parent.models.keys())  # Dynamically populate models
        layout.addWidget(self.model_combobox)   

        # Add OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Set the layout
        self.setLayout(layout)

    def get_selected_model(self):
        """Return the selected model"""
        return self.model_combobox.currentText()

# GTI Tole
class CameraSelectorDialog(QDialog):
    # Variable de clase para almacenar las cámaras detectadas
    available_cameras = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar Cámara")

        # Crear el layout
        layout = QVBoxLayout()

        # Obtener la lista de cámaras disponibles (solo la primera vez)
        if CameraSelectorDialog.available_cameras is None:
            CameraSelectorDialog.available_cameras = self.get_available_cameras()

        # Obtener la lista de cámaras disponibles (solo la primera vez)
        #self.available_cameras = self.get_available_cameras()

        # Crear el ComboBox con las cámaras detectadas
        self.camera_combobox = QComboBox(self)
        self.camera_combobox.addItems(CameraSelectorDialog.available_cameras)
        layout.addWidget(self.camera_combobox)

        # Botones de Aceptar y Cancelar
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Set the layout
        self.setLayout(layout)

    def get_available_cameras(self):
        """
        Escanea los dispositivos de captura de OpenCV para identificar cámaras disponibles.
        Retorna una lista con los identificadores de cámara detectados.
        """
        max_cameras = 10
        cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)

            if cap.read()[0]:
                cameras.append(str(i))
                cap.release()

        print(f"Cameras found: {cameras}")
        return cameras if cameras else ["No se encontraron cámaras"]


    def get_selected_camera(self):
        """Devuelve el identificador de la cámara seleccionada."""
        return self.camera_combobox.currentText()

class FireReportForm(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reporte de Incendio")
        self.setFixedSize(400, 575)

        layout = QVBoxLayout()

        # Form fields
        self.location_label = QLabel("Localización del incendio")
        layout.addWidget(self.location_label)
        self.location_input = QLineEdit(self)
        self.location_input.setPlaceholderText("Localización del incendio")
        layout.addWidget(self.location_input)

        self.area_label = QLabel("Área del incendio (m²)")
        layout.addWidget(self.area_label)
        self.area_input = QLineEdit(self)
        self.area_input.setPlaceholderText("Área del incendio (m²)")
        layout.addWidget(self.area_input)

        self.altura_label = QLabel("Altura del incendio (m)")
        layout.addWidget(self.altura_label)
        self.height_input = QLineEdit(self)
        self.height_input.setPlaceholderText("Altura del incendio (m)")
        layout.addWidget(self.height_input)

        self.municipio_label = QLabel("Municipio")
        layout.addWidget(self.municipio_label)
        self.municipio_input = QLineEdit(self)
        self.municipio_input.setPlaceholderText("Municipio")
        layout.addWidget(self.municipio_input)

        self.estado_label = QLabel("Estado")
        layout.addWidget(self.estado_label)
        self.estado_input = QLineEdit(self)
        self.estado_input.setPlaceholderText("Estado")
        layout.addWidget(self.estado_input)

        self.localidad_label = QLabel("Localidad")
        layout.addWidget(self.localidad_label)
        self.localidad_input = QLineEdit(self)
        self.localidad_input.setPlaceholderText("Localidad")
        layout.addWidget(self.localidad_input)

        self.tipo_label = QLabel("Indique tipo de incendio")
        layout.addWidget(self.tipo_label)
        self.fire_type_combo = QComboBox(self)
        self.fire_type_combo.addItems(["Controlado", "No Controlado"])
        layout.addWidget(self.fire_type_combo)

        self.personas_label = QLabel("Hay personas alrededor?")
        layout.addWidget(self.personas_label)
        self.people_around_combo = QComboBox(self)
        self.people_around_combo.addItems(["Sí", "No"])
        layout.addWidget(self.people_around_combo)

        # Send button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.send_report)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def send_report(self):
        """Send SMS with the filled report data to all phone numbers in the database."""
        location = self.location_input.text()
        area = self.area_input.text()
        height = self.height_input.text()
        municipio = self.municipio_input.text()
        estado = self.estado_input.text()
        localidad = self.localidad_input.text()
        fire_type = self.fire_type_combo.currentText()
        people_around = self.people_around_combo.currentText()

        # Validate required fields
        if not location or not area or not height or not municipio or not estado or not localidad:
            QMessageBox.warning(self, "Error", "Por favor, rellena todos los campos.")
            return

        # Get current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a new report
        new_report = {
            'location': location,
            'area': area,
            'height': height,
            'municipio': municipio,
            'estado': estado,
            'localidad': localidad,
            'fire_type': fire_type,
            'people_around': people_around,
            'timestamp': current_time  # Add timestamp to the report

        }

        # Push the new report to the database
        ref.push(new_report)

        # Format the alert message
        message = (
            f"🚨 Alerta de Incendio 🚨\n\n"
            f"Fecha y hora: {current_time}\n"
            f"📍 Ubicación: {location}, {localidad}, {municipio}, {estado}\n"
            f"🔥 Área afectada: {area} m²\n"
            f"📏 Altura de llamas: {height} m\n"
            f"🚒 Tipo: {fire_type}\n"
            f"⚠️ Precaución: Evitar la zona.\n"
            f"👥 Personas alrededor: {'Sí' if people_around == 'Sí' else 'No'}"
        )
        
        email_sender = "greentechtestin01@gmail.com"
        email_passoword = "rywz relm smcy fkmz"
        #email_receiver = "andrescabral108@gmail.com"
        
        # Retrieve all phone numbers from the database
        contacts = contacts_ref.get()
        if contacts:
            for contact_id, contact in contacts.items():
                email_receiver = contact.get("email")
                #phone_number = contact.get("phone_number")
                try:
                    subject = "Alerta de Incendio"
                    em = EmailMessage()
                    em["From"] = email_sender
                    em["To"] = email_receiver
                    em["Subject"] = subject
                    em.set_content(message)

                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                        smtp.login(email_sender, email_passoword)
                        smtp.sendmail(email_sender, email_receiver, em.as_string())
                            
                except Exception as e:
                    print(f"Failed to send email to {email_receiver}: {e}")
        
        QMessageBox.information(self, "Enviado", "El reporte ha sido enviado exitosamente a todos los contactos.")
        self.accept()

class ContactForm(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Contact")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        # Phone number label and input
        self.phone_label = QLabel("Email Address")
        layout.addWidget(self.phone_label)
        self.phone_input = QLineEdit(self)
        self.phone_input.setPlaceholderText("JohnSmith@gmail.com")  # Placeholder format for international numbers
        layout.addWidget(self.phone_input)

        # Save button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_contact)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def save_contact(self):
        """Save the entered phone number to Firebase."""
        phone_number = self.phone_input.text().strip()

        # Validate that a phone number is entered
        if not phone_number:
            QMessageBox.warning(self, "Error", "Please enter an email address")
            return

        # Save phone number to Firebase under 'contacts'
        contact = {'email': phone_number}
        contacts_ref.push(contact)  # Push the contact to the database

        QMessageBox.information(self, "Contact Added", "The phone number has been saved successfully.")
        self.accept()

# Running the app
if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    print(torch.cuda.device_count())
    app = QApplication([])
    window = ObjectDetectionApp()
    window.show()
    app.exec()