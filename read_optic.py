import cv2
from ultralytics import YOLO
import torch

# Use webcam (uncomment for live camera)
cap = cv2.VideoCapture(1)  
video_type = "cam"

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# 🔁 Load your YOLOv8 model
model = YOLO('Models/fire_l.pt')  # Make sure the path is correct

# 🎥 Video source
#video_path = 'Video_Prueba.mp4'
#cap = cv2.VideoCapture(video_path)
#video_type = "vid"

# Defines GPU or CPU in App
device = torch.device(0 if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")

def enhance_contrast_color(img):
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge channels back
    limg = cv2.merge((cl, a, b))
    # Convert LAB back to BGR
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

if not cap.isOpened():
    print("Cannot open video or camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #processed_frame = enhance_contrast_color(frame)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Frame width: {frame_width}, frame height: {frame_height}")
    
      # 🔍 Run inference
    results = model.predict(frame, conf=0.5)

    # 🖼️ Draw results on frame
    annotated_frame = results[0].plot()

    # 🪟 Display the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # 🛑 Press 'q' to quit
    if video_type == "vid":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    elif video_type == "cam":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
