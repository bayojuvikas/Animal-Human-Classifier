import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import timm

# -------------------------
# Configuration
# -------------------------
YOLO_MODEL_PATH = "models/yolo_animals_humans/weights/best.pt"
CLASSIFIER_PATH = "models/classifier_effnet.pth"
VIDEO_DIR = "test_videos"
OUTPUT_DIR = "outputs"

# -------------------------
# Load Detection Model (YOLOv8)
# -------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

# -------------------------
# Load Classification Model (EfficientNet)
# -------------------------
classifier = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=torch.device('cpu')))
classifier.eval()

# -------------------------
# Transform for classifier
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# Label map (update as per your classes.yaml if needed)
# -------------------------
label_map = {
    0: "animal",  # YOLO class 0
    1: "human"    # YOLO class 1
}

# -------------------------
# Start Monitoring test_videos/
# -------------------------
print("🚀 Watching for new videos in test_videos/ ...")

processed_files = set()

while True:
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith((".mp4", ".avi", ".mov")) and filename not in processed_files:
            input_path = os.path.join(VIDEO_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
            processed_files.add(filename)

            print(f"\n📽️ Processing {filename}...")

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLOv8 detection
                results = yolo_model(frame)[0]

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = label_map.get(cls_id)

                    if label is None:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = frame[y1:y2, x1:x2]

                    # Skip empty crops
                    if cropped.size == 0:
                        continue

                    # Classify using classifier
                    img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(img_pil).unsqueeze(0)
                    with torch.no_grad():
                        pred = classifier(img_tensor)
                        final_label = "animal" if torch.argmax(pred) == 0 else "human"

                    # Draw box and label
                    color = (0, 255, 0) if final_label == "human" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, color, 2)

                out.write(frame)

            cap.release()
            out.release()
            print(f"✅ Output saved to {output_path}")

    # Check every 5 seconds
    time.sleep(5)
