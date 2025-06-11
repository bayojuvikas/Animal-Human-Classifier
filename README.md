## 🧠 Animal & Human Detection and Classification System

This project implements a complete end-to-end system to detect, classify, and track humans and animals in video files using deep learning and computer vision techniques. Built to fulfill the requirements of an AI Engineer assignment at **Maharshi Industries Pvt. Ltd.**

![Architecture](https://img.shields.io/badge/YOLOv8-DETECTION-blue) ![ResNet18](https://img.shields.io/badge/ResNet18-CLASSIFICATION-green) ![DeepSORT](https://img.shields.io/badge/DeepSORT-TRACKING-red) ![W\&B](https://img.shields.io/badge/w%26b-Logging-orange)

---

### 📁 Directory Structure

```
project/
├── datasets/             # Datasets used for training
├── models/               # Trained detection and classification models
├── test_videos/          # Drop test videos here
├── outputs/              # Annotated output videos
└── animal_human_detection.py  # Main script
```

---

### 🎯 Features

* ✅ Object detection using custom-trained **YOLOv8**
* ✅ Classification using fine-tuned **ResNet18**
* ✅ Real-time object tracking with **Deep SORT**
* ✅ Fully automated video processing pipeline
* ✅ Training metrics logged using **Weights & Biases**
* ✅ Plug-and-play inference: just drop a video in `test_videos/`

---

### 🔍 Dataset Choices

**1. Object Detection**

* **Source**: [Roboflow](https://roboflow.com)
* **Classes**: `"human"` and `"animal"`
* **Format**: YOLOv8-compatible `.yaml` and annotations
* **Reason**: Accurate bounding boxes, easy export, good variety

**2. Classification**

* **Datasets**:

  * [Animals-10 Dataset (Kaggle)](https://www.kaggle.com/alessiocorrado99/animals10)
  * [Human Faces Dataset (Kaggle)](https://www.kaggle.com/datasets)
* **Task**: Binary classification between `human` and `animal`
* **Reason**: High-quality labeled images, balanced classes

---

### 🤖 Model Choices

| Task           | Model     | Framework            | Why?                                                   |
| -------------- | --------- | -------------------- | ------------------------------------------------------ |
| Detection      | YOLOv8    | Ultralytics          | Lightweight, fast, accurate, easy Roboflow integration |
| Classification | ResNet18  | PyTorch              | Strong baseline for binary image classification        |
| Tracking       | Deep SORT | deep\_sort\_realtime | Reduces flickering, adds track consistency & IDs       |

---

### 🧪 Training Summary

* **Detection**:

  * Framework: Ultralytics YOLOv8
  * Loss: \~1.29
  * mAP\@50: 87.4
  * Tool: Weights & Biases (wandb)

* **Classification**:

  * Model: ResNet18 (fine-tuned)
  * Accuracy: 95.2
  * Loss: 0.183

---

### 🚀 Inference Usage

#### Step 1: Place a video file in `test_videos/`

```bash
cp your_video.mp4 test_videos/
```

#### Step 2: Run the main script

```bash
python animal_human_detection.py
```

#### Step 3: Get your result

Annotated video will be saved in `outputs/annotated_your_video.mp4`

---

### ✨ Sample Output

<img src="https://user-images.githubusercontent.com/yourusername/sample.gif" width="700"/>

---

### 🧰 Dependencies

```bash
pip install ultralytics torch torchvision opencv-python deep_sort_realtime
```

For wandb logging:

```bash
pip install wandb
```

---

### ⚠️ Challenges Faced

* Kaggle download restrictions required dataset shifting to Roboflow
* Frame-to-frame box instability resolved using Deep SORT
* Handling tiny crops during classification to avoid empty tensors

---

### 🚧 Future Improvements

* Replace ResNet18 with ViT or EfficientNet for better classification
* Add audio alerts for detected humans
* Create a Streamlit web interface for real-time monitoring
* Export video with track logs (CSV of IDs and positions)

---

### 🧑‍💻 Author

**Bayoju Vikas**
🔗 [LinkedIn](https://www.linkedin.com/in/bayoju-vikas-81578726a)
