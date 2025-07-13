# 🧠 Image Segmentation with Transformers, CNNs, and Vision-Language Models

This repository contains multiple **image and video segmentation** implementations using state-of-the-art models in computer vision:

* DeepLabV3 (CNN)
* FCN-ResNet101 (CNN)
* YOLO11-Segmentation (Ultralytics)
* SAM2 (Segment Anything Model)
* FastSAM (Lightweight SAM variant)

---

## 📂 Project Structure

```bash
📁 Image_Segmentation

├── YOLO11_Image_Segmentation.ipynb

├── YOLO11_Segmentation_in_Video.ipynb

├── SAM2_Segmentation_in_Video.ipynb

├── FAST-SAM.ipynb

├── Deeplabv3_resnet101_Segmentation_in_Video.ipynb

├── Pytorch_Image_Segmentation.ipynb
```

---

## 🔍 Implementations Overview

### ⚡ YOLO11 Segmentation (Real-Time)

* Based on Ultralytics YOLOv11 segmentation variant (`yolo11n-seg.pt`)
* Runs on images and videos
* Ultra-fast predictions and annotated output via `.plot()`
* Video support using `cv2.VideoCapture()`
* Confidence threshold: *auto-handled*

---

### ⚡ SAM2 (Segment Anything Model v2)

* Vision-Language segmentation by Meta AI
* High-accuracy, zero-shot segmentation
* Loads `sam2.1_b.pt` and processes each video frame
* Output is a segmentation overlay on the video
* Flip and segment frame-by-frame in real-time

---

### ⚡ FastSAM

* Lightweight alternative to SAM
* High-speed segmentation with retina masks
* Ideal for inference on CPU or low-power devices
* Uses `FastSAM-s.pt` checkpoint
* Inputs: URLs or image files

---

### ⚡ DeepLabV3 + ResNet101

* CNN-based semantic segmentation from PyTorch
* Pretrained on COCO/Cityscapes datasets
* Uses `DeepLabV3_ResNet101_Weights.DEFAULT`
* Torch transforms with resizing, normalization
* Output: pixel-wise class map + color-mapped result

---

### ⚡ FCN + ResNet101

* Fully Convolutional Network for pixel-wise labeling
* Side-by-side comparison with DeepLabV3
* Input: static image
* Output: color maps for both models

---

## 🧪 Requirements

Install all dependencies using:

```bash
pip install torch torchvision ultralytics opencv-python matplotlib Pillow
```

If using Google Colab, also run:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics
```

---

## 📌 Notes

* All notebooks are optimized for **Google Colab GPU**
* Replace paths with your own images/videos in Google Drive
* Supports **real-time and static image segmentation**

---
