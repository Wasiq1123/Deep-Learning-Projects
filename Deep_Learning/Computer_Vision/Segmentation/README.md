# 🧠 Image Segmentation with Transformers, CNNs, and Vision-Language Models

This repository contains multiple image and video segmentation implementations using state-of-the-art models in computer vision:

* [**DeepLabV3 (CNN Semantic Segmentation)**](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html)
* [**FCN-ResNet101 (Fully Convolutional Network)**](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet101.html)
* [**YOLOv11-Seg (Real-Time Segmentation)**](https://github.com/ultralytics/ultralytics)
* [**SAM2 (Segment Anything Model v2)**](https://segment-anything.com/)
* [**FastSAM (Lightweight SAM by Ultralytics)**](https://github.com/ultralytics/ultralytics)

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

### ⚡ [YOLOv11 Segmentation (Real-Time)](https://github.com/ultralytics/ultralytics)

* Based on Ultralytics YOLOv11 segmentation (`yolo11n-seg.pt`)
* Ultra-fast video/image segmentation with `.plot()`
* Integrated with OpenCV for frame-wise rendering
* Video and image inference supported

---

### ⚡ [SAM2 (Segment Anything Model v2)](https://segment-anything.com/)

* Powerful vision-language segmentation model by Meta AI
* Supports zero-shot segmentation
* Highly accurate and generalized
* Processes each video frame individually

---

### ⚡ [FastSAM (Ultralytics)](https://github.com/ultralytics/ultralytics)

* Lightweight, fast alternative to SAM2
* Ideal for inference on CPUs or embedded systems
* Supports retina masks, high-resolution inputs
* Real-time static image segmentation

---

### ⚡ [DeepLabV3 + ResNet101](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html)

* CNN-based semantic segmentation from PyTorch
* Pretrained on COCO/Cityscapes
* Outputs class-wise segmentation maps
* Best for dense, pixel-level labeling tasks

---

### ⚡ [FCN-ResNet101](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet101.html)

* Fully Convolutional Network for semantic segmentation
* Often used for comparison with DeepLabV3
* Lightweight and easy to deploy
* Ideal for educational and experimentation purposes

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

Let me know if you also want a Markdown version or GitHub upload-ready format!
