# 🧠 Object Detection with Transformers, CNNs, and Vision-Language Models

This repository contains multiple object detection implementations using state-of-the-art models in computer vision:

- [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872)
- [Faster R-CNN (Region-based CNN)](https://arxiv.org/abs/1506.01497)
- [Qwen2.5-VL (Vision-Language Large Model)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [YOLOv8 (You Only Look Once)](https://github.com/ultralytics/ultralytics)

---

## 📂 Project Structure

📁 Object_Detection
├── DETR_Object_Detection.ipynb

├── FasterRCNN_Object_Detection.ipynb

├── Qwen2.5_VL_Object_Detection.ipynb

├── YOLOv8_Video_Object_Detection.ipynb


---

## 🔍 Implementations Overview

### ⚡ DETR (DEtection TRansformer)
- Uses Transformer architecture with a ResNet-50 backbone
- Outputs object bounding boxes and class labels
- Implemented using HuggingFace's `transformers` and `DetrImageProcessor`
- Supports high-accuracy end-to-end detection
- Threshold: `0.86`

### ⚡ Faster R-CNN (CNN-based Detector)
- ResNet50-FPN backbone pre-trained on COCO dataset
- Implemented with PyTorch and TorchVision
- Simple yet accurate method for real-time detection
- Confidence threshold: `0.6`

### ⚡ Qwen2.5-VL (Vision-Language Multimodal Model)
- Accepts image and text prompt together
- Performs **Zero-shot object grounding**
- Output: JSON containing label and bounding box
- Uses `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face

### ⚡ YOLOv8 (Video Object Detection)
- Real-time detection on video frames using `ultralytics` YOLOv8n model
- Supports multiple classes and live rendering
- Confidence threshold: `0.5`
- Frame-by-frame predictions and plotting via `cv2`

---

## 🧪 Requirements

Install dependencies using pip:

```bash
pip install torch torchvision transformers qwen-vl-utils ultralytics matplotlib opencv-python

If you're using Google Colab:
!pip install qwen-vl-utils ultralytics
