# 🧠 Image Understanding with Vision Transformers

This repository contains transformer-based implementations for image classification and object detection using state-of-the-art deep learning models:

* **DETR (DEtection TRansformer)**
* **ViT (Vision Transformer for Image Classification)**

---

## 📂 Project Structure

📁 `Vision_Transformers`

```
├── Detr Transformer.ipynb          # DETR object detection using Hugging Face Transformers
├── ViT Image Classification.ipynb  # Vision Transformer for classifying natural images
├── .gitkeep
```

---

## 🔍 Implementations Overview

### ⚡ DETR (DEtection TRansformer)

* Uses a **ResNet-50** CNN backbone combined with a **Transformer encoder-decoder**
* End-to-end object detection: predicts boxes and labels directly, no need for anchors or NMS
* Built using Hugging Face's `transformers` and `DetrImageProcessor`

📌 **Highlights:**

* High-accuracy bounding boxes and class predictions
* Hungarian matching for object association
* Zero post-processing detection pipeline
* Threshold: `0.86`

---

### ⚡ ViT (Vision Transformer)

* Treats images as sequences of patches instead of using convolutions
* Pre-trained on large-scale datasets like ImageNet
* Efficient for high-level visual understanding and classification

📌 **Highlights:**

* Patch-based transformer encoder
* Strong performance on clean and large image datasets
* Hugging Face `ViTFeatureExtractor` and `ViTModel` used
* Outputs class logits and embeddings

---

## 🧪 Requirements

Install dependencies using pip:

```bash
pip install torch torchvision transformers matplotlib opencv-python
```

If you're using Google Colab:

```python
!pip install transformers
```

---

## 💡 Applications

✔️ **DETR:**

* Robotics object perception
* Satellite image annotation
* Autonomous systems (visual detection)

✔️ **ViT:**

* Image classification
* Medical imaging
* E-commerce visual tagging

---

## 🔗 References

* [DETR Paper (Facebook AI)](https://arxiv.org/abs/2005.12872)
* [Vision Transformer Paper (Google)](https://arxiv.org/abs/2010.11929)
* [Hugging Face DETR](https://huggingface.co/facebook/detr-resnet-50)
* [Hugging Face ViT](https://huggingface.co/google/vit-base-patch16-224)

---
