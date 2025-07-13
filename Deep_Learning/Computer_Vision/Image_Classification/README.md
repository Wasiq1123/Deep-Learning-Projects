# 🖼️ Image Classification with ViT, ResNet50, VGG16, and CNN (MNIST)

This folder contains a set of experiments to compare image classification performance using different deep learning models and techniques. The models include:

- 🔬 Vision Transformer (ViT) from HuggingFace Transformers
- 🧠 ResNet50 and VGG16 from TensorFlow/Keras Applications
- ✏️ Custom CNN trained on MNIST for handwritten digit recognition

---

## 📌 Project Highlights

### ✅ 1. Vision Transformer (ViT)
- 📦 Pretrained Model: [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224)
- 📷 Task: Predict image class from a single input image
- 🛠️ Framework: PyTorch + HuggingFace
- 🧠 Feature Extractor and Inference via `ViTFeatureExtractor` and `ViTForImageClassification`
- 📊 Output: Top-1 predicted class using `argmax(-1)`

➡️ **Notebook**: `ViT Image Classification.ipynb`

---

### ✅ 2. ResNet50 vs. VGG16 (TensorFlow)
- 📷 Task: Classify custom image of crocodile vs. alligator
- ⚙️ Framework: TensorFlow / Keras Applications
- 📊 Metrics Compared:
  - Inference Time
  - Number of Parameters
  - Top-1 Prediction and Confidence Score
- 🧪 ResNet50: [`tf.keras.applications.ResNet50`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
- 🧪 VGG16: [`tf.keras.applications.VGG16`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16)

➡️ **Notebook**: `Tf Resnet and VGG image classification comparison.ipynb`

---

### ✅ 3. MNIST Handwritten Digit Classification
- 📚 Dataset: `keras.datasets.mnist`
- 🔧 Architecture: Simple CNN with Conv2D → MaxPooling → Dense layers
- 🧠 Trained on grayscale (28x28) digit images
- 📈 Evaluation: Accuracy, Confusion Matrix, and Classification Report
- 🧪 Inference: `model.predict()` and argmax over softmax output

➡️ **Notebook**: `Hand_Written_Dataset Recognition.ipynb`

---

## 📊 Comparative Results

| Model      | Inference Time (s) | Parameters      | Dataset       | Accuracy / Prediction         |
|------------|--------------------|------------------|---------------|-------------------------------|
| ViT        | Fast (~0.61s)      | ~85M             | Custom Image  | Top-1 label (e.g., Tabby Cat) |
| ResNet50   | Fast (~0.36s)      | ~25.6M           | Custom Image  | Crocodile 🐊                   |
| VGG16      | Slower (~0.49s)    | ~138M            | Custom Image  | Alligator 🐊                   |
| CNN (MNIST)| Very Fast          | ~1.1M            | MNIST         | ~99% test accuracy            |

---

## 💻 Environment
- Python 3.8+
- TensorFlow 2.11+
- PyTorch
- HuggingFace Transformers
- OpenCV
- NumPy, Matplotlib

To install requirements:
```bash
pip install torch torchvision transformers tensorflow opencv-python matplotlib
