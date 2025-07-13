# 🧠 Vision-Language Models (VLMs) with CLIP and Qwen2.5-VL

This repository showcases multiple Vision-Language Model implementations using state-of-the-art architectures like **CLIP** and **Qwen2.5-VL** for tasks such as image captioning, object grounding, and token-level embedding analysis.

---

## 📁 Project Structure

Vision_Language_Models/
├── .gitkeep
├── CLIP Tokenization.ipynb
├── Owen2.5 Zero Shot Object Detection.ipynb
├── Owen 2.5 VLMs Image Captioning.ipynb

---

## 🔍 Implementations Overview

### ⚡ `CLIP Tokenization.ipynb`
- **Model**: `openai/clip-vit-base-patch32`
- **Task**: Compute cosine similarity between text and image embeddings.
- **Features**:
  - Tokenizes and embeds image and text inputs.
  - Calculates pairwise similarities between all pairs (text-text, image-image).
  - Useful for analyzing multi-class similarity and semantic alignment.

### ⚡ `Owen 2.5 VLMs Image Captioning.ipynb`
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Task**: Zero-shot image captioning
- **Features**:
  - Prompts are passed in a multimodal chat template (text + image).
  - Automatically generates detailed natural-language captions.
  - Built with Hugging Face + `qwen-vl-utils`.

### ⚡ `Owen2.5 Zero Shot Object Detection.ipynb`
- **Model**: Same as above
- **Task**: Zero-shot object grounding and localization
- **Features**:
  - Outputs object label + bounding boxes from just image and prompt.
  - Supports JSON-style outputs for integration into downstream systems.
  - Powerful for tasks where no retraining is needed.

---

## 🔧 Installation

Install the dependencies with:

```bash
pip install torch torchvision transformers qwen-vl-utils matplotlib opencv-python
For Qwen2.5-VL (if using Colab):

python
Copy
Edit
!pip install qwen-vl-utils
from huggingface_hub import login
login(token="your_huggingface_token")
