# 🧠 Vision-Language Models (VLMs) with CLIP and Qwen2.5-VL

This repository showcases powerful **Vision-Language Models** combining image and text understanding, featuring **CLIP** and **Qwen2.5-VL** for tasks such as image captioning, object grounding, and semantic similarity.

---

## 📂 Project Structure

📁 `Vision_Language_Models`

```
├── .gitkeep
├── CLIP Tokenization.ipynb
├── Owen2.5 Zero Shot Object Detection.ipynb
├── Owen 2.5 VLMs Image Captioning.ipynb
```

---

## 🔍 Implementations Overview

### ⚡ CLIP Tokenization & Embedding

**Notebook**: `CLIP Tokenization.ipynb`
**Model**: `openai/clip-vit-base-patch32`

📌 **Highlights**:

* Tokenizes both images and text to generate embeddings
* Computes **cosine similarity** for image-image and text-text pairs
* Explores cross-modal relationships using `torch.nn.functional.cosine_similarity`
* Ideal for **zero-shot classification**, **semantic alignment**, and **embedding analysis**

---

### ⚡ Qwen2.5-VL Image Captioning

**Notebook**: `Owen 2.5 VLMs Image Captioning.ipynb`
**Model**: `Qwen/Qwen2.5-VL-3B-Instruct`

📌 **Highlights**:

* Accepts **multimodal prompts**: (Image + Text)
* Generates **zero-shot image captions** in natural language
* Uses HuggingFace Transformers + `qwen-vl-utils`
* Easy-to-use template-based chat formatting

---

### ⚡ Qwen2.5-VL Zero-Shot Object Detection

**Notebook**: `Owen2.5 Zero Shot Object Detection.ipynb`
**Model**: `Qwen/Qwen2.5-VL-3B-Instruct`

📌 **Highlights**:

* Performs **object grounding** using only visual input + descriptive prompt
* Outputs **JSON with bounding boxes** and **semantic labels**
* Perfect for **zero-shot localization** in unseen scenarios
* No training/fine-tuning required

---

## 🧪 Requirements

Install dependencies using pip:

```bash
pip install torch torchvision transformers qwen-vl-utils matplotlib opencv-python
```

If you're using **Google Colab** with Hugging Face models:

```python
!pip install qwen-vl-utils

from huggingface_hub import login
login(token="your_huggingface_token")
```

---

## 💡 Applications

✔️ Zero-shot **image captioning**
✔️ **Visual grounding** without retraining
✔️ **Vision-language alignment** for embedding analysis
✔️ Integration into **multimodal AI agents**, smart search, or chat-based systems

---

## 🔗 References

* [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020)
* [CLIP on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
* [Qwen2.5-VL Hugging Face Page](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
* [qwen-vl-utils GitHub](https://github.com/QwenLM/Qwen-VL)

---
