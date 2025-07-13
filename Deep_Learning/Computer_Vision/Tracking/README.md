## 🧠 Real-Time Object Tracking with YOLOv8, YOLO11, and SORT

This repository showcases various real-time object tracking implementations using the following state-of-the-art tools:

* 🔗 [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
* 🔗 [YOLO11](https://github.com/ultralytics/ultralytics) (custom variant for fast detection)
* 🔗 [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort)
* 🔗 [Supervision Library](https://github.com/roboflow/supervision) for annotation and overlay

---

## 📂 Project Structure

```
📁 Tracking
├── Multi Tracking of Objects.ipynb            # 🔥 Multi-threaded tracker on 2+ video streams
├── Yolov8 tracking.ipynb                      # YOLOv8 video tracking (with flip & resize)
├── SORTTracker and YOLO11.py                  # Webcam-based real-time tracking using SORT
├── .gitkeep
```

---

## 🔍 Tracking Implementations Overview

### ⚔️ Multi-Tracking (⭐ Highlight Feature)

* **Track multiple videos simultaneously** using Python's `threading` module
* Each video stream runs a separate YOLOv8 model for detection + `.track()`
* Great for **security systems**, **multi-camera robotics**, or **surveillance control centers**
* Easily extendable to handle different model types (segmentation/detection)

---

### ⚡ [YOLOv8 with Built-in Tracking](https://docs.ultralytics.com/modes/track/)

* Uses `model.track(source=..., persist=True)`
* Frame-by-frame annotations using `.plot()`
* Supported for video files, webcams, or streams
* Ideal for lightweight deployments and quick setup

---

### ⚡ [YOLO11 + SORT Tracking](https://github.com/abewley/sort)

* Combines fast detection (YOLO11) with classic SORT tracking
* Stable tracking IDs using Kalman Filters and Hungarian Matching
* Annotated results with tracker IDs rendered using `supervision.LabelAnnotator`
* Efficient for real-time webcam tracking with minimum latency

---

## 💻 Requirements

Install dependencies:

```bash
pip install ultralytics supervision opencv-python
```

Or in Google Colab:

```python
!pip install ultralytics supervision
from google.colab import drive
drive.mount('/content/drive')
```
---

## 📌 Key Use Cases

✔️ Smart surveillance systems
✔️ Multi-camera monitoring
✔️ Robotics with camera navigation
✔️ Edge AI-based live tracking
✔️ Sports analytics & motion prediction
