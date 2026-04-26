# 🇧🇩 Bangladeshi Taka Note Detection Using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-orange)](YOUR_GOOGLE_DRIVE_LINK_HERE)

> **Module 16 Assignment** — Detecting and classifying Bangladeshi Taka currency notes (and optionally coins) using a fine-tuned YOLOv8 object detection model.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Model Configuration](#model-configuration)
- [Results](#results)
- [Bonus: Coin Detection](#bonus-coin-detection)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## 📖 Project Overview

This project trains a **YOLOv8** object detection model to identify and classify Bangladeshi Taka currency notes across **9 denominations**:

| Denomination | Label |
|---|---|
| 2 Taka | `2tk` |
| 5 Taka | `5tk` |
| 10 Taka | `10tk` |
| 20 Taka | `20tk` |
| 50 Taka | `50tk` |
| 100 Taka | `100tk` |
| 200 Taka | `200tk` |
| 500 Taka | `500tk` |
| 1000 Taka | `1000tk` |

The model is fine-tuned from a pretrained YOLOv8 checkpoint and evaluated on a held-out test set.

---

## 🎬 Demo

> Sample inference output with bounding boxes:

![Inference Sample](assets/inference_sample.jpg)

*More sample images are available in the [`results/`](results/) folder.*

---

## 📦 Dataset

### Collection

Images were collected from:
- Online open-source image repositories
- Captured using a mobile phone camera

Each denomination class includes images with variation in:
- **Background** (plain, textured, real-world surfaces)
- **Lighting conditions** (bright, dim, natural, artificial)
- **Orientation and scale** (flat, angled, partially overlapping)

### Download

📁 **[Google Drive Dataset Link](YOUR_GOOGLE_DRIVE_LINK_HERE)**

### Dataset Split Statistics

| Split | Images | Percentage |
|---|---|---|
| Training | ~XXX | ~70% |
| Validation | ~XXX | ~20% |
| Test | ~XXX | ~10% |
| **Total** | **~XXX** | **100%** |

### Per-Class Image Count

| Class | # Images |
|---|---|
| 2 Taka | XX |
| 5 Taka | XX |
| 10 Taka | XX |
| 20 Taka | XX |
| 50 Taka | XX |
| 100 Taka | XX |
| 200 Taka | XX |
| 500 Taka | XX |
| 1000 Taka | XX |

### Annotation

All images were annotated using **[Roboflow](https://roboflow.com/)** / **[MakeSense.ai](https://www.makesense.ai/)** and exported in **YOLO format** (`.txt` label files).

---

## 📁 Dataset Structure

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**`data.yaml`** example:
```yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images

nc: 9
names: ['2tk', '5tk', '10tk', '20tk', '50tk', '100tk', '200tk', '500tk', '1000tk']
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/taka-note-detection.git
cd taka-note-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
ultralytics>=8.0.0
torch>=1.13.0
torchvision
opencv-python
matplotlib
pandas
numpy
PyYAML
```

### 4. Download the Dataset

Download the dataset from the [Google Drive link](YOUR_GOOGLE_DRIVE_LINK_HERE) and place it in the `dataset/` directory.

---

## 🚀 Usage

### Training

```bash
python train.py
```

Or directly with the Ultralytics CLI:

```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=taka_detector
```

### Evaluation

```bash
python evaluate.py
```

Or:

```bash
yolo detect val \
  model=runs/detect/taka_detector/weights/best.pt \
  data=dataset/data.yaml \
  split=test
```

### Inference

**On a single image:**

```bash
python inference.py --source path/to/image.jpg --weights runs/detect/taka_detector/weights/best.pt
```

**On a folder of images:**

```bash
yolo detect predict \
  model=runs/detect/taka_detector/weights/best.pt \
  source=dataset/test/images/ \
  conf=0.5 \
  save=True
```

---

## 🛠️ Model Configuration

| Parameter | Value |
|---|---|
| Base Model | `yolov8n.pt` (pretrained) |
| Epochs | 50 |
| Batch Size | 16 |
| Image Resolution | 640 × 640 |
| Optimizer | SGD (default) |
| Learning Rate | 0.01 |
| Number of Classes | 9 |

Training logs and model weights are saved to:

```
runs/
└── detect/
    └── taka_detector/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        └── confusion_matrix.png
```

---

## 📊 Results

### Evaluation Metrics (Test Set)

| Metric | Score |
|---|---|
| mAP@0.5 | X.XX |
| mAP@0.5:0.95 | X.XX |
| Precision | X.XX |
| Recall | X.XX |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Training Curves

![Training Results](results/results.png)

### Sample Detections

| Input Image | Detection Output |
|---|---|
| ![](assets/sample1_input.jpg) | ![](assets/sample1_output.jpg) |
| ![](assets/sample2_input.jpg) | ![](assets/sample2_output.jpg) |

---

## 🪙 Bonus: Coin Detection

As part of the optional bonus task, the model was extended to also detect **Bangladeshi coins**:

| Denomination | Label |
|---|---|
| 2 Taka Coin | `coin_2tk` |
| 5 Taka Coin | `coin_5tk` |

The model was retrained with the additional coin classes. Updated results and inference samples for coin detection are available in [`results/bonus/`](results/bonus/).

```bash
yolo detect train \
  data=dataset/data_with_coins.yaml \
  model=runs/detect/taka_detector/weights/best.pt \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  name=taka_detector_with_coins
```

---

## 🗂️ Project Structure

```
taka-note-detection/
│
├── dataset/                    # Dataset (download from Google Drive)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/                       # Training outputs (auto-generated)
│   └── detect/
│       └── taka_detector/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── results/                    # Evaluation results & sample images
│   ├── confusion_matrix.png
│   ├── results.png
│   └── bonus/
│
├── assets/                     # README images
│
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── inference.py                # Inference script
├── requirements.txt
├── notebook.ipynb              # Colab/Jupyter notebook (includes Drive link)
└── README.md
```

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — model framework
- [Roboflow](https://roboflow.com/) — annotation and dataset management
- [MakeSense.ai](https://www.makesense.ai/) — alternative annotation tool
- Bangladesh Bank — reference images for currency notes

---

## 📬 Contact

**Author:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

*This project was submitted as part of Module 16 Assignment on Object Detection using YOLO.*
