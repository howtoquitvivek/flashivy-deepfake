# **Flashivy-Deepfake v1**

Deepfake detection using deep learning. This version establishes a **baseline CNN model** trained on images extracted from FaceForensics++ videos.

---

## 📚 Table of Contents

* [🚀 What This Tool Does (v1)](#-what-this-tool-does-v1)
* [📂 Dataset Used (v1)](#-dataset-used-v1)

  * [Download REAL videos](#download-real-videos)
  * [Download FAKE videos](#download-fake-videos)
* [⚙️ Environment Setup](#-environment-setup)

  * [PyTorch](#pytorch)
  * [Other Dependencies](#other-dependencies)
  * [CUDA Check](#cuda-check)
* [📌 Version Scope](#-version-scope)
* [📊 Results](#-results)
* [🔍 Explainability (Grad-CAM)](#-explainability-grad-cam)

  * [Run Grad-CAM](#run-grad-cam)
* [▶️ Inference](#-inference)

  * [🔍 Running Inference](#-running-inference)
  * [📤 Output](#-output)
  * [📊 Interpretation of Confidence](#-interpretation-of-confidence)
  * [⚠️ Important Note on Generalization](#-important-note-on-generalization)
* [📄 License & Usage](#-license--usage)

---

## 🚀 What This Tool Does (v1)

* Classifies **face images** as **Real** or **Deepfake**
* Learns visual artifacts introduced by face-swap manipulation
* Designed as a foundation for explainability and video-level models in later versions

> **v1 is image-based** (frames extracted from videos).

---

## 📂 Dataset Used (v1)

### **FaceForensics++ (Image Frames)**

Images are extracted from the **FaceForensics++** dataset using the official downloader.

**Composition:**

* **Real:** Frames from pristine YouTube videos
* **Fake:** Frames from **DeepFakes** manipulated videos
* **Compression:** `c23`
* **Balanced dataset** (equal number of real and fake frames after extraction)

FaceForensics++ is a standard benchmark dataset widely used in deepfake detection research.

---

## 📥 Dataset Download

The dataset is downloaded using the official FaceForensics++ script.
To obtain access or use the original script included in this repository, users must fill out the Google Form and accept the terms available on the
[FaceForensics website](http://niessnerlab.org/projects/roessler2018faceforensics.html).

> Note: The dataset itself is not included in this repository. Users must download the data directly from the official source in accordance with the FaceForensics++ Terms of Service.

### Download REAL videos

```bash
python faceforensics_download_v4.py ff_data \
  --dataset original \
  --compression c23 \
  --type videos \
  --num_videos 25 \
  --server EU2
```

### Download FAKE videos

```bash
python faceforensics_download_v4.py ff_data \
  --dataset Deepfakes \
  --compression c23 \
  --type videos \
  --num_videos 25 \
  --server EU2
```

Image frames are extracted from videos, and **face crops are generated using the OpenCV Haar Cascade classifier (`haarcascade_frontalface_default.xml`)** for training.

---

## ⚙️ Environment Setup

### PyTorch

* **PyTorch 2.2+**
* **CUDA 12.1**
* Optimized for **RTX 4060 Laptop GPU**

Install PyTorch manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Other Dependencies

```bash
pip install -r requirements.txt
```

---

### CUDA Check

```bash
python check_cuda.py
```

---

## 📌 Version Scope

**Included:**

* Image-based deepfake classification
* CNN-based training pipeline
* FaceForensics++ data
* Grad-CAM explainability

**Not included:**

* Video-level modeling
* Cross-dataset testing

---

## 📊 Results

The model achieves high validation accuracy on the FaceForensics++ dataset due to the presence of clear manipulation artifacts in face-swap deepfakes. This version focuses on establishing a strong image-based baseline, while robustness across datasets is addressed in later versions.

---

## 🔍 Explainability (Grad-CAM)

To interpret the model’s predictions, Grad-CAM is used to visualize which facial regions contribute most to the real vs fake classification.

Grad-CAM heatmaps indicate that the model primarily focuses on facial regions such as the eyes, mouth, cheeks, and skin texture, which are known to contain manipulation artifacts in face-swap deepfakes.

### Run Grad-CAM

```bash
PYTHONPATH=. python - <<EOF
from explainability.grad_cam import plot_real_vs_fake

plot_real_vs_fake(
    "data/faces/real/IMAGE_NAME.jpg",
    "data/faces/fake/IMAGE_NAME.jpg"
)
EOF
```

![Grad-CAM Comparison](assets/gradcam_comparison.png)

*Example Grad-CAM visualization comparing real and fake face images.*

Grad-CAM visualizations show that for real images, the model focuses on identity-related facial regions such as the eyes, while for fake images, attention is distributed across facial textures and blending regions, indicating the presence of manipulation artifacts.

---

## ▶️ Inference

This repository provides a **command-line inference script** to perform deepfake prediction on a **single face image** using the trained **v1 CNN model**.

### 🔍 Running Inference

Use the following command to run inference:

```bash
PYTHONPATH=. python predict.py --image path/to/image.jpg
```

#### Input Requirements

* The input image should ideally be **face-cropped**
* Images similar to those used during training yield the best results
* For optimal performance, use images from the `data/faces/` directory

---

### 📤 Output

The script outputs:

* **Predicted label:** `REAL` or `FAKE`
* **Confidence score:** Model certainty for the predicted class

#### Example Output

```text
Prediction : FAKE
Confidence : 0.9821
```

---

### 📊 Interpretation of Confidence

* The confidence score represents the **softmax probability** of the predicted class.
* It reflects the model’s certainty **within the training distribution** (FaceForensics++).
* A **high confidence score does not guarantee correctness** on images from:

  * Unseen datasets
  * Different deepfake generation techniques

---

### ⚠️ Important Note on Generalization

The **v1 model is trained exclusively on the FaceForensics++ DeepFakes subset**.

While it achieves high accuracy on **in-distribution samples**, it may struggle to generalize to deepfakes created using **other methods or datasets**.

In particular:

* **Modern or visually clean deepfakes**
* Deepfakes lacking **FaceForensics++-style artifacts**

may be incorrectly classified as `REAL`, sometimes even with **high confidence**.

This limitation highlights the well-known **dataset bias problem** in image-based deepfake detection.

---

### 📄 License & Usage

* **Code License:** MIT License
* **Dataset Usage:** FaceForensics++ data is used strictly in accordance with its Terms of Service
* **Intended Use:** Academic and educational purposes only
