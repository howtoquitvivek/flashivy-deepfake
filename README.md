# **Falshivy-Deepfake v1**

Deepfake detection using deep learning.
This version establishes a **baseline CNN model** trained on images extracted from FaceForensics++ videos.

---

## 🚀 What This Tool Does (v1)

- Classifies **face images** as **Real** or **Deepfake**
- Learns visual artifacts introduced by face-swap manipulation
- Designed as a foundation for explainability and video-level models in later versions

> **v1 is image-based** (frames extracted from videos).

---

## 📂 Dataset Used (v1)

### **FaceForensics++ (Image Frames)**

Images are extracted from the **FaceForensics++** dataset using the official downloader.

**Composition**

- Real: frames from pristine YouTube videos
- Fake: frames from **DeepFakes** manipulated videos
- Compression: `c23`
- Balanced dataset

FaceForensics++ is a standard benchmark dataset widely used in deepfake detection research.

---

## 📥 Dataset Download

Dataset is downloaded using the official FaceForensics++ script.
To obtain the script, fill out the Google Form available on the  
[FaceForensics website](http://niessnerlab.org/projects/roessler2018faceforensics.html).

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

Image frames and face crops are generated from these videos for training.

---

## ⚙️ Environment Setup

### PyTorch

- **PyTorch 2.2+**
- **CUDA 12.1**
- Optimized for **RTX 4060 Laptop GPU**

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

**Included**

- Image-based deepfake classification
- CNN-based training pipeline
- FaceForensics++ data

**Not included**

- Video-level modeling
- Explainability (Grad-CAM)
- Cross-dataset testing

These are planned for future versions.

---

## 📄 License & Usage

For **academic and educational use only**.
Dataset usage follows the FaceForensics++ Terms of Service.