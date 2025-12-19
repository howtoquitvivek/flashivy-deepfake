# **Flashivy-Deepfake v1**

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
- Grad-CAM explainability

**Not included**

- Video-level modeling
- Cross-dataset testing


---

## 📊 Results

The model achieves high validation accuracy on the FaceForensics++ dataset due to the presence of clear manipulation artifacts in face-swap deepfakes. This version focuses on establishing a strong image-based baseline, while robustness across datasets is addressed in later versions.

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

![Grad-CAM Comparison](assets/gradcam_comparison.png)
Grad-CAM visualizations show that for real images, the model focuses on identity-related facial regions such as the eyes, while for fake images, attention is distributed across facial textures and blending regions, indicating the presence of manipulation artifacts.
```

---

## 📄 License & Usage

For **academic and educational use only**.
Dataset usage follows the FaceForensics++ Terms of Service.