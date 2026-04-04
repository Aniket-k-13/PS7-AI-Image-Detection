# 🔍 PS7 — AI-Generated Image Detection & Artifact Identification
### Neural Nexus AI/ML Hackathon 2026

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-98.58%25-brightgreen)
![F1](https://img.shields.io/badge/F1%20Score-0.9854-brightgreen)
![AUC](https://img.shields.io/badge/ROC--AUC-0.9989-brightgreen)

---

## 🚀 Live Demo

## 🧪 Test the Model — Google Colab Demo

No installation needed. Just open and run:

👉 **[Open Demo Notebook in Colab](https://drive.google.com/file/d/143tXRwb6RR6MIuiRiXSSXCWuIqE_6PBY/view?usp=sharing)**

Steps:
1. Runtime → Change runtime type → T4 GPU
2. Run all cells in order
3. Upload any image in Cell 3 (u can use any image to test the model just go to any ai platform genrate any image and give it our model, also u can take few random photos from ur phone camera and give it to our model, it will accuratel detectect if the image is real or ai genrated}
4. Get instant prediction!

Model downloads automatically (~364MB, ~1 min).

---

## download Model

the model Size is over the limit of github so we are not able to add it to repository, U can downlaod the the model using the link provided below
[link to download model:https://drive.google.com/file/d/1N_4mZUGXTMa_8nbbAJswuUtrLHKxgM_m/view?usp=sharing]

## 📌 Problem Statement

Advancements in generative AI (Stable Diffusion, Midjourney, DALL-E) have enabled the creation of photorealistic synthetic images indistinguishable from real photographs. This raises serious concerns about digital fraud, misinformation, and trust in visual media.

**Our system:**
- Classifies any image as **REAL** or **AI-GENERATED** with 98.58% accuracy
- Identifies **exactly where** fake artifacts appear using Grad-CAM heatmaps
- Provides a **live web interface** anyone can use instantly

---

## 🏆 Results

| Metric | Score |
|---|---|
| **Accuracy** | **98.58%** |
| **F1 Score** | **0.9854** |
| **ROC-AUC** | **0.9989** |
| **Precision** | 0.9851 |
| **Recall** | 0.9857 |
| Misclassification rate | 1.4% (142/10,000) |

> **Beats the original CIFAKE paper** — Bird & Lotfi (IEEE Access, 2024) achieved 92.98% using a standard CNN. Our hybrid model improves this by **+5.6%**.

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
        │
        ├──► EfficientNet-B0 ──► Local features  (1280-d)
        │     Detects: texture artifacts, GAN noise,
        │              blurring at object boundaries
        │
        ├──► ViT-Base/16     ──► Global features  (768-d)
        │     Detects: structural inconsistencies,
        │              wrong proportions, context mismatches
        │
        └──► Concatenate (2048-d)
                    │
             Linear(2048 → 512)
             BatchNorm1d + ReLU + Dropout(0.3)
             Linear(512 → 2)
                    │
              FAKE  /  REAL
```

---

## 🔬 Explainability — Grad-CAM

Grad-CAM highlights the exact pixel regions that influenced the decision.

- 🔴 Red/yellow = high artifact confidence regions
- 🔵 Blue = low influence regions

Common AI artifacts detected: unnatural texture repetition, blurring at boundaries, inconsistent lighting, unnaturally smooth surfaces.

---
<img width="949" height="1965" alt="image" src="https://github.com/user-attachments/assets/06e52cb1-26b6-4cd4-8912-099c49309241" />



## 📁 Project Structure

```
PS7-AI-Image-Detection/
├── src/
│   ├── app.py                       # Streamlit web app
│   └── model.pth                    # Trained model (364MB)
├── PS7_CIFAKE_Detection.ipynb       # Full training notebook
├── Dockerfile                       # Docker container
├── requirements.txt                 # Dependencies
├── run.sh                           # Linux/Mac one-click run
├── run.bat                          # Windows one-click run
└── README.md
```

---

## ⚡ Quick Start

**Linux/Mac:**
```bash
chmod +x run.sh && ./run.sh
```

**Windows:**
```
run.bat
```

**Docker:**
```bash
docker build -t ai-detector .
docker run -p 7860:7860 ai-detector
```

**Manual:**
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

**Train from scratch (Colab T4):**
```
1. Open PS7_CIFAKE_Detection.ipynb in Google Colab
2. Runtime → T4 GPU → Run all cells
```

---

## 📦 Dataset

**CIFAKE** — 120,000 images (50k REAL + 50k FAKE train, 10k each test)

FAKE images generated with Stable Diffusion v1.4.

Source: [Kaggle — birdy654/cifake](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

---

## 📊 Training

| Parameter | Value |
|---|---|
| Epochs | 18 (6 frozen + 12 fine-tuned) |
| Optimizer | Adam (lr=1e-4) |
| Scheduler | OneCycleLR |
| Batch size | 32 |
| Image size | 224×224 |
| GPU | Google Colab T4 |

---

## 📈 Model Comparison

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| ResNet50 baseline | 87.6% | 0.871 | 0.945 |
| EfficientNet-B0 only | 91.2% | 0.909 | 0.972 |
| ViT-Base only | 90.3% | 0.899 | 0.968 |
| Bird & Lotfi 2024 (paper) | 92.98% | — | — |
| **Hybrid EffNet+ViT (ours)** | **98.58%** | **0.9854** | **0.9989** |

---

## 🛠️ Tech Stack

Python 3.10 · PyTorch 2.0 · EfficientNet-B0 · ViT-Base/16 · timm · Grad-CAM · Streamlit · Docker · Hugging Face Spaces

---

## 📚 Citations

```bibtex
@article{bird2024cifake,
  title={CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images},
  author={Bird, Jordan J. and Lotfi, Ahmad},
  journal={IEEE Access},
  volume={12},
  pages={15642--15650},
  year={2024},
  doi={10.1109/ACCESS.2024.3356122}
}
```

Pre-trained models: EfficientNet-B0 (Tan & Le, ICML 2019), ViT-Base/16 (Dosovitskiy et al., ICLR 2021), Grad-CAM (Selvaraju et al., ICCV 2017), timm (Ross Wightman, 2019).

---

## 👥 Team — Neural Nexus Hackathon 2026 | Problem Statement 7
