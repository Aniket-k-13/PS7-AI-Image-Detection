<img width="1188" height="390" alt="image" src="https://github.com/user-attachments/assets/466951ff-5b4e-458e-8476-810b82a9f118" /># 🔍 PS7 — AI-Generated Image Detection & Artifact Identification
### Neural Nexus AI/ML Hackathon 2026

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-98.58%25-brightgreen)
![F1](https://img.shields.io/badge/F1%20Score-0.9854-brightgreen)
![AUC](https://img.shields.io/badge/ROC--AUC-0.9989-brightgreen)

---

## 📌 Problem Statement

Advancements in generative AI have enabled the creation of highly realistic synthetic images. While this improves content creation, it raises serious concerns about authenticity and misuse. Detecting AI-generated images and identifying artifacts is essential for maintaining trust in digital media.

**Our system:**
- Classifies any image as **REAL** or **AI-generated** with 98.58% accuracy
- Identifies **exactly where** the fake artifacts are using Grad-CAM heatmaps
- Provides a fully deployed **web interface** for real-time detection

---

## ⚙️ Model Loading
Due to size limitations, the trained model is hosted on Google Drive and automatically downloaded during app startup.

## 🚀 Live Demo
https://yourapp.streamlit.app

## ⚙️ Model Loading
Model is automatically downloaded from Google Drive on first run.


⚠️ First run may take ~30 seconds.
## 🏆 Results

| Metric | Score |
|---|---|
| **Accuracy** | **98.58%** |
| **F1 Score** | **0.9854** |
| **ROC-AUC** | **0.9989** |
| **Precision** | 0.9851 |
| **Recall** | 0.9857 |
| Misclassification rate | 1.4% |

> **Beats the original CIFAKE paper** (Bird & Lotfi, 2024) which achieved 92.98% using a standard CNN. Our hybrid architecture improves this by ~5.6%.

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
        │
        ├──► EfficientNet-B0 ──► Local features  (1280-d)
        │     └─ Detects texture artifacts, noise,
        │        blurring at object boundaries
        │
        ├──► ViT-Base/16     ──► Global features  (768-d)
        │     └─ Detects structural inconsistencies,
        │        wrong proportions, context mismatches
        │
        └──► Concatenate (2048-d)
                    │
             Linear(2048→512)
             BatchNorm1d(512)
             ReLU
             Dropout(0.3)
             Linear(512→2)
                    │
              FAKE  /  REAL
```

**Why this works better than a single model:**
- AI-generated images leave **local texture artifacts** that EfficientNet detects well
- **Global structural inconsistencies** are captured by ViT's attention mechanism
- Together they catch what neither can alone

---

## 🔬 Explainability — Grad-CAM

Grad-CAM highlights the exact regions that influenced the model's decision.

- 🔴 **Red/yellow** = high artifact confidence regions
- 🔵 **Blue** = low influence regions

Common AI artifacts detected:
- Unnatural texture repetition (GAN mode collapse)
- Blurring at object boundaries (diffusion edge artifacts)
- Inconsistent lighting and shadow direction
- Unnaturally smooth surfaces

---

## 📁 Project Structure

```
PS7-AI-Image-Detection/
├── PS7_CIFAKE_Detection.ipynb   # Full training & evaluation notebook
├── app.py                        # Streamlit web deployment
├── run.bat                       # Windows — one-click run
├── run.sh                        # Linux/Mac — one-click run
├── requirements.txt              # All dependencies
└── README.md                     # This file
```

---

## ⚡ Quick Start

### Option 1 — One-click script (recommended)

**Linux / Mac:**
```bash
chmod +x run.sh && ./run.sh
```

**Windows:**
```
run.bat
```

### Option 2 — Manual

```bash
git clone https://github.com/YOUR_USERNAME/PS7-AI-Image-Detection.git
cd PS7-AI-Image-Detection
pip install -r requirements.txt
streamlit run app.py
```

### Option 3 — Train from scratch (Google Colab T4 GPU)

```
1. Open PS7_CIFAKE_Detection.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Training takes ~5 hours
```

---

## 📦 Dataset

**CIFAKE** — Real and AI-Generated Synthetic Images

| Split | REAL | FAKE | Total |
|---|---|---|---|
| Train | 50,000 | 50,000 | 100,000 |
| Test | 10,000 | 10,000 | 20,000 |

Source: [Kaggle — birdy654/cifake](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

---

## 🛠️ Technologies

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.0 |
| Models | EfficientNet-B0 + ViT-Base/16 (timm) |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Training | Mixed Precision AMP + OneCycleLR |
| Evaluation | scikit-learn |
| Deployment | Streamlit |

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 18 (6 frozen + 12 fine-tuned) |
| Learning rate | 1e-4 (OneCycleLR) |
| Optimizer | Adam (weight decay 1e-5) |
| Dropout | 0.3 |

---

## 📈 Model Comparison

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| ResNet50 (baseline) | 87.6% | 0.871 | 0.945 |
| EfficientNet-B0 only | 91.2% | 0.909 | 0.972 |
| ViT-Base only | 90.3% | 0.899 | 0.968 |
| Bird & Lotfi 2024 (paper) | 92.98% | — | — |
| **Hybrid EffNet+ViT (ours)** | **98.58%** | **0.9854** | **0.9989** |

---

## 📊 Visual Results

Below are key outputs from training:

- Training & ROC curves
  <img width="1188" height="390" alt="image" src="https://github.com/user-attachments/assets/30113e43-9914-4cc1-84ac-912522a62450" />
<img width="590" height="590" alt="image" src="https://github.com/user-attachments/assets/f3648891-0760-4f5a-a294-fe9cbdb17e9e" />

- Confusion matrix
  <img width="492" height="484" alt="image" src="https://github.com/user-attachments/assets/0c4d5cb8-d66f-4d58-b149-7c2b2adb3622" />

- Grad-CAM visualizations
  <img width="949" height="1965" alt="image" src="https://github.com/user-attachments/assets/1e76f85a-b394-44ea-8704-ca37df85c66f" />


> Full outputs are available inside the notebook.

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

Pre-trained models: EfficientNet-B0 (Tan & Le, ICML 2019), ViT-Base/16 (Dosovitskiy et al., ICLR 2021), timm library (Ross Wightman, 2019), Grad-CAM (Selvaraju et al., ICCV 2017)

---

## 👥 Team — Neural Nexus Hackathon 2026 | Problem Statement 7
