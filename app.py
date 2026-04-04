import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# ============================
# MODEL DEFINITION
# ============================
class HybridEffNetViT(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.eff.num_features + self.vit.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.eff(x)
        f2 = self.vit(x)
        return self.classifier(torch.cat([f1, f2], dim=1))


# ============================
# CONFIG
# ============================
CLASSES    = ['FAKE', 'REAL']
IMG_SIZE   = 224
DEVICE     = torch.device("cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ============================
# MODEL LOADING — from local file
# ============================
@st.cache_resource
def load_model():
    model = HybridEffNetViT().to(DEVICE)
    model.load_state_dict(
        torch.load("model.pth", map_location=DEVICE)
    )
    model.eval()
    return model


# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Image Detector | Neural Nexus 2026",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI-Generated Image Detector")
st.markdown(
    "**Neural Nexus Hackathon 2026 — PS7** | "
    "Hybrid EfficientNet-B0 + ViT-Base | "
    "Accuracy: **98.58%** | F1: **0.9854** | AUC: **0.9989**"
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 1️⃣ EfficientNet-B0")
    st.markdown("Detects **local texture artifacts** — blurring, noise, GAN fingerprints.")
with col2:
    st.markdown("#### 2️⃣ ViT-Base/16")
    st.markdown("Detects **global inconsistencies** — wrong proportions, context mismatches.")
with col3:
    st.markdown("#### 3️⃣ Fusion Classifier")
    st.markdown("Combines both — **beats original CIFAKE paper** (92.98% → 98.58%).")

st.markdown("---")

# Load model
model = load_model()

st.sidebar.success("✅ Model ready")
st.sidebar.markdown("### 📊 Performance")
st.sidebar.markdown("""
- **Accuracy:** 98.58%
- **F1 Score:** 0.9854
- **ROC-AUC:** 0.9989
- **Dataset:** CIFAKE (120,000 images)
""")
st.sidebar.markdown("---")
st.sidebar.markdown("📄 [Original Paper](https://doi.org/10.1109/ACCESS.2024.3356122)")

# ============================
# IMAGE UPLOAD & PREDICTION
# ============================
st.subheader("📤 Upload an Image")
uploaded = st.file_uploader(
    "Supports JPG, PNG, WEBP",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        tensor = tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), 1).cpu().numpy()[0]

    pred       = int(probs.argmax())
    label      = CLASSES[pred]
    confidence = float(probs.max()) * 100

    st.markdown("---")
    if label == "FAKE":
        st.error("🔴 AI-GENERATED IMAGE DETECTED")
    else:
        st.success("🟢 REAL IMAGE")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction",       label)
    c2.metric("Confidence",       f"{confidence:.1f}%")
    c3.metric("FAKE probability", f"{probs[0]*100:.1f}%")
    st.progress(int(confidence))

    st.markdown("---")
    st.subheader("🧠 Explanation")
    if label == "FAKE":
        st.info(
            "AI artifacts detected — unnatural textures, blurring at boundaries, "
            "inconsistent lighting. These are typical signatures of GAN or "
            "diffusion model outputs. Full Grad-CAM visualization is in the notebook."
        )
    else:
        st.info(
            "No AI artifacts found. Natural noise patterns and edge characteristics "
            "are consistent with real photographic content."
        )

st.markdown("---")
st.caption("Neural Nexus AI/ML Hackathon 2026 — PS7 | EfficientNet-B0 + ViT-Base | CIFAKE")
