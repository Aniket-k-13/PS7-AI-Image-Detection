import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import io
import os
import gdown

# ============================
# MODEL DEFINITION
# ============================
class HybridEffNetViT(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        eff_out = self.eff.num_features

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        vit_out = self.vit.num_features

        self.classifier = nn.Sequential(
            nn.Linear(eff_out + vit_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f_eff = self.eff(x)
        f_vit = self.vit(x)
        fused = torch.cat([f_eff, f_vit], dim=1)
        return self.classifier(fused)

# ============================
# CONFIG
# ============================
CLASSES = ['FAKE', 'REAL']
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ============================
# MODEL LOADING
# ============================
MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1N_4mZUGXTMa_8nbbAJswuUtrLHKxgM_m"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = HybridEffNetViT(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ============================
# PREDICTION
# ============================
def predict(model, img_pil):
    tensor = tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, 1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, probs

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Generated Image Detector")
st.markdown("Hybrid EfficientNet-B0 + ViT | Accuracy: **98.58%**")

# ============================
# LOAD MODEL
# ============================
model = load_model()
st.success(f"✅ Model loaded on {DEVICE}")

# ============================
# IMAGE UPLOAD
# ============================
st.subheader("📤 Upload Image")

uploaded_img = st.file_uploader(
    "Upload image (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_img is not None:
    img_pil = Image.open(uploaded_img).convert("RGB")

    st.image(img_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        pred, probs = predict(model, img_pil)

    label = CLASSES[pred]
    confidence = float(probs.max()) * 100

    if label == "FAKE":
        st.error(f"🔴 AI-GENERATED IMAGE — {confidence:.2f}% confidence")
    else:
        st.success(f"🟢 REAL IMAGE — {confidence:.2f}% confidence")

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("FAKE Probability", f"{probs[0]*100:.2f}%")
    col2.metric("REAL Probability", f"{probs[1]*100:.2f}%")

    st.progress(int(confidence))

# ============================
# FOOTER
# ============================
st.markdown("---")
st.caption("Neural Nexus Hackathon 2026 | AI Image Detection")
