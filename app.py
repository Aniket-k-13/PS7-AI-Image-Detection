import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import gdown
import os

# ============================
# MODEL
# ============================
class HybridEffNetViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Linear(self.eff.num_features + self.vit.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.eff(x)
        f2 = self.vit(x)
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

# ============================
# CONFIG
# ============================
DEVICE = torch.device("cpu")
CLASSES = ["FAKE", "REAL"]
IMG_SIZE = 224

# ============================
# MODEL LOADING
# ============================
MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1N_4mZUGXTMa_8nbbAJswuUtrLHKxgM_m"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = HybridEffNetViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ============================
# IMAGE PREPROCESS (MANUAL)
# ============================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return torch.tensor(img, dtype=torch.float32)

# ============================
# UI
# ============================
st.title("🔍 AI Image Detector (FINAL FIXED VERSION)")
st.write("No Pillow. No torchvision. Works on Python 3.14.")

model = load_model()
st.success("✅ Model Loaded")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR")

    tensor = preprocess(img)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    pred = np.argmax(probs)
    confidence = probs[pred] * 100

    if CLASSES[pred] == "FAKE":
        st.error(f"🔴 FAKE — {confidence:.2f}%")
    else:
        st.success(f"🟢 REAL — {confidence:.2f}%")
