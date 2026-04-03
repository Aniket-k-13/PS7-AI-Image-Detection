import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import io
import os

# ============================
# MODEL DEFINITION
# ============================
class HybridEffNetViT(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        eff_out  = self.eff.num_features   # 1280
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        vit_out  = self.vit.num_features   # 768
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
CLASSES  = ['FAKE', 'REAL']
IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


import gdown
import os

MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = HybridEffNetViT(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict(model, img_pil):
    tensor = tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, 1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, probs, tensor


def get_gradcam(model, tensor, pred):
    target_layer  = [model.eff.conv_head]
    cam           = GradCAM(model=model, target_layers=target_layer)
    targets       = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    return grayscale_cam


# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Image Detector | Neural Nexus 2026",
    page_icon="🔍",
    layout="wide"
)

# ============================
# HEADER
# ============================
st.title("🔍 AI-Generated Image Detector")
st.markdown(
    "**Neural Nexus Hackathon 2026 — Problem Statement 7** | "
    "Hybrid EfficientNet-B0 + ViT-Base | Accuracy: **98.58%** | F1: **0.9854**"
)
st.markdown("---")

# ============================
# SIDEBAR
# ============================
st.sidebar.header("⚙️ Model Setup")
st.sidebar.markdown("Upload your trained `.pth` model file to get started.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown("""
- **Architecture:** EfficientNet-B0 + ViT-Base/16
- **Dataset:** CIFAKE (120,000 images)
- **Accuracy:** 98.58%
- **F1 Score:** 0.9854
- **ROC-AUC:** 0.9989
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Citation")
st.sidebar.markdown(
    "Bird & Lotfi, *CIFAKE*, IEEE Access 2024. "
    "[Paper](https://doi.org/10.1109/ACCESS.2024.3356122)"
)

# ============================
# MAIN CONTENT
# ============================
st.markdown("### 🧠 How it works")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1️⃣ EfficientNet-B0")
    st.markdown("Scans for **local texture artifacts** — blurring, noise patterns, and GAN fingerprints.")

with col2:
    st.markdown("#### 2️⃣ ViT-Base/16")
    st.markdown("Checks **global structure** — wrong proportions, misaligned shadows.")

with col3:
    st.markdown("#### 3️⃣ Grad-CAM")
    st.markdown("Highlights **artifact regions** with heatmaps.")

# Save model to temp file

model = load_model()
st.sidebar.success(f"✅ Model loaded ({DEVICE})")

# ============================
# IMAGE UPLOAD
# ============================
st.subheader("📤 Upload Image for Detection")
uploaded_img = st.file_uploader(
    "Choose an image — JPG, PNG, WEBP",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_img is not None:
    img_pil = Image.open(uploaded_img).convert("RGB")

    # Run prediction
    with st.spinner("🔍 Analyzing image..."):
        pred, probs, tensor   = predict(model, img_pil)
        grayscale_cam         = get_gradcam(model, tensor, pred)
        orig_np               = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        cam_overlay           = show_cam_on_image(
            orig_np.astype(np.float32), grayscale_cam, use_rgb=True
        )

    # ---- Result banner ----
    label      = CLASSES[pred]
    confidence = float(probs.max()) * 100

    if label == "FAKE":
        st.error(f"🔴 **AI-GENERATED IMAGE DETECTED** — Confidence: {confidence:.1f}%")
    else:
        st.success(f"🟢 **REAL IMAGE** — Confidence: {confidence:.1f}%")

    # ---- Metrics ----
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Prediction",      label)
    col_b.metric("Confidence",      f"{confidence:.1f}%")
    col_c.metric("FAKE probability", f"{probs[0]*100:.1f}%")
    col_d.metric("REAL probability", f"{probs[1]*100:.1f}%")

    st.progress(int(confidence))

    # ---- Images ----
    st.markdown("---")
    st.subheader("🖼️ Visual Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_pil, caption="Original Image", use_column_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(grayscale_cam, cmap='jet')
        ax.axis('off')
        ax.set_title("Artifact Heatmap", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🔴 Red = detected artifact regions")

    with col3:
        st.image(cam_overlay, caption="Grad-CAM Overlay", use_column_width=True)

    # ---- Explanation ----
    st.markdown("---")
    st.subheader("🧠 Explanation")
    if label == "FAKE":
        st.markdown(
            "The model detected **AI-generation artifacts** in this image. "
            "The red/yellow regions in the heatmap show where these artifacts were found. "
            "Common signs include: unnatural texture repetition, blurring at object boundaries, "
            "inconsistent lighting, and unnaturally smooth surfaces — all typical signatures "
            "of GAN or diffusion model outputs."
        )
    else:
        st.markdown(
            "The model found **no significant AI artifacts** in this image. "
            "The attention is spread across natural features like edges and textures, "
            "consistent with real photographic content. "
            "The background and fine details show natural noise patterns expected in real photos."
        )

    # ---- Download ----
    st.markdown("---")
    buf = io.BytesIO()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(orig_np);     axs[0].set_title("Original");          axs[0].axis("off")
    axs[1].imshow(grayscale_cam, cmap='jet'); axs[1].set_title("Artifact Heatmap"); axs[1].axis("off")
    axs[2].imshow(cam_overlay); axs[2].set_title(f"Pred: {label} ({confidence:.1f}%)"); axs[2].axis("off")
    plt.suptitle(f"AI Image Detection Result — {label} ({confidence:.1f}% confidence)", fontsize=13)
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    st.download_button(
        label="⬇️ Download Full Analysis",
        data=buf,
        file_name="ai_detection_result.png",
        mime="image/png"
    )

# ============================
# FOOTER
# ============================
st.markdown("---")
st.caption(
    "Model: Hybrid EfficientNet-B0 + ViT-Base/16 | "
    "Dataset: CIFAKE (120,000 images) | "
    "Explainability: Grad-CAM | "
    "Neural Nexus AI/ML Hackathon 2026 — PS7"
)
