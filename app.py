import gradio as gr
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# ============================
# MODEL DEFINITION — v3 MultiModal
# Must match training exactly
# ============================
class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.dct_net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(128*4*4, out_dim), nn.ReLU()
        )
        self.noise_net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(128*4*4, out_dim), nn.ReLU()
        )

    def get_dct_map(self, x):
        gray = (0.299*x[:,0]+0.587*x[:,1]+0.114*x[:,2]).unsqueeze(1)
        fft  = torch.fft.fft2(gray)
        mag  = torch.log(torch.abs(torch.fft.fftshift(fft)) + 1e-8)
        mn   = mag.amin(dim=(-2,-1), keepdim=True)
        mx   = mag.amax(dim=(-2,-1), keepdim=True)
        return (mag - mn) / (mx - mn + 1e-8)

    def get_noise_map(self, x):
        gray  = (0.299*x[:,0]+0.587*x[:,1]+0.114*x[:,2]).unsqueeze(1)
        blur  = nn.functional.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        noise = gray - blur
        mn    = noise.amin(dim=(-2,-1), keepdim=True)
        mx    = noise.amax(dim=(-2,-1), keepdim=True)
        return (noise - mn) / (mx - mn + 1e-8)

    def forward(self, x):
        return torch.cat(
            [self.dct_net(self.get_dct_map(x)),
             self.noise_net(self.get_noise_map(x))], dim=1
        )


class MultiModalDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.eff  = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.vit  = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.freq = FrequencyFeatureExtractor(out_dim=256)
        total_dim = self.eff.num_features + self.vit.num_features + 512
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 1024), nn.BatchNorm1d(1024),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(
            torch.cat([self.eff(x), self.vit(x), self.freq(x)], dim=1)
        )


# ============================
# CONFIG
# ============================
CLASSES  = ['FAKE', 'REAL']
IMG_SIZE = 224
DEVICE   = torch.device('cpu')

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ============================
# LOAD MODEL
# ============================
def load_model():
    # Check multiple possible paths
    for path in [
        'src/hybrid_effnet_vit_v3_best.pth',
        'hybrid_effnet_vit_v3_best.pth',
        'src/model.pth',
        'model.pth',
    ]:
        if os.path.exists(path):
            m = MultiModalDetector().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            print(f'✅ Model loaded from {path}')
            return m
    raise FileNotFoundError('No model file found. Upload model to src/ folder.')

model = load_model()


# ============================
# PREDICT FUNCTION
# ============================
def predict(image):
    if image is None:
        return "Please upload an image.", {}, ""

    tensor = tf(image.convert('RGB')).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        probs  = torch.softmax(logits, 1).cpu().numpy()[0]

    pred       = int(probs.argmax())
    label      = CLASSES[pred]
    confidence = float(probs.max()) * 100

    if label == 'FAKE':
        result = f"🔴 AI-GENERATED IMAGE\nConfidence: {confidence:.1f}%"
        explanation = (
            "This image shows signs of AI generation. "
            "The model detected anomalies in both visual texture patterns "
            "(EfficientNet + ViT) and frequency domain fingerprints (DCT + noise residual). "
            "Common indicators: unnatural textures, inconsistent lighting, "
            "or diffusion model artifacts."
        )
    else:
        result = f"🟢 REAL IMAGE\nConfidence: {confidence:.1f}%"
        explanation = (
            "No significant AI-generation artifacts detected. "
            "Visual features, frequency spectrum, and noise residual patterns "
            "are consistent with real photographic content from a camera sensor."
        )

    scores = {
        'FAKE (AI-Generated)': float(round(probs[0], 4)),
        'REAL (Authentic)':    float(round(probs[1], 4)),
    }

    return result, scores, explanation


# ============================
# GRADIO UI
# ============================
with gr.Blocks(title="AI Image Detector v3 | MultiModal") as demo:

    gr.Markdown("""
    # 🔍 AI-Generated Image Detector — v3 MultiModal
    **Detects AI-generated images using visual + frequency domain analysis**

    | Model | Architecture | Accuracy | F1 | AUC |
    |---|---|---|---|---|
    | v3 MultiModal (this) | EfficientNet + ViT + DCT + Noise | **91.00%** | **0.8956** | **0.9776** |
    | v1 (CIFAKE only) | EfficientNet + ViT | 98.58% | 0.9854 | 0.9989 |
    | Bird & Lotfi 2024 (paper) | CNN | 92.98% | — | — |

    > v3 trades some CIFAKE accuracy for **better generalization** to real-world photos and diverse AI generators.
    """)

    with gr.Row():
        gr.Markdown("**🖼️ EfficientNet-B0**\nDetects local texture artifacts and GAN fingerprints")
        gr.Markdown("**🧠 ViT-Base/16**\nDetects global structural inconsistencies")
        gr.Markdown("**📊 DCT Frequency**\nDetects unnatural frequency patterns in AI images")
        gr.Markdown("**🔬 Noise Residual**\nDetects absence of real camera sensor noise")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type='pil', label='📤 Upload Image (JPG, PNG, WEBP)')
            submit_btn  = gr.Button('🔍 Analyze Image', variant='primary', size='lg')

        with gr.Column(scale=1):
            result_text    = gr.Textbox(label='Prediction', lines=2, interactive=False)
            confidence_bar = gr.Label(label='Confidence Scores', num_top_classes=2)
            explanation    = gr.Textbox(label='Explanation', lines=4, interactive=False)

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[result_text, confidence_bar, explanation]
    )

    gr.Markdown("""
    ---
    ### ⚠️ Important Note on Model Scope
    This model was trained on:
    - **140k Real/Fake Faces** (StyleGAN vs real people)
    - **CIFAKE** (Stable Diffusion v1.4 fakes)
    - **AI vs Human Generated** dataset

    It works best on **face images and CIFAKE-style content**.
    Highly realistic modern AI images (Gemini, DALL-E 3) may still fool the model —
    this is an **active research problem** in the field.

    ---
    ### 📚 Citation
    > Bird & Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*,
    IEEE Access, 2024. [DOI: 10.1109/ACCESS.2024.3356122](https://doi.org/10.1109/ACCESS.2024.3356122)

    *Neural Nexus Hackathon 2026 — PS7 | EfficientNet-B0 + ViT-Base/16 + DCT + Noise Residual*
    """)

demo.launch(ssr_mode=False)
