import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# ✅ Debe ir antes de cualquier otro uso de Streamlit
st.set_page_config(page_title="Detector de Señas", layout="centered")

# --- CONFIGURACIÓN GENERAL ---
IMG_SIZE = 128
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, NUM_CLASSES)
    )
    model.load_state_dict(torch.load("modelo_resnet_lengua_senas.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
CLASSES = [chr(ord('A') + i) for i in range(NUM_CLASSES)]

# --- TRANSFORMACIÓN DE IMAGEN ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- FUNCIÓN DE PREDICCIÓN ---
def predict(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = transforms.ToPILImage()(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = output.max(1)
    return CLASSES[pred.item()]

# --- PROCESAMIENTO DEL VIDEO ---
class SignLanguageDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Recuadro central (224x224)
        box_size = 224
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Recortar imagen en el centro
        center_crop = img[y1:y2, x1:x2]

        if center_crop.shape[0] != 0 and center_crop.shape[1] != 0:
            try:
                letra = predict(center_crop)
                cv2.putText(img, f"Letra: {letra}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            except Exception as e:
                cv2.putText(img, "Error en predicción", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Dibujar el recuadro de guía
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        return img

# --- INTERFAZ STREAMLIT ---
st.title("🤟 Detector de Lenguaje de Señas en Vivo")
st.markdown("Coloca tu mano dentro del recuadro azul. El modelo intentará predecir la letra que estás señalando.")

webrtc_streamer(key="sign-detect", video_transformer_factory=SignLanguageDetector)
