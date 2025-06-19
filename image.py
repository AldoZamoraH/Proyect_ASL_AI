import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURACIÓN ---
MODEL_PATH = "yolov5/runs/train/yolov5_senas/weights/best.pt"  # Ruta al modelo entrenado

# --- CARGAR MODELO YOLOv5 ---
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    return model

model = load_model()

# --- TÍTULO DE APP ---
st.title("Detector de Señas con YOLOv5 - Imágenes")
st.markdown("Sube una imagen con una seña manual y el modelo intentará reconocerla.")

# --- CARGA DE IMAGEN ---
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer imagen como arreglo de OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Inference con YOLOv5
    results = model(image_bgr)
    detections = results.pandas().xyxy[0]  # Pandas DataFrame con detecciones

    letra_principal = "No detectado"

    # Dibujar cajas y etiquetas
    if not detections.empty:
        detections = detections.sort_values(by="confidence", ascending=False)
        letra_principal = detections.iloc[0]['name']  # Letra más confiable

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']
            conf = row['confidence']
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Mostrar imagen procesada
    st.image(image_np, caption="Resultado de la detección", channels="RGB")
    st.markdown(f"### Letra detectada: **{letra_principal}**")
