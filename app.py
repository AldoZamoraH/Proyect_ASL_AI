import streamlit as st
import torch
import cv2
import numpy as np

# --- CONFIGURACIÓN ---
MODEL_PATH = "yolov5/runs/train/yolov5_senas/weights/best.pt"  # Ruta al modelo entrenado

# --- CARGAR MODELO YOLOv5 ---
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    return model

model = load_model()

# --- TÍTULO DE APP ---
st.title("Detector de Señas con YOLOv5")
st.markdown("Muestra una letra con tu mano y el modelo YOLOv5 intentará detectarla en vivo.")

# --- INICIAR CÁMARA ---
run = st.checkbox("Iniciar cámara")
FRAME_WINDOW = st.image([])
letra_detectada = st.empty()  # Caja vacía para mostrar la letra detectada

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("No se pudo capturar el frame de la cámara.")
        break

    # Inference con YOLOv5
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Pandas DataFrame con detecciones

    letra_principal = "No detectado"

    # Dibujar cajas y etiquetas
    if not detections.empty:
        # Ordenar por confianza descendente
        detections = detections.sort_values(by="confidence", ascending=False)
        letra_principal = detections.iloc[0]['name']  # Tomar la letra con mayor confianza

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']
            conf = row['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Mostrar resultado en Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # Mostrar letra detectada debajo de la cámara
    letra_detectada.markdown(f"### Letra detectada: **{letra_principal}**")

# Liberar recursos
if cap:
    cap.release()
