import streamlit as st
import torch
import cv2
import numpy as np
import time

# --- CONFIGURACIÓN ---
MODEL_PATH = "yolov5/runs/train/yolov5_senas/weights/best.pt"

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    return model

model = load_model()

# --- TÍTULO APP ---
st.title("Detector de Señas con YOLOv5")
st.markdown("Muestra una letra con tu mano. Si se mantiene 1 segundos, se añadirá abajo como texto.")

# --- ELEMENTOS STREAMLIT ---
run = st.checkbox("Iniciar cámara")
FRAME_WINDOW = st.image([])
letra_actual = st.empty()
letra_confirmada = st.empty()

# Variables de control
letra_anterior = None
tiempo_inicio = None
texto_detectado = ""

cap = None
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ No se pudo abrir la cámara. Asegúrate de que esté conectada y no esté siendo usada por otra aplicación.")
        run = False

while run and cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ No se pudo capturar el frame de la cámara.")
        break

    # --- DETECCIÓN ---
    results = model(frame)
    detections = results.pandas().xyxy[0]

    letra_detectada = None  # por defecto

    if not detections.empty:
        detections = detections.sort_values(by="confidence", ascending=False)
        letra_detectada = detections.iloc[0]['name']

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']
            conf = row['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # --- CONTROL DE TIEMPO PARA CONFIRMAR LETRA ---
    tiempo_actual = time.time()
    if letra_detectada is not None:
        if letra_detectada == letra_anterior:
            if tiempo_inicio and (tiempo_actual - tiempo_inicio >= 1):
                texto_detectado += letra_detectada
                letra_confirmada.markdown(f"### Texto confirmado: **{texto_detectado}**")
                tiempo_inicio = None
                letra_anterior = None
        else:
            letra_anterior = letra_detectada
            tiempo_inicio = tiempo_actual

    # --- MOSTRAR EN STREAMLIT ---
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    if letra_detectada:
        letra_actual.markdown(f"### Letra detectada: **{letra_detectada}**")
    else:
        letra_actual.markdown("### Letra detectada: **No detectado**")

# --- LIBERAR RECURSOS ---
if cap:
    cap.release()
