# Detector de Lenguaje de Señas con YOLOv5

Esta aplicación permite detectar letras del lenguaje de señas en tiempo real usando un modelo YOLOv5 entrenado. La interfaz está implementada en **Streamlit**, lo que facilita su uso directamente desde el navegador.

## Características

- Detección de letras del lenguaje de señas mediante la cámara web.
- Visualización del video en tiempo real con cuadros y etiquetas sobre las detecciones.
- Confirmación de letras si se mantienen durante 1 segundo.
- Construcción dinámica del texto confirmado con las letras detectadas.

## 🖥Tecnologías

- **Streamlit**: interfaz gráfica.
- **YOLOv5**: modelo de detección de letras entrenado.
- **OpenCV**: captura y procesamiento de video.
- **PyTorch**: ejecución del modelo.

## ⚙Requisitos

- Python 3.8 o superior.
- Dependencias:
    pip install streamlit torch opencv-python-headless numpy

## Cómo ejecutar
- streamlit run app.py
