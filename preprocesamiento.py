import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

IMG_SIZE = 128
INPUT_DIR = 'dataset'
OUTPUT_DIR = 'dataset_preprocesado'

# Limpiar y crear carpeta salida
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

for clase in os.listdir(INPUT_DIR):
    clase_path = os.path.join(INPUT_DIR, clase)
    if not os.path.isdir(clase_path):
        continue

    save_dir = os.path.join(OUTPUT_DIR, clase)
    os.makedirs(save_dir, exist_ok=True)

    for filename in tqdm(os.listdir(clase_path), desc=f'Procesando {clase}'):
        img_path = os.path.join(clase_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Convertir a escala de grises (opcional)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0  # Normalizar entre 0 y 1

        # Guardar en uint8 para visualización o datasets que usen imágenes
        final_img = (normalized * 255).astype(np.uint8)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, final_img)

# Limpieza: eliminar archivos que no son carpetas y carpetas vacías
for nombre in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, nombre)
    if not os.path.isdir(path):
        print(f"Eliminando archivo no válido: {path}")
        os.remove(path)
    elif len(os.listdir(path)) == 0:
        print(f"Eliminando carpeta vacía: {path}")
        shutil.rmtree(path)

# Nota: para usar estas imágenes en PyTorch, usa transformaciones para convertir
# las imágenes a tensor y normalizarlas, ver Código 1.
