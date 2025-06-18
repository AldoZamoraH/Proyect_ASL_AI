import os
import cv2
import random
from tqdm import tqdm

IMG_SIZE = 128
DATASET_DIR = 'dataset'  

def augment_image(img):
    augmented = []

    # Imagen original
    augmented.append(img)

    # Flip horizontal
    augmented.append(cv2.flip(img, 1))

    # Rotación aleatoria ±15 grados
    rows, cols = img.shape[:2]
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    augmented.append(rotated)

    # Saturación aumentada (solo si es color)
    if len(img.shape) == 3 and img.shape[2] == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.add(hsv[..., 1], 40)  # aumentar saturación
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented.append(saturated)

    return augmented

for subset in ['train', 'val']:
    subset_path = os.path.join(DATASET_DIR, subset)
    for clase in os.listdir(subset_path):
        clase_path = os.path.join(subset_path, clase)
        if not os.path.isdir(clase_path):
            continue

        for filename in tqdm(os.listdir(clase_path), desc=f'Procesando {subset} {clase}'):
            file_path = os.path.join(clase_path, filename)
            img = cv2.imread(file_path)
            if img is None:
                continue

            # Redimensionar
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Imagen en escala de grises con 3 canales
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Aumentaciones en color y gris
            variantes = augment_image(img) + augment_image(gray_3ch)

            # Guardar imágenes aumentadas (incluyendo la original)
            base_name = os.path.splitext(filename)[0]
            for i, var_img in enumerate(variantes):
                save_path = os.path.join(clase_path, f"{base_name}_aug{i}.jpg")
                cv2.imwrite(save_path, var_img)
