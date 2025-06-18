import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- CONFIGURACIÓN ---
IMG_SIZE = 128
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CARGA DEL MODELO ---
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
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Imagen no encontrada: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)

    predicted_class = CLASSES[predicted.item()]
    print(f"✅ Letra predicha: {predicted_class}")

# --- USO ---
if __name__ == "__main__":
    # Cambia aquí la ruta a tu imagen
    ruta_imagen = "image.jpg"
    predict_image(ruta_imagen)
