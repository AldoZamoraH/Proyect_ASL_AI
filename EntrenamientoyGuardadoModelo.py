import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Detectar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Configuraciones
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 26

# Transformaciones para entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet stats
                         std=[0.229, 0.224, 0.225])
])

# Para validación (usamos las mismas transformaciones que entrenamiento aquí)
val_transform = train_transform

# Cargar dataset completo sin separar
dataset = datasets.ImageFolder('dataset_preprocesado', transform=train_transform)

# DataLoader (solo entrenamiento aquí)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Clases detectadas:", dataset.classes)
print("Total de clases:", len(dataset.classes))

# Modelo ResNet50 preentrenado
model = models.resnet50(pretrained=True)

# Congelar todas las capas
for param in model.parameters():
    param.requires_grad = False

# Descongelar últimas capas (últimos 3 bloques)
for child in list(model.children())[-3:]:
    for param in child.parameters():
        param.requires_grad = True

# Modificar capa final
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, NUM_CLASSES)
)

model = model.to(device)

# Definir pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Entrenamiento (por ejemplo 5 epochs para prueba)
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")

# Guardar modelo entrenado
torch.save(model.state_dict(), 'modelo_resnet_lengua_senas.pth')

# --- PREDICCIÓN DE UNA IMAGEN NUEVA ---

def predict_image(image_path, model, classes, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = output.max(1)

    return classes[pred.item()]

# Ejemplo de uso:
ruta_imagen = 'q1.jpg'  # Cambia esta ruta
letra_predicha = predict_image(ruta_imagen, model, dataset.classes, device)
print(f"La letra predicha para la imagen es: {letra_predicha}")
