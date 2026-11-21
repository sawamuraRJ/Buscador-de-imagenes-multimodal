
import torch
import open_clip

# 1. Definir el dispositivo
dispositivo = "cpu"

# 2. Cargar el modelo OpenCLIP
modelo, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=dispositivo
)

# 3. Cargar el tokenizador
tokenizador = open_clip.get_tokenizer("ViT-B-32")

# 4. Mostrar información
modelo.eval()

print("Modelo cargado correctamente")
print("Arquitectura: ViT-B-32")
print("Pesos: laion2b_s34b_b79k")
print("Dispositivo:", dispositivo)
print("Modelo en modo evaluación:", not modelo.training)
