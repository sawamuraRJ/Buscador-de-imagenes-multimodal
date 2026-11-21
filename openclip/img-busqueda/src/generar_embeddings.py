
import json
import torch
import open_clip
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RUTA_MANIFEST = Path("Dataset_procesado/manifest.jsonl")
RUTA_SALIDA = Path("embeddings")

DISPOSITIVO = "cpu"
TAMANO_BATCH = 16

# ==========================================
# 2. CARGAR EL MODELO OPENCLIP
# ==========================================

modelo, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=DISPOSITIVO
)

modelo.eval()

print("Modelo OpenCLIP cargado.")

# ==========================================
# 3. LEER EL MANIFIESTO
# ==========================================

with open(RUTA_MANIFEST, "r", encoding="utf-8") as archivo:
    registros = [json.loads(line) for line in archivo]

print("Total de imágenes:", len(registros))

# ==========================================
# 4. PREPARAR ALMACENAMIENTO
# ==========================================

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

embeddings = []
ids = []
nombres = []

# ==========================================
# 5. PROCESAR IMÁGENES POR LOTES
# ==========================================

for inicio in tqdm(
    range(0, len(registros), TAMANO_BATCH),
    desc="Generando embeddings"
):

    lote = registros[inicio:inicio + TAMANO_BATCH]

    imagenes = []
    registros_validos = []

    for registro in lote:
        ruta = Path(registro["file_path"])

        try:
            with Image.open(ruta) as imagen:
                imagen = imagen.convert("RGB")
                tensor = preprocess(imagen)

            imagenes.append(tensor)
            registros_validos.append(registro)

        except Exception as error:
            print(f"Error procesando {ruta}: {error}")

    if not imagenes:
        continue

    batch = torch.stack(imagenes).to(DISPOSITIVO)

    # Extraer características de las imágenes
    with torch.no_grad():
        vectores = modelo.encode_image(batch)

        # Normalizar los vectores
        vectores = vectores / vectores.norm(
            dim=-1, keepdim=True
        )

    embeddings.append(vectores.cpu().numpy())

    for registro in registros_validos:
        ids.append(registro["image_id"])
        nombres.append(registro["file_name"])

# ==========================================
# 6. UNIR Y GUARDAR RESULTADOS
# ==========================================

if embeddings:
    matriz_embeddings = np.concatenate(embeddings, axis=0)

    np.save(
        RUTA_SALIDA / "embeddings_imagenes.npy",
        matriz_embeddings
    )

    np.save(
        RUTA_SALIDA / "ids_imagenes.npy",
        np.array(ids)
    )

    with open(
        RUTA_SALIDA / "metadata.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            nombres,
            archivo,
            ensure_ascii=False,
            indent=4
        )

    # ======================================
    # 7. MOSTRAR RESULTADOS
    # ======================================

    print("\nPROCESAMIENTO FINALIZADO")
    print("Imágenes procesadas:", len(ids))
    print("Forma de los embeddings:", matriz_embeddings.shape)
    print("Dimensiones por imagen:", matriz_embeddings.shape[1])
    print("Archivos guardados en:", RUTA_SALIDA)

else:
    print("No se pudieron generar embeddings.")
