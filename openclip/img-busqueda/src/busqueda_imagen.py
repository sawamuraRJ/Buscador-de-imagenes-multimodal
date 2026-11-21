
import json
import time
import faiss
import torch
import open_clip
import numpy as np

from pathlib import Path
from PIL import Image

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RUTA_INDICE = Path("embeddings/indice_hnsw.faiss")
RUTA_IDS = Path("embeddings/ids_imagenes.npy")
RUTA_METADATA = Path("embeddings/metadata.json")
RUTA_MANIFEST = Path("Dataset_procesado/manifest.jsonl")

DISPOSITIVO = "cpu"
TOP_K = 6
EF_SEARCH = 64

# ==========================================
# 2. CARGAR EL MODELO OPENCLIP
# ==========================================

modelo, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=DISPOSITIVO
)

modelo.eval()

print("Modelo OpenCLIP cargado correctamente.")

# ==========================================
# 3. CARGAR EL ÍNDICE FAISS
# ==========================================

indice = faiss.read_index(str(RUTA_INDICE))

# Accedemos al índice HNSW interno
indice_hnsw = faiss.downcast_index(indice.index)

indice_hnsw.hnsw.efSearch = EF_SEARCH

print("Índice FAISS cargado.")
print("Total de imágenes indexadas:", indice.ntotal)

# ==========================================
# 4. CARGAR LOS METADATOS
# ==========================================

ids = np.load(RUTA_IDS)

with open(RUTA_METADATA, "r", encoding="utf-8") as archivo:
    nombres = json.load(archivo)

id_a_nombre = {
    int(image_id): nombre
    for image_id, nombre in zip(ids, nombres)
}

id_a_ruta = {}

with open(RUTA_MANIFEST, "r", encoding="utf-8") as archivo:
    for linea in archivo:
        registro = json.loads(linea)
        id_a_ruta[registro["image_id"]] = registro["file_path"]

print("Metadatos cargados correctamente.")

# ==========================================
# 5. FUNCIÓN DE BÚSQUEDA IMAGEN A IMAGEN
# ==========================================

def buscar_imagenes(ruta_consulta, top_k=TOP_K):

    ruta_consulta = Path(ruta_consulta)

    if not ruta_consulta.is_file():
        print("ERROR: No se encontró la imagen indicada.")
        return

    inicio = time.perf_counter()

    # Abrimos y preprocesamos la imagen
    try:
        with Image.open(ruta_consulta) as imagen:
            imagen = imagen.convert("RGB")
            imagen_tensor = preprocess(imagen).unsqueeze(0)

    except Exception as error:
        print("ERROR al procesar la imagen:", error)
        return

    # Generamos el embedding de la imagen
    imagen_tensor = imagen_tensor.to(DISPOSITIVO)

    with torch.no_grad():
        vector = modelo.encode_image(imagen_tensor)

        # Normalizamos el embedding
        vector = vector / vector.norm(dim=-1, keepdim=True)

    # Convertimos a NumPy float32
    vector = np.ascontiguousarray(
        vector.cpu().numpy(),
        dtype=np.float32
    )

    # Buscamos imágenes similares en FAISS
    k = min(top_k, indice.ntotal)

    puntuaciones, resultados_ids = indice.search(vector, k)

    tiempo_total = time.perf_counter() - inicio

    # ======================================
    # 6. MOSTRAR LOS RESULTADOS
    # ======================================

    print("\n" + "=" * 55)
    print("IMAGEN DE CONSULTA:", ruta_consulta)
    print("=" * 55)

    for posicion, (image_id, puntuacion) in enumerate(
        zip(resultados_ids[0], puntuaciones[0]), start=1
    ):

        if image_id == -1:
            continue

        nombre = id_a_nombre.get(int(image_id), "Desconocido")
        ruta = id_a_ruta.get(int(image_id), "Ruta no encontrada")

        print(f"\nResultado {posicion}")
        print("ID:", image_id)
        print("Archivo:", nombre)
        print("Ruta:", ruta)
        print("Similitud:", round(float(puntuacion), 4))

    print("\nTiempo total de búsqueda:", round(tiempo_total, 4), "segundos")


# ==========================================
# 7. INTERACCIÓN CON EL USUARIO
# ==========================================

if __name__ == "__main__":

    print("\nBUSCADOR MULTIMODAL - IMAGEN A IMAGEN")
    print("Escribe 'salir' para terminar.")

    while True:

        ruta = input("\nIngresa la ruta de una imagen: ").strip()

        # Quitamos comillas si se pega una ruta de Windows
        ruta = ruta.strip('"').strip("'")

        if ruta.lower() == "salir":
            print("Buscador finalizado.")
            break

        if not ruta:
            print("Por favor, ingresa una ruta.")
            continue

        buscar_imagenes(ruta)
