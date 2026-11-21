# BÚSQUEDA TEXTO - IMAGEN

import json
import time
import faiss
import torch
import open_clip
import numpy as np

from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
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

modelo, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=DISPOSITIVO
)

tokenizador = open_clip.get_tokenizer("ViT-B-32")

modelo.eval()

print("Modelo OpenCLIP cargado correctamente.")

# ==========================================
# 3. CARGAR EL ÍNDICE FAISS
# ==========================================

indice = faiss.read_index(str(RUTA_INDICE))

# Accedemos al índice HNSW interno
indice_hnsw = faiss.downcast_index(indice.index)

# Configuramos la búsqueda HNSW
#indice.index.hnsw.efSearch = EF_SEARCH
indice_hnsw.hnsw.efSearch = EF_SEARCH

print("Índice FAISS cargado.")
print("Total de imágenes indexadas:", indice.ntotal)

# ==========================================
# 4. CARGAR METADATOS DE LAS IMÁGENES
# ==========================================

ids = np.load(RUTA_IDS)

with open(RUTA_METADATA, "r", encoding="utf-8") as archivo:
    nombres = json.load(archivo)

# Relacionamos cada ID con su nombre de archivo
id_a_nombre = {
    int(image_id): nombre
    for image_id, nombre in zip(ids, nombres)
}

# Cargamos las rutas originales desde el manifiesto
id_a_ruta = {}

with open(RUTA_MANIFEST, "r", encoding="utf-8") as archivo:
    for linea in archivo:
        registro = json.loads(linea)
        id_a_ruta[registro["image_id"]] = registro["file_path"]

print("Metadatos cargados correctamente.")

# ==========================================
# 5. FUNCIÓN DE BÚSQUEDA POR TEXTO
# ==========================================

def buscar_imagenes(consulta, top_k=TOP_K):

    inicio = time.perf_counter()

    # Convertimos el texto en tokens
    tokens = tokenizador([consulta]).to(DISPOSITIVO)

    # Generamos el embedding del texto
    with torch.no_grad():
        vector = modelo.encode_text(tokens)

        # Normalizamos el vector
        vector = vector / vector.norm(dim=-1, keepdim=True)

    # Convertimos el embedding a formato NumPy float32
    vector = np.ascontiguousarray(
        vector.cpu().numpy(),
        dtype=np.float32
    )

    # Buscamos las imágenes más similares en FAISS
    k = min(top_k, indice.ntotal)

    puntuaciones, resultados_ids = indice.search(vector, k)

    tiempo_total = time.perf_counter() - inicio

    # ======================================
    # 6. MOSTRAR LOS RESULTADOS
    # ======================================

    print("\n" + "=" * 55)
    print("CONSULTA:", consulta)
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

    print("\nBUSCADOR MULTIMODAL - TEXTO A IMAGEN")
    print("Escribe 'salir' para terminar.")

    while True:

        consulta = input("\n¿Qué imagen deseas encontrar?: ").strip()

        if consulta.lower() == "salir":
            print("Buscador finalizado.")
            break

        if not consulta:
            print("Por favor, escribe una descripción.")
            continue

        buscar_imagenes(consulta)
