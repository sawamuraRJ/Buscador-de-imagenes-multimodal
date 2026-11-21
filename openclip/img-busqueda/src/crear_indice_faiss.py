
import faiss
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RUTA_EMBEDDINGS = Path("embeddings/embeddings_imagenes.npy")
RUTA_IDS = Path("embeddings/ids_imagenes.npy")
RUTA_INDICE = Path("embeddings/indice_hnsw.faiss")

DIMENSIONES = 512

M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64

# ==========================================
# 2. CARGAR EMBEDDINGS E IDS
# ==========================================

embeddings = np.load(RUTA_EMBEDDINGS)
ids = np.load(RUTA_IDS).astype(np.int64)

print("Embeddings cargados:", embeddings.shape)
print("IDs cargados:", ids.shape)

# ==========================================
# 3. VALIDAR DATOS
# ==========================================

assert embeddings.ndim == 2
assert embeddings.shape[1] == DIMENSIONES
assert len(embeddings) == len(ids)
assert np.isfinite(embeddings).all()

# FAISS trabaja con matrices float32 contiguas
embeddings = np.ascontiguousarray(
    embeddings,
    dtype=np.float32
)

# ==========================================
# 4. CREAR ÍNDICE HNSW
# ==========================================

indice_base = faiss.IndexHNSWFlat(
    DIMENSIONES,
    M,
    faiss.METRIC_INNER_PRODUCT
)

indice_base.hnsw.efConstruction = EF_CONSTRUCTION
indice_base.hnsw.efSearch = EF_SEARCH

# Asociar IDs originales de COCO
indice = faiss.IndexIDMap2(indice_base)

# ==========================================
# 5. AGREGAR VECTORES AL ÍNDICE
# ==========================================

indice.add_with_ids(embeddings, ids)

print("\nÍndice HNSW construido.")
print("Total de vectores:", indice.ntotal)

# ==========================================
# 6. GUARDAR ÍNDICE
# ==========================================

faiss.write_index(indice, str(RUTA_INDICE))

print("Índice guardado en:", RUTA_INDICE)
print("M:", M)
print("efConstruction:", EF_CONSTRUCTION)
print("efSearch:", EF_SEARCH)

print("\nPROCESO FINALIZADO CORRECTAMENTE.")
