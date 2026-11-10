"""Configuración central del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
THUMBS_DIR = ROOT / "data" / "thumbs"
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
PATHS_PATH = INDEX_DIR / "paths.json"
FRONTEND_DIR = ROOT / "frontend"

# EmbeddingGemma 2: embeddings multimodales de texto e imagen en el mismo espacio.
EMBEDDING_MODEL = "google/embeddinggemma-2"
MODEL_INFO_PATH = INDEX_DIR / "model.json"

# Procesamiento por lotes
BATCH_SIZE = 8

# Índice HNSW (aproximado): M = vecinos por nodo, EF = calidad de construcción/búsqueda
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

TOP_K = 6
THUMB_SIZE = 256
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for d in (IMAGES_DIR, THUMBS_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)
