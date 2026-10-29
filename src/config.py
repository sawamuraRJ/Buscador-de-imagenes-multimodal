"""Configuración central del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
THUMBS_DIR = ROOT / "data" / "thumbs"
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
PATHS_PATH = INDEX_DIR / "paths.json"
FRONTEND_DIR = ROOT / "frontend"

# Modelo Open CLIP (puedes cambiarlo por uno más grande sin tocar el resto)
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Procesamiento por lotes
BATCH_SIZE = 64
NUM_WORKERS = 2

# Índice HNSW (aproximado): M = vecinos por nodo, EF = calidad de construcción/búsqueda
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

TOP_K = 6
THUMB_SIZE = 256
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for d in (IMAGES_DIR, THUMBS_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)
