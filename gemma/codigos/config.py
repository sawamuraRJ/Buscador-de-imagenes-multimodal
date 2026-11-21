"""Configuración central del proyecto."""
from pathlib import Path
#rutas de las carpetas
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
IMAGES_DIR = ROOT / "data" / "images"
THUMBS_DIR = ROOT / "data" / "thumbs"
INDEX_DIR = ROOT / "data" / "index"
HNSW_INDEX_PATH = INDEX_DIR / "hnsw.index"
PATHS_DB_PATH = INDEX_DIR / "paths.sqlite3"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# OpenCLIP conserva sus artefactos y dataset dentro de su propio proyecto.
OPENCLIP_ROOT = PROJECT_ROOT / "openclip" / "img-busqueda"
OPENCLIP_INDEX_PATH = OPENCLIP_ROOT / "embeddings" / "indice_hnsw.faiss"
OPENCLIP_MANIFEST_PATH = OPENCLIP_ROOT / "dataset_procesado" / "manifest.jsonl"
OPENCLIP_THUMBS_DIR = OPENCLIP_ROOT / "dataset_procesado" / "thumbs"
OPENCLIP_IMAGE_DIR = OPENCLIP_ROOT / "dataset" / "val2017"

# EmbeddingGemma 2: se indica el modelo  y el otro guarda informaicion del modelo usado para el indice
EMBEDDING_MODEL = "google/embeddinggemma-2"
MODEL_INFO_PATH = INDEX_DIR / "model.json"

# Procesamiento por lotes
BATCH_SIZE = 8

# Parámetros del índice FAISS HNSW.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
TOP_K = 6
THUMB_SIZE = 256
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for d in (IMAGES_DIR, THUMBS_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True) #crea carpetas si no existen
