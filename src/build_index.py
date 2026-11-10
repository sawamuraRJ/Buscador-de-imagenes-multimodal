"""Genera embeddings de todas las imágenes, miniaturas y el índice FAISS.

Uso:
    python -m src.build_index          # construye el índice desde cero
    python -m src.build_index --add    # agrega solo imágenes nuevas (incremental)
"""
import argparse
import json

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config
from .embeddings import Embedder


def list_images():
    files = [p for p in config.IMAGES_DIR.rglob("*") if p.suffix.lower() in config.IMAGE_EXTS]
    return sorted(p.relative_to(config.IMAGES_DIR).as_posix() for p in files)


def load_image(rel):
    """Carga una imagen y crea su miniatura; devuelve None si está dañada."""
    try:
        img = Image.open(config.IMAGES_DIR / rel).convert("RGB")
        thumb_path = (config.THUMBS_DIR / rel).with_suffix(".jpg")
        if not thumb_path.exists():
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb = img.copy()
            thumb.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE))
            thumb.save(thumb_path, "JPEG", quality=85)
        return img
    except Exception as e:
        print(f"[aviso] se omite {rel}: {e}")
        return None


def make_index(dim: int):
    index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true", help="agregar solo imágenes nuevas")
    args = ap.parse_args()

    all_paths = list_images()
    if not all_paths:
        raise SystemExit(f"No hay imágenes en {config.IMAGES_DIR}. Ejecuta scripts/download_dataset.py")

    if args.add and config.INDEX_PATH.exists():
        if not config.MODEL_INFO_PATH.exists():
            raise SystemExit("El índice actual no tiene modelo registrado. Reconstrúyelo sin --add.")
        model_info = json.loads(config.MODEL_INFO_PATH.read_text())
        if model_info.get("model") != config.EMBEDDING_MODEL:
            raise SystemExit("El índice se creó con otro modelo. Reconstrúyelo sin --add.")
        index = faiss.read_index(str(config.INDEX_PATH))
        known = json.loads(config.PATHS_PATH.read_text())
        new_paths = [p for p in all_paths if p not in set(known)]
        print(f"Índice existente: {len(known)} imágenes. Nuevas: {len(new_paths)}")
    else:
        index, known, new_paths = None, [], all_paths

    if not new_paths:
        print("Nada que agregar.")
        return

    embedder = Embedder()
    print(f"Dispositivo: {embedder.device} | Imágenes a procesar: {len(new_paths)}")
    vectors, valid_paths = [], []
    batch_images, batch_paths = [], []
    for rel in tqdm(new_paths, desc="Embeddings"):
        img = load_image(rel)
        if img is not None:
            batch_images.append(img)
            batch_paths.append(rel)
        if len(batch_images) == config.BATCH_SIZE:
            vectors.append(embedder.embed_image_batch(batch_images))
            valid_paths.extend(batch_paths)
            batch_images, batch_paths = [], []

    if batch_images:
        vectors.append(embedder.embed_image_batch(batch_images))
        valid_paths.extend(batch_paths)

    if not vectors:
        raise SystemExit("No se encontraron imágenes válidas para indexar.")

    vectors = np.concatenate(vectors).astype("float32")
    if index is None:
        index = make_index(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(config.INDEX_PATH))
    config.PATHS_PATH.write_text(json.dumps(known + valid_paths))
    config.MODEL_INFO_PATH.write_text(json.dumps({"model": config.EMBEDDING_MODEL}, indent=2))
    print(f"Listo. Total en el índice: {index.ntotal} imágenes.")


if __name__ == "__main__":
    main()
