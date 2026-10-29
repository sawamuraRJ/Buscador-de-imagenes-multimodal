"""Genera embeddings de todas las imágenes, miniaturas y el índice FAISS.

Uso:
    python -m src.build_index          # construye el índice desde cero
    python -m src.build_index --add    # agrega solo imágenes nuevas (incremental)
"""
import argparse
import json

import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config
from .embeddings import Embedder


def list_images():
    files = [p for p in config.IMAGES_DIR.rglob("*") if p.suffix.lower() in config.IMAGE_EXTS]
    return sorted(p.relative_to(config.IMAGES_DIR).as_posix() for p in files)


class ImageDataset(Dataset):
    def __init__(self, rel_paths, preprocess):
        self.rel_paths = rel_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        rel = self.rel_paths[i]
        try:
            img = Image.open(config.IMAGES_DIR / rel).convert("RGB")
            thumb_path = (config.THUMBS_DIR / rel).with_suffix(".jpg")
            if not thumb_path.exists():
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                t = img.copy()
                t.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE))
                t.save(thumb_path, "JPEG", quality=85)
            return self.preprocess(img), True
        except Exception as e:  # imagen corrupta
            print(f"[aviso] se omite {rel}: {e}")
            return torch.zeros(3, 224, 224), False


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
    loader = DataLoader(
        ImageDataset(new_paths, embedder.preprocess),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    vectors, valid_paths, cursor = [], [], 0
    for batch, ok in tqdm(loader, desc="Embeddings"):
        feats = embedder.embed_image_batch(batch)
        ok = ok.numpy().astype(bool)
        vectors.append(feats[ok])
        valid_paths.extend(p for p, v in zip(new_paths[cursor:cursor + len(ok)], ok) if v)
        cursor += len(ok)

    vectors = np.concatenate(vectors).astype("float32")
    if index is None:
        index = make_index(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(config.INDEX_PATH))
    config.PATHS_PATH.write_text(json.dumps(known + valid_paths))
    print(f"Listo. Total en el índice: {index.ntotal} imágenes.")


if __name__ == "__main__":
    main()
