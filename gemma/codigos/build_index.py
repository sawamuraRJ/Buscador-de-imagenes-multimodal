"""Genera embeddings por lotes, miniaturas e índice FAISS HNSW con rutas SQLite.

Uso:
    python -m gemma.codigos.build_index          # construye el índice desde cero
    python -m gemma.codigos.build_index --add    # agrega solo imágenes nuevas (incremental)
"""
import argparse
import json
import os
from pathlib import Path

import faiss
from PIL import Image
from tqdm import tqdm

from . import config
from .embeddings import Embedder
from . import hnsw, rutas


def iter_images():
    """Recorre imágenes en orden estable sin guardar todas sus rutas en una lista."""
    for carpeta, subcarpetas, archivos in os.walk(config.IMAGES_DIR):
        subcarpetas.sort()
        for nombre in sorted(archivos):
            ruta = os.path.join(carpeta, nombre)
            if Path(ruta).suffix.lower() in config.IMAGE_EXTS:
                yield Path(ruta).relative_to(config.IMAGES_DIR).as_posix()


def load_image(rel):
    """Abre una imagen RGB y crea su miniatura; devuelve None si está dañada."""
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


def main():
    """Genera embeddings e inserta cada lote en HNSW; --add procesa nuevas imágenes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true", help="agregar solo imágenes nuevas")
    args = ap.parse_args()

    conexion_rutas = rutas.conectar()
    try:
        cantidad_rutas = rutas.contar(conexion_rutas)

        if args.add and config.HNSW_INDEX_PATH.exists():
            if not config.MODEL_INFO_PATH.exists():
                raise SystemExit("El índice actual no tiene modelo registrado. Reconstrúyelo sin --add.")
            model_info = json.loads(config.MODEL_INFO_PATH.read_text())
            if model_info.get("model") != config.EMBEDDING_MODEL:
                raise SystemExit("El índice se creó con otro modelo. Reconstrúyelo sin --add.")
            if model_info.get("index") != "HNSW":
                raise SystemExit("El índice existente no es HNSW. Reconstrúyelo sin --add.")
            index_hnsw = hnsw.cargar()
            if cantidad_rutas != index_hnsw.ntotal:
                raise SystemExit("La base de rutas no coincide con HNSW. Revisa la migración antes de agregar imágenes.")
            print(f"Índice existente: {cantidad_rutas} imágenes. Buscando imágenes nuevas.")
        else:
            index_hnsw = None
            conexion_rutas.execute("DELETE FROM imagenes")

        embedder = Embedder()
        print(f"Dispositivo: {embedder.device} | Procesando imágenes por lotes de hasta {config.BATCH_SIZE}")
        lote_imagenes, lote_rutas = [], []
        total_agregadas = 0

        def agregar_lote():
            """Calcula e inserta un lote, liberando sus imágenes y vectores al terminar."""
            nonlocal index_hnsw, total_agregadas, lote_imagenes, lote_rutas
            if not lote_imagenes:
                return
            vectores = embedder.embed_image_batch(lote_imagenes)
            if index_hnsw is None:
                index_hnsw = hnsw.crear(vectores.shape[1])
            id_inicial = index_hnsw.ntotal
            index_hnsw.add(vectores)
            rutas.guardar_lote(conexion_rutas, id_inicial, lote_rutas)
            total_agregadas += len(lote_rutas)
            lote_imagenes, lote_rutas = [], []

        for ruta_relativa in tqdm(iter_images(), desc="Imágenes", unit="img"):
            if args.add and rutas.buscar_ruta(conexion_rutas, ruta_relativa):
                continue
            imagen = load_image(ruta_relativa)
            if imagen is None:
                continue
            lote_imagenes.append(imagen)
            lote_rutas.append(ruta_relativa)
            if len(lote_imagenes) >= config.BATCH_SIZE:
                agregar_lote()

        agregar_lote()
        if total_agregadas == 0:
            if index_hnsw is None:
                raise SystemExit(f"No hay imágenes válidas en {config.IMAGES_DIR}.")
            conexion_rutas.commit()
            print("No hay imágenes nuevas para agregar.")
            return

        faiss.write_index(index_hnsw, str(config.HNSW_INDEX_PATH))
        conexion_rutas.commit()
        config.MODEL_INFO_PATH.write_text(json.dumps({
            "model": config.EMBEDDING_MODEL,
            "index": "HNSW",
            "hnsw_m": config.HNSW_M,
            "hnsw_ef_construction": config.HNSW_EF_CONSTRUCTION,
        }, indent=2))
        print(f"Listo. Se agregaron {total_agregadas}; total en HNSW: {index_hnsw.ntotal} imágenes.")
    finally:
        conexion_rutas.close()


if __name__ == "__main__":
    main()
