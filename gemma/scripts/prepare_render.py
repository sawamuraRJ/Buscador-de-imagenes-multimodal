"""Descarga los datos e índices que necesita la aplicación en Render.

Se ejecuta durante el build, no cuando llega cada consulta. Los datos grandes
no se guardan en GitHub: Render los incorpora al entorno desplegado.
"""
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from gemma.codigos import config


COCO_VAL_URL = "https://images.cocodataset.org/zips/val2017.zip"
OPENCLIP_ROOT = config.OPENCLIP_ROOT
OPENCLIP_IMAGES = config.OPENCLIP_IMAGE_DIR
OPENCLIP_MANIFEST = config.OPENCLIP_MANIFEST_PATH
OPENCLIP_EMBEDDINGS = OPENCLIP_ROOT / "embeddings"
OPENCLIP_INDEX = config.OPENCLIP_INDEX_PATH


def _prepare_gemma() -> None:
    """Descarga una muestra de Caltech-256 y construye el índice Gemma."""
    index_path = config.HNSW_INDEX_PATH
    model_info_path = config.MODEL_INFO_PATH
    if index_path.is_file() and model_info_path.is_file():
        print("Índice Gemma ya preparado; se omite su reconstrucción.")
        return

    count = int(os.environ.get("GEMMA_IMAGE_COUNT", "5000"))
    if count < 1:
        raise ValueError("GEMMA_IMAGE_COUNT debe ser un entero positivo")

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "gemma.scripts.download_caltech256", "--count", str(count)],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "gemma.codigos.build_index"], check=True)


def _read_openclip_records(limit: int) -> list[dict]:
    if not OPENCLIP_MANIFEST.is_file():
        raise FileNotFoundError(f"Falta el manifiesto OpenCLIP: {OPENCLIP_MANIFEST}")
    with OPENCLIP_MANIFEST.open(encoding="utf-8") as manifest:
        records = [json.loads(line) for line in manifest if line.strip()]
    if limit < 1:
        raise ValueError("OPENCLIP_IMAGE_COUNT debe ser un entero positivo")
    return records[:limit]


def _download_missing_coco(records: list[dict]) -> None:
    OPENCLIP_IMAGES.mkdir(parents=True, exist_ok=True)
    missing = [
        record for record in records
        if not (OPENCLIP_IMAGES / (record.get("file_name") or Path(record["file_path"]).name)).is_file()
    ]
    if not missing:
        print("Imágenes OpenCLIP ya disponibles.")
        return

    expected = {record.get("file_name") or Path(record["file_path"]).name for record in missing}
    print(f"Descargando el archivo COCO val2017 para extraer {len(expected)} imágenes...")
    with tempfile.TemporaryDirectory(prefix="coco-val2017-") as temp_dir:
        archive = Path(temp_dir) / "val2017.zip"
        urllib.request.urlretrieve(COCO_VAL_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            members = {
                Path(info.filename).name: info
                for info in zipped.infolist()
                if not info.is_dir() and Path(info.filename).name in expected
            }
            absent = expected - members.keys()
            if absent:
                raise RuntimeError(f"El archivo COCO no contiene {len(absent)} imágenes del manifiesto")
            for name, info in tqdm(members.items(), total=len(members), desc="Extrayendo COCO"):
                target = OPENCLIP_IMAGES / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)


def _build_openclip(records: list[dict]) -> None:
    if OPENCLIP_INDEX.is_file():
        try:
            existing = faiss.read_index(str(OPENCLIP_INDEX))
            if existing.ntotal:
                print(f"Índice OpenCLIP ya preparado: {existing.ntotal} vectores.")
                return
        except RuntimeError:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    batch_size = int(os.environ.get("OPENCLIP_BATCH_SIZE", "16"))
    if batch_size < 1:
        raise ValueError("OPENCLIP_BATCH_SIZE debe ser un entero positivo")

    vectors: list[np.ndarray] = []
    valid_ids: list[int] = []
    valid_names: list[str] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Embeddings OpenCLIP"):
        batch_records = records[start : start + batch_size]
        tensors, usable = [], []
        for record in batch_records:
            name = record.get("file_name") or Path(record["file_path"]).name
            try:
                with Image.open(OPENCLIP_IMAGES / name) as image:
                    tensors.append(preprocess(image.convert("RGB")))
                usable.append(record)
            except (OSError, ValueError) as error:
                print(f"[aviso] se omite {name}: {error}")
        if not tensors:
            continue

        with torch.inference_mode():
            batch = torch.stack(tensors).to(device)
            batch_vectors = model.encode_image(batch)
            batch_vectors = batch_vectors / batch_vectors.norm(dim=-1, keepdim=True)
        vectors.append(np.ascontiguousarray(batch_vectors.cpu().numpy(), dtype=np.float32))
        valid_ids.extend(int(record["image_id"]) for record in usable)
        valid_names.extend(record.get("file_name") or Path(record["file_path"]).name for record in usable)

    if not vectors:
        raise RuntimeError("No se pudieron generar embeddings OpenCLIP")

    matrix = np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)
    base = faiss.IndexHNSWFlat(matrix.shape[1], config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    base.hnsw.efSearch = config.HNSW_EF_SEARCH
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(matrix, np.asarray(valid_ids, dtype=np.int64))

    OPENCLIP_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OPENCLIP_INDEX))
    np.save(OPENCLIP_EMBEDDINGS / "embeddings_imagenes.npy", matrix)
    np.save(OPENCLIP_EMBEDDINGS / "ids_imagenes.npy", np.asarray(valid_ids, dtype=np.int64))
    (OPENCLIP_EMBEDDINGS / "metadata.json").write_text(
        json.dumps(valid_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Índice OpenCLIP listo: {index.ntotal} vectores.")


def main() -> None:
    """Prepara ambos índices antes de que Uvicorn inicie la API."""
    _prepare_gemma()
    limit = int(os.environ.get("OPENCLIP_IMAGE_COUNT", "5000"))
    records = _read_openclip_records(limit)
    _download_missing_coco(records)
    _build_openclip(records)


if __name__ == "__main__":
    main()
