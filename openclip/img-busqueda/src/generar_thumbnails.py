"""Genera miniaturas JPEG para el dataset OpenCLIP usado por la página web."""
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "val2017"
MANIFEST = ROOT / "dataset_procesado" / "manifest.jsonl"
THUMBS = ROOT / "dataset_procesado" / "thumbs"
SIZE = (256, 256)


def main():
    THUMBS.mkdir(parents=True, exist_ok=True)
    creadas = 0
    with MANIFEST.open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            name = record.get("file_name") or Path(record["file_path"]).name
            source = DATASET / name
            target = THUMBS / f"{Path(name).stem}.jpg"
            if not source.is_file() or target.is_file():
                continue
            try:
                with Image.open(source) as image:
                    thumb = ImageOps.fit(image.convert("RGB"), SIZE)
                    thumb.save(target, "JPEG", quality=82, optimize=True)
                creadas += 1
            except (OSError, ValueError) as error:
                print(f"No se pudo procesar {source}: {error}")
    print(f"Miniaturas nuevas: {creadas}. Carpeta: {THUMBS}")


if __name__ == "__main__":
    main()
