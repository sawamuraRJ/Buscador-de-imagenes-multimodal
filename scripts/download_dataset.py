"""Descarga COCO val2017 (5000 imágenes) en data/images/.

Uso:  python scripts/download_dataset.py
(Pesa ~800 MB. Si prefieren otro dataset, copien sus imágenes a data/images/)
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "images"
URL = "http://images.cocodataset.org/zips/val2017.zip"
ZIP = ROOT / "data" / "val2017.zip"

DEST.mkdir(parents=True, exist_ok=True)
print("Descargando COCO val2017...")
urllib.request.urlretrieve(URL, ZIP)
print("Descomprimiendo...")
with zipfile.ZipFile(ZIP) as z:
    z.extractall(ROOT / "data")
for f in (ROOT / "data" / "val2017").glob("*.jpg"):
    shutil.move(str(f), DEST / f.name)
shutil.rmtree(ROOT / "data" / "val2017", ignore_errors=True)
ZIP.unlink(missing_ok=True)
print(f"Listo: {len(list(DEST.glob('*.jpg')))} imágenes en {DEST}")
