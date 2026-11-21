
import json
from pathlib import Path
from PIL import Image

# Cargar manifiesto
ruta_manifest = Path("Dataset_procesado/manifest.jsonl")

with open(ruta_manifest, "r", encoding="utf-8") as f:
    registros = [json.loads(line) for line in f]

# Validar imágenes
imagenes_danadas = []

for registro in registros:
    ruta = Path(registro["file_path"])

    try:
        with Image.open(ruta) as imagen:
            imagen.verify()
    except Exception:
        imagenes_danadas.append(str(ruta))

# Mostrar resultados
print("Total de registros:", len(registros))
print("Imágenes válidas:", len(registros) - len(imagenes_danadas))
print("Imágenes dañadas:", len(imagenes_danadas))

if not imagenes_danadas:
    print("RESULTADO: Todas las imágenes están correctas.")
else:
    print("Archivos dañados:")
    for ruta in imagenes_danadas:
        print(ruta)
