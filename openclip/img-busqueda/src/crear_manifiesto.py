
import os
import json
from pathlib import Path

# 1. Definir las rutas del proyecto
RUTA_DATASET = Path("dataset/val2017")
RUTA_SALIDA = Path("Dataset_procesado/manifest.jsonl")

# 2. Buscar las imágenes
extensiones = {".jpg", ".jpeg", ".png"}

imagenes = sorted([
    archivo for archivo in RUTA_DATASET.iterdir()
    if archivo.suffix.lower() in extensiones
])

# 3. Crear el manifiesto
RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

with open(RUTA_SALIDA, "w", encoding="utf-8") as archivo_salida:

    for imagen in imagenes:

        registro = {
            "image_id": int(imagen.stem),
            "file_name": imagen.name,
            "file_path": str(imagen)
        }

        archivo_salida.write(
            json.dumps(registro) + "\n"
        )

# 4. Mostrar resultados
print("Total de imágenes:", len(imagenes))
print("Manifiesto creado en:", RUTA_SALIDA)
