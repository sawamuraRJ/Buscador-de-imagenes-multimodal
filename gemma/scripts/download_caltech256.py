"""Descarga Caltech-256 y exporta una muestra de imágenes a gemma/data/images."""
import argparse
import random

from torchvision.datasets import Caltech256

from gemma.codigos import config


def main():
    """Descarga la fuente y exporta una muestra reproducible sin duplicarla."""
    parser = argparse.ArgumentParser(
        description="Descarga imágenes variadas de Caltech-256 para el buscador."
    )
    parser.add_argument(
        "--count", type=int, default=5000,
        help="cantidad de imágenes a exportar (por defecto: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="semilla para que la selección aleatoria sea reproducible",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count debe ser mayor que cero")

    dataset_root = config.ROOT / "data" / "datasets"
    destino = config.IMAGES_DIR / "caltech256"
    destino.mkdir(parents=True, exist_ok=True)

    print("Descargando o abriendo Caltech-256; el archivo completo ocupa cerca de 1.2 GB.")
    dataset = Caltech256(root=str(dataset_root), download=True)
    if args.count > len(dataset):
        parser.error(f"Caltech-256 contiene solo {len(dataset)} imágenes")

    indices = random.Random(args.seed).sample(range(len(dataset)), args.count)
    indices.sort()
    guardadas = omitidas = 0
    for numero, indice in enumerate(indices, start=1):
        archivo = destino / f"caltech_{indice:06d}.jpg"
        if archivo.exists():
            omitidas += 1
            continue
        imagen, _etiqueta = dataset[indice]
        imagen.convert("RGB").save(archivo, format="JPEG", quality=95)
        guardadas += 1
        if numero % 250 == 0 or numero == len(indices):
            print(f"Progreso: {numero}/{len(indices)} seleccionadas")

    print(f"Listo. Nuevas: {guardadas}; ya existentes: {omitidas}; carpeta: {destino}")
    print("Para indexarlas ejecuta: python -m gemma.codigos.build_index --add")


if __name__ == "__main__":
    main()
