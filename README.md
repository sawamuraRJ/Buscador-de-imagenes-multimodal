# Buscador de imágenes multimodal (texto e imagen)

Proyecto final de Tópicos Avanzados en Machine Learning (2026-II).
Busca imágenes por consulta de texto ("a sleeping cat") o por similitud con una imagen subida.

**Stack:** Open CLIP (embeddings) · FAISS HNSW (índice) · FastAPI (API) · HTML/JS (interfaz)

## Estructura

```
├── src/
│   ├── config.py        # rutas y parámetros
│   ├── embeddings.py    # Open CLIP: texto e imagen
│   ├── build_index.py   # embeddings por lotes + miniaturas + índice FAISS
│   ├── search.py        # búsqueda en el índice
│   └── api.py           # FastAPI + servidor del frontend
├── frontend/            # index.html (texto), image.html (imagen)
├── scripts/
│   ├── download_dataset.py   # descarga COCO val2017 (5000 imágenes)
│   └── benchmark.py          # Flat vs HNSW vs IVF (para el informe)
├── data/                # NO se sube a Git (images, thumbs, index)
└── requirements.txt
```

## Instalación

```bash
git clone <URL_DEL_REPO>
cd busqueda-imagenes
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Uso

```bash
python scripts/download_dataset.py     # 1. descargar imágenes (una vez)
python -m src.build_index              # 2. generar embeddings e índice (una vez)
uvicorn src.api:app --reload           # 3. levantar el servidor
```

Abrir http://127.0.0.1:8000 (interfaz) o http://127.0.0.1:8000/docs (API).

Para agregar imágenes nuevas sin reconstruir todo: copiarlas a `data/images/` y ejecutar
`python -m src.build_index --add`.

## Decisiones de escalabilidad

- Embeddings por lotes con `DataLoader` (batch y workers configurables).
- Vectores normalizados en float32; similitud coseno = producto interno.
- Índice HNSW de FAISS (búsqueda aproximada sub-lineal) en lugar de fuerza bruta.
- Modelo e índice se cargan una sola vez al iniciar el servidor.
- Miniaturas de 256 px para la grilla de resultados.
- Caché LRU de embeddings de consultas de texto repetidas.
- Indexación incremental (`--add`).
- `scripts/benchmark.py` compara Flat, HNSW e IVF con hasta 500 000 vectores.
- Para millones de imágenes: IVF-PQ o base vectorial (Qdrant/Milvus), GPU y varios workers de Uvicorn.

## Equipo

| Integrante | Rol |
|---|---|
| Persona 1 | Datos y embeddings |
| Persona 2 | Índice, búsqueda y API |
| Persona 3 | Frontend, pruebas e informe |
