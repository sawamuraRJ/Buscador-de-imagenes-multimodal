# 🔎 Buscador de imágenes multimodal

Sistema de búsqueda de imágenes por **consulta de texto** (por ejemplo, *"a sleeping cat"*) o por **similitud con una imagen** subida desde el computador. Se basa en embeddings de CLIP y en un índice vectorial aproximado (FAISS HNSW) para que la búsqueda siga siendo rápida a medida que crece el dataset.

> Proyecto final del curso **Tópicos Avanzados en Machine Learning** (2026-II).

## Descripción

El sistema convierte cada imagen del dataset en un vector (embedding) usando **Open CLIP**. CLIP proyecta texto e imágenes al mismo espacio vectorial, por lo que un mismo índice permite dos tipos de consulta:

1. **Texto → imagen:** la consulta se convierte en un vector de texto y se buscan las imágenes más cercanas.
2. **Imagen → imagen:** la imagen subida se convierte en un vector y se buscan las más parecidas.

La similitud se calcula como producto interno sobre vectores normalizados, que equivale a la similitud coseno. La interfaz devuelve las **6 imágenes más similares**, tal como piden los mockups del enunciado.

## Características

- Búsqueda multimodal: texto a imagen e imagen a imagen.
- Dos interfaces web: `index.html` (texto) e `image.html` (imagen).
- API REST con FastAPI y documentación automática en `/docs`.
- Dataset de más de 1000 imágenes (por defecto, COCO val2017 con 5000).
- Funciona 100 % en local, con CPU o GPU.

## Tecnologías

| Componente | Herramienta |
|---|---|
| Embeddings | Open CLIP (`ViT-B-32`) |
| Índice vectorial | FAISS (`IndexHNSWFlat`) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS y JavaScript |
| Dataset | COCO val2017 |
| Lenguaje | Python 3.10+ |

## Arquitectura

```
Fase offline (una vez)                  Fase online (cada consulta)
──────────────────────                  ───────────────────────────
Imágenes ──► Open CLIP ──► vectores     Texto o imagen ──► Open CLIP ──► vector
                │                                                          │
                ▼                                                          ▼
         Índice FAISS HNSW  ◄───────────────── búsqueda top-6 ◄───────────┘
                                                      │
                                                      ▼
                                          API FastAPI ──► Interfaz web
```

## Estructura del repositorio

```
├── src/
│   ├── config.py        # rutas y parámetros
│   ├── embeddings.py    # Open CLIP: texto e imagen
│   ├── build_index.py   # embeddings por lotes, miniaturas e índice
│   ├── search.py        # búsqueda en el índice
│   └── api.py           # FastAPI + servidor del frontend
├── frontend/            # index.html, image.html, style.css, app.js
├── scripts/
│   ├── download_dataset.py   # descarga COCO val2017
│   └── benchmark.py          # comparación Flat vs HNSW vs IVF
├── data/                # imágenes, miniaturas e índice (no se suben a Git)
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/USUARIO/busqueda-imagenes.git
cd busqueda-imagenes

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Uso

```bash
# 1. Descargar el dataset (una sola vez, ~800 MB)
python scripts/download_dataset.py

# 2. Generar embeddings, miniaturas e índice (una sola vez)
python -m src.build_index

# 3. Iniciar el servidor
uvicorn src.api:app --reload
```

Luego abre en el navegador:

- Interfaz: http://127.0.0.1:8000
- Documentación de la API: http://127.0.0.1:8000/docs

La primera ejecución descarga el modelo CLIP, así que requiere conexión a internet.

### Usar un dataset propio

Copia tus imágenes (`.jpg`, `.png`, `.webp`, `.bmp`) a `data/images/` y ejecuta `python -m src.build_index`. Se admiten subcarpetas.

### Agregar imágenes sin reconstruir el índice

```bash
python -m src.build_index --add
```

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/search/text?q=...&k=6` | Búsqueda por texto |
| POST | `/search/image` | Búsqueda por imagen (campo `file`) |
| GET | `/health` | Estado del servicio y cantidad de imágenes indexadas |

Ejemplo de respuesta:

```json
{
  "query": "a sleeping cat",
  "resultados": [
    { "image": "/images/000000.jpg", "thumb": "/thumbs/000000.jpg", "score": 0.31 }
  ]
}
```

## Escalabilidad

- **Procesamiento por lotes:** los embeddings se generan con `DataLoader`, con batch size y workers configurables.
- **Índice aproximado:** HNSW ofrece búsqueda sub-lineal en lugar de comparar contra todas las imágenes.
- **Vectores normalizados en `float32`:** menos memoria y similitud coseno directa.
- **Carga única:** el modelo y el índice se cargan una sola vez al iniciar el servidor.
- **Miniaturas:** la interfaz carga versiones de 256 px y no las imágenes originales.
- **Caché de consultas:** las consultas de texto repetidas no se vuelven a procesar.
- **Indexación incremental:** `--add` incorpora imágenes nuevas sin reconstruir todo.
- **Benchmark:** `python scripts/benchmark.py` compara índices exacto (Flat) y aproximados (HNSW, IVF) con hasta 500 000 vectores, midiendo tiempo de construcción, latencia por consulta y recall.

### Camino hacia millones de imágenes

Para volúmenes mayores se puede pasar a IVF-PQ (compresión de vectores) o a una base de datos vectorial como Qdrant o Milvus, usar GPU para los embeddings y desplegar varios workers de Uvicorn detrás de un balanceador.

## Resultados

> *Completar con los resultados del benchmark y ejemplos de búsquedas (capturas de pantalla) antes de entregar.*

| Imágenes | Índice | Tiempo por consulta (ms) | Recall@6 |
|---|---|---|---|
| 1 000 | | | |
| 10 000 | | | |
| 100 000 | | | |

## Equipo

| Integrante | Rol |
|---|---|
| Nombre 1 | Datos y embeddings |
| Nombre 2 | Índice, búsqueda y API |
| Nombre 3 | Frontend, pruebas e informe |

## Referencias

- [Open CLIP](https://pypi.org/project/open-clip-torch/)
- [CLIP en Hugging Face](https://huggingface.co/docs/transformers/model_doc/clip)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- [COCO Dataset](https://cocodataset.org/)
