
"importaciones "
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import config
from .embeddings import Embedder
from .search import Buscador
from openclip.codigos.backend import OpenClipBackend

"guarda los objetos de la api"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y el índice al iniciar la API, y libera el estado al cerrarla."""
    # El modelo y el índice se cargan UNA sola vez al iniciar
    state["embedder"] = Embedder()
    state["buscador"] = Buscador()
    state["openclip"] = OpenClipBackend()
    yield
    state.clear() #vacia el estado de la aplicacion

#crea la aplicacion
app = FastAPI(title="Buscador de imágenes multimodal", lifespan=lifespan)

"""Devuelve el estado de la API y cuántas imágenes contiene el índice."""
@app.get("/health")
def health():

    return {
        "status": "ok",
        "imagenes_indexadas": state["buscador"].indice.ntotal,
        "imagenes_openclip": state["openclip"].index.ntotal,
        "indice": "HNSW",
    }

  #"""Convierte una consulta de texto en vector y devuelve imágenes similares."""
@app.get("/search/text")
def search_text(q: str, k: int = config.TOP_K, model: str = "gemma"):

    if not q.strip():
        raise HTTPException(400, "Escribe una consulta.")
    if model == "openclip":
        vector = state["openclip"].embed_text(q)
        resultados = state["openclip"].buscar(vector, k)
    elif model == "gemma":
        vector = state["embedder"].embed_text(q)
        resultados = state["buscador"].buscar(vector, k)
    else:
        raise HTTPException(400, "Modelo no válido.")
    return {"query": q, "modelo": model, "indice": "HNSW", "resultados": resultados}


@app.post("/search/image")
def search_image(file: UploadFile = File(...), k: int = config.TOP_K, model: str = "gemma"):
    """Recibe una imagen, la convierte en vector y busca imágenes similares."""
    try:
        img = Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida.")
    if model == "openclip":
        vector = state["openclip"].embed_image(img)
        resultados = state["openclip"].buscar(vector, k)
    elif model == "gemma":
        vector = state["embedder"].embed_image(img)
        resultados = state["buscador"].buscar(vector, k)
    else:
        raise HTTPException(400, "Modelo no válido.")
    return {"modelo": model, "indice": "HNSW", "resultados": resultados}


# Archivos estáticos (el mount del frontend va al final)
app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/thumbs", StaticFiles(directory=config.THUMBS_DIR), name="thumbs")
app.mount("/openclip/images", StaticFiles(directory=config.OPENCLIP_IMAGE_DIR), name="openclip-images")
config.OPENCLIP_THUMBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/openclip/thumbs", StaticFiles(directory=config.OPENCLIP_THUMBS_DIR), name="openclip-thumbs")
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
