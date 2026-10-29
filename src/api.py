"""API FastAPI + servidor del frontend.

Ejecutar:  uvicorn src.api:app --reload
Docs:      http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import config
from .embeddings import Embedder
from .search import SearchEngine

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo y el índice se cargan UNA sola vez al iniciar
    state["embedder"] = Embedder()
    state["engine"] = SearchEngine()
    yield
    state.clear()


app = FastAPI(title="Buscador de imágenes multimodal", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "imagenes_indexadas": state["engine"].index.ntotal}


@app.get("/search/text")
def search_text(q: str, k: int = config.TOP_K):
    if not q.strip():
        raise HTTPException(400, "Escribe una consulta.")
    vector = state["embedder"].embed_text(q)
    return {"query": q, "resultados": state["engine"].search(vector, k)}


@app.post("/search/image")
def search_image(file: UploadFile = File(...), k: int = config.TOP_K):
    try:
        img = Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida.")
    vector = state["embedder"].embed_image(img)
    return {"resultados": state["engine"].search(vector, k)}


# Archivos estáticos (el mount del frontend va al final)
app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/thumbs", StaticFiles(directory=config.THUMBS_DIR), name="thumbs")
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
