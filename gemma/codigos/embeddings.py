"""Extracción de embeddings multimodales con EmbeddingGemma 2."""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    """Carga EmbeddingGemma 2 y ofrece métodos para vectorizar textos e imágenes."""

    def __init__(self):
        """Selecciona CPU o GPU y carga el modelo con los codificadores necesarios."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},
        )

    def embed_image_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Convierte un lote de imágenes en vectores float32 normalizados."""
        inputs = [{"image": image.convert("RGB")} for image in images]
        return self.model.encode(
            inputs,
            batch_size=len(inputs),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Convierte una sola imagen en un vector de características."""
        return self.embed_image_batch([img])

    def _embed_text_uncached(self, text: str) -> np.ndarray:
        """Convierte texto en vector sin consultar ni guardar en la caché."""
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[None, :].astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        """Devuelve el embedding del texto, reutilizando consultas guardadas."""
        return _cached_text(self, text.strip().lower()).copy()


@lru_cache(maxsize=512)
def _cached_text(embedder: Embedder, text: str) -> np.ndarray:
    """Guarda hasta 512 embeddings de texto, para acelerar consultas repetidas."""
    return embedder._embed_text_uncached(text)
