"""Extracción de embeddings multimodales con EmbeddingGemma 2."""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    def __init__(self):
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
        """Codifica una lista de imágenes -> vectores float32 normalizados."""
        inputs = [{"image": image.convert("RGB")} for image in images]
        return self.model.encode(
            inputs,
            batch_size=len(inputs),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def embed_image(self, img: Image.Image) -> np.ndarray:
        return self.embed_image_batch([img])

    def _embed_text_uncached(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[None, :].astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        return _cached_text(self, text.strip().lower()).copy()


@lru_cache(maxsize=512)
def _cached_text(embedder: Embedder, text: str) -> np.ndarray:
    """Caché de consultas repetidas."""
    return embedder._embed_text_uncached(text)
