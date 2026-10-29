"""Extracción de embeddings con Open CLIP (texto e imagen en el mismo espacio)."""
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

from . import config


class Embedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.MODEL_NAME, pretrained=config.PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    @torch.no_grad()
    def embed_image_batch(self, batch: torch.Tensor) -> np.ndarray:
        """batch: tensor (N,3,H,W) ya preprocesado -> (N,D) float32 normalizado."""
        feats = self.model.encode_image(batch.to(self.device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def embed_image(self, img: Image.Image) -> np.ndarray:
        tensor = self.preprocess(img.convert("RGB")).unsqueeze(0)
        return self.embed_image_batch(tensor)

    @torch.no_grad()
    def _embed_text_uncached(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        return _cached_text(self, text.strip().lower()).copy()


@lru_cache(maxsize=512)
def _cached_text(embedder: Embedder, text: str) -> np.ndarray:
    """Caché de consultas repetidas."""
    return embedder._embed_text_uncached(text)
