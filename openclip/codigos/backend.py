"""Modelo y búsqueda OpenCLIP usando los artefactos de su dataset independiente."""
import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps

from gemma.codigos import config


class OpenClipBackend:
    def __init__(self):
        if not config.OPENCLIP_INDEX_PATH.is_file():
            raise FileNotFoundError(f"No existe el índice OpenCLIP: {config.OPENCLIP_INDEX_PATH}")
        if not config.OPENCLIP_MANIFEST_PATH.is_file():
            raise FileNotFoundError(f"No existe el manifiesto OpenCLIP: {config.OPENCLIP_MANIFEST_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.index = faiss.read_index(str(config.OPENCLIP_INDEX_PATH))
        faiss.downcast_index(self.index.index).hnsw.efSearch = config.HNSW_EF_SEARCH

        self.paths = {}
        with config.OPENCLIP_MANIFEST_PATH.open(encoding="utf-8") as manifest:
            for line in manifest:
                record = json.loads(line)
                image_id = int(record["image_id"])
                file_name = record.get("file_name") or Path(record["file_path"]).name
                image_path = config.OPENCLIP_IMAGE_DIR / file_name
                if image_path.is_file():
                    self.paths[image_id] = (file_name, image_path)

    @torch.inference_mode()
    def embed_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        vector = self.model.encode_text(tokens)
        vector = vector / vector.norm(dim=-1, keepdim=True)
        return np.ascontiguousarray(vector.cpu().numpy(), dtype=np.float32)

    @torch.inference_mode()
    def embed_image(self, image):
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        vector = self.model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)
        return np.ascontiguousarray(vector.cpu().numpy(), dtype=np.float32)

    def buscar(self, vector, cantidad):
        scores, ids = self.index.search(vector, min(max(1, cantidad), self.index.ntotal))
        results = []
        for score, image_id in zip(scores[0], ids[0]):
            if image_id < 0 or int(image_id) not in self.paths:
                continue
            file_name, image_path = self.paths[int(image_id)]
            thumb_name = f"{Path(file_name).stem}.jpg"
            thumb_path = config.OPENCLIP_THUMBS_DIR / thumb_name
            if not thumb_path.is_file():
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(image_path) as source:
                    thumb = ImageOps.fit(source.convert("RGB"), (config.THUMB_SIZE, config.THUMB_SIZE))
                    thumb.save(thumb_path, "JPEG", quality=82, optimize=True)
            results.append({
                "image": f"/openclip/images/{file_name}",
                "thumb": f"/openclip/thumbs/{thumb_name}",
                "score": round(float(score), 4),
            })
        return results
