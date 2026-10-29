"""Búsqueda por similitud sobre el índice FAISS."""
import json

import faiss
import numpy as np

from . import config


class SearchEngine:
    def __init__(self):
        if not config.INDEX_PATH.exists():
            raise FileNotFoundError("No existe el índice. Ejecuta: python -m src.build_index")
        self.index = faiss.read_index(str(config.INDEX_PATH))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
        self.paths = json.loads(config.PATHS_PATH.read_text())

    def search(self, vector: np.ndarray, k: int = config.TOP_K):
        k = max(1, min(k, self.index.ntotal))
        scores, ids = self.index.search(vector, k)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            rel = self.paths[i]
            thumb = (config.THUMBS_DIR / rel).with_suffix(".jpg").relative_to(config.THUMBS_DIR).as_posix()
            results.append({
                "image": f"/images/{rel}",
                "thumb": f"/thumbs/{thumb}",
                "score": round(float(score), 4),
            })
        return results
