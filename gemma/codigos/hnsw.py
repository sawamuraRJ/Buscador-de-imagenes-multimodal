
import faiss

from . import config


def crear(dimension: int):
    """Crea el grafo HNSW y configura calidad de construcción y búsqueda."""
    indice = faiss.IndexHNSWFlat(
        dimension, config.HNSW_M, faiss.METRIC_INNER_PRODUCT
    )
    indice.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    indice.hnsw.efSearch = config.HNSW_EF_SEARCH
    return indice


def cargar():
    """Carga HNSW desde disco y aplica el parámetro de búsqueda."""
    indice = faiss.read_index(str(config.HNSW_INDEX_PATH))
    indice.hnsw.efSearch = config.HNSW_EF_SEARCH
    return indice
