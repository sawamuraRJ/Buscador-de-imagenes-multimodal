"""Búsqueda por similitud sobre el índice FAISS."""
import json

import numpy as np

from . import config
from . import hnsw, rutas


class Buscador:
    """Busca imágenes similares usando el índice vectorial FAISS."""

    def __init__(self):
        """Comprueba el modelo y carga el índice FAISS junto con las rutas."""
        if not config.HNSW_INDEX_PATH.exists():
            raise FileNotFoundError("No existe el índice HNSW. ")
        if not config.MODEL_INFO_PATH.exists():
            raise FileNotFoundError("El índice no registra su modelo.")
        model_info = json.loads(config.MODEL_INFO_PATH.read_text())
        if model_info.get("model") != config.EMBEDDING_MODEL:
            raise ValueError("El índice pertenece a otro modelo.")
        if model_info.get("index") != "HNSW":
            raise ValueError("El índice no está registrado como HNSW. Reconstrúyelo ")
        self.indice = hnsw.cargar()
        conexion = rutas.conectar()
        try:
            total_rutas = rutas.contar(conexion)
        finally:
            conexion.close()
        if total_rutas != self.indice.ntotal:
            raise ValueError("La base de rutas no coincide con HNSW. Ejecuta la migración del índice.")

    def buscar(self, vector_consulta: np.ndarray, cantidad: int = config.TOP_K):
        """Encuentra los vectores más cercanos y arma los datos para la interfaz."""
        cantidad = max(1, min(cantidad, self.indice.ntotal))
        similitudes, posiciones = self.indice.search(vector_consulta, cantidad)
        ids = [int(posicion) for posicion in posiciones[0] if posicion >= 0]
        rutas_encontradas = rutas.obtener_rutas(ids)
        resultados = []
        for similitud, posicion in zip(similitudes[0], posiciones[0]):
            if posicion < 0:
                continue
            ruta_relativa = rutas_encontradas[int(posicion)]
            ruta_miniatura = (
                (config.THUMBS_DIR / ruta_relativa)
                .with_suffix(".jpg")
                .relative_to(config.THUMBS_DIR)
                .as_posix()
            )
            resultados.append({
                # Se conservan estas claves porque son el contrato que usa el frontend.
                "image": f"/images/{ruta_relativa}",
                "thumb": f"/thumbs/{ruta_miniatura}",
                "score": round(float(similitud), 4),
            })
        return resultados
