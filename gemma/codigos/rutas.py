"""Guarda y consulta la ruta de cada vector mediante SQLite local."""
import sqlite3

from . import config


def conectar():
    """Abre la base local y asegura la tabla de relación ID-ruta."""
    conexion = sqlite3.connect(config.PATHS_DB_PATH, timeout=30)
    conexion.execute("PRAGMA journal_mode=WAL")
    conexion.execute(
        "CREATE TABLE IF NOT EXISTS imagenes ("
        "id INTEGER PRIMARY KEY, ruta TEXT NOT NULL UNIQUE)"
    )
    return conexion


def contar(conexion):
    """Devuelve cuántas rutas están asociadas actualmente a vectores."""
    return conexion.execute("SELECT COUNT(*) FROM imagenes").fetchone()[0]


def buscar_ruta(conexion, ruta):
    """Indica si una ruta ya está asociada a un vector."""
    return conexion.execute(
        "SELECT 1 FROM imagenes WHERE ruta = ?", (ruta,)
    ).fetchone() is not None


def guardar_lote(conexion, id_inicial, rutas):
    """Asocia en bloque las rutas nuevas con sus IDs consecutivos de FAISS."""
    conexion.executemany(
        "INSERT INTO imagenes (id, ruta) VALUES (?, ?)",
        ((id_inicial + offset, ruta) for offset, ruta in enumerate(rutas)),
    )


def obtener_rutas(ids):
    """Obtiene solo las rutas necesarias para los resultados de una búsqueda."""
    if not ids:
        return {}
    with sqlite3.connect(config.PATHS_DB_PATH, timeout=30) as conexion:
        marcas = ",".join("?" for _ in ids)
        filas = conexion.execute(
            f"SELECT id, ruta FROM imagenes WHERE id IN ({marcas})", ids
        ).fetchall()
    return {identificador: ruta for identificador, ruta in filas}
