
import numpy as np
import json

# Cargar embeddings
embeddings = np.load("embeddings/embeddings_imagenes.npy")

# Cargar identificadores
ids = np.load("embeddings/ids_imagenes.npy")

# Cargar nombres
with open("embeddings/metadata.json", "r", encoding="utf-8") as f:
    nombres = json.load(f)

# Verificar información
print("Cantidad de embeddings:", len(embeddings))
print("Dimensiones:", embeddings.shape)
print("Cantidad de IDs:", len(ids))
print("Cantidad de nombres:", len(nombres))

# Verificar correspondencia
assert len(embeddings) == len(ids) == len(nombres)

# Verificar valores numéricos
assert np.isfinite(embeddings).all()

# Verificar normalización
normas = np.linalg.norm(embeddings, axis=1)

print("Norma mínima:", normas.min())
print("Norma máxima:", normas.max())

print("\nRESULTADO: Embeddings verificados correctamente.")
