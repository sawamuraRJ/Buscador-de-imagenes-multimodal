"""Benchmark de escalabilidad con vectores sintéticos (para el informe).

Compara índice exacto (Flat) vs aproximados (HNSW, IVF) con 1k, 10k, 100k y 500k vectores.
Uso:  python scripts/benchmark.py
"""
import time

import faiss
import numpy as np

D = 512
N_QUERIES = 200
rng = np.random.default_rng(0)


def norm(x):
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


print(f"{'N':>8} | {'índice':<8} | {'build (s)':>9} | {'ms/consulta':>11} | {'recall@6':>8}")
print("-" * 58)
for n in (1_000, 10_000, 100_000, 500_000):
    data = norm(rng.standard_normal((n, D)).astype("float32"))
    queries = norm(rng.standard_normal((N_QUERIES, D)).astype("float32"))

    flat = faiss.IndexFlatIP(D)
    t = time.time(); flat.add(data); b_flat = time.time() - t
    t = time.time(); _, truth = flat.search(queries, 6); q_flat = (time.time() - t) / N_QUERIES * 1000
    print(f"{n:>8} | {'Flat':<8} | {b_flat:>9.2f} | {q_flat:>11.3f} | {1.0:>8.2f}")

    hnsw = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100; hnsw.hnsw.efSearch = 64
    t = time.time(); hnsw.add(data); b = time.time() - t
    t = time.time(); _, ids = hnsw.search(queries, 6); q = (time.time() - t) / N_QUERIES * 1000
    rec = np.mean([len(set(a) & set(b_)) / 6 for a, b_ in zip(ids, truth)])
    print(f"{n:>8} | {'HNSW':<8} | {b:>9.2f} | {q:>11.3f} | {rec:>8.2f}")

    nlist = max(16, int(np.sqrt(n)))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, nlist, faiss.METRIC_INNER_PRODUCT)
    t = time.time(); ivf.train(data); ivf.add(data); b = time.time() - t
    ivf.nprobe = 8
    t = time.time(); _, ids = ivf.search(queries, 6); q = (time.time() - t) / N_QUERIES * 1000
    rec = np.mean([len(set(a) & set(b_)) / 6 for a, b_ in zip(ids, truth)])
    print(f"{n:>8} | {'IVF':<8} | {b:>9.2f} | {q:>11.3f} | {rec:>8.2f}")
