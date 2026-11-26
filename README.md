# Buscador multimodal de imágenes

La aplicación web permite elegir entre **Gemma** y **OpenCLIP** y buscar tanto por texto como por imagen. Los modelos, índices y datasets se mantienen independientes:

- Gemma: `gemma/codigos`, `gemma/scripts` y `gemma/data` (`images`, `thumbs` e `index`).
- OpenCLIP: `openclip/img-busqueda/dataset`, `embeddings` y `dataset_procesado`.

OpenCLIP usa el índice HNSW y el manifiesto ya preparados en `openclip/img-busqueda`. Las miniaturas para la página se crean al mostrar resultados; para prepararlas en lote:

```powershell
python openclip/img-busqueda/src/generar_thumbnails.py
```

Instala las dependencias del proyecto y ejecuta la misma API y página para ambos modelos:

```powershell
pip install -r requirements.txt
uvicorn gemma.codigos.api:app --reload
```

Abre `http://127.0.0.1:8000`. En el primer inicio, OpenCLIP descarga sus pesos `ViT-B-32` si todavía no están en la caché local.
