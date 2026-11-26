# Buscador multimodal de imágenes

La aplicación web permite elegir entre **Gemma** y **OpenCLIP** y buscar tanto por texto como por imagen. Los modelos, índices y datasets se mantienen independientes:

- Gemma: `gemma/codigos`, `gemma/scripts` y `gemma/data` (`images`, `thumbs` e `index`).
- OpenCLIP: `openclip/img-busqueda/dataset`, `embeddings` y `dataset_procesado`.

OpenCLIP usa un índice HNSW y el manifiesto de `openclip/img-busqueda`. Las miniaturas para la página se crean al mostrar resultados; para prepararlas en lote:

```powershell
python openclip/img-busqueda/src/generar_thumbnails.py
```

Instala las dependencias del proyecto y ejecuta la misma API y página para ambos modelos:

```powershell
pip install -r requirements.txt
uvicorn gemma.codigos.api:app --reload
```

Abre `http://127.0.0.1:8000`. En el primer inicio, OpenCLIP descarga sus pesos `ViT-B-32` si todavía no están en la caché local.

## Despliegue en Render

El archivo `render.yaml` configura el servicio web. Al desplegar desde el Blueprint de Render, el comando de build instala las dependencias y ejecuta `gemma.scripts.prepare_render`: descarga Caltech-256, crea el índice Gemma, descarga las imágenes COCO val2017 indicadas por el manifiesto y genera el índice OpenCLIP. Los datos e índices se crean en Render y no se suben a GitHub.

En Render, selecciona **New → Blueprint** y conecta este repositorio. El plan configurado es `2c-4g` porque la API carga ambos modelos en memoria; el servicio gratuito de 512 MB no es suficiente. El build también descarga varios conjuntos de datos y puede tardar bastante. Se pueden reducir las muestras con las variables `GEMMA_IMAGE_COUNT` y `OPENCLIP_IMAGE_COUNT`, pero los valores configurados por defecto mantienen 5000 imágenes por modelo.
