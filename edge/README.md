# Edge: captura e inferencia local (P4-06)

Captura una foto con la cámara, la recorta, la preprocesa igual que P3, la clasifica con el
modelo local y guarda el resultado como evento del
[contrato v1](../docs/p4/contrato-evento-edge.md). La ruta de captura no usa la red.

## Instalación

Desde la raíz del repositorio, con **Python 3.12 de 64 bits**, el de la laptop edge. `onnxruntime` va fijo
en 1.30.0, la versión con la que #4 verificó la paridad, y no tiene wheel para Python 3.10:

```bash
python -m pip install -r edge/requirements.txt       # app
python -m pip install -r edge/requirements-dev.txt   # app + tests
```

## Modelo

La app solo carga el modelo desde la caché local (`edge/models/`, ignorada por git) y
verifica su SHA-256 antes de arrancar: debe coincidir con `model.sha256` de la config y con
`model_sha256` de `models/edge_registry.json` (`model.registry`). Si algo no coincide, falta
o no se puede leer, aborta con `ModelVerificationError`.

**Modelo real (por defecto):** la variante INT8 estática de #4 (`1.0.0-int8`, PR #22,
SHA `dffa9cf2…0392`), registrada en `models/edge_registry.json` con su `VersionId`. Su
calidad (accuracy en val, 0 diferencias de clase en las 20 referencias de P4-01) está en
`reports/p4/conversion/conversion_log.json`.

```bash
cp edge/config.example.yaml edge/config.yaml
aws s3api get-object --bucket mlops-p3-models-222629887955   --key models/edge/1.0.0-int8/model_int8.onnx   --version-id 8xfGLRI8Mlq07SyM9hsODKBfuC1c24ig edge/models/model_int8.onnx
python -m edge status          # debe decir "sha256      : OK"
```

El ONNX FP32 (`1.0.0-fp32`, `reports/p4/conversion/parity_fp32.json`) sigue sirviendo para
comparar: cambia `path`, `version` y `sha256` en `edge/config.yaml` y pon `registry: null`,
porque el registro edge solo tiene la variante INT8.

**Modelo genérico (respaldo):** ResNet18 con arquitectura de P3 y pesos aleatorios. Sirve
para probar runtime y latencia sin el archivo real, no para clasificar. Este comando genera
`edge/models/generic-resnet18.onnx` y deja `edge/config.yaml` apuntando a él con su SHA:

```bash
python -m edge.tools.generic_resnet18 --setup
```

## Configuración

`edge/config.yaml` está ignorado por git y no lleva secretos. Valores por defecto
(aprobados por el PM en #6):

| Clave | Valor | Nota |
|---|---|---|
| `device_id` | `edge-laptop-01` | |
| `camera.name` | `GENERAL WEBCAM` | Cámara USB de la laptop por nombre DirectShow; si está definido, `index` se ignora. `python -m edge cameras` lista los nombres |
| `camera.index` | `1` | Solo si `camera.name: null`; el orden de los índices cambia entre arranques |
| `camera.api` | `auto` | DirectShow (`cv2.CAP_DSHOW`) en Windows |
| `camera.width` / `height` | `null` | Resolución nativa de la cámara |
| `camera.warmup_frames` | `10` | Se descartan al abrir y antes de cada captura (autoexposición) |
| `crop` | `{center_square: 0.8}` | Cuadrado centrado, 80% del lado menor. También `null` o `{x, y, width, height}` en píxeles |
| `model.registry` | `../models/edge_registry.json` | El SHA también debe coincidir con `model_sha256` del registro (#6) |
| `capture.mode` | `manual` | `manual`: Enter captura, `q` sale. `interval`: cada `interval_seconds` (10), Ctrl+C sale |

Si la cámara no abre, revisa `python -m edge cameras` y ajusta `camera.name` (o `camera.index`).

## Uso

```bash
python -m edge run                          # modo de la config
python -m edge run --mode interval --count 5
python -m edge status                       # config, SHA del modelo y último evento
```

Por captura se escriben en `edge/data/` (ignorada por git):

| Archivo | Contenido |
|---|---|
| `images/{capture_id}.jpg` | Frame completo (`image_key` del evento) |
| `crops/{capture_id}.jpg` | Región usada para inferir, si hay recorte |
| `captures.jsonl` | Una línea por captura: `{"event", "image_path", "crop_path"}`. Se escribe al final, así que una línea indica una captura completa |
| `edge.log` | Arranque (verificación del SHA), capturas y errores |

Al reiniciar, la app sigue escribiendo en el mismo log y no repite `capture_id`.

## Módulos

| Módulo | Qué hace |
|---|---|
| `config.py` | Lee y valida `config.yaml` |
| `camera.py` | Cámara con OpenCV (BGR) |
| `preprocess.py` | BGR→RGB, recorte, resize 128×128 bilineal con antialias (Pillow, como torchvision), /255, normalización ImageNet, CHW |
| `backend.py` | `InferenceBackend` (`load`, `predict(tensor) -> probs`) y la implementación de ONNX Runtime. Otro runtime = otra clase en `BACKENDS` |
| `pipeline.py` | Una captura: inferencia, evento validado y escritura en disco |
| `event_validator.py` | Valida el evento contra el contrato v1 |

## Paridad

- **Preprocesamiento vs torchvision:** `edge/tests/test_preprocess.py` compara contra el
  pipeline de P3 con tolerancia 1e-5, primero con imágenes aleatorias y después con las 20
  referencias de `reports/p4/reference/parity_reference.csv`. Se salta si no hay
  torch/torchvision o si faltan los recortes en `data/derived/crops/`.
- **Salida del modelo vs baseline de P4-01:** la paridad de clases del INT8 ya la demuestra
  #4 (PR #22) en `reports/p4/conversion/conversion_log.json`
  (`quality.parity_reference_class_mismatches: []`). Para repetirla con el preprocesamiento
  y el backend de la app (INT8: 0 diferencias de clase, máx |Δp| = 0.070; FP32: 0
  diferencias, máx |Δp| = 1.04e-6):

  ```bash
  python -m edge.tools.parity_model --model edge/models/model_int8.onnx --json reports/p4/parity/edge.json
  ```

## Tests

```bash
python -m unittest discover -s edge/tests -v
```

No usan cámara, red ni modelos reales: el modelo ONNX diminuto se construye en el test.

## Uploader a S3 (P4-05)

`edge/uploader.py` sube imagen y luego evento con `If-None-Match: *` y devuelve
`sent`, `already_sent` o `failed`. Detalles, prueba de doble envío y comandos de
lectura en [`docs/p4/aws-capturas.md`](../docs/p4/aws-capturas.md).
