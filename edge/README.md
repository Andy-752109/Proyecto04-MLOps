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
verifica su SHA-256 antes de arrancar; si no coincide, aborta.

**Modelo real (por defecto):** el ONNX FP32 de #4 (`1.0.0-fp32`). Su SHA está en
`reports/p4/conversion/parity_fp32.json` y ya va en `config.example.yaml`.

```bash
cp edge/config.example.yaml edge/config.yaml
# copia model_fp32.onnx (te lo pasa Karen) a edge/models/
python -m edge status          # debe decir "sha256      : OK"
```

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
| `camera.index` | `1` | Cámara USB de la laptop; la webcam integrada suele ser `0` |
| `camera.api` | `auto` | DirectShow (`cv2.CAP_DSHOW`) en Windows |
| `camera.width` / `height` | `null` | Resolución nativa de la cámara |
| `camera.warmup_frames` | `10` | Se descartan al abrir y antes de cada captura (autoexposición) |
| `crop` | `{center_square: 0.8}` | Cuadrado centrado, 80% del lado menor. También `null` o `{x, y, width, height}` en píxeles |
| `capture.mode` | `manual` | `manual`: Enter captura, `q` sale. `interval`: cada `interval_seconds` (10), Ctrl+C sale |

Si la cámara no abre, prueba otro `camera.index`.

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
- **Salida del modelo vs baseline de P4-01:**

  ```bash
  python -m edge.tools.parity_model --model edge/models/model.onnx --json reports/p4/parity/edge.json
  ```

## Tests

```bash
python -m unittest discover -s edge/tests -v
```

No usan cámara, red ni modelos reales: el modelo ONNX diminuto se construye en el test.
