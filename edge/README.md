# Edge: captura e inferencia local (P4-06)

Captura una foto con la cámara, la recorta, la preprocesa igual que P3, la clasifica con el
modelo local y guarda el resultado como evento del
[contrato v1](../docs/p4/contrato-evento-edge.md). La ruta de captura no usa la red.

## Instalación

Desde la raíz del repositorio, con **Python 3.12 de 64 bits**, crea un venv propio. `onnxruntime` va fijo
en 1.30.0, la versión con la que #4 verificó la paridad, y requiere Python >=3.11. Para el recorrido completo del evaluador **sin cámara ni AWS**, sigue el [inicio rápido del README principal](../README.md#inicio-rápido-para-evaluadores-sin-cámara-y-sin-aws):

```powershell
py -3.12 -m venv .venv-edge
.\.venv-edge\Scripts\Activate.ps1
python -m pip install -r edge/requirements.txt
python -c "import jsonschema, onnxruntime, cv2, boto3, yaml; print('deps OK')"
```

Ejecuta los comandos `python -m edge` con ese venv activo. Para los tests, instala además `edge/requirements-dev.txt` en el mismo venv. El Release INT8 está en el repositorio privado: su descarga mediante GitHub CLI autenticado y la verificación SHA se documentan en el inicio rápido.

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
| `camera.warmup_frames` | `10` | Cuadros que se leen y descartan al abrir y antes de cada captura (autoexposición) |
| `camera.settle_seconds` | `0.5` | Tiempo mínimo leyendo cuadros antes de capturar, para no devolver el cuadro de la escena anterior (DirectShow) |
| `crop` | `{center_square: 0.8}` | Cuadrado centrado, 80% del lado menor. También `null` o `{x, y, width, height}` en píxeles |
| `model.registry` | `../models/edge_registry.json` | El SHA también debe coincidir con `model_sha256` del registro (#6) |
| `capture.mode` | `manual` | `manual`: Enter captura, `q` sale. `interval`: cada `interval_seconds` (10), Ctrl+C sale |
| `aws.bucket` | `mlops-p4-edge-captures-222629887955` | Destino del envío (P4-09); `""` = solo local |
| `aws.profile` | `mlops-p3` | Perfil SSO local que asume `MLOpsP3`; `null` = `AWS_PROFILE` o la cadena por defecto |
| `aws.region` | `us-east-1` | |

Si la cámara no abre, revisa `python -m edge cameras` y ajusta `camera.name` (o `camera.index`).

## Uso

```bash
python -m edge run                          # modo de la config; envía cada captura a S3
python -m edge run --mode interval --count 5
python -m edge run --image RUTA.jpg         # sin cámara: clasifica ese JPG (1 captura; --count N la repite)
python -m edge gui                          # ventana con vista en vivo y botón para capturar
python -m edge status                       # config, SHA, último evento y estado de envíos
python -m edge retry <capture_id>           # reenvía el mismo evento de esa captura
python -m edge retry --pending              # reenvía todas las que no están sent/already_sent
python -m edge export [--out DIR]           # events_local.csv y events_local.json
```

### Sin cámara: `run --image`

`--image RUTA` usa un JPG (o PNG) en lugar de la cámara, para recorrer la app completa en una
máquina sin cámara, sin AWS o ambas. Es `Camera` cambiada por `ImageSource`
(`edge/camera.py`); todo lo demás es igual a una captura real: verificación del SHA y del
registro, recorte de la config, preprocesamiento, inferencia, evento validado (contrato v1,
`schema_version: "1"`), `images/`, `crops/` y `captures.jsonl`.

```bash
python -m edge run --image reports/p4/evidence/prueba-edge-pantalla/images/5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110.jpg
```

- Sin `--count` hace **una** captura y termina: no pide Enter ni espera `interval_seconds`.
  `--count N` repite la misma imagen N veces, sin pausa, y cada una tiene su `capture_id`.
- No se combina con `--mode`: el modo es `image` y aparece así en el log
  (`modo image, dispositivo …, imagen RUTA`).
- El frame debe ser el **completo**, como el que guarda la app en `images/`: el recorte
  (`crop.center_square` o el de la config) se aplica igual que con la cámara. No le pases un
  recorte de `crops/`.
- Con `aws.bucket: ""` (o `--bucket ""`) la captura queda solo en `edge/data/`. Con bucket
  configurado se envía como cualquier otra, con el `device_id` de la config.
- Una ruta que no existe, una carpeta o un archivo que no es imagen terminan con exit 1 y un
  mensaje claro, sin traceback.

### Interfaz gráfica: `gui`

```bash
python -m edge gui                   # cámara de la config
python -m edge gui --image RUTA.jpg  # sin cámara: la vista en vivo es esa imagen
python -m edge gui --bucket ""       # solo local, sin enviar a S3
```

Abre una ventana en lugar de pedir Enter en la terminal. Hace lo mismo que `run`: cada foto pasa
por la verificación del modelo, el recorte de la config, la inferencia y el evento validado
(contrato v1), se guarda en `edge/data/` y se envía a S3 en segundo plano. Solo cambia cómo se
pide la foto.

- **Vista en vivo** (~30 imágenes por segundo) de hacia dónde apunta la cámara. Con un `crop` en
  la config, lo que queda fuera de la zona que ve el modelo se oscurece y la zona lleva un borde
  blanco (casilla *Marcar la zona que ve el modelo*).
- **Capturar foto** (o la tecla Espacio) toma el último cuadro de la vista: la foto es lo que se
  ve. A la derecha aparecen la zona clasificada, la clase, la probabilidad de cada clase, los
  tiempos y el `capture_id`, que se puede seleccionar y copiar para buscarlo en el portal.
- **Captura automática cada N s**: casilla desactivada al abrir, aunque `capture.mode` sea
  `interval`; `capture.interval_seconds` es el valor inicial de los segundos.
- **Esta sesión**: una fila por captura con su estado de envío (*enviando…*, *enviada*, *ya
  enviada*, *FALLÓ*). Las que fallan se reenvían con `python -m edge retry --pending`.
- Al cerrar la ventana espera los envíos en cola (máx. 60 s). Si se cierra otra vez, sale sin
  esperar y esos envíos quedan `pending`.
- Una cámara que no abre o se desconecta, o un modelo que no verifica, se muestran en la propia
  ventana (y en `edge.log`); el botón queda desactivado hasta que haya imagen y modelo.

**Requisitos.** `tkinter`, que ya trae el instalador de Python para Windows de python.org. En
Linux: `sudo apt install python3-tk`; en macOS con Homebrew: `brew install python-tk@3.12`. Sin
`tkinter`, `gui` termina con un mensaje; el resto de los comandos no lo necesita.

**Necesita pantalla.** La ventana se abre en la pantalla de la laptop, no en la terminal desde la
que se lanza. Por SSH sirven `status`, `retry`, `export`, `run` y `run --image`, pero `gui`
debe lanzarse desde la sesión de la laptop. Sin pantalla termina con exit 1 y un mensaje que lo
dice.

Por captura se escriben en `edge/data/` (ignorada por git):

| Archivo | Contenido |
|---|---|
| `images/{capture_id}.jpg` | Frame completo (`image_key` del evento) |
| `crops/{capture_id}.jpg` | Región usada para inferir, si hay recorte |
| `captures.jsonl` | Una línea por captura: `{"event", "image_path", "crop_path"}`. Se escribe al final, así que una línea indica una captura completa |
| `uploads.jsonl` | Un registro por intento de envío: `{"capture_id", "upload_status", "error", "upload_ms", "bucket", "at"}`. Separado de la inferencia |
| `edge.log` | Arranque (verificación del SHA), capturas, envíos y errores |
| `export/` | Salida de `python -m edge export` |

Al reiniciar, la app sigue escribiendo en el mismo log y no repite `capture_id`.

## Módulos

| Módulo | Qué hace |
|---|---|
| `config.py` | Lee y valida `config.yaml` |
| `camera.py` | Cámara con OpenCV (BGR) e `ImageSource`, un archivo de imagen con la misma interfaz (`run --image`). `read()` vacía cuadros viejos antes de cada captura; `read_live()` lee el siguiente cuadro del flujo, para la vista en vivo |
| `preprocess.py` | BGR→RGB, recorte, resize 128×128 bilineal con antialias (Pillow, como torchvision), /255, normalización ImageNet, CHW |
| `backend.py` | `InferenceBackend` (`load`, `predict(tensor) -> probs`) y la implementación de ONNX Runtime. Otro runtime = otra clase en `BACKENDS` |
| `pipeline.py` | Una captura: inferencia, evento validado y escritura en disco |
| `event_validator.py` | Valida el evento contra el contrato v1 |
| `uploader.py` | Sube imagen y evento a S3 sin duplicar (P4-05) |
| `sync.py` | Envío en segundo plano, `uploads.jsonl` y reintentos (P4-09) |
| `export.py` | `edge export`: eventos locales y estado de envío a CSV y JSON |
| `preview.py` | Vista en vivo sin tkinter: `FrameGrabber` (hilo que lee la cámara y guarda el último cuadro) y `render_preview` (ajusta el cuadro a la ventana y marca la zona recortada) |
| `gui.py` | `edge gui`: la ventana (tkinter) sobre `EdgeApp` y `BackgroundSender` |

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

  `parity_model` necesita un CSV de referencia y las imágenes a las que apunta; el comando anterior usa las 20 referencias cuyos recortes se obtienen por DVC. Para una inferencia **sin cámara ni AWS** desde un clon, el [inicio rápido](../README.md#inicio-rápido-para-evaluadores-sin-cámara-y-sin-aws) usa un CSV de una fila y un recorte histórico ya versionado. Esa comparación reproduce una predicción, no mide accuracy independiente.

## Tests

```bash
python -m unittest discover -s edge/tests -v
```

No usan cámara, red ni modelos reales: el modelo ONNX diminuto se construye en el test. `run --image`
se prueba de punta a punta con ese modelo (`edge/tests/test_pipeline.py`). Las pruebas de la
ventana (`edge/tests/test_gui.py`) la manejan por código y necesitan `tkinter` y una pantalla;
sin ellas se saltan, y el resto corre igual.

## Uploader a S3 (P4-05)

`edge/uploader.py` sube imagen y luego evento con `If-None-Match: *` y devuelve
`sent`, `already_sent` o `failed`. Detalles, prueba de doble envío y comandos de
lectura en [`docs/p4/aws-capturas.md`](../docs/p4/aws-capturas.md).

## Envío desde la app (P4-09)

Con `aws.bucket` configurado, `run` pasa cada captura ya guardada a un hilo de fondo que
la sube con el uploader; la siguiente captura no espera. Cada intento queda en
`uploads.jsonl` con `pending` → `sent` / `already_sent` / `failed`, y un fallo se ve en
consola y en `edge.log` con el comando para reintentar. `retry` reenvía el evento guardado
en `captures.jsonl`, con el mismo `capture_id`, sin volver a inferir. No hay reenvío
automático: lo que falló o quedó `pending` se reenvía con `retry --pending`.

Procedimiento de falla y reintento, y evidencia para #9:
[`docs/p4/integracion-edge-aws.md`](../docs/p4/integracion-edge-aws.md).
