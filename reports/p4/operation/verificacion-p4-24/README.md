# P4-24: verificación del recorrido de la app sin cámara (`edge run --image`)

Registra qué se ejecutó para comprobar [#58](https://github.com/Andy-752109/Proyecto04-MLOps/issues/58) y qué falta. Procedimiento: el [inicio rápido para evaluadores](../../../../README.md#inicio-rápido-para-evaluadores-sin-cámara-y-sin-aws). No se abrió cámara ni se usó AWS en ninguna prueba (`aws.bucket: ""`).

| Prueba | Estado |
|---|---|
| Tests de `edge/tests/` (incluye `ImageSource` y `run --image` de punta a punta) | Ejecutada en macOS |
| Recorrido del inicio rápido desde un clon nuevo de la rama, en macOS | Ejecutada el 16 de octubre de 2026 |
| Recorrido del inicio rápido desde un clon nuevo en **Windows** | Ejecutada el 16 de octubre de 2026 por AleSarahi; ver [la última sección](#clon-nuevo-en-windows-reportado-por-alesarahi) |

## Tests

Con el venv de `edge/requirements-dev.txt` (Python 3.12, `onnxruntime 1.30.0`, `opencv-python-headless 5.0.0`):

```text
python -m unittest discover -s edge/tests
Ran 160 tests ... OK (skipped=2)
```

Los 14 tests nuevos cubren: `ImageSource` devuelve BGR `uint8` HxWx3 sin pérdida de canales, convierte grises a 3 canales, entrega copias independientes y abre rutas con tildes; error (`CameraError`) con ruta inexistente, carpeta, archivo que no es imagen y archivo vacío; y `run --image` con `write_tiny_model` y `write_config`: una captura sin cámara ni Enter ni sockets, evento que pasa `validate_event` con `schema_version: "1"`, `images/` y `crops/` escritos, `--count 3`, `status`, exit 1 sin traceback con ruta mala y `--image` con `--mode` rechazado (exit 2).

## Clon nuevo en macOS

**Entorno:** macOS 26.6.2 arm64 (`Darwin 25.6.0`), Python 3.12.12. `gh repo clone Andy-752109/Proyecto04-MLOps -- --branch p4-24-edge-run-image` a un directorio temporal nuevo; commit `bbfd5de3b3a9b1ac3d70d78538b1d3fe7a3a2a16`, árbol limpio. Venv `.venv-edge` nuevo dentro del clon.

Pasos ejecutados, en este orden:

1. `python3.12 -m venv .venv-edge` y `python -m pip install -r edge/requirements.txt`; `python -c "import jsonschema, onnxruntime, cv2, boto3, yaml; print('deps OK')"` imprimió `deps OK`.
2. `cp edge/config.example.yaml edge/config.yaml` y `aws.bucket: ""` (comprobado con `yaml.safe_load`: cadena vacía).
3. El modelo **no se descargó del Release**: se copió a `edge/models/model_int8.onnx` el `model_int8.onnx` ya presente en el equipo de quien ejecutó la prueba (la opción A del inicio rápido). `shasum -a 256` dio `dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392`, igual al registro.
4. `python -m edge status` antes de capturar: `sha256      : OK`, `capturas    : 0`, `bucket      : (vacío: solo local)`.
5. El recorrido nuevo, con `</dev/null` para comprobar que no espera Enter:

```text
$ python -m edge run --image reports/p4/evidence/prueba-edge-pantalla/images/5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110.jpg
2026-10-16 18:17:45,306 INFO modelo model_int8.onnx verificado (sha256=dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392)
2026-10-16 18:17:45,325 INFO runtime onnxruntime-1.30.0 listo, modelo 1.0.0-int8
2026-10-16 18:17:45,325 INFO log edge/data/captures.jsonl: 0 capturas previas
2026-10-16 18:17:45,325 INFO modo image, dispositivo edge-laptop-01, imagen reports/p4/evidence/prueba-edge-pantalla/images/5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110.jpg
2026-10-16 18:17:45,325 WARNING sin aws.bucket: las capturas solo se guardan en local
2026-10-16 18:17:45,640 INFO captura 4b2016f5-6e73-43de-a011-b385dcb04593 dog 0.9881
[1] DOG 0.988 | 18:17:45 local / 00:17:45 UTC | pre 1.1 ms | inf 1.8 ms | 4b2016f5-6e73-43de-a011-b385dcb04593
2026-10-16 18:17:45,640 INFO fin: 1 capturas en esta sesión
(exit 0)

$ python -m edge status
dispositivo : edge-laptop-01
modo        : manual (intervalo 10 s)
cámara      : GENERAL WEBCAM
recorte     : {'center_square': 0.8}
modelo      : edge/models/model_int8.onnx (1.0.0-int8, onnxruntime)
sha256      : OK
capturas    : 1 en edge/data/captures.jsonl
bucket      : (vacío: solo local)
envíos      : desactivados; configura aws.bucket para enviar o reintentar
última      : 2026-10-17T00:17:45.376040+00:00 dog 0.988 (4b2016f5-6e73-43de-a011-b385dcb04593)

$ python -m edge export
1 capturas -> edge/data/export/events_local.json y edge/data/export/events_local.csv
(exit 0)
```

(En la consola real las rutas de `status` y `export` salen absolutas; aquí se acortaron.)

Comprobaciones sobre los archivos resultantes:

- `edge/data/` contenía `images/4b2016f5-….jpg`, `crops/4b2016f5-….jpg`, `captures.jsonl` (1 línea), `edge.log` y `export/`; `events_local.csv` tenía cabecera y 1 fila, y `events_local.json`, 1 elemento.
- El evento de `captures.jsonl` pasó `edge.event_validator.validate_event`: `schema_version: "1"`, `predicted_class: dog`, `confidence: 0.9881`, `crop: {x: 176, y: 36, width: 288, height: 288, frame_width: 640, frame_height: 360}`. Es el mismo recorte que el de la [captura histórica](../../evidence/prueba-edge-pantalla/captures.jsonl) de ese frame, que también dio `dog` (0.9887); la diferencia de confianza es la del frame guardado como JPEG frente al de la cámara.
- `git status --short` en el clon quedó vacío: `edge/data/`, `edge/models/`, `edge/config.yaml` y `.venv-edge` están ignorados.
- `python -m edge run --image reports/no-existe.jpg`: `ERROR no existe la imagen reports/no-existe.jpg`, exit 1, sin traceback.

## Clon nuevo en Windows (reportado por AleSarahi)

**Responsable:** AleSarahi. **Fuente:** [revisión aprobada del PR #59](https://github.com/Andy-752109/Proyecto04-MLOps/pull/59), publicada el 17 de octubre de 2026 a las 00:50 UTC (16 de octubre, 18:50 CST). Las marcas de hora de la consola indican que la ejecución fue el **16 de octubre de 2026** (18:45, hora local). Los resultados proceden de los fragmentos de consola que Ale incluyó en la revisión, no de un log independiente adjunto. Las rutas absolutas de su carpeta de usuario se abreviaron como `…\Proyecto04-MLOps`; el resto de la salida está tal como la reportó.

**Entorno:** Windows 11 Pro (NT 10.0.26200), Windows PowerShell 5.1.26100.9711, Python 3.12.14 en `.venv-edge` (el `sys.executable` estaba dentro de `.venv-edge\Scripts`). Clon nuevo con `gh repo clone … --branch p4-24-edge-run-image` en una carpeta nueva, en el commit `1cc2c16`, es decir, la cabeza de la rama cuando se hizo la prueba. Siguió **solo** el inicio rápido de esa rama.

| Paso | Resultado reportado por Ale |
|---|---|
| Modelo | Opción B: descargado del Release `p4-int8-1.0.0`. `Get-FileHash` dio `DFFA9CF2670F0F9BB96139AE08779E2BA3AF87691791AD770061E3F666290392`, igual a `$ExpectedSha`. |
| `deps OK` | Impreso. |
| `python -m edge status` antes de capturar | `sha256      : OK`, `capturas    : 0`, `bucket      : (vacío: solo local)`. |
| `python -m edge.tools.parity_model …` | `base=dog edge=dog`, `class_mismatches: 0`, código de salida 0. |

Salida del recorrido nuevo:

```text
> python -m edge run --image reports/p4/evidence/prueba-edge-pantalla/images/5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110.jpg
2026-10-16 18:45:21,351 INFO modelo model_int8.onnx verificado (sha256=dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392)
2026-10-16 18:45:21,450 INFO runtime onnxruntime-1.30.0 listo, modelo 1.0.0-int8
2026-10-16 18:45:21,450 INFO log …\Proyecto04-MLOps\edge\data\captures.jsonl: 0 capturas previas
2026-10-16 18:45:21,450 INFO modo image, dispositivo edge-laptop-01, imagen reports\p4\evidence\prueba-edge-pantalla\images\5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110.jpg
2026-10-16 18:45:21,450 WARNING sin aws.bucket: las capturas solo se guardan en local
2026-10-16 18:45:21,800 INFO captura 306758ba-6fdf-4fa3-af5e-05a70928984d dog 0.9875
[1] DOG 0.987 | 18:45:21 local / 00:45:21 UTC | pre 7.1 ms | inf 4.5 ms | 306758ba-6fdf-4fa3-af5e-05a70928984d
2026-10-16 18:45:21,800 INFO fin: 1 capturas en esta sesión
$LASTEXITCODE = 0

> python -m edge status
dispositivo : edge-laptop-01
modo        : manual (intervalo 10 s)
cámara      : GENERAL WEBCAM
recorte     : {'center_square': 0.8}
modelo      : …\Proyecto04-MLOps\edge\models\model_int8.onnx (1.0.0-int8, onnxruntime)
sha256      : OK
capturas    : 1 en …\Proyecto04-MLOps\edge\data\captures.jsonl
bucket      : (vacío: solo local)
envíos      : desactivados; configura aws.bucket para enviar o reintentar
última      : 2026-10-17T00:45:21.624142+00:00 dog 0.987 (306758ba-6fdf-4fa3-af5e-05a70928984d)
$LASTEXITCODE = 0

> python -m edge export
1 capturas -> …\Proyecto04-MLOps\edge\data\export\events_local.json y …\Proyecto04-MLOps\edge\data\export\events_local.csv
$LASTEXITCODE = 0

> python -m edge run --image reports\no-existe.jpg
... ERROR no existe la imagen reports\no-existe.jpg
$LASTEXITCODE = 1   (sin traceback)

> git status --short
(vacío)
```

Ale también reportó que `run` no pidió Enter (la entrada estándar de esa sesión estaba cerrada); que `edge/data/` quedó con `images/` y `crops/` (un JPG en cada una), `captures.jsonl`, `edge.log` y `export/`, y que el CSV tenía cabecera y 1 fila; que el evento pasa `validate_event` (`schema_version: "1"`, `dog`, confianza `0.98748`, recorte `{x: 176, y: 36, width: 288, height: 288, frame_width: 640, frame_height: 360}`, igual que en macOS); y que el CI "Edge contract (unittest offline)" ya estaba en verde.

**Lo que difiere de la prueba en macOS:** el modelo vino del Release (opción B) y no de una copia local; las rutas del log y de `status` salen con `\`; la confianza fue 0.9875 en Windows frente a 0.9881 en macOS (las probabilidades no tienen por qué coincidir entre plataformas) y los tiempos de preprocesamiento e inferencia fueron mayores. La clase y el recorte coinciden.

**Problema encontrado:** `py -3.12 -m venv .venv-edge` falló con `No suitable Python runtime found` porque Python 3.12 estaba instalado con uv y no desde python.org. Ale creó el venv con `py -V:Astral/CPython3.12.14 -m venv .venv-edge`; el resto del inicio rápido no cambió. Se añadió a la tabla de problemas del [README principal](../../../../README.md#inicio-rápido-para-evaluadores-sin-cámara-y-sin-aws).

**No cubierto:** la ruta opcional con cámara, el envío a S3 y Python 3.10.
