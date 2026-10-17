# P4-24: verificación del recorrido de la app sin cámara (`edge run --image`)

Registra qué se ejecutó para comprobar [#58](https://github.com/Andy-752109/Proyecto04-MLOps/issues/58) y qué falta. Procedimiento: el [inicio rápido para evaluadores](../../../../README.md#inicio-rápido-para-evaluadores-sin-cámara-y-sin-aws). No se abrió cámara ni se usó AWS en ninguna prueba (`aws.bucket: ""`).

| Prueba | Estado |
|---|---|
| Tests de `edge/tests/` (incluye `ImageSource` y `run --image` de punta a punta) | Ejecutada en macOS |
| Recorrido del inicio rápido desde un clon nuevo de la rama, en macOS | Ejecutada el 16 de octubre de 2026 |
| Recorrido del inicio rápido desde un clon nuevo en **Windows** | **Pendiente**: requiere una máquina Windows; ver [la última sección](#windows-desde-clon-nuevo-pendiente) |

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

## Windows desde clon nuevo (pendiente)

**No se ha ejecutado.** La prueba de macOS no sustituye a la de Windows: aquí no se ejercitó PowerShell, `Activate.ps1`, `Get-FileHash` ni las rutas con `\` del log. Pendiente de quien tenga una máquina Windows (p. ej. AleSarahi, como en P4-23): seguir **solo** el inicio rápido de la rama `p4-24-edge-run-image` desde un clon nuevo hasta `python -m edge export`, y pegar aquí y en el PR, sin credenciales:

- versión de Windows, de PowerShell y de Python (`python --version` dentro de `.venv-edge`);
- de dónde se tomó el modelo (Release u opción A) y su `Get-FileHash`;
- la salida de `python -m edge run --image …`, `python -m edge status` y `python -m edge export`, con sus códigos de salida (`$LASTEXITCODE`);
- el resultado de `python -m edge run --image reports\no-existe.jpg` (mensaje y exit 1);
- `git status --short` vacío al final.
