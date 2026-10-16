# P4-23: verificación local de inferencia sin cámara ni AWS

Esta evidencia registra una **prueba local en macOS**, realizada el 15 de octubre de 2026 desde el checkout de la rama `p4-23-readme-edge-reproducible`. No se inició cámara ni se consultó AWS. El [inicio rápido para evaluadores](../../../../README.md#inicio-rápido-para-evaluadores-sin-cámara-y-sin-aws) contiene el procedimiento propuesto para Windows PowerShell.

## Procedimiento ejecutado

- Python 3.12.14 en un venv nuevo fuera del repositorio, sin `--system-site-packages`.
- `python -m pip install -r edge/requirements.txt`: terminó correctamente en ese venv.
- `python -c "import jsonschema, onnxruntime, cv2, boto3, yaml; print('deps OK')"`: imprimió `deps OK`; ONNX Runtime reportó versión `1.30.0`.
- Descarga autenticada con GitHub CLI de `model_int8.onnx` del Release privado `p4-int8-1.0.0` a `edge/models/model_int8.onnx` (ruta ignorada por Git).
- SHA-256 calculado del ONNX: `dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392`, igual a [`models/edge_registry.json`](../../../../models/edge_registry.json).
- Se preparó un CSV temporal de una fila a partir de [`resultados.csv`](../../evidence/prueba-edge-pantalla/resultados.csv) y [`captures.jsonl`](../../evidence/prueba-edge-pantalla/captures.jsonl). Su contenido se conserva en [`referencia.csv`](referencia.csv), apuntando al [JPEG versionado](../../evidence/prueba-edge-pantalla/crops/5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110.jpg).

Desde la raíz del repositorio se ejecutó `python -m edge.tools.parity_model` con `--model edge/models/model_int8.onnx`, `--reference` al CSV temporal, `--crops reports/p4/evidence/prueba-edge-pantalla` y `--json` a un archivo temporal. El comando devolvió **código 0** y el resumen indicó `references: 1`, `class_mismatches: 0`, `runtime: onnxruntime-1.30.0`. La clase nueva fue **`dog`** (`prob_cat=0.011918750126189204`, `prob_dog=0.9880812498738109`). La predicción histórica también fue `dog` (`prob_cat=0.011341811907719998`, `prob_dog=0.9886581880922799`); `max_abs_delta_prob` fue `0.0005769382184692053`.

Después se repitió el mismo comando usando el [`referencia.csv` versionado](referencia.csv) como `--reference`. También devolvió **código 0**, `edge=dog` y `class_mismatches: 0`; el JSON se guardó fuera del repositorio.

La clase real `dog` procede del orden de imágenes mostrado durante P4-06, según su [reporte](../../evidence/prueba-edge-pantalla/README.md). El CSV usa la **predicción histórica del mismo modelo como referencia de paridad**. Esta prueba demuestra que se puede repetir una inferencia sin cámara ni AWS; no es una medición independiente de exactitud. Las probabilidades no tienen por qué coincidir exactamente entre plataformas.

## Validación desde clon nuevo en macOS

**Ejecutada el 15 de octubre de 2026 (CST)** en macOS arm64 (`Darwin 27.0.0`). Se creó un directorio temporal nuevo, se clonó el repositorio privado desde GitHub y se comprobó que el clon estaba limpio en `main`, commit `dfe74c5d74f55edf114faff1fce47fea4905a646`. Como los cambios de P4-23 aún no están publicados, se transfirieron al clon **únicamente** `README.md`, `edge/README.md` y `reports/p4/operation/verificacion-p4-23/referencia.csv`. El commit clonado **no** contenía esos cambios; los otros archivos, el modelo y el venv no se copiaron del checkout principal.

El clon se creó con `gh repo clone Andy-752109/Proyecto04-MLOps /tmp/p4-23-clon-limpio.Zq2RBI/Proyecto04-MLOps`. Después se ejecutaron estos comandos relevantes desde su raíz:

```bash
python3.12 --version
python3.12 -m venv .venv-edge
.venv-edge/bin/python --version
.venv-edge/bin/python -m pip --version
.venv-edge/bin/python -m pip install -r edge/requirements.txt
.venv-edge/bin/python -c "import jsonschema, onnxruntime, cv2, boto3, yaml; print('deps OK')"
cp edge/config.example.yaml edge/config.yaml
# En la copia local se estableció aws.bucket: ""; no se ejecutó AWS CLI.
mkdir -p edge/models
gh release download p4-int8-1.0.0 --repo Andy-752109/Proyecto04-MLOps --pattern model_int8.onnx --dir edge/models
shasum -a 256 edge/models/model_int8.onnx
.venv-edge/bin/python -m edge status
.venv-edge/bin/python -m edge.tools.parity_model \
  --model edge/models/model_int8.onnx \
  --reference reports/p4/operation/verificacion-p4-23/referencia.csv \
  --crops reports/p4/evidence/prueba-edge-pantalla \
  --json /tmp/p4-23-clon-limpio.Zq2RBI/resultado.json
```

La edición de `edge/config.yaml` se hizo sobre el archivo ignorado por Git y se verificó con `yaml.safe_load`: el bucket era la cadena vacía. El venv recién creado usó **Python 3.12.14**, con `pip` dentro de `.venv-edge`, distinto del Python base y sin `--system-site-packages`. La instalación terminó correctamente; entre las versiones instaladas estuvieron `jsonschema 4.26.0`, `onnxruntime 1.30.0`, `opencv-python-headless 5.0.0.93`, `boto3 1.43.114` y `PyYAML 6.0.3`. La comprobación de imports imprimió **`deps OK`**.

El modelo se descargó nuevamente al clon con GitHub CLI autenticado. Su SHA-256 calculado fue `dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392`, igual al del registro. La salida relevante y real de `status` fue:

```text
sha256      : OK
bucket      : (vacío: solo local)
envíos      : desactivados; configura aws.bucket para enviar o reintentar
```

La inferencia procesó **una imagen** y terminó con **código 0**. La consola imprimió `OK  5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110 base=dog edge=dog max_dp=5.77e-04`. El JSON generado registró `references: 1`, `class_mismatches: 0`, `edge_class: dog`, `prob_cat: 0.011918750126189204` y `prob_dog: 0.9880812498738109`. No se abrió cámara ni se usó AWS.

## Verificación Windows reportada por AleSarahi

**Responsable:** AleSarahi. **Fuente:** [revisión aprobada del PR #57](https://github.com/Andy-752109/Proyecto04-MLOps/pull/57#pullrequestreview-5543598031), publicada el **16 de octubre de 2026 a las 05:52 UTC** (15 de octubre, 23:52 CST). Ale no indicó la fecha exacta de ejecución; esta es la fecha del reporte, no una fecha de prueba inferida. Los resultados siguientes proceden de los fragmentos de consola que incluyó en la revisión, no de una ejecución realizada por Heri ni de un log independiente adjunto.

Ale indicó **Windows 11 Pro x64** y **Windows PowerShell 5.1**. Usó una carpeta nueva y clonó `p4-23-readme-edge-reproducible` en el commit `6bbded3`; el Python global 3.10 no se usó. `py -3.12 -m venv .venv-edge` falló con `No suitable Python runtime found` porque Python 3.12 estaba instalado mediante uv. Ale creó el venv con `py -V:Astral/CPython3.12.14 -m venv .venv-edge`; `Activate.ps1` funcionó sin cambiar la política de ejecución. La consola mostró `Python 3.12.14`, `pip 25.0.1` y un `python.exe` dentro de `.venv-edge`.

Los comandos y resultados que Ale consignó en la revisión fueron:

| Paso | Resultado reportado por Ale |
|---|---|
| `python -m pip install -r edge/requirements.txt` | Código 0. |
| `python -c "import jsonschema, onnxruntime, cv2, boto3, yaml; print('deps OK')"` | `deps OK`. |
| `gh release download p4-int8-1.0.0` con `--pattern model_int8.onnx --dir edge/models` | Descarga del Release privado; la revisión abrevia los demás argumentos del comando. No se probó copiar un modelo preentregado. |
| `Get-FileHash -Algorithm SHA256 edge/models/model_int8.onnx` | `DFFA9CF2670F0F9BB96139AE08779E2BA3AF87691791AD770061E3F666290392`, igual al SHA esperado. |
| `python -m edge status` | `sha256      : OK`; `bucket      : (vacío: solo local)`; `envíos      : desactivados; configura aws.bucket para enviar o reintentar`; `capturas    : 0` antes de inferir. |
| `python -m edge.tools.parity_model --model edge/models/model_int8.onnx --reference reports/p4/operation/verificacion-p4-23/referencia.csv --crops reports/p4/evidence/prueba-edge-pantalla --json "$env:TEMP\p4-23-resultado.json"` | `OK  5ce6c2d4-be72-4e3a-a1b7-ab3d8b583110 base=dog edge=dog max_dp=5.77e-04`; `runtime: onnxruntime-1.30.0`, `references: 1`, `class_mismatches: 0`, código de salida 0. |

Ale editó la copia local de `edge/config.yaml` para poner `aws.bucket: ""` y reportó `git status --short` vacío al final; el venv, la configuración y el modelo estaban ignorados. No publicó las probabilidades individuales del JSON, la salida completa de instalación ni un registro de consola independiente, por lo que aquí no se atribuyen esos datos a Windows. Tampoco probó la ruta opcional con cámara, la instalación con Python 3.10 ni la **opción de copiar un modelo preentregado** añadida después de su prueba. La comparación de una imagen verifica inferencia y concordancia con la predicción histórica, no accuracy independiente.
