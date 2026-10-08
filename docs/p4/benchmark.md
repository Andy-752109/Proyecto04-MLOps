# P4-11 — benchmark local de tamaño y latencia

## Objetivo y procedencia

Comparar el checkpoint PyTorch P3 `1.0.0` con `1.0.0-int8` (ONNX Runtime CPU) **en el mismo
dispositivo**. La evidencia final debe ejecutarla Karen en `edge-laptop-01`, ASUS GL553VD,
Intel Core i5-7300HQ, 8 GB RAM, Windows 10, conectada a corriente y sin otros programas
abiertos. La temperatura puede anotarse si está disponible. Las pruebas de desarrollo en otras
máquinas no son evidencia de este ticket.

El script no usa cámara, red, AWS ni uploader. La preparación previa de modelos y recortes sí
puede requerir SSO/DVC, pero debe concluir antes de iniciar el benchmark.
El original es un checkpoint PyTorch de pesos FP32; el optimizado es ONNX con cuantización
estática INT8 QDQ por canal. Ambos reciben el mismo tensor de entrada RGB float32
`[1,3,128,128]`; el formato de pesos cambia, la entrada no.

## Preparación en la laptop edge

Todos los bloques PowerShell siguientes se ejecutan **desde la raíz del checkout** en Windows 10,
con Python 3.12 de 64 bits. La raíz debe contener `app/`, `edge/`, `models/` y `reports/`;
ejecutar allí `python -m edge.benchmark` permite importar
`from app.edge_model.baseline import ...` sin modificar `PYTHONPATH`. Preparar modelos y datos
con red antes de medir. ONNX Runtime en Windows requiere Microsoft Visual C++ Redistributable
x64, según `decision-runtime.md`. Conviene clonar en una ruta corta, por ejemplo
`C:\mlops\Proyecto04-MLOps`: las dependencias de DVC/pip pueden superar el límite tradicional
de longitud de rutas de Windows.

1. Instalar las dependencias edge y **PyTorch CPU** en el Python que ejecutará el benchmark.
   Las versiones están fijadas en `app/pyproject.toml`; el índice CPU evita instalar una
   variante CUDA. La salida esperada es `2.14.0+cpu`, `0.29.0+cpu` y `CUDA: False`.

   ```powershell
   python -m pip install -r edge/requirements.txt
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   ```

2. Iniciar SSO y recuperar el paquete original de
   `s3://mlops-p3-models-222629887955/models/releases/v1.0.0/model_release_v1.0.0.tar.gz`.
   `app/verify_reload.py` lee `models/registry.json`, descarga exactamente el `VersionId`
   `V8H6fouL5HiaUvrxAz1gUI5RbRD008VH`, verifica el SHA-256 del tar y lo extrae en la raíz
   como `eval_v1.0.0/`. Su entorno `app/.venv` es independiente del Python del benchmark.

   ```powershell
   aws sso login --profile mlops-p3
   if ($LASTEXITCODE -ne 0) { throw 'Falló el inicio de sesión SSO' }
   Push-Location app
   uv sync --locked --no-build
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   uv run python verify_reload.py --version 1.0.0 --profile mlops-p3
   if ($LASTEXITCODE -ne 0) { throw 'Falló la verificación del paquete original' }
   Pop-Location
   if (-not (Test-Path 'eval_v1.0.0/artifacts/checkpoint/best.pt' -PathType Leaf)) { throw 'Falta eval_v1.0.0/artifacts/checkpoint/best.pt' }
   ```

   El paquete también debe contener `config.json` y `class_map.json`. El benchmark vuelve a
   verificar el SHA de `best.pt` contra `models/registry.json` y `reports/selection.json`.

3. Materializar directamente la salida cacheada `data/derived/crops` de la etapa `crops` de
   `dvc.yaml`, ya sincronizada con el remote `prod`. `uvx` ejecuta DVC `3.67.1` en un entorno
   aislado con Python 3.12: no depende de `py -3.12`, de `Activate.ps1` ni de cambiar la
   ExecutionPolicy. El perfil se configura **solo localmente** en `.dvc/config.local`.
   Los 20 paths de `parity_reference.csv` son relativos a `data/derived/crops/` y
   corresponden a validation, no a los IDs de calibración de train.

   ```powershell
   Get-Command uvx -ErrorAction Stop | Out-Null
   uvx --python 3.12 --from 'dvc[s3]==3.67.1' dvc --version
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   uvx --python 3.12 --from 'dvc[s3]==3.67.1' dvc remote modify --local prod profile mlops-p3
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   uvx --python 3.12 --from 'dvc[s3]==3.67.1' dvc pull -r prod crops
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   if (-not (Test-Path 'data/derived/crops/images' -PathType Container)) { throw 'Falta data/derived/crops/images' }
   git diff --exit-code -- dvc.lock
   if ($LASTEXITCODE -ne 0) { throw 'dvc.lock cambió durante la preparación' }
   ```

   `dvc pull` descarga y coloca la salida de la etapa sin ejecutar `dvc repro` ni actualizar
   `dvc.lock`. El preflight del paso 5 comprueba los **20 JPEG** del CSV; el benchmark los carga
   antes de cronometrar y registra el SHA-256 de cada uno.

4. Descargar `s3://mlops-p3-models-222629887955/models/edge/1.0.0-int8/model_int8.onnx`
   **por el VersionId del archivo ONNX**, distinto del VersionId del paquete, conforme a
   `models/edge_registry.json` y `edge/README.md`. El benchmark verifica `model_sha256` y
   su relación con el checkpoint original. Los binarios no entran a Git.

   ```powershell
   New-Item -ItemType Directory -Force edge/models | Out-Null
   aws s3api get-object --bucket mlops-p3-models-222629887955 --key models/edge/1.0.0-int8/model_int8.onnx --version-id 8xfGLRI8Mlq07SyM9hsODKBfuC1c24ig --profile mlops-p3 edge/models/model_int8.onnx
   if ($LASTEXITCODE -ne 0) { throw 'Falló la descarga del modelo INT8' }
   if (-not (Test-Path 'edge/models/model_int8.onnx' -PathType Leaf)) { throw 'Falta edge/models/model_int8.onnx' }
   ```

5. Ejecutar este **preflight desde la raíz**, ya fuera de `.venv-dvc`, con el mismo `python`
   que usará el benchmark. Termina con error si faltan dependencias, archivos o alguno de los
   20 recortes fijos.

   ```powershell
   @'
   import csv
   import platform
   import sys
   from pathlib import Path
   import torch
   import torchvision
   import onnxruntime
   from app.edge_model.baseline import load_package

   assert sys.version_info[:2] == (3, 12) and platform.architecture()[0] == '64bit', 'Se requiere Python 3.12 de 64 bits'
   assert torch.__version__ == '2.14.0+cpu', f'torch CPU inesperado: {torch.__version__}'
   assert torchvision.__version__ == '0.29.0+cpu', f'torchvision CPU inesperado: {torchvision.__version__}'
   assert not torch.cuda.is_available(), 'CUDA debe estar deshabilitado'
   assert onnxruntime.__version__ == '1.30.0', f'onnxruntime inesperado: {onnxruntime.__version__}'
   for path in (Path('eval_v1.0.0/artifacts/checkpoint/best.pt'), Path('edge/models/model_int8.onnx'), Path('reports/p4/reference/parity_reference.csv')):
       assert path.is_file(), f'Falta {path}'
   crops = Path('data/derived/crops')
   assert crops.is_dir(), f'Falta {crops}'
   with Path('reports/p4/reference/parity_reference.csv').open(newline='', encoding='utf-8') as f:
       paths = [crops / row['path'] for row in csv.DictReader(f)]
   assert len(paths) == 20, f'Se esperaban 20 referencias, hay {len(paths)}'
   missing = [str(path) for path in paths if not path.is_file()]
   assert not missing, f'Faltan recortes: {missing}'
   print('Preflight OK:', sys.version.split()[0], torch.__version__, torchvision.__version__, onnxruntime.__version__, len(paths), 'recortes')
   '@ | python -
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   ```

6. Conectar la laptop a corriente, cerrar otros programas y comprobar que la salida aún no
   contiene `latency_raw.csv` ni `summary.json`: el script no sobrescribe evidencia anterior.

Desde la raíz del repositorio, en PowerShell, ejecutar en **una sola línea**:

```powershell
python -m edge.benchmark --reference reports/p4/reference/parity_reference.csv --crops data/derived/crops --original-package eval_v1.0.0 --int8-model edge/models/model_int8.onnx --warmups 10 --measurements 100 --threads 1 --device-id edge-laptop-01 --power-state connected --other-processes none --output-dir reports/p4/benchmark
```

`--power-state` y `--other-processes` deben describir las condiciones reales; no se detectan
automáticamente. Puede añadirse `--temperature-c N` si se dispone de una lectura confiable.
El ID lógico, hostname, sistema, procesador, versiones de runtime e hilos quedan en el resumen.
El comando se niega a producir evidencia final fuera de Windows. PyTorch usa CPU, `eval()`,
`inference_mode()` y un hilo intra-op; ONNX usa `CPUExecutionProvider` y un hilo intra-op.

## Qué mide cada fila

La entrada es el **recorte RGB `uint8` ya decodificado**. Por ejecución, `perf_counter_ns()`
marca `t0` antes de `edge.preprocess.preprocess`, `t1` después de obtener el tensor float32
`[1,3,128,128]`, y `t2` tras la predicción y softmax:

- `preprocess_ms = (t1-t0)/1e6`: resize bilineal 128×128, escala a [0,1], normalización
  ImageNet y CHW. Ambas variantes llaman **la misma función**.
- `inference_ms = (t2-t1)/1e6`: conversión mínima del tensor para PyTorch, forward y softmax;
  para INT8, `OnnxRuntimeBackend.predict` y softmax.
- `total_ms = (t2-t0)/1e6`: suma de ambas fases, sin redondeos intermedios.

Quedan fuera cámara, recorte desde frame completo, lectura de disco, decodificación JPEG,
verificación y carga de modelos, persistencia del evento, upload y red. Por tanto, estos tiempos
no son la latencia de una captura completa. Las dos variantes reciben los mismos 20 recortes
en el mismo orden. Cada una hace 10 warmups (primeros diez IDs) y 100 mediciones (20 IDs ×
cinco vueltas); en cada pareja se alterna cuál runtime se ejecuta primero. El orden del CSV es
determinista. `rep` es 1–110 por variante y `is_warmup` identifica las primeras diez filas.

## Artefactos y recálculo

`reports/p4/benchmark/latency_raw.csv` guarda exclusivamente:

`variant,rep,is_warmup,input_id,preprocess_ms,inference_ms,total_ms`.

`summary.json` registra dispositivo, condiciones declaradas, runtimes y hilos, hashes de
entradas/modelos, conteos, tamaños y p50/p95 de las tres duraciones **por variante**. Los
campos `checkpoint_sha256` y `model_sha256` identifican respectivamente el archivo PyTorch
original y el ONNX INT8. Los
percentiles usan solo las 100 filas `is_warmup=false`: ordenar valores, calcular posición
`(n-1)×q` e interpolar linealmente entre vecinos (`q=0.50` y `q=0.95`). Cada cifra puede
recalcularse desde el CSV; el resumen guarda su SHA-256.

El tamaño original es el de `best.pt`; el optimizado, el de `model_int8.onnx`. Son formatos
distintos, nombrados explícitamente. La reducción se calcula con los bytes de esos archivos
en la laptop: `100 × (original_bytes - optimized_bytes) / original_bytes`.

Si el modelo original, PyTorch u otra precondición falla, `summary.json` queda con
`status: incomplete` y el tipo/mensaje del **error real**. Si ambos artefactos pasaron la
verificación, conserva sus bytes y la reducción de tamaño. No crea percentiles ni un CSV
completo ficticio; la comparación de latencia sigue pendiente. No publicar un reporte
`incomplete` como resultado final.

## Recursos ↔ calidad (P4-10, PR #28)

El benchmark no necesita que P4-10 esté integrado. Si existe
`reports/p4/quality/metrics_val.json`, lo lee de forma opcional: comprueba `split=val`, orden de
clases y SHA de ambos modelos, y agrega a `quality_evidence` ruta, SHA del JSON, 131 muestras,
accuracy/F1 macro de ambas variantes, caída en puntos porcentuales y desacuerdos de clase.
Si falta, escribe `status: pending`; si su procedencia no coincide, `status: invalid` sin copiar
métricas. No hay valores de calidad codificados en el script.

Al redactar la evidencia final, enlazar `metrics_val.json` y `comparison_val.csv` y completar:

| Recurso o calidad | Original P3 | INT8 | Interpretación |
|---|---:|---:|---|
| Tamaño en bytes | Desde `summary.json` | Desde `summary.json` | Reducción calculada, no estimada |
| Preprocess p50/p95 | Desde CSV | Desde CSV | Mismo algoritmo y entradas |
| Inference p50/p95 | Desde CSV | Desde CSV | PyTorch CPU vs ONNX CPU |
| Total p50/p95 | Desde CSV | Desde CSV | Sin upload |
| Accuracy/F1 macro en validation | Desde `metrics_val.json` | Desde `metrics_val.json` | Misma validación de 131 recortes |
| Diferencias de clase | — | Desde `metrics_val.json` | Separar calidad de latencia |

No trasladar resultados de la laptop donde se midió calidad a la columna de latencia del
dispositivo edge. La evidencia de P4-10 sirve para interpretar el intercambio, pero no
reemplaza las mediciones locales de este benchmark.

## Upload por separado (P4-09, PR #29)

P4-09 registra cada intento en `edge/data/uploads.jsonl`, separado de `captures.jsonl`:
`capture_id`, `upload_status`, `error`, `upload_ms`, `bucket`, `at`. `upload_ms` abarca validación
del evento, lectura del JPEG y dos operaciones de S3; **no es solo tiempo de red**. Un
`sent` después de reintento puede tener la imagen ya subida, así que conservar el estado y
contexto del intento al citarlo. `edge export` refleja el **último** intento, mientras que
`uploads.jsonl` conserva todo el historial. Para P4-11 citar una medición real y su
`capture_id` desde ese log cuando Karen la aporte. No usar el número ilustrativo de la
documentación de P4-09 ni volver a subir objetos solo para este benchmark. Upload nunca se
incorpora a `total_ms`, p50 o p95 locales.
