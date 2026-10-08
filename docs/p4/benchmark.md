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

1. Usar Python 3.12 de 64 bits. Instalar `edge/requirements.txt` (incluye
   `onnxruntime==1.30.0`) y PyTorch CPU `2.14.0` + torchvision CPU `0.29.0` del entorno `ml`
   de `app/pyproject.toml`. Confirmar que importan ambas bibliotecas. ONNX Runtime en Windows
   requiere Microsoft Visual C++ Redistributable x64, según `decision-runtime.md`.
2. Tener el paquete original verificado en `eval_v1.0.0/`: `config.json`, `class_map.json` y
   `artifacts/checkpoint/best.pt`. `app/verify_reload.py --version 1.0.0` lo obtiene por
   `VersionId` y verifica el SHA del paquete. El benchmark verifica de nuevo el SHA del
   checkpoint contra `models/registry.json` y `reports/selection.json`.
3. Tener `edge/models/model_int8.onnx` obtenido según `models/edge_registry.json`. El benchmark
   verifica su `model_sha256` y que deriva del checkpoint original. Los binarios no entran a Git.
4. Materializar los 20 JPEG indicados por `reports/p4/reference/parity_reference.csv` bajo
   `data/derived/crops/images/`. Son referencias de **validation**, diez por clase; no usar
   `calibration_ids.csv`, que pertenece a train. El script valida el conjunto fijo, carga los
   20 JPEG antes de cronometrar y registra el SHA-256 de cada uno.
5. Comprobar alimentación y cerrar otros programas. Verificar que el directorio de salida aún
   no contiene `latency_raw.csv` ni `summary.json`: el script no sobrescribe evidencia anterior.

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
