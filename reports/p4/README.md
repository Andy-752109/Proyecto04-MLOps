# Índice de evidencias P4

Matriz construida con el issue P4-13 y los criterios documentados en los tickets P4. No se encontró una rúbrica o prompt P4 completo versionado en el repositorio; por ello no se atribuyen números de rúbrica no comprobados. **Registrado** significa que existe una evidencia versionada; **pendiente** exige verificación adicional, no equivale a fallo funcional. Las rutas son relativas a este archivo.

| Criterio | Evidencia | Ruta | Estado |
|---|---|---|---|
| Continuidad P3 | Release, checkpoint, clases y procedencia | [modelo-origen](../../docs/p4/modelo-origen.md), [registro P3](../../models/registry.json) | Registrado |
| Conversión INT8 | FP32, cuantización y calibración train | [conversión](conversion/conversion_log.json), [calibración](conversion/calibration_ids.csv) | Registrado |
| Trazabilidad de modelo | SHA y versión de ambos artefactos | [registro edge](../../models/edge_registry.json), [recarga P3](evidence/verify_reload_1.0.0.txt) | Registrado |
| Paridad | PyTorch ↔ ONNX FP32 y referencias | [paridad FP32](conversion/parity_fp32.json), [referencias](reference/parity_reference.csv) | Registrado |
| Calidad | Accuracy y F1 de validation | [métricas](quality/metrics_val.json), [comparación](../../docs/p4/comparacion-calidad.md) | Registrado |
| Ejecución física | Estado, cámara y log edge | [estado](operation/paso1_status.txt), [log](operation/edge.log) | Registrado |
| Cámara | Nombre y prueba con imágenes | [decisión](../../docs/p4/decision-runtime.md), [prueba](evidence/prueba-edge-pantalla/README.md) | Registrado; modelo comercial exacto no consta |
| Clasificación `cat`/`dog` | 20 fotos etiquetadas, 18 aciertos y 2 errores | [ground truth](operation/ground_truth.csv), [resumen](operation/resumen.md) | Registrado; origen fuera del dataset pendiente de PM |
| Funcionamiento offline | Tres capturas fallidas en envío, conservadas localmente | [antes](operation/paso4_s3_antes.txt), [registros](operation/paso4_failed_antes.jsonl), [guía](../../docs/p4/validacion-e2e.md) | Registrado |
| Envío S3 | 57 eventos S3 con IDs y contenido iguales a los eventos locales; imagen y evento (dos objetos) comprobados explícitamente para los tres IDs del paso offline y los cinco IDs trazados | [eventos S3](operation/events_s3.json), [locales](operation/events_local.json), [objetos offline](operation/paso4_s3_despues.txt), [cinco IDs](operation/trazabilidad.md) | Registrado para esos alcances; no hay verificación individual versionada de las 57 imágenes |
| Contrato de evento | Esquema v1 e imagen vinculada por UUID | [contrato](../../docs/p4/contrato-evento-edge.md), [eventos locales](operation/events_local.json) | Registrado |
| Idempotencia | Segundo reintento `already_sent` | [reintento repetido](operation/paso5_retry_repetido.txt), [objetos](operation/paso5_s3_despues.txt) | Registrado |
| API | Lectura de S3 por `ml-api` y tests | [código](../../app/ml_api/edge_captures.py), [tests](../../app/tests/test_ml_api_edge_captures.py) | Registrado |
| Portal | Una captura visible en screenshot, cinco IDs resumidos en trazabilidad; el reporte generado registra coincidencia de metadatos para 57 | [código](../../frontend/src/edge/pages/Captures.tsx), [captura](operation/paso7_portal_d9bd109d.jpg), [trazabilidad](operation/trazabilidad.md), [resumen](operation/resumen.md) | Registrado; respuesta cruda de la API no versionada |
| Benchmark | Corrida real, protocolo, muestras y resumen | [CSV](benchmark/latency_raw.csv), [JSON](benchmark/summary.json), [log](benchmark/run.log), [lectura](../../docs/p4/benchmark.md) | Registrado |
| 20 capturas | 20 filas etiquetadas del paso 2 | [ground truth](operation/ground_truth.csv), [resumen](operation/resumen.md) | Registrado |
| Prueba de 5 minutos | 31 capturas, 5 min 26 s | [consola](operation/paso3_consola.txt), [acta](../../docs/p4/validacion-e2e.md) | Registrado |
| Reinicio | 56 anteriores + nueva, 57 IDs únicos | [reinicio](operation/paso6_reinicio.txt), [únicos](operation/paso6_unicos.txt) | Registrado |
| Reintentos | De `failed` a `sent` para mismos IDs | [retry offline](operation/paso4_retry.txt), [retry error](operation/paso5_retry.txt) | Registrado |
| Ausencia de duplicados | 57 UUID locales únicos; `already_sent` y dos objetos para el ID reintentado explícitamente | [únicos](operation/paso6_unicos.txt), [reintento repetido](operation/paso5_retry_repetido.txt), [objetos](operation/paso5_s3_despues.txt) | Registrado para los IDs comprobados |
| Pruebas E2E | Siete pasos y cinco IDs local → S3 → portal | [validación](../../docs/p4/validacion-e2e.md), [trazabilidad](operation/trazabilidad.md) | Ensayo registrado; corrida final y confirmación PM pendientes |
| Documentación de entrega | Instalación, ficha y demo | [README](../../README.md#proyecto-4--clasificación-en-edge), [ficha](../../docs/p4/ficha-entrega.md), [guion](../../docs/p4/guion-demo.md) | Preparado para revisión |
| Revisión en clon limpio | Plantilla de reproducción | [checklist](../../docs/p4/revision-clon-limpio.md) | Pendiente de ejecución por Karen o Ale |

Los logs originales de `operation/` contienen rutas de ejecución específicas de la carpeta de usuario de Windows; son rutas locales, no credenciales ni rutas necesarias para reproducir el proyecto. Se conservan para no alterar la evidencia histórica. No se incorporaron enlaces a comentarios o issues individuales como evidencia suplementaria porque no se verificó una URL específica que sustituya los artefactos versionados. Para la entrega, el equipo debe aportar la rúbrica P4 completa si existe fuera del repositorio y confirmar que las 20 fotografías de P4-12 no pertenecen al dataset de entrenamiento.
