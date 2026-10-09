# Índice de evidencias P4

Matriz construida con el issue P4-13 y los criterios documentados en los tickets P4. La columna **ID** indica el requisito mínimo (M1–M3) y el criterio de la rúbrica (1.1–6.3) del prompt de evaluación P4 que respalda cada fila. **Registrado** significa que existe una evidencia versionada; **pendiente** exige verificación adicional, no equivale a fallo funcional. Las rutas son relativas a este archivo.

| ID | Criterio | Evidencia | Ruta | Estado |
|---|---|---|---|---|
| M1 · 1.1 | Continuidad P3 | Release, checkpoint, clases y procedencia | [modelo-origen](../../docs/p4/modelo-origen.md), [registro P3](../../models/registry.json) | Registrado |
| M1 · 1.2 | Conversión INT8 | FP32, cuantización y calibración train | [conversión](conversion/conversion_log.json), [calibración](conversion/calibration_ids.csv) | Registrado |
| M1 · 1.1 | Trazabilidad de modelo | SHA y versión de ambos artefactos | [registro edge](../../models/edge_registry.json), [recarga P3](evidence/verify_reload_1.0.0.txt) | Registrado |
| 1.2, 2.2 | Paridad | PyTorch ↔ ONNX FP32 y referencias | [paridad FP32](conversion/parity_fp32.json), [referencias](reference/parity_reference.csv) | Registrado |
| 1.3 | Calidad | Accuracy y F1 de validation | [métricas](quality/metrics_val.json), [comparación](../../docs/p4/comparacion-calidad.md) | Registrado |
| M2 · 2.3 | Ejecución física | Estado, cámara y log edge | [estado](operation/paso1_status.txt), [log](operation/edge.log) | Registrado |
| M2 · 2.1 | Cámara | Nombre y prueba con imágenes | [decisión](../../docs/p4/decision-runtime.md), [prueba](evidence/prueba-edge-pantalla/README.md) | Registrado; modelo comercial exacto no consta |
| M2 · 2.3 | Clasificación `cat`/`dog` | 20 fotos etiquetadas, 18 aciertos y 2 errores | [ground truth](operation/ground_truth.csv), [resumen](operation/resumen.md) | Registrado; fotos de Google, fuera del dataset ([PM](https://github.com/Andy-752109/Proyecto04-MLOps/pull/34#issuecomment-6072364988)) |
| M2 · 2.3 | Funcionamiento offline | Tres capturas fallidas en envío, conservadas localmente | [antes](operation/paso4_s3_antes.txt), [registros](operation/paso4_failed_antes.jsonl), [guía](../../docs/p4/validacion-e2e.md) | Registrado |
| M3 · 4.1–4.2 | Envío S3 | 57 eventos S3 con IDs y contenido iguales a los eventos locales; imagen y evento (dos objetos) comprobados explícitamente para los tres IDs del paso offline y los cinco IDs trazados | [eventos S3](operation/events_s3.json), [locales](operation/events_local.json), [objetos offline](operation/paso4_s3_despues.txt), [cinco IDs](operation/trazabilidad.md) | Registrado para esos alcances; no hay verificación individual versionada de las 57 imágenes |
| 4.1, 6.3 | Contrato de evento | Esquema v1 e imagen vinculada por UUID | [contrato](../../docs/p4/contrato-evento-edge.md), [eventos locales](operation/events_local.json) | Registrado |
| 4.3 | Idempotencia | Segundo reintento `already_sent` | [reintento repetido](operation/paso5_retry_repetido.txt), [objetos](operation/paso5_s3_despues.txt) | Registrado |
| M3 · 5.2–5.3 | API | Lectura de S3 por `ml-api` y tests | [código](../../app/ml_api/edge_captures.py), [tests](../../app/tests/test_ml_api_edge_captures.py) | Registrado |
| M3 · 5.1–5.3 | Portal | Una captura visible en screenshot, cinco IDs resumidos en trazabilidad; el reporte generado registra coincidencia de metadatos para 57 | [código](../../frontend/src/edge/pages/Captures.tsx), [captura](operation/paso7_portal_d9bd109d.jpg), [trazabilidad](operation/trazabilidad.md), [resumen](operation/resumen.md) | Registrado; respuesta cruda de la API no versionada |
| 3.1–3.3 | Benchmark | Corrida real, protocolo, muestras y resumen | [CSV](benchmark/latency_raw.csv), [JSON](benchmark/summary.json), [log](benchmark/run.log), [lectura](../../docs/p4/benchmark.md) | Registrado |
| 3.2 | Tiempo de subida | 57 envíos `sent` de P4-12, medidos aparte de la latencia local: p50 1,814 ms, p95 4,521 ms (mín. 565, máx. 6,076) | [uploads](operation/uploads.jsonl), [lectura](../../docs/p4/benchmark.md#upload-por-separado-p4-09-pr-29) | Registrado |
| 2.4 | 20 capturas | 20 filas etiquetadas del paso 2 | [ground truth](operation/ground_truth.csv), [resumen](operation/resumen.md) | Registrado |
| 2.4 | Prueba de 5 minutos | 31 capturas, 5 min 26 s | [consola](operation/paso3_consola.txt), [acta](../../docs/p4/validacion-e2e.md) | Registrado |
| 2.4 | Reinicio | 56 anteriores + nueva, 57 IDs únicos | [reinicio](operation/paso6_reinicio.txt), [únicos](operation/paso6_unicos.txt) | Registrado |
| 4.3 | Reintentos | De `failed` a `sent` para mismos IDs | [retry offline](operation/paso4_retry.txt), [retry error](operation/paso5_retry.txt) | Registrado |
| 4.3 | Ausencia de duplicados | 57 UUID locales únicos; `already_sent` y dos objetos para el ID reintentado explícitamente | [únicos](operation/paso6_unicos.txt), [reintento repetido](operation/paso5_retry_repetido.txt), [objetos](operation/paso5_s3_despues.txt) | Registrado para los IDs comprobados |
| M2–M3 · 4.1, 5.2, 6.2 | Pruebas E2E | Siete pasos y cinco IDs local → S3 → portal | [validación](../../docs/p4/validacion-e2e.md), [trazabilidad](operation/trazabilidad.md) | Registrado; corrida final verificada en vivo por la PM ([decisión](https://github.com/Andy-752109/Proyecto04-MLOps/pull/34#issuecomment-6072364988)) |
| 6.1, 6.3 | Documentación de entrega | Instalación, ficha y demo | [README](../../README.md#proyecto-4--clasificación-en-edge), [ficha](../../docs/p4/ficha-entrega.md), [guion](../../docs/p4/guion-demo.md) | Preparado para revisión |
| 6.1 | Revisión en clon limpio | Clon nuevo, Docker sin caché y sin `.env` previo | [checklist y registro](../../docs/p4/revision-clon-limpio.md), [confirmación](https://github.com/Andy-752109/Proyecto04-MLOps/pull/35#issuecomment-6073416756) | Ejecutada por Karen (commit `26841f1`): funciona |

Los logs originales de `operation/` contienen rutas de ejecución específicas de la carpeta de usuario de Windows; son rutas locales, no credenciales ni rutas necesarias para reproducir el proyecto. Se conservan para no alterar la evidencia histórica. Las decisiones de la PM (corrida final, procedencia de las fotos y fotos en pantalla) y la revisión en clon limpio se enlazan a sus comentarios en los PR #34 y #35; complementan los artefactos versionados, no los sustituyen.
