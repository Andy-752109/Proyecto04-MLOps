# Ficha de entrega — Proyecto 4: Clasificación en Edge

## Objetivo y equipo

Trasladar el clasificador P3 de gatos y perros a una laptop con cámara, clasificar sin depender de la red y sincronizar fotografía y evento a AWS para consulta desde Capturas Edge.

| Integrante | Responsabilidad |
|---|---|
| Karen | Modelo de origen (#1), decisión de hardware y runtime (#3), variante INT8 (#4), bucket AWS (#5), comparación de calidad (#10), corrida del benchmark y validación E2E en la laptop (#11, #12) |
| Ale | Uploader a S3 (#5), app edge (#6), integración AWS y reintentos (#9) |
| Heri | Contrato del evento y convención de nombres (#2), API (#7), Capturas Edge (#8), script del benchmark (#11), documentación (#13) |
| PM | Planeación y coordinación, aprobación de PRs, fotografías y verificación de evidencias |

## Dispositivo y software

| Campo | Valor y fuente |
|---|---|
| Dispositivo | ASUS GL553VD, `edge-laptop-01` ([decisión de runtime](decision-runtime.md), [benchmark](../../reports/p4/benchmark/summary.json)) |
| Cámara | USB `GENERAL WEBCAM` (UVC, VID `0x1B3F`, PID `0x2002`), 640×360 con DirectShow; la integrada `USB2.0 UVC HD Webcam` es otra cámara ([decisión](decision-runtime.md), [configuración](../../edge/config.example.yaml)) |
| CPU y RAM | Intel Core i5-7300HQ @ 2.50 GHz; 7.89 GB reportados por Windows, 8 GB nominales ([benchmark](benchmark.md)) |
| Sistema operativo | Windows 10 Home Single Language, 64 bits, build 19045 ([decisión](decision-runtime.md)) |
| Runtimes | Python 3.12.10; PyTorch 2.14.0+cpu; torchvision 0.29.0+cpu; ONNX Runtime 1.30.0 ([benchmark](../../reports/p4/benchmark/summary.json)) |

## Modelo, entrada y optimización

| Campo | Valor y fuente |
|---|---|
| Origen P3 | `1.0.0`, ResNet-18, checkpoint PyTorch `best.pt`, SHA-256 `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9` ([origen](modelo-origen.md)) |
| Variante | `1.0.0-int8`, ONNX Runtime CPU, SHA-256 `dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392` ([registro](../../models/edge_registry.json)) |
| Técnica | Exportación ONNX FP32 con paridad; cuantización estática INT8 QDQ por canal y calibración con 100 recortes de train ([conversión](../../reports/p4/conversion/conversion_log.json), [IDs](../../reports/p4/conversion/calibration_ids.csv)) |
| Entrada y clases | RGB float32 `[1,3,128,128]`, normalización ImageNet; salida `cat`, `dog` en ese orden ([origen](modelo-origen.md), [contrato](contrato-evento-edge.md)) |
| Tamaño | Original 45,304,641 bytes; INT8 11,431,238 bytes; reducción 74.76806404889071 % ([benchmark](../../reports/p4/benchmark/summary.json)) |
| Calidad val | 131 muestras; accuracy original e INT8 0.9618320610687023; F1 macro en ambos 0.9618231625575566; caída de accuracy 0 pp; 0 desacuerdos de clase ([métricas](../../reports/p4/quality/metrics_val.json)) |
| Latencia total | Original p50 24.155000 ms, p95 24.727215 ms; INT8 p50 8.910200 ms, p95 9.436275 ms. Incluye preprocesamiento e inferencia, excluye cámara, disco y red ([CSV](../../reports/p4/benchmark/latency_raw.csv), [protocolo](benchmark.md)) |
| Tiempo de subida (aparte) | 57 envíos `sent` de P4-12: p50 1,814 ms, p95 4,521 ms (mín. 565, máx. 6,076); en la prueba de fallo y reintento de P4-09, 2,782 ms. Incluye validación del evento, lectura del JPEG y las dos operaciones de S3; no se suma a la latencia local ([uploads.jsonl](../../reports/p4/operation/uploads.jsonl), [benchmark](benchmark.md#upload-por-separado-p4-09-pr-29), [evidencia #9](https://github.com/Andy-752109/Proyecto04-MLOps/issues/9#issuecomment-6071641928)) |

## AWS, evento y portal

El paquete P3 y la variante INT8 están en el bucket versionado `mlops-p3-models-222629887955`, con rutas y `VersionId` en [`models/registry.json`](../../models/registry.json) y [`models/edge_registry.json`](../../models/edge_registry.json). Las capturas van al bucket privado `mlops-p4-edge-captures-222629887955`, región `us-east-1`: `edge-captures/v1/images/{capture_id}.jpg` y `edge-captures/v1/events/{capture_id}.json` ([infraestructura](aws-capturas.md)). La app edge escribe primero el evento local y usa un hilo de envío; `If-None-Match: *` impide reemplazar un objeto ya subido ([integración](integracion-edge-aws.md)). El [contrato v1](contrato-evento-edge.md) define UUID, dispositivo, hora, versión, clase, probabilidades, confianza, recorte y `image_key`.

Para abrir el portal desde la computadora de desarrollo, prepara `.env` local según [README](../../README.md#abrir-capturas-edge-y-verificar-una-captura), inicia `aws sso login --profile mlops-p3`, ejecuta `docker compose up -d --build frontend ml-api` y abre `http://localhost:8080/edge/captures`. `ml-api` consulta S3; el frontend no lee AWS directamente. La infraestructura P3 en `terraform/` forma parte del repositorio, mientras que el bucket de capturas P4 se documenta como creación por AWS CLI, sin Terraform ([AWS](aws-capturas.md)).

## Validación E2E y evidencias

La [corrida final P4-12 del 8 de octubre](validacion-e2e.md), verificada en vivo por la PM, registra 57 IDs locales únicos y 57 eventos S3 con contenido igual al local. El [resumen generado](../../reports/p4/operation/resumen.md) reporta metadatos coincidentes para los 57 en el portal, aunque no se versionó la respuesta cruda de la API; hay [cinco IDs trazados](../../reports/p4/operation/trazabilidad.md) y una captura de pantalla. En la corrida, 20 fotos etiquetadas (10 `cat`, 10 `dog`) dieron 18 aciertos y 2 errores; son imágenes de Google, no del dataset ([confirmación de la PM](https://github.com/Andy-752109/Proyecto04-MLOps/pull/34#issuecomment-6072364988)), y una secuencia de 31 capturas duró 5 min 26 s. Se documentaron corte de red, reintentos, un segundo intento `already_sent` y reinicio. [Índice de evidencias](../../reports/p4/README.md) y [guion de demo](guion-demo.md).

## Limitaciones

- En P4-12 las fotos se mostraron a la cámara desde una pantalla, no impresas como se había planeado (se ve el borde del monitor en el portal). La PM lo aceptó: el recorrido captura → inferencia local → S3 → portal no cambia por el medio de la foto ([decisión](https://github.com/Andy-752109/Proyecto04-MLOps/pull/34#issuecomment-6072364988)).
- El benchmark mide preprocesamiento e inferencia sobre 20 recortes de referencia, no cámara ni carga de modelo; la subida se mide aparte en `uploads.jsonl`. La calidad de validation usa 131 muestras; la prueba física de 20 fotos tuvo 2 errores.
- El portal requiere red, SSO vigente y acceso al bucket; la clasificación local continúa sin red.
