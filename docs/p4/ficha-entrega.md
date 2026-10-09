# Ficha de entrega — Proyecto 4: Clasificación en Edge

## Objetivo y equipo

Trasladar el clasificador P3 de gatos y perros a una laptop con cámara, clasificar sin depender de la red y sincronizar fotografía y evento a AWS para consulta desde Capturas Edge.

| Integrante | Responsabilidad comprobable en el repositorio | Pendiente de confirmar |
|---|---|---|
| Heri | Documentación y evidencias P4-13 (issue #13) | Alcance exacto de su rol durante la demo |
| Karen | Corrida oficial del benchmark P4-11 en `edge-laptop-01` ([registro](benchmark.md)) | Rol en la demo y revisión desde clon limpio |
| Ale | Revisión desde clon limpio propuesta por P4-13 | Participación efectiva y otras responsabilidades |
| PM/equipo | Fotografías y validación de P4-12 ([acta](validacion-e2e.md)) | Confirmar que las 20 fotos no están en el dataset |

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

## AWS, evento y portal

El paquete P3 y la variante INT8 están en el bucket versionado `mlops-p3-models-222629887955`, con rutas y `VersionId` en [`models/registry.json`](../../models/registry.json) y [`models/edge_registry.json`](../../models/edge_registry.json). Las capturas van al bucket privado `mlops-p4-edge-captures-222629887955`, región `us-east-1`: `edge-captures/v1/images/{capture_id}.jpg` y `edge-captures/v1/events/{capture_id}.json` ([infraestructura](aws-capturas.md)). La app edge escribe primero el evento local y usa un hilo de envío; `If-None-Match: *` impide reemplazar un objeto ya subido ([integración](integracion-edge-aws.md)). El [contrato v1](contrato-evento-edge.md) define UUID, dispositivo, hora, versión, clase, probabilidades, confianza, recorte y `image_key`.

Para abrir el portal desde la computadora de desarrollo, prepara `.env` local según [README](../../README.md#abrir-capturas-edge-y-verificar-una-captura), inicia `aws sso login --profile mlops-p3`, ejecuta `docker compose up -d --build frontend ml-api` y abre `http://localhost:8080/edge/captures`. `ml-api` consulta S3; el frontend no lee AWS directamente. La infraestructura P3 en `terraform/` forma parte del repositorio, mientras que el bucket de capturas P4 se documenta como creación por AWS CLI, sin Terraform ([AWS](aws-capturas.md)).

## Validación E2E y evidencias

El [ensayo P4-12 del jueves 8](validacion-e2e.md) registra 57 IDs locales únicos y 57 eventos S3 con contenido igual al local. El [resumen generado](../../reports/p4/operation/resumen.md) reporta metadatos coincidentes para los 57 en el portal, aunque no se versionó la respuesta cruda de la API; hay [cinco IDs trazados](../../reports/p4/operation/trazabilidad.md) y una captura de pantalla. Dentro del ensayo, 20 fotos etiquetadas (10 `cat`, 10 `dog`) dieron 18 aciertos y 2 errores, y una secuencia de 31 capturas duró 5 min 26 s. Se documentaron corte de red, reintentos, un segundo intento `already_sent` y reinicio. La corrida final del viernes 9 queda para verificación del PM. [Índice de evidencias](../../reports/p4/README.md) y [guion de demo](guion-demo.md).

## Limitaciones y pendientes

- La confirmación del PM de que las fotografías de P4-12 no pertenecen al dataset sigue abierta en [validación E2E](validacion-e2e.md).
- La corrida final prevista para el viernes 9 todavía no consta en las evidencias versionadas.
- La revisión por Karen o Ale desde un clon limpio todavía no se ha ejecutado; hay [checklist](revision-clon-limpio.md).
- El benchmark mide preprocesamiento e inferencia sobre 20 recortes de referencia, no cámara, carga de modelo ni subida. La calidad de validation usa 131 muestras; la prueba física de 20 fotos tuvo 2 errores.
- El portal requiere red, SSO vigente y acceso al bucket; la clasificación local continúa sin red.
