# Guion de demostración P4 (10–15 minutos)

Recorrido dirigido de la **fase 4B, «Observar el recorrido»**, del archivo original `prompt-evaluacion-proyecto-4.md` (recibido por separado). La fase 4A trata de recalcular archivos y no se numera aquí. La [arquitectura P4](../../README.md#proyecto-4--clasificación-en-edge) sigue P3 → ONNX INT8 → cámara/edge → S3 → API → portal. Los nueve pasos siguientes están **documentados, no ejecutados en esta sesión**. El [ensayo histórico](#ensayo-realizado) no equivale a una demo observada por el evaluador. Ale tiene la laptop física y prepara el dispositivo, la cámara y el modelo local; confirma su disponibilidad y apoya la verificación. Heri prepara la verificación de S3, el portal AWS, la trazabilidad de capturas y las evidencias técnicas. La persona que ejecutará los pasos en la evaluación ante los profesores del ITESO se confirmará con el equipo; preparar la laptop no implica operarla durante la demostración. Si falla una prueba en vivo, indicar que se cambia a un respaldo histórico y no atribuirle resultados nuevos.

## Ensayo realizado

| Campo | Registro |
|---|---|
| Fecha | Jueves 8 de octubre de 2026 |
| Participantes | Karen y Ale; la PM estuvo presente como observadora |
| Equipos | Mac de Karen para el portal; `edge-laptop-01` con la cámara `GENERAL WEBCAM` |
| Duración aproximada | 20 minutos |
| Incidencias y ajustes | Sin fallas de operación. El modelo asignó la clase contraria a 2 fotos; se dejan visibles como errores reales del clasificador. Las fotos se mostraron a la cámara desde una pantalla, no impresas; la PM lo aceptó |

### Verificación P4-21

La [corrida P4-21 (#50)](https://github.com/Andy-752109/Proyecto04-MLOps/issues/50) es una **verificación interna**: las fotos las eligió el equipo y nadie externo la observó. No sustituye la demostración observada por el evaluador ni convierte en observados los pasos de la [matriz](#matriz-de-cobertura-de-la-fase-4b).

| Campo | Registro P4-21 |
|---|---|
| Responsable | Ale |
| Fecha | Martes 13 de octubre de 2026, 15:22–16:22 (UTC−06:00) |
| Commit | `02c82d5` (`main`, merge del #51), clon nuevo |
| Entorno | `edge-laptop-01`, Windows 10, Python 3.12.10, cámara `GENERAL WEBCAM`; portal AWS en el navegador |
| Duración | Parte B (B.1–B.9): 33 min 31 s, con la primera instalación de DVC y el `dvc pull` de recortes; B.1–B.8: unos 22 min |
| Pasos ejecutados | B.1–B.9 completos, todos PASS |
| Capturas generadas | 5: dos en B.2, una sin WiFi en B.3, una en B.4 y una tras reiniciar en B.8 |
| IDs comprobados | `6c21a87e…` laptop → S3 → portal (B.5–B.6); `52ae33a2…` `failed` → `sent` → `already_sent` (B.7) |
| Resultados de S3 | Evento de S3 igual al local; exactamente 2 objetos para el ID reintentado |
| Resultados del portal | 75 capturas válidas; `6c21a87e…` primera tras Actualizar, cat 94.1 %, misma hora UTC que el evento |
| Incidencias | `KeyCount` salía `null` en B.5 y B.7 (corregido en este guion); SSO con `--use-device-code` |
| Evidencias | [Registro y consola](../../reports/p4/operation/verificacion-p4-21/README.md), [captura del portal](../../reports/p4/operation/verificacion-p4-21/portal_aws.png) |
| Conclusión | El guion se puede ejecutar completo desde un clon limpio de `main` con el portal en AWS |

## Preparación antes de iniciar el cronómetro

- En la laptop Windows, tener materializados el paquete P3 `1.0.0` y el ONNX `1.0.0-int8`, `edge/config.yaml` ajustado, cámara `GENERAL WEBCAM` conectada y `python -m edge status` con `sha256: OK`. `edge run` coteja además el registro de modelos al arrancar. Confirmar con [`edge/README.md`](../../edge/README.md#instalación) las dependencias de Python antes de empezar.
- Tener **dos terminales** en la laptop: A para `python -m edge run` (queda ocupada esperando Enter) y B para `python -m edge status` y `python -m edge retry --pending`. Preparar cómo mostrar a la cámara las fotos que elija el evaluador y anotar los UUID que se generen.
- Leer `aws.bucket` y `aws.profile` de `edge/config.yaml` en la laptop; `python -m edge status` muestra el bucket. Si hay perfil configurado, iniciar `aws sso login --profile <perfil_real>` en la laptop edge y en el equipo que consultará S3; comprobar identidad y lectura con `aws sts get-caller-identity --profile <perfil_real>` y `aws s3api list-objects-v2 --bucket <bucket_real> --prefix edge-captures/v1/events/ --max-keys 1 --profile <perfil_real>`. Si `aws.profile` es `null`, usar la cadena AWS configurada sin `--profile`. El ejemplo versionado usa `mlops-p3`, pero la configuración real manda.
- Preparar **antes de la demostración** las credenciales del portal entregadas por privado. En el navegador, abrir **<http://18.213.105.30/edge/captures>**, autenticarse y confirmar que Capturas Edge responde. El portal de evaluación corre en [AWS EC2 (`us-east-1`)](portal-aws.md); el navegador no necesita Docker ni SSO. No mostrar contraseñas ni respuestas completas de la API, que incluyen URLs prefirmadas.
- Tener abiertos [ficha](ficha-entrega.md), [índice](../../reports/p4/README.md), [calidad](comparacion-calidad.md), [benchmark](benchmark.md) y [acta E2E](validacion-e2e.md). Los reportes de 131 muestras de validación, 20 capturas y tiempos de benchmark están listos para consulta, pero la fase 4B solo pide una muestra dirigida nueva; no repetir el benchmark. La demostración offline puede requerir varios segundos para que el intento cambie de `pending` a `failed`; si excede el tiempo asignado, mostrar el respaldo y declarar ese paso sin comprobar en vivo.
- Antes de abrir Experiments o Training, advertir que **en el despliegue AWS actual pueden mostrar errores porque MLflow no está disponible**. Capturas Edge y Models no dependen de MLflow ([detalle](portal-aws.md)).

## Secuencia cronometrada

| Minuto estimado | Paso y apoyo previsto | Acción y resultado observable | Respaldo y actuación si falla |
|---|---|---|---|
| 0–1 | **B.1** · Ale (preparación)/Heri | Mostrar laptop y cámara físicas; `python -m edge cameras` y `python -m edge status`. Identificar origen P3, INT8, ruta ONNX, SHA y runtime según [detalle B.1](#b1-dispositivo-y-modelo). | [Estado P4-12](../../reports/p4/operation/paso1_status.txt) y [ficha](ficha-entrega.md). Si falta el modelo, seguir su recuperación documentada; no afirmar que cargó. |
| 1–2½ | **B.2** · demostrador Edge (Ale apoya) | `python -m edge run`. El evaluador elige dos objetos o fotos de clases distintas, su orden y posición; Enter por captura. Mostrar imagen, predicción, confianza, `capture_id` y estado de envío. | [20 etiquetas](../../reports/p4/operation/ground_truth.csv) y [errores históricos](../../reports/p4/operation/resumen.md). Si falla la cámara, ejecutar `python -m edge cameras`, revisar `camera.name` y, si persiste, declarar la captura no observada. |
| 2½–4 | **B.3** · demostrador Edge (Ale apoya) | Con el modelo ya cargado y A aún ejecutándose, desconectar WiFi de forma reversible; presentar objeto nuevo y pulsar Enter. Mostrar inferencia e ID locales, envío `pending` y luego `failed` si concluye el intento. | [Fallo offline P4-12](../../reports/p4/operation/paso4_failed_antes.jsonl). Si tarda o no se observa, mostrar estado real y decir qué falta; no presentar respaldo como ejecución actual. |
| 4–5 | **B.4** · demostrador Edge (Ale apoya) | Reconectar WiFi, comprobar conectividad y, **antes de reintentar la captura offline**, generar una captura **nueva** con Enter. Mostrar inferencia, nuevo ID y envío `sent`. | [Captura nueva P4-12](../../reports/p4/operation/paso4_captura_nueva.txt). Si falla red o SSO, conservar evento local y dejar el envío como no verificado. |
| 5–6 | **B.5** · Heri/demostrador Edge | Elegir el ID nuevo de B.4; cotejar fotografía local, evento local y las claves de imagen/JSON en S3 con [consultas B.5](#b5-trazabilidad-de-una-captura); mostrar coincidencia de ID y metadatos. | [Cinco IDs trazados hasta AWS](../../reports/p4/operation/portal-aws/trazabilidad.md). Si falla la lectura S3, explicar que la persistencia actual no se comprobó. |
| 6–7 | **B.6** · Heri | Abrir el portal AWS desde su navegación principal → **Capturas Edge**, localizar el ID de B.4, imagen, clase, confianza, fecha y dispositivo; pulsar **Actualizar** y comprobar permanencia y nuevas capturas. | [Captura histórica AWS](../../reports/p4/operation/portal-aws/portal_aws_capturas.png). Si no carga, señalar el fallo y no atribuir al portal la sesión actual. |
| 7–9 | **B.7** · demostrador Edge/Heri | Con el ID fallido de B.3, cerrar A con `q` y esperar su salida; reintentar ese mismo ID, observar `sent`; repetir para `already_sent` y comprobar dos claves actuales con [B.7](#b7-reintento-idempotente-y-dos-objetos-s3). | [Fallo](../../reports/p4/operation/paso5_falla.txt), [reintento](../../reports/p4/operation/paso5_retry.txt) y [dos objetos](../../reports/p4/operation/paso5_s3_despues.txt) históricos. Si no se produjo `failed`, no simularlo con otro método. |
| 9–11 | **B.8** · demostrador Edge (Ale apoya) | Tras el cierre con `q`, reiniciar `python -m edge run`; mostrar `N capturas previas`, capturar otro objeto y consultar `python -m edge status`. Salir con `q`. | [Reinicio P4-12](../../reports/p4/operation/paso6_reinicio.txt). Si no reinicia, mostrar error y marcar la prueba en vivo incompleta. |
| 11–14 | **B.9** · demostrador Edge/Heri | El evaluador elige algunas filas del CSV; ejecutar la reproducción existente de las **20** referencias con [`parity_model`](#b9-paridad-de-referencias-conocidas), señalar las filas elegidas y comparar clase/probabilidad. | [CSV](../../reports/p4/reference/parity_reference.csv) y [conversión histórica](../../reports/p4/conversion/conversion_log.json). Si faltan recortes/ONNX, seguir recuperación documentada o declarar que no se reprodujo. |
| 14–15 | Cierre · Heri | Ofrecer [calidad](comparacion-calidad.md), [benchmark](benchmark.md), [ficha](ficha-entrega.md) e [índice](../../reports/p4/README.md); distinguir resultados observados, ensayo y pendientes. | Si el tiempo se agota, entregar los archivos y registrar qué paso no se observó; no darlo por cumplido. |

Los minutos son una guía; SSO, cámara o red pueden consumir más tiempo. La corrida P4-12 (8 de octubre) fue un ensayo/validación interna verificada por la PM; **no marca estos pasos como observados en la evaluación**. Si una predicción en vivo falla, dejarla visible: no repetir la captura para sustituirla ni cambiar el resultado manualmente; explicar que es la inferencia real. No ejecutar conversión, benchmark ni infraestructura durante la demo.

## Procedimientos de la fase 4B

Ejecutar los comandos de esta sección desde la **raíz del repositorio en la laptop Windows**. La numeración corresponde a «Fase 4 · B. Observar el recorrido» del prompt original; los enlaces locales señalan los materiales del equipo.

### B.1 Dispositivo y modelo

Ale prepara y pone a disposición la laptop `edge-laptop-01` y su cámara USB `GENERAL WEBCAM`; la persona designada para ejecutar la demostración muestra el equipo y ejecuta `python -m edge cameras` y `python -m edge status`, con apoyo de Ale durante la verificación. En la salida de `status`, señalar dispositivo, cámara, ruta local `edge/models/model_int8.onnx`, versión `1.0.0-int8`, runtime `onnxruntime` y `sha256: OK`; `edge run` comprueba también el registro antes de inferir. Heri abre [origen P3 `1.0.0`](modelo-origen.md), [registro P3](../../models/registry.json), [registro INT8](../../models/edge_registry.json) y [conversión](../../reports/p4/conversion/conversion_log.json) para contrastar la procedencia y los SHA completos, sin tomarlos por sí solos como prueba de parentesco. El backend local usa ONNX Runtime en CPU, clase `cat`/`dog`, recorte según `edge/config.yaml`, RGB 128×128, escala `/255` y normalización ImageNet ([manual Edge](../../edge/README.md#modelo), [preprocesamiento](../../edge/preprocess.py)). Respaldo: [estado P4-12](../../reports/p4/operation/paso1_status.txt) y [ficha](ficha-entrega.md). Si el ONNX falta o el SHA no coincide, seguir la [recuperación versionada](../../edge/README.md#modelo) y dejar pendiente la inferencia en vivo hasta verificarlo; no improvisar una conversión.

### B.2 Objetos elegidos por el evaluador

La persona designada para ejecutar la demostración deja `python -m edge run` activo en A. El evaluador elige **objetos o fotografías, orden y posición**, con al menos dos clases; puede cambiar la posición entre capturas. La persona que ejecuta muestra cada objeto a la cámara y pulsa Enter, con apoyo de Ale si hace falta. Para cada una señala fotografía local en `edge/data/images/{capture_id}.jpg`, predicción y confianza en la consola/evento, UUID en `edge/data/captures.jsonl` y estado de envío en `edge/data/uploads.jsonl` o `python -m edge status` ([manual Edge](../../edge/README.md#uso)). No ocultar errores ni sustituir predicciones con capturas históricas. Si no abre la cámara, revisar `python -m edge cameras` y `camera.name`; usar [evidencia P4-12](../../reports/p4/operation/resumen.md) solo como respaldo identificado.

### B.3 Inferencia sin red

**Después de cargar el modelo**, apagar WiFi temporalmente sin alterar la configuración permanente del sistema y mantener A en ejecución. Capturar un objeto nuevo: debe haber imagen, clase, confianza e ID locales aunque el envío quede `pending` o termine `failed`. En B consultar `python -m edge status` y, para el ID concreto, `edge/data/uploads.jsonl`. Respaldo: [fallo offline P4-12](../../reports/p4/operation/paso4_failed_antes.jsonl). Si el intento sigue pendiente, decirlo; esperar el `failed` antes de reconectar si se quiere usar este ID para B.7.

### B.4 Reconexión y captura nueva

Reconectar WiFi y comprobar conectividad; **generar una captura nueva** en A y mostrar su inferencia, ID y envío `sent` en el registro local. Respaldo: [captura nueva P4-12](../../reports/p4/operation/paso4_captura_nueva.txt). Reconectar **no** reenvía por sí solo las capturas offline: la pendiente se reintenta explícitamente en B.7. Si la conexión o las credenciales fallan, conservar la captura local, señalar el estado real y no afirmar que llegó a AWS.

### B.5 Trazabilidad de una captura

Heri toma el `capture_id` **real de B.4**. La persona que ejecuta los pasos Edge muestra `edge/data/images/{capture_id}.jpg` y el evento correspondiente de `edge/data/captures.jsonl`; cotejar clase, confianza, hora con zona, dispositivo, versión e `image_key`. Fijar el bucket y perfil **reales** de `edge/config.yaml`; dejar `$AwsProfile` vacío solo si `aws.profile` es `null`. Estas consultas son de **solo lectura**:

```powershell
$Bucket = '<bucket real>'
$AwsProfile = '' # Sustituir por el perfil real si aws.profile no es null
$AwsProfileArgs = @(if ($AwsProfile) { '--profile'; $AwsProfile })
$CaptureId = '<UUID nuevo de B.4>'
aws s3api list-objects-v2 --bucket $Bucket --prefix "edge-captures/v1/images/${CaptureId}.jpg" --query 'Contents[].Key' --output json @AwsProfileArgs
aws s3api list-objects-v2 --bucket $Bucket --prefix "edge-captures/v1/events/${CaptureId}.json" --query 'Contents[].Key' --output json @AwsProfileArgs
aws s3 cp "s3://$Bucket/edge-captures/v1/events/${CaptureId}.json" - @AwsProfileArgs
aws s3api head-object --bucket $Bucket --key "edge-captures/v1/images/${CaptureId}.jpg" @AwsProfileArgs
```

Comprobar los nombres completos de ambas claves y cotejar el JSON S3 con el evento local; `head-object` confirma la fotografía en AWS. En B.6 se coteja esa misma foto/ID con el portal. Respaldo: [trazabilidad histórica AWS](../../reports/p4/operation/portal-aws/trazabilidad.md). Si no hay lectura S3, mostrar error y dejar la persistencia actual sin verificar; no exhibir respuestas de API con URLs prefirmadas.

### B.6 Portal AWS

Heri abre el [portal AWS](http://18.213.105.30/edge/captures), entra a **Capturas Edge** por la navegación, localiza el `capture_id` de B.4 y señala imagen, clase, confianza, fecha, dispositivo y versión. Pulsa **Actualizar** y comprueba que el registro permanece y que aparecen las capturas nuevas de la sesión. Contrasta con el evento de B.5. Respaldo: [captura histórica AWS](../../reports/p4/operation/portal-aws/portal_aws_capturas.png) y [verificación P4-15](portal-aws.md#verificación-evidencia-de-p4-15). Consultar la [documentación técnica del portal](portal-aws.md) cuando sea necesario. Si no carga o no aparece el ID, decir qué se vio y dejar la comprobación pendiente. Antes de navegar a Experiments o Training, recordar que pueden mostrar errores en AWS por ausencia de MLflow; Capturas Edge y Models tienen otro backend.

### B.7 Reintento idempotente y dos objetos S3

Usar el `capture_id` **real que falló de forma controlada en B.3** al cortar temporalmente el WiFi. La imagen y la inferencia ya quedaron locales; mostrar para ese ID `pending` y después `failed` en `edge/data/uploads.jsonl`. Con la red restaurada, salir de A con `q` y Enter y esperar a que terminen los envíos en curso. Reintentar **ese mismo ID** manualmente: debe pasar a `sent`; repetir exactamente el mismo comando: debe dar `already_sent`. No usar `retry --pending` si se quiere demostrar un ID concreto, y no suponer reenvío automático al reconectar. Si B.3 no produjo un fallo comprobable, B.7 queda pendiente; no inventar otra forma de provocarlo.

La sintaxis es `python -m edge retry <capture_id>`. En PowerShell, usar el UUID anotado, sin los signos `< >`:

```powershell
$CaptureId = '<UUID fallido de B.3>'
python -m edge retry $CaptureId
python -m edge retry $CaptureId
```

Esperado en consola y `edge/data/uploads.jsonl`: el mismo ID con `failed`, luego `sent`, luego `already_sent`; no hay una nueva inferencia. `python -m edge status` resume los estados, pero el registro JSONL permite comprobar el ID concreto. Reutilizar `$Bucket` y `$AwsProfileArgs` de B.5; si la consulta se hace en otra terminal, volver a definirlos allí con el bucket y perfil reales:

```powershell
aws s3api list-objects-v2 --bucket $Bucket `
  --prefix "edge-captures/v1/images/${CaptureId}" `
  --query 'Contents[].Key' `
  --output json @AwsProfileArgs

aws s3api list-objects-v2 --bucket $Bucket `
  --prefix "edge-captures/v1/events/${CaptureId}" `
  --query 'Contents[].Key' `
  --output json @AwsProfileArgs
```

Comprobar los **nombres completos**: exactamente `edge-captures/v1/images/${CaptureId}.jpg` y `edge-captures/v1/events/${CaptureId}.json`, una sola clave por consulta, **dos claves actuales en total**. El prefijo va sin extensión para que aparezca cualquier otro objeto con el mismo ID; no se cuentan versiones históricas de S3. No usar `KeyCount`: AWS CLI v2 pagina `list-objects-v2` y lo devuelve `null` (visto en la [corrida P4-21](../../reports/p4/operation/verificacion-p4-21/README.md#incidencias)). La imagen es el frame JPEG y el JSON es el evento. El uploader usa `If-None-Match: *`, de modo que el reintento no debe duplicarlos. Si falla S3 o el SSO, mostrar el estado local y el error; si no se recupera, señalar como **respaldo histórico** el [fallo controlado P4-12](../../reports/p4/operation/paso5_falla.txt), el [primer reintento](../../reports/p4/operation/paso5_retry.txt), el [segundo reintento](../../reports/p4/operation/paso5_retry_repetido.txt) y la [consulta de objetos](../../reports/p4/operation/paso5_s3_despues.txt), sin atribuirlos a la sesión actual.

### B.8 Reinicio en vivo

1. El programa `python -m edge run` de B.2–B.4 ya capturó objetos nuevos y se cerró con `q` y Enter antes de B.7. Confirmar que terminó el cierre y los envíos en cola; si aún sigue abierto, salir ahora con `q` y esperar.
2. Volver a ejecutar `python -m edge run` desde la raíz. Mostrar la línea del log **`N capturas previas`**. `N` es el historial real de esa laptop, no un número fijado para la demo.
3. Presentar otra foto elegida por el evaluador, pulsar Enter y mostrar nueva predicción, confianza e ID.
4. En B ejecutar `python -m edge status` para comprobar que aumentó el total y continúa el historial; cerrar A con `q` y Enter y esperar su salida segura.

Si no abre la cámara, revisar `python -m edge cameras` y `camera.name` en `edge/config.yaml`. Si no se recupera, mostrar el [reinicio histórico P4-12](../../reports/p4/operation/paso6_reinicio.txt) indicando que **no fue el reinicio de esta sesión**.

### B.9 Paridad de referencias conocidas

El evaluador elige algunas filas de [`parity_reference.csv`](../../reports/p4/reference/parity_reference.csv) para contrastarlas. El script existente procesa **las 20 entradas conocidas completas** y se señalan en la salida las filas elegidas; no se inventa un modo de ejecutar solo esas filas. Esto reproduce una muestra dirigida de la comparación de calidad, sin recalcular toda la validación ni repetir el benchmark. No clasifica una imagen nueva elegida por el evaluador: para esa imagen se usa `python -m edge run`.

Antes de ejecutarlo, confirmar Python y las dependencias de [`edge/requirements.txt`](../../edge/requirements.txt), el ONNX local, el CSV y **todos** los JPEG que el CSV enumera bajo `data/derived/crops/`:

```powershell
if (-not (Test-Path 'edge/models/model_int8.onnx' -PathType Leaf)) { throw 'Falta el ONNX local' }
if (-not (Test-Path 'reports/p4/reference/parity_reference.csv' -PathType Leaf)) { throw 'Falta el CSV de referencias' }
$Missing = @(Import-Csv 'reports/p4/reference/parity_reference.csv' | Where-Object {
  -not (Test-Path (Join-Path 'data/derived/crops' $_.path) -PathType Leaf)
})
if ($Missing.Count) { throw "Faltan $($Missing.Count) recortes de referencia" }

python -m edge.tools.parity_model `
  --model edge/models/model_int8.onnx `
  --reference reports/p4/reference/parity_reference.csv `
  --crops data/derived/crops `
  --json edge/data/parity-demo.json
```

La herramienta usa el preprocesamiento y backend Edge. `OK` indica que la **clase** coincide con la referencia; `DIF` indica diferencia de clase. El resumen JSON muestra SHA del modelo, cantidad de referencias, diferencias de clase y máxima diferencia absoluta de probabilidad; informar los valores observados, sin prometer cero diferencias. Sale con código 1 si hay diferencias de clase.

Si faltan recortes, seguir la [recuperación DVC existente](benchmark.md#preparación-en-la-laptop-edge): configurar el perfil local del remote `prod` y ejecutar allí el `dvc pull -r prod crops` documentado, **sin** repetir conversión ni benchmark. Si no es posible materializarlos, mostrar el [CSV](../../reports/p4/reference/parity_reference.csv) y la [evidencia histórica de conversión](../../reports/p4/conversion/conversion_log.json); explicar que **la paridad no se reprodujo en vivo**. Si falta el ONNX, seguir la [descarga por VersionId y comprobación SHA](../../edge/README.md#modelo); no afirmar inferencia en vivo hasta recuperarlo.

## Evidencia histórica para consulta, fuera del cronómetro

| Tema | Registro conservado | Alcance |
|---|---|---|
| Origen y conversión | [Recarga P3](../../reports/p4/evidence/verify_reload_1.0.0.txt), [conversión INT8](../../reports/p4/conversion/conversion_log.json), [paridad FP32](../../reports/p4/conversion/parity_fp32.json) | Procedencia y técnica de INT8 QDQ por canal; los hashes no sustituyen la comparación de origen y variante. |
| Calidad | [Comparación](comparacion-calidad.md), [métricas val](../../reports/p4/quality/metrics_val.json), [CSV comparativo](../../reports/p4/quality/comparison_val.csv) | 126/131 aciertos en ambos, accuracy 0.961832, F1 macro 0.961823, caída de 0 pp; son 131 muestras de validación, no predicciones de la demo. |
| Recursos | [Benchmark](benchmark.md), [resumen](../../reports/p4/benchmark/summary.json), [tiempos crudos](../../reports/p4/benchmark/latency_raw.csv), [log](../../reports/p4/benchmark/run.log) | Tamaño −74.768 %; p50 total 24.155 → 8.9102 ms, p95 24.727215 → 9.436275 ms en el mismo edge; no se repite durante B.9. |
| Operación P4-12 | [Acta E2E](validacion-e2e.md), [resumen](../../reports/p4/operation/resumen.md), [cinco IDs](../../reports/p4/operation/trazabilidad.md), [captura del portal local](../../reports/p4/operation/paso7_portal_d9bd109d.jpg) | 20 fotos etiquetadas, 18 aciertos y 2 errores; 57 UUID únicos en la corrida. El portal local es respaldo de P4-12, no prueba de la demo AWS actual. |
| Portal AWS | [Verificación P4-15](portal-aws.md#verificación-evidencia-de-p4-15), [trazabilidad AWS](../../reports/p4/operation/portal-aws/trazabilidad.md), [captura AWS](../../reports/p4/operation/portal-aws/portal_aws_capturas.png) | Registros anteriores; comprobar los nuevos IDs en vivo mediante B.5–B.6. |

## Matriz de cobertura de la fase 4B

Esta relación con los requisitos M1–M3 de la fase 1 y los criterios de la rúbrica de la fase 2 indica **qué se puede observar**, no asigna puntos ni declara criterios aprobados. `DOCUMENTADO` significa que existe el procedimiento en este guion; `VERIFICADO EN ENSAYO` se limita a los registros históricos enlazados; `OBSERVADO EN DEMO` solo se anotará tras una evaluación real. **Ninguno de los nueve pasos está marcado aquí como observado en la demo evaluada.**

| Paso y requisito del prompt | Criterio relacionado | Procedimiento y evidencia observable | Respaldo histórico | Estado actual |
|---|---|---|---|---|
| B.1 Dispositivo, cámara, origen, variante, runtime | M1, M2; 1.1, 1.2, 2.2, 6.1 | [B.1](#b1-dispositivo-y-modelo): laptop/cámara físicas, `cameras`, `status`, SHA/versión, registros P3/INT8 | [Estado P4-12](../../reports/p4/operation/paso1_status.txt), [ficha](ficha-entrega.md) | DOCUMENTADO; hay registro de ensayo; demo pendiente |
| B.2 Objetos/orden elegidos por evaluador y capturas nuevas | M2; 2.1, 2.3 | [B.2](#b2-objetos-elegidos-por-el-evaluador): `run`, Enter, imagen, clase, confianza, ID, envío | [Capturas P4-12](../../reports/p4/operation/events_local.csv), [errores](../../reports/p4/operation/resumen.md) | DOCUMENTADO; hay capturas de ensayo; demo pendiente |
| B.3 Captura y clasificación tras desconectar red | M2; 2.3 | [B.3](#b3-inferencia-sin-red): nuevo ID e inferencia local con `pending`/`failed` | [Fallo offline P4-12](../../reports/p4/operation/paso4_failed_antes.jsonl) | DOCUMENTADO; hay registro de ensayo; demo pendiente |
| B.4 Captura **nueva** y envío tras reconectar | M3; 4.1, 4.2 | [B.4](#b4-reconexión-y-captura-nueva): nuevo ID, clase y `sent` | [Captura nueva P4-12](../../reports/p4/operation/paso4_captura_nueva.txt) | DOCUMENTADO; hay registro de ensayo; demo pendiente |
| B.5 Seguir imagen e ID hasta AWS y portal | M3; 4.1, 4.2, 5.2 | [B.5](#b5-trazabilidad-de-una-captura): foto/evento local, ambas claves S3 y mismo ID en B.6 | [Trazabilidad AWS](../../reports/p4/operation/portal-aws/trazabilidad.md) | DOCUMENTADO; hay registro histórico; demo pendiente |
| B.6 Navegación, refresco, permanencia y novedades | M3; 5.1–5.3 | [B.6](#b6-portal-aws): Capturas Edge desde menú, imagen/metadatos y Actualizar | [Captura AWS P4-15](../../reports/p4/operation/portal-aws/portal_aws_capturas.png) | DOCUMENTADO; hay verificación histórica; demo pendiente |
| B.7 Fallo reversible, mismo ID, sin duplicación | 4.3 | [B.7](#b7-reintento-idempotente-y-dos-objetos-s3): `failed` → `sent` → `already_sent`, dos claves actuales | [Reintentos P4-12](../../reports/p4/operation/paso5_retry_repetido.txt), [S3](../../reports/p4/operation/paso5_s3_despues.txt) | DOCUMENTADO; hay registro de ensayo; demo pendiente |
| B.8 Reinicio y nueva clasificación | 2.4 | [B.8](#b8-reinicio-en-vivo): `q`, `run`, `N capturas previas`, nuevo ID y `status` | [Reinicio P4-12](../../reports/p4/operation/paso6_reinicio.txt) | DOCUMENTADO; hay registro de ensayo; demo pendiente |
| B.9 Reproducción de muestra elegida | 2.2, 6.2; 1.3 se evalúa con la comparación completa aparte | [B.9](#b9-paridad-de-referencias-conocidas): evaluador elige filas, script procesa 20, cotejar `OK`/`DIF` y probabilidad | [Referencias](../../reports/p4/reference/parity_reference.csv), [conversión](../../reports/p4/conversion/conversion_log.json) | DOCUMENTADO; hay evidencia técnica histórica; demo pendiente |

La [comparación de calidad](comparacion-calidad.md) y el [benchmark](benchmark.md) respaldan por separado 1.3 y 3.1–3.3; el guion no afirma haberlos recalculado durante la demo. La [configuración Edge](../../edge/README.md#configuración) y la [ficha](ficha-entrega.md) apoyan 6.3. Ante cualquier fallo, mostrar primero el estado o error observado, indicar qué parte no se ejecutó y pasar al respaldo identificado como histórico; nunca sustituir resultados de esta sesión por evidencias anteriores.
