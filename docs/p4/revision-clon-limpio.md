# Revisión desde clon limpio — checklist P4

## Checklist preparado

Revisor sugerido: Karen o Ale. Registrar una ejecución real en la sección siguiente; ninguna casilla representa PASS hasta hacerla. Usar un directorio nuevo y una identidad AWS autorizada. No borrar datos, volúmenes, modelos o carpetas de otro checkout.

| Paso | Acción desde el clon limpio | Resultado esperado | PASS/FAIL y nota |
|---|---|---|---|
| 1 | Clonar `https://github.com/Andy-752109/Proyecto04-MLOps.git`; registrar `git rev-parse HEAD` y `git status --short` | Commit conocido, árbol limpio | Sin ejecutar |
| 2 | Leer [README P4](../../README.md#proyecto-4--clasificación-en-edge), [manual edge](../../edge/README.md) y esta ficha | Dependencias y permisos claros | Sin ejecutar |
| 3 | En Windows 10 con Python 3.12, instalar `python -m pip install -r edge/requirements.txt` | Dependencias disponibles | Sin ejecutar |
| 4 | Copiar `edge/config.example.yaml` a `edge/config.yaml`; ejecutar `python -m edge cameras` y ajustar `camera.name` | Cámara USB identificada | Sin ejecutar |
| 5 | `aws sso login --profile mlops-p3`; recuperar P3 con `app/verify_reload.py` según [README](../../README.md#modelos-p3-e-int8) | Checkpoint verificado | Sin ejecutar |
| 6 | Crear `edge/models/`; descargar INT8 por `VersionId` según [registro](../../models/edge_registry.json); `python -m edge status` | `sha256: OK` | Sin ejecutar |
| 7 | Consultar [conversión](../../reports/p4/conversion/conversion_log.json), [calidad](../../reports/p4/quality/metrics_val.json) y [benchmark](../../reports/p4/benchmark/summary.json) | Cifras y rutas localizables | Sin ejecutar |
| 8 | `python -m edge run` con gato y perro; revisar `edge/data/captures.jsonl` | Clasificación local y UUID por captura | Sin ejecutar |
| 9 | Si el revisor tiene permiso para generar capturas: comprobar offline y `retry --pending` según [guía](integracion-edge-aws.md); si no, inspeccionar evidencia P4-12 | Estado de envío y trazabilidad explicables | Sin ejecutar |
| 10 | En computadora de desarrollo, completar `.env` local; iniciar SSO; `docker compose up -d --build frontend ml-api`; abrir `http://localhost:8080/edge/captures` | Portal carga capturas desde AWS | Sin ejecutar |
| 11 | Comparar un UUID local, evento e imagen S3 y fila del portal; consultar [trazabilidad](../../reports/p4/operation/trazabilidad.md) | Mismo ID y metadatos | Sin ejecutar |
| 12 | Registrar cualquier error, corrección, limitación y conclusión; no modificar infraestructura AWS | Revisión reproducible | Sin ejecutar |

## Revisión realmente ejecutada

**Estado:** ejecutada por Karen ([registro](https://github.com/Andy-752109/Proyecto04-MLOps/pull/35#issuecomment-6073416756)).

| Campo | Registro |
|---|---|
| Fecha y hora (zona) | Jueves 8 de octubre de 2026, 19:30, hora del centro de México (UTC−6; 2026-10-09 01:30 UTC) |
| Persona que revisa | Karen |
| Sistema operativo y versión | Microsoft Windows 10 Home Single Language |
| Commit probado (`git rev-parse HEAD`) | `26841f1` (rama `p4-13-evidencias-entrega`) |
| Pasos seguidos y diferencias frente al checklist | Clon nuevo del PR, todo levantado desde cero con la caché de Docker vacía y sin `.env` previo |
| Resultado global PASS/FAIL | PASS: funciona |
| Errores encontrados y evidencia | Ninguno reportado |
| Correcciones aplicadas o propuestas | Ninguna |
| Conclusión y firma/confirmación | Karen confirma en el [PR #35](https://github.com/Andy-752109/Proyecto04-MLOps/pull/35#issuecomment-6073416756) que el proyecto se levanta y funciona desde un clon limpio |

La revisión no debe inferirse de los tests automatizados ni de la corrida P4-12: requiere un clon nuevo y un registro de la persona revisora.
