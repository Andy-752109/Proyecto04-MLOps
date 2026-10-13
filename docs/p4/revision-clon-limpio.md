# Revisión desde clon limpio — checklist P4

## Checklist preparado

Revisor sugerido: Karen o Ale. Registrar una ejecución real en la sección siguiente; ninguna casilla representa PASS hasta hacerla. Usar un directorio nuevo y una identidad AWS autorizada. No borrar datos, volúmenes, modelos o carpetas de otro checkout.

| Paso | Acción desde el clon limpio | Resultado esperado | PASS/FAIL y nota |
|---|---|---|---|
| 1 | Clonar `https://github.com/Andy-752109/Proyecto04-MLOps.git`; registrar `git rev-parse HEAD` y `git status --short` | Commit conocido, árbol limpio | PASS (P4-21): `02c82d5` |
| 2 | Leer [README P4](../../README.md#proyecto-4--clasificación-en-edge), [manual edge](../../edge/README.md) y esta ficha | Dependencias y permisos claros | PASS (P4-21): se siguió solo la sección P4 del README |
| 3 | En Windows 10 con Python 3.12, instalar `python -m pip install -r edge/requirements.txt` | Dependencias disponibles | PASS (P4-21): Python 3.12.10; las dependencias ya estaban en el Python global |
| 4 | Copiar `edge/config.example.yaml` a `edge/config.yaml`; ejecutar `python -m edge cameras` y ajustar `camera.name` | Cámara USB identificada | PASS (P4-21): `GENERAL WEBCAM`; la plantilla ya traía ese nombre |
| 5 | `aws sso login --profile mlops-p3`; recuperar P3 con `app/verify_reload.py` según [README](../../README.md#modelos-p3-e-int8) | Checkpoint verificado | PASS parcial (P4-21): SSO con `--use-device-code`; `verify_reload.py` no se ejecutó (no lo necesita la corrida edge) |
| 6 | Crear `edge/models/`; descargar INT8 por `VersionId` según [registro](../../models/edge_registry.json); `python -m edge status` | `sha256: OK` | PASS (P4-21): `sha256: OK` |
| 7 | Consultar [conversión](../../reports/p4/conversion/conversion_log.json), [calidad](../../reports/p4/quality/metrics_val.json) y [benchmark](../../reports/p4/benchmark/summary.json) | Cifras y rutas localizables | No ejecutado en P4-21; B.9 reprodujo la paridad (20 referencias, 0 diferencias) |
| 8 | `python -m edge run` con gato y perro; revisar `edge/data/captures.jsonl` | Clasificación local y UUID por captura | PASS (P4-21): 5 capturas con clase y UUID |
| 9 | Si el revisor tiene permiso para generar capturas: comprobar offline y `retry --pending` según [guía](integracion-edge-aws.md); si no, inspeccionar evidencia P4-12 | Estado de envío y trazabilidad explicables | PASS (P4-21): `failed` sin WiFi; `retry <capture_id>` → `sent` → `already_sent` |
| 10 | Abrir el [portal AWS](portal-aws.md) `http://18.213.105.30/edge/captures` con las credenciales entregadas por privado y pulsar Actualizar (alternativa de desarrollo: `docker compose up -d --build frontend ml-api` y `http://localhost:8080/edge/captures`) | Portal carga capturas desde AWS | PASS (P4-21): portal AWS, 75 capturas |
| 11 | Comparar un UUID local, evento e imagen S3 y fila del portal; consultar [trazabilidad](../../reports/p4/operation/trazabilidad.md) | Mismo ID y metadatos | PASS (P4-21): `6c21a87e…` igual en laptop, S3 y portal |
| 12 | Registrar cualquier error, corrección, limitación y conclusión; no modificar infraestructura AWS | Revisión reproducible | PASS (P4-21): [registro](../../reports/p4/operation/verificacion-p4-21/README.md) |

## Revisión realmente ejecutada

**Estado:** última ejecución por Ale el 13 de octubre (P4-21, [#50](https://github.com/Andy-752109/Proyecto04-MLOps/issues/50)) sobre `main` con el portal en AWS. La de Karen del 8 de octubre se conserva abajo como antecedente.

### P4-21: 13 de octubre de 2026

| Campo | Registro |
|---|---|
| Fecha y hora (zona) | 13 de octubre de 2026, 15:22:03–16:22:25, UTC−06:00 (hora del centro de México) |
| Persona que revisa | Ale |
| Sistema operativo y versión | Windows 10 (NT 10.0.19045) en `edge-laptop-01`, Python 3.12.10, AWS CLI 2.37.9 |
| Commit probado (`git rev-parse HEAD`) | `02c82d5d0eb099bf1297693e6b2e7563cacb7b51` (`main`, merge del #51) |
| Pasos seguidos y diferencias frente al checklist | Clon nuevo en una carpeta nueva, sin reutilizar `config.yaml`, modelos ni datos del checkout anterior. Pasos 1–6 y 8–12 del checklist y el [guion de demo](guion-demo.md) B.1–B.9 completo; el paso 7 no se ejecutó. Diferencias: SSO con `--use-device-code`; el Python global ya tenía las dependencias |
| Resultado global PASS/FAIL | PASS |
| Errores encontrados y evidencia | En el guion, `--query '{count:KeyCount,...}'` devolvía `"count": null` (B.5 y B.7). Evidencia: [registro y consola](../../reports/p4/operation/verificacion-p4-21/README.md) |
| Correcciones aplicadas o propuestas | Guion corregido en el PR de P4-21: las consultas listan solo `Contents[].Key`. Paso 10 de este checklist actualizado al portal AWS |
| Conclusión y firma/confirmación | Un clon limpio de `main` se prepara siguiendo solo el README y el guion de demo se ejecuta completo con el portal en AWS. Verificación interna, no la demo de evaluación. Firma: Ale |

### P4-13: 8 de octubre de 2026 (antecedente)

| Campo | Registro |
|---|---|
| Fecha y hora (zona) | 8 de octubre de 2026, antes de las 21:06, hora del centro de México (hora de su [registro](https://github.com/Andy-752109/Proyecto04-MLOps/pull/35#issuecomment-6073416756); el commit probado es de las 19:46) |
| Persona que revisa | Karen |
| Sistema operativo y versión | macOS, en la Mac de Karen (no en la laptop edge) |
| Commit probado (`git rev-parse HEAD`) | `26841f1` (rama `p4-13-evidencias-entrega`) |
| Pasos seguidos y diferencias frente al checklist | Clon nuevo del PR, todo levantado desde cero con la caché de Docker vacía y sin `.env` previo |
| Resultado global PASS/FAIL | PASS: funciona |
| Errores encontrados y evidencia | Ninguno reportado |
| Correcciones aplicadas o propuestas | Ninguna |
| Conclusión y firma/confirmación | Karen confirma en el [PR #35](https://github.com/Andy-752109/Proyecto04-MLOps/pull/35#issuecomment-6073416756) que el proyecto se levanta y funciona desde un clon limpio |

La revisión no debe inferirse de los tests automatizados ni de la corrida P4-12: requiere un clon nuevo y un registro de la persona revisora.
