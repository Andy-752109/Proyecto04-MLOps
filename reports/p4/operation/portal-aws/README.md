# Evidencia P4-16: laptop edge → S3 → portal desplegado en AWS

Recorrido completo hasta el portal **desplegado en EC2** (`us-east-1`, ver
[`docs/p4/portal-aws.md`](../../../../docs/p4/portal-aws.md)), en `http://18.213.105.30`.
Repite solo la parte del portal de la validación E2E de P4-12 (que usó `localhost:8080`);
el resto de esa validación sigue en [`../`](../).

## Trazabilidad de 5 `capture_id`

Dos capturas nuevas de `edge-laptop-01` del 11 de octubre (un perro y un gato reales) y tres
de la corrida final de P4-12 (8 de octubre), incluido uno de sus errores reales. Las cinco
están en el log local, en S3 con el evento idéntico y en el portal de AWS con los metadatos
iguales: ver [`trazabilidad.md`](trazabilidad.md).

Las fechas `captured_at` y `LastModified` de S3 están en UTC; `edge.log` usa hora local
UTC−6. Por eso la captura del gato `0ee4aa1b…` figura como **12/10/2026 01:38 UTC** en
el evento y como **11/10/2026 19:38 UTC−6** en el log: es la misma captura.

| `capture_id` | Origen | Real | Predicho |
|---|---|---|---|
| `d9bd109d…` | P4-12, foto 2 | dog | cat 0.778 (error real) |
| `3e60aa20…` | P4-12, foto 5 | dog | dog 0.995 |
| `45730efb…` | P4-12, foto 12 | cat | cat 0.998 |
| `015513c6…` | nueva (#37), perro | dog | dog 0.999 |
| `0ee4aa1b…` | nueva, gato | cat | cat 0.999 |

Las fotos nuevas se mostraron a la cámara en una pantalla, igual que en P4-12. En la misma
sesión del 11 de octubre se tomaron otras tres capturas del mismo perro (`085a6b5a…`,
`c5a3b470…`, `48289c2d…`), que también están en S3 y en el portal; `085a6b5a…` salió `cat`
0.811 y se deja visible como error real del clasificador.

## Archivos

| Archivo | Contenido |
|---|---|
| `captures.jsonl`, `uploads.jsonl`, `edge.log` | Copiados de `edge/data/` de la laptop edge el 11 de octubre (62 capturas, 62 `capture_id` únicos, todas `sent`/`already_sent`); las 57 de P4-12 son idénticas a [`../captures.jsonl`](../captures.jsonl) |
| `ground_truth.csv` | Clase real de los 5 IDs |
| `s3_consulta.txt` | Consulta de lectura en S3 (`list-objects-v2`): evento e imagen de cada ID, con `LastModified` y tamaño |
| `events_s3.json` | Evento de cada ID leído de S3 (`aws s3 cp`) |
| `portal_items.json` | Respuesta del portal de AWS (`GET /ml-api/edge/captures?limit=100`) para los 5 IDs, **sin `image_url`** (URL prefirmada) |
| `trazabilidad.md` | Tabla log local / S3 / portal generada por el script |
| `portal_aws_capturas.png` | Capturas Edge en el navegador, con la URL de AWS visible y las dos capturas nuevas arriba (gato `0ee4aa1b…` y perro `015513c6…`) |

## Cómo se generó

Con el perfil SSO `mlops-p3` (rol `MLOpsP3`) para leer S3 y el usuario del portal en variables de entorno
(la contraseña se entrega por privado y no se escribe en ningún archivo):

```bash
AWS_PROFILE=mlops-p3 PORTAL_USER=evaluador PORTAL_PASSWORD=... \
  python3 scripts/p4_e2e_report.py --out reports/p4/operation/portal-aws \
  --portal http://18.213.105.30 --solo-trazabilidad \
  d9bd109d-784e-426f-acb5-8743e44fef1e 3e60aa20-2b0e-4ed7-bf02-2ff3f2d6f02e \
  45730efb-5920-4eb1-9874-c41369252764 015513c6-2ef3-4000-9e1a-45cdf4358fd5 \
  0ee4aa1b-52d1-49db-8265-e042d954c4e5
```

## Observación: reloj de la laptop

En `015513c6…` el `LastModified` de S3 (23:39:37) es ~2 s **anterior** al `captured_at`
(23:39:38.9), y en el portal "Recibida" aparece antes que "Capturada". En P4-12 el
`LastModified` llegaba 2–10 s **después**, sin desviación apreciable; aquí el reloj de la
laptop edge va adelantado al menos unos 2 s respecto a AWS. No afecta la trazabilidad, que se
hace por `capture_id`.
