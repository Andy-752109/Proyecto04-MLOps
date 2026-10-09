# Validación operativa E2E (P4-12)

Corrida en el dispositivo edge (laptop + cámara USB) que demuestra el recorrido completo:
captura → inferencia local → S3 → portal. Los fallos reales se registran, no se ocultan;
los hallazgos se corrigen en el issue de origen.

- Ensayo: jueves 8 de octubre. Corrida final: viernes 9 (el PM la verifica en vivo).
- Evidencia: `reports/p4/operation/` (ver "Exportaciones").

## Preparación

```bash
git pull
python -m pip install -r edge/requirements.txt
aws sso login --profile mlops-p3                   # renovar antes de la corrida
python -m edge status                              # sha256: OK, bucket, envíos
python -m edge cameras                             # confirmar camera.name
```

Variables y la función `objetos` (0 o exactamente 2 objetos por captura) están en
[`integracion-edge-aws.md`](integracion-edge-aws.md#preparación-en-la-laptop-edge).

Antes de empezar, anota la hora y deja `edge/data/` vacío (o apunta la hora de inicio para
filtrar), para que la exportación contenga solo esta corrida.

## Fotografías

El PM entrega las fotografías de gatos y perros. Anota la etiqueta real de cada una en
`reports/p4/operation/ground_truth.csv` (`capture_id,true_class,foto`) al capturarla, y el
PM confirma que **ninguna pertenece al dataset**.

## Checklist de la corrida

Marca cada paso con la hora y la evidencia (log, captura de pantalla o clip).

| # | Paso | Evidencia (`reports/p4/operation/`) | Resultado |
|---|---|---|---|
| 1 | Dispositivo, modelo, variante y runtime visibles | `paso1_status.txt`, `edge.log` | ✅ `edge-laptop-01`, `1.0.0-int8`, onnxruntime 1.30.0, `sha256: OK` |
| 2 | ≥20 capturas, gatos y perros | `ground_truth.csv`, `events_local.*` | ✅ 20 capturas (10 perros, 10 gatos), 18 aciertos, 2 errores (ver `resumen.md`); 20 `sent` |
| 3 | 5 minutos en modo intervalo | `paso3_consola.txt` | ✅ 5 min 26 s (01:02:34–01:08:00 UTC), 31 capturas, 31 `sent`, sin errores |
| 4 | Red cortada → 3 capturas → red → captura nueva | `paso4_*` | ✅ 3 capturas `failed` (`EndpointConnectionError`, 0 objetos en S3); `retry --pending`: `sent 3`, 2 objetos cada una; captura nueva `79714e50…` `sent` |
| 5 | Falla de envío → reintento del mismo ID → un solo registro | `paso5_*` | ✅ `NoSuchBucket` visible; `retry` → `sent`; segundo `retry` → `already_sent`; siempre 2 objetos |
| 6 | Reinicio → captura nueva | `paso6_*` | ✅ "56 capturas previas", captura `6a8e867f…` `sent`; 57 capturas, 57 `capture_id` únicos |
| 7 | Seguimiento de 5 IDs: log, S3 y portal | `trazabilidad.md`, `paso7_portal_d9bd109d.jpg` | ✅ los 5 IDs coinciden en los tres lugares |

Notas de la corrida:

- Las 57 capturas de la corrida están en S3 (evento igual al local) y en el portal con
  metadatos iguales. S3 y el portal listan 67 porque había 10 capturas de pruebas anteriores
  (5–8 de octubre, de P4-09 y la demo) que no son de esta corrida. Las 12 de la laptop de antes de la
  corrida se apartaron en `edge/data_pre_p4-12`.
- Errores reales del modelo, sin ocultar ni repetir: la foto 2 (pomerania, predicho `cat`
  0.778) y la foto 16 (gato, predicho `dog` 0.910).
- Reloj del dispositivo: `LastModified` de S3 llega 2–10 s después de `captured_at`, sin
  desviación apreciable (primer envío de cada sesión y reintentos tardan más).
- Los pasos 3 a 6 usaron una foto fija de gato; sus capturas no entran en el conteo de acierto
  del paso 2 (`paso4_clase_real.csv` anota la clase real de las del paso 4 en adelante).
- El portal se levantó con `docker compose up -d --build frontend ml-api` (perfil SSO
  `mlops-p3`) y se leyó en `http://localhost:8080/edge/captures`.

## Trazabilidad de 5 IDs

Tabla completa, generada por `scripts/p4_e2e_report.py`, en
[`trazabilidad.md`](../../reports/p4/operation/trazabilidad.md): fotos 2, 5, 9, 12 y 19 (tres
perros y dos gatos, incluido el error de la foto 2). Cada ID tiene 2 objetos en S3, el evento de
S3 es igual al del log local y el portal muestra los mismos metadatos.

## Exportaciones (`reports/p4/operation/`)

- `events_local.csv` / `events_local.json`: `python -m edge export --out reports/p4/operation`
- `events_s3.json`: `LastModified` y contenido del evento de cada captura de la corrida en S3
- `trazabilidad.md` y `resumen.md`: `AWS_PROFILE=mlops-p3 python3 scripts/p4_e2e_report.py <5 capture_id>` (con el portal en :8080)
- `captures.jsonl`, `uploads.jsonl`, `edge.log`, `status_final.txt`: copiados de la laptop edge
- `paso*`: salida de cada paso, con hora

## Criterios de aceptación

- [x] Pasos 1–7 cumplidos, con evidencia por paso (ensayo; la corrida final del viernes la verifica el PM en vivo)
- [x] ≥20 capturas en S3 y en el portal con metadatos coincidentes (57)
- [ ] El PM confirma que las fotografías no pertenecen al dataset
