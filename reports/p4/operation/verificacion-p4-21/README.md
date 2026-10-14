# Verificación P4-21: clon limpio y guion de demo en la laptop edge

Corrida **interna** del 13 de octubre de 2026 ([#50](https://github.com/Andy-752109/Proyecto04-MLOps/issues/50)).
**No es la demo de evaluación**: las fotos las eligió el equipo y nadie externo la observó.
Comprueba que un clon nuevo de `main` (`02c82d5`) funciona siguiendo solo el README y que el
[guion de demo](../../../../docs/p4/guion-demo.md) se puede ejecutar completo, B.1–B.9, con el
portal en AWS.

| Archivo | Contenido |
|---|---|
| [`consola.txt`](consola.txt) | Salidas de las dos terminales de PowerShell, en orden |
| [`portal_aws.png`](portal_aws.png) | Capturas Edge en `http://18.213.105.30` con la captura `6c21a87e…` arriba |

## Entorno

- `edge-laptop-01`, Windows 10 (NT 10.0.19045), Python 3.12.10, AWS CLI 2.37.9, cámara USB
  `GENERAL WEBCAM`.
- Clon nuevo en `C:\Users\karen\p4-21\Proyecto04-MLOps`; no se reutilizaron `edge/config.yaml`,
  `edge/models/` ni `edge/data/` del checkout anterior de la laptop.
- Se ejecutó `python -m pip install -r edge/requirements.txt` sin errores, pero el Python global
  de la laptop ya tenía las dependencias de corridas anteriores (`pip` respondió "already
  satisfied"). El README no pide un entorno virtual; el clon es nuevo, el intérprete no, así que
  **no se comprobó una instalación de dependencias desde cero**.
- SSO con el perfil `mlops-p3` (rol `MLOpsP3`), iniciado con `--use-device-code` y aprobado desde
  otra computadora.

## Tiempos (hora local, UTC−06:00)

| Tramo | Inicio | Fin | Duración |
|---|---|---|---|
| Parte A: clon y preparación (README) | 15:22:03 | antes de 15:48:54 | < 27 min |
| Parte B: guion B.1–B.9 | 15:48:54 | 16:22:25 | 33 min 31 s |
| Total | 15:22:03 | 16:22:25 | 60 min 22 s |

La parte B incluye la primera instalación de DVC con `uvx` (27 s) y el `dvc pull` de los
recortes (~33 s); sin eso, B.1–B.8 tomaron unos 22 minutos.

## Capturas generadas

| # | `capture_id` | Paso | Clase y confianza | Envío |
|---|---|---|---|---|
| 1 | `dda5c747-c802-4eb8-99a6-7338f5d702ea` | B.2 | cat 0.951 | `sent` |
| 2 | `e80a5695-d8d4-4aaa-b612-d82d860ab701` | B.2 | dog 0.938 | `sent` |
| 3 | `52ae33a2-7c24-487c-abb5-11ae9bce9557` | B.3, sin WiFi | dog 1.000 | `failed` → `sent` → `already_sent` (B.7) |
| 4 | `6c21a87e-d5f1-451f-9a2f-aeb2ae34ab96` | B.4, WiFi de vuelta | cat 0.941 | `sent`; trazada en B.5–B.6 |
| 5 | `9c584721-b375-4a5b-8a61-f86c7378d245` | B.8, tras reiniciar | dog 0.999 | `sent` |

Cinco fotos distintas de gatos y perros, una por captura. No se anotó la clase real de cada foto
por separado: esta corrida comprueba el recorrido, no la exactitud (eso está en la
[comparación de calidad](../../../../docs/p4/comparacion-calidad.md)).

## Resultado por paso

| Paso | Resultado | Lo observado |
|---|---|---|
| B.1 | PASS | `cameras` lista `GENERAL WEBCAM`; `status` con `1.0.0-int8`, `onnxruntime` y `sha256: OK`; `run` verifica el SHA `dffa9cf2…0392` al arrancar |
| B.2 | PASS | Gato y perro con clase, confianza y UUID; envío `sent` en 1–3 s |
| B.3 | PASS | Sin WiFi la captura 3 se clasificó; el envío falló en 1.4 s (`EndpointConnectionError`); `status`: `failed 1` |
| B.4 | PASS | Con WiFi, la captura 4 llegó como `sent` en 1.1 s; la 3 siguió `failed` (no hay reenvío automático) |
| B.5 | PASS | En S3, una clave de imagen y una de evento para `6c21a87e…`; el evento de S3 es igual al de `captures.jsonl` (hora, clase, probabilidades, SHA, recorte, `image_key`) |
| B.6 | PASS | Tras Actualizar, 75 capturas válidas; `6c21a87e…` primera, cat 94.1 %, capturada 21:54:11.985144 UTC, `edge-laptop-01`, `1.0.0-int8` |
| B.7 | PASS | `python -m edge retry 52ae33a2…` → `sent`; repetido → `already_sent` (no se ejecutó la variante `retry --pending`). `uploads.jsonl`: `pending → failed → sent → already_sent`. S3: exactamente 2 objetos (imagen y evento) |
| B.8 | PASS | Al reiniciar, "4 capturas previas"; captura 5 `sent`; `status`: 5 capturas, `sent 4`, `already_sent 1`, `failed 0` |
| B.9 | PASS | `dvc pull -r prod crops` sin cambios en `dvc.lock`; `parity_model`: 20 referencias, 0 diferencias de clase, máxima diferencia de probabilidad 0.0701, igual que la [conversión](../../conversion/conversion_log.json) (0 diferencias) |

## Incidencias

- **Guion, B.5:** con AWS CLI 2.37.9, `--query '{count:KeyCount,...}'` devolvió `"count": null`,
  aunque `Contents[].Key` sí mostró la clave esperada. Es lo observado en esta corrida, no una
  regla general de S3. El guion ahora lista solo `Contents[].Key` y compara las claves completas
  de imagen y evento con las del mismo `capture_id`; si devuelve `[]` o `null`, ese objeto no
  quedó verificado. En B.7 ya se usó esa consulta.
- **README, SSO:** se usó `aws sso login --profile mlops-p3 --use-device-code` porque el segundo
  factor de la cuenta no está en la laptop edge. Con el comando del README tal cual se abre el
  navegador de la laptop; no es un error del README.
- Ningún paso necesitó cambios de código.
