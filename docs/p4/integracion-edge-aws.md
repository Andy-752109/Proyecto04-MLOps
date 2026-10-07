# Integración edge → AWS: envío, falla y reintento (P4-09)

La app edge (#6) clasifica y guarda cada captura en local; después la sube a S3 con el
uploader de P4-05 (#5), en un hilo de fondo. **La inferencia nunca depende del envío**: sin
red, sin credenciales o con el bucket mal configurado, la captura se clasifica y se guarda
igual, y el envío queda `failed` con el error visible.

## Estados de envío

Cada intento se agrega a `edge/data/uploads.jsonl`, separado del log de inferencia
(`captures.jsonl`):

```json
{"capture_id": "…", "upload_status": "sent", "error": null, "upload_ms": 412.3, "bucket": "mlops-p4-edge-captures-222629887955", "at": "2026-10-08T17:02:11.123+00:00"}
```

| `upload_status` | Significado |
|---|---|
| `pending` | En cola para el hilo de envío (o la app se cerró antes de intentarlo) |
| `sent` | Imagen y evento creados en este intento |
| `already_sent` | El evento ya existía en S3 (HTTP 412): no se sobrescribió nada |
| `failed` | Sin red, sin credenciales, sin permisos o bucket inexistente; `error` dice por qué |

El estado de una captura es el de su último registro; `python -m edge status` los cuenta.
Con `aws.bucket` vacío (o `run --bucket ""`) la app trabaja solo en local: no se envía nada y
`status` muestra los envíos como desactivados en lugar de contarlos como `pending`.
Un `failed` también aparece en consola y en `edge.log`:

```
ERROR envío <capture_id> FALLÓ (<bucket>): NoSuchBucket: The specified bucket does not exist; reintenta con: python -m edge retry <capture_id>
```

## Reintento sin duplicados

`python -m edge retry <capture_id>` (o `--pending` para todas las que no están `sent` ni
`already_sent`) lee el evento de `captures.jsonl` y reenvía **exactamente el mismo evento**,
con el mismo `capture_id` y la misma imagen; nunca se vuelve a inferir. El uploader escribe
con `If-None-Match: *` (imagen → evento), así que en S3 queda **un** evento y **una**
imagen por `capture_id`, aunque se reintente varias veces o la falla haya ocurrido entre la
imagen y el evento.

Fuera de alcance (#9): cola persistente y reenvío automático al reconectar. Lo capturado sin
red queda `failed` hasta que se corre `retry --pending`.

## Preparación en la laptop edge

```bash
git pull
python -m pip install -r edge/requirements.txt     # incluye boto3
aws sso login --profile mlops-p3                   # renovar antes de cada sesión
cp edge/config.example.yaml edge/config.yaml       # o agrega la sección aws: a tu config
python -m edge status                              # sha256: OK, bucket y envíos
```

El nombre del perfil es libre (en `aws.profile`); debe asumir el rol `MLOpsP3` en la cuenta
`222629887955`. Variables para los comandos de abajo:

```bash
export AWS_PROFILE=mlops-p3
export BUCKET=mlops-p4-edge-captures-222629887955
objetos() {  # objetos de S3 de una captura: debe haber 0 o exactamente 2
  aws s3api list-objects-v2 --bucket "$BUCKET" --prefix edge-captures/v1/ \
    --query "Contents[?contains(Key, '$1')].[Key,Size,LastModified]" --output table
}
```

## 1. Captura del dispositivo en S3 y en el portal

```bash
python -m edge run --mode interval --count 1      # consola: "envío <id> sent"
objetos <capture_id>                              # events/<id>.json e images/<id>.jpg
```

El evento en S3 es el mismo que el del log local (mismos metadatos):

```bash
aws s3 cp "s3://$BUCKET/edge-captures/v1/events/<capture_id>.json" evento-s3.json
python -c "import json,sys; s=json.load(open('evento-s3.json')); l=[json.loads(x)['event'] for x in open('edge/data/captures.jsonl') if json.loads(x)['event']['capture_id']==s['capture_id']][0]; print('iguales' if s==l else 'DISTINTOS')"
```

Luego abre **Capturas Edge** en el portal (local, leyendo AWS) y toma captura de pantalla con
el mismo `capture_id` (hito del #9).

## 2. Falla provocada → error visible → reintento → un solo registro

**Opción A, bucket inexistente por configuración** (reversible: solo afecta esa corrida):

```bash
python -m edge run --mode interval --count 1 --bucket mlops-p4-edge-captures-no-existe-222629887955
# consola y edge.log: "envío <id> FALLÓ (...): NoSuchBucket: ..."
python -m edge status                             # envíos: ... failed 1
objetos <capture_id>                              # ANTES: no hay objetos de ese id
python -m edge retry <capture_id>                 # "<id> sent"
objetos <capture_id>                              # DESPUÉS: exactamente un evento y una imagen
```

**Opción B, red cortada** (también sirve para el paso 4 de #12):

```bash
# apagar el WiFi
python -m edge run --mode interval --count 3      # se clasifican las 3; cada envío falla
                                                  # (EndpointConnectionError, ~10 s, sin frenar)
# encender el WiFi
python -m edge retry --pending                    # resumen: sent 3
```

Si la sesión SSO expira a mitad de la prueba, el envío queda `failed` con el error del token
(error visible); se renueva con `aws sso login --profile mlops-p3` y se corre
`python -m edge retry --pending`.

## 3. Reintentar algo ya enviado

```bash
python -m edge retry <capture_id>                 # "<id> already_sent"
objetos <capture_id>                              # sigue habiendo un evento y una imagen,
                                                  # con el LastModified del primer envío
```

## 4. Exportación

```bash
python -m edge export --out reports/p4/evidence/integracion-edge-aws
```

- `events_local.json`: `{"event", "image_path", "crop_path", "upload"}` por captura; el
  evento es el mismo que se sube a S3.
- `events_local.csv`: una fila por captura con el evento aplanado y `upload_status`,
  `upload_error`, `upload_ms`, `upload_attempts`, `upload_bucket` y `upload_last_at`.

`upload_attempts` cuenta solo los intentos reales de subida (`sent`, `already_sent` o
`failed`); el registro `pending` que se escribe al encolar no es un intento. Por ejemplo,
`pending` → `failed` → `sent` son 2 intentos.

## Evidencia para #9

En `reports/p4/evidence/integracion-edge-aws/` (sin credenciales ni URLs prefirmadas):

1. Fragmento de `edge.log` y `uploads.jsonl` con los estados (`pending`, `failed`, `sent`,
   `already_sent`) del `capture_id` de la prueba.
2. Salida de `objetos <capture_id>` antes y después del reintento.
3. `events_local.csv` y `events_local.json` de `python -m edge export`.
4. Captura de pantalla de Capturas Edge con el mismo `capture_id`.
