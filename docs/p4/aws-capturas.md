# Persistencia de capturas edge en AWS (P4-05)

| Dato | Valor |
|---|---|
| Cuenta | `222629887955` |
| Bucket | `mlops-p4-edge-captures-222629887955` (`us-east-1`), separado del de modelos |
| Acceso | SSO con el permission set `MLOpsP3` (Plan A); sin llaves de larga duración |
| Prefijo | `edge-captures/v1/` |

Las claves y la semántica vienen del [contrato v1](contrato-evento-edge.md).

## Creación del bucket y permisos

Pendiente: los documenta Karen (dueña de la cuenta) al crear el bucket.

## Perfil SSO local

```bash
aws configure sso --profile mlops-p4     # cuenta 222629887955, rol MLOpsP3, us-east-1
aws sso login --profile mlops-p4
aws sts get-caller-identity --profile mlops-p4 --query Account --output text
```

`aws configure sso` es interactivo: en Windows corre en PowerShell o cmd, no en Git Bash.

## Uploader

`edge/uploader.py` sube una captura en dos escrituras condicionales (`If-None-Match: *`):

1. imagen → `edge-captures/v1/images/{capture_id}.jpg` (`image/jpeg`);
2. evento → `edge-captures/v1/events/{capture_id}.json` (`application/json`).

| Resultado | Cuándo |
|---|---|
| `sent` | El evento se creó en esta llamada |
| `already_sent` | El evento ya existía (HTTP 412): la captura ya estaba completa |
| `failed` | Evento inválido, imagen ilegible, sin red, sin credenciales, sin permisos, etc. Va con `error` |

Siempre devuelve `upload_ms`, que solo va al log local, nunca al evento.

- **Reintentos:** si se cae entre la imagen y el evento, el reintento recibe 412 en la
  imagen (no la sobrescribe) y crea el evento. Como el evento va al final, en S3 nunca
  queda un evento sin su imagen.
- **Contenido idéntico:** el evento se serializa en JSON canónico (llaves ordenadas), así
  un reintento manda exactamente los mismos bytes.
- **Credenciales:** solo la cadena por defecto de boto3 (perfil SSO o variables de
  `aws configure export-credentials`). Los errores se reducen a código y mensaje de AWS.
- **Sin red:** timeouts de 5 s de conexión y 15 s de lectura, con 2 intentos. Falla rápido y
  el reintento queda para P4-09.

Uso desde Python:

```python
from edge.uploader import Uploader, make_s3_client

uploader = Uploader("mlops-p4-edge-captures-222629887955", make_s3_client("mlops-p4"))
result = uploader.upload(event, image_path)   # UploadResult(status, capture_id, upload_ms, error)
```

## Prueba de doble envío

Se sube dos veces el evento de ejemplo. Lo esperado es `sent` y luego `already_sent`, con
un solo objeto de cada tipo:

```bash
for i in 1 2; do
  python -m edge.uploader --bucket mlops-p4-edge-captures-222629887955 --profile mlops-p4 \
    --event contracts/examples/valid-cat-no-crop.json --image contracts/examples/test-capture.jpg
done
```

## Comandos de solo lectura (evaluador)

```bash
BUCKET=mlops-p4-edge-captures-222629887955
ID=8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65    # capture_id del ejemplo

aws s3api list-objects-v2 --bucket $BUCKET --prefix edge-captures/v1/ \
  --query 'Contents[].{Key:Key,Size:Size,LastModified:LastModified}' --output table
aws s3api head-object --bucket $BUCKET --key edge-captures/v1/images/$ID.jpg
aws s3api head-object --bucket $BUCKET --key edge-captures/v1/events/$ID.json
aws s3 cp s3://$BUCKET/edge-captures/v1/events/$ID.json -    # contenido del evento
```

`LastModified` del objeto de evento es el `received_at` que deriva el lector (P4-07).

## Tests

```bash
python -m pip install -r edge/requirements.txt
python -m unittest discover -s edge/tests -v
```

`edge/tests/test_uploader.py` no usa AWS:

- `botocore.stub.Stubber` comprueba los parámetros exactos de cada `put_object` (orden,
  claves, `ContentType` e `IfNoneMatch`) y el manejo de 412 y 403;
- un S3 en memoria con la semántica de `If-None-Match` prueba el doble envío, la caída
  antes de la imagen, la caída entre imagen y evento, y la falta de credenciales.
