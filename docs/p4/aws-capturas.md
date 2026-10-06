# Persistencia de capturas edge en AWS (P4-05)

Bucket S3 privado y separado del de modelos donde el uploader edge guarda cada
captura: la imagen en `edge-captures/v1/images/{capture_id}.jpg` y el evento en
`edge-captures/v1/events/{capture_id}.json` (ver
[`contrato-evento-edge.md`](contrato-evento-edge.md)). Se crea con la AWS CLI, sin
Terraform.

| Propiedad | Valor |
|---|---|
| Bucket | `mlops-p4-edge-captures-222629887955` |
| Región | `us-east-1` |
| Acceso público | bloqueado (las 4 opciones) |
| Cifrado | SSE-S3 (`AES256`) con bucket key |
| Versionado | habilitado |
| Transporte | solo HTTPS (política `Deny` con `aws:SecureTransport=false`) |
| Permisos `MLOpsP3` | `s3:PutObject`, `s3:GetObject` sobre `edge-captures/*` y `s3:ListBucket` con prefijo `edge-captures/*`; **sin** `DeleteObject` |

Ninguna credencial de larga duración: el acceso es por SSO (perfil `mlops-p3`).

## Quién ejecuta qué

Crear el bucket y fijar su política requiere un perfil con permisos de administración
sobre S3 en la cuenta (no `MLOpsP3`, que es de solo uso). Lo ejecuta la dueña de la
cuenta una sola vez. Los demás solo usan los comandos de verificación.

```bash
aws sso login --profile <perfil-admin>   # sesión SSO vigente
export AWS_PROFILE=<perfil-admin>
export AWS_REGION=us-east-1
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export BUCKET="mlops-p4-edge-captures-${ACCOUNT_ID}"
```

## 1. Crear el bucket

`us-east-1` no admite `LocationConstraint`.

```bash
aws s3api create-bucket --bucket "$BUCKET" --region us-east-1
```

## 2. Bloquear acceso público

```bash
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

## 3. Cifrado SSE-S3

```bash
aws s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
      "BucketKeyEnabled": true
    }]
  }'
```

## 4. Versionado

```bash
aws s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled
```

## 5. Política del bucket: solo HTTPS y permisos de MLOpsP3

El rol que IAM Identity Center crea para el permission set `MLOpsP3` se llama
`AWSReservedSSO_MLOpsP3_<sufijo>` y vive bajo la ruta
`/aws-reserved/sso.amazonaws.com/`. Como el sufijo cambia, la política usa `Principal: "*"` acotado por `aws:PrincipalArn`
(solo ese rol, de esta cuenta). No se usa `arn:aws:iam::<cuenta>:root` como principal:
eso solo *delega* a IAM y exigiría además una política de identidad en el permission set,
que es justo lo que se evita aquí. Al ser la misma cuenta, no hace falta modificar el
permission set.

```bash
cat > /tmp/p4-edge-captures-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ],
      "Condition": {"Bool": {"aws:SecureTransport": "false"}}
    },
    {
      "Sid": "MLOpsP3ReadWriteCaptures",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/edge-captures/*",
      "Condition": {
        "ArnLike": {
          "aws:PrincipalArn": "arn:aws:iam::${ACCOUNT_ID}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_MLOpsP3_*"
        }
      }
    },
    {
      "Sid": "MLOpsP3ListCaptures",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${BUCKET}",
      "Condition": {
        "StringLike": {"s3:prefix": "edge-captures/*"},
        "ArnLike": {
          "aws:PrincipalArn": "arn:aws:iam::${ACCOUNT_ID}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_MLOpsP3_*"
        }
      }
    }
  ]
}
EOF
aws s3api put-bucket-policy --bucket "$BUCKET" --policy file:///tmp/p4-edge-captures-policy.json
rm /tmp/p4-edge-captures-policy.json
```

`DeleteObject` no se concede. Si el permission set `MLOpsP3` ya tuviera un `s3:*`
amplio sobre este bucket, habría que acotarlo en Identity Center.

## 6. Verificación (evidencia para #5)

Los de configuración con el perfil admin:

```bash
aws s3api get-public-access-block --bucket "$BUCKET"
aws s3api get-bucket-policy-status --bucket "$BUCKET"
aws s3api get-bucket-versioning --bucket "$BUCKET"
aws s3api get-bucket-encryption --bucket "$BUCKET"
```

Esperado: las cuatro opciones de bloqueo en `true`, `IsPublic: false`,
`Status: Enabled` y `SSEAlgorithm: AES256`.

Acceso de `MLOpsP3` (cualquier dev):

```bash
aws sso login --profile mlops-p3
export AWS_PROFILE=mlops-p3
export BUCKET=mlops-p4-edge-captures-222629887955

echo prueba > /tmp/acceso.txt
aws s3api put-object --bucket "$BUCKET" --key edge-captures/v1/acceso-prueba.txt --body /tmp/acceso.txt
aws s3api get-object --bucket "$BUCKET" --key edge-captures/v1/acceso-prueba.txt /tmp/acceso-leido.txt
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix edge-captures/
rm /tmp/acceso.txt /tmp/acceso-leido.txt

# Debe fallar con AccessDenied (sin DeleteObject y fuera del prefijo):
aws s3api delete-object --bucket "$BUCKET" --key edge-captures/v1/acceso-prueba.txt
aws s3api put-object --bucket "$BUCKET" --key otro-prefijo/x.txt --body /dev/null
```

Con el versionado activo, el objeto de prueba queda en el bucket (no se puede borrar con
`MLOpsP3`); es inocuo y está fuera de `events/` e `images/`.

## 7. Uploader (`edge/uploader.py`)

Sube una captura en dos escrituras condicionales (`If-None-Match: *`): primero la imagen
(`image/jpeg`) y después el evento (`application/json`).

| Resultado | Cuándo |
|---|---|
| `sent` | El evento se creó en esta llamada |
| `already_sent` | El evento ya existía (HTTP 412): la captura ya estaba completa |
| `failed` | Evento inválido, imagen ilegible, sin red, sin credenciales o sin permisos; va con `error` |

Siempre devuelve `upload_ms`, que va solo al log local y nunca al evento.

- **Reintentos:** si se cae entre la imagen y el evento, el reintento recibe 412 en la imagen
  (no la sobrescribe) y crea el evento. Como el evento va al final, en S3 nunca queda un
  evento sin su imagen.
- **Contenido idéntico:** el evento se serializa en JSON canónico (llaves ordenadas): un
  reintento manda los mismos bytes.
- **Credenciales:** solo la cadena por defecto de boto3 (perfil SSO o credenciales
  temporales de `aws configure export-credentials`). Los errores se reducen a código y
  mensaje de AWS.
- **Sin red:** timeouts de 5 s (conexión) y 15 s (lectura), con 2 intentos. Falla rápido; el
  reintento es de P4-09.

```python
from edge.uploader import Uploader, make_s3_client

uploader = Uploader("mlops-p4-edge-captures-222629887955", make_s3_client("mlops-p3"))
result = uploader.upload(event, image_path)   # UploadResult(status, capture_id, upload_ms, error)
```

### Prueba de doble envío

Esperado: un solo objeto de cada tipo, y el segundo envío `already_sent`.

```bash
for i in 1 2; do
  python -m edge.uploader --bucket mlops-p4-edge-captures-222629887955 --profile mlops-p3     --event contracts/examples/valid-cat-no-crop.json --image contracts/examples/test-capture.jpg
done
```

### Tests (sin AWS)

```bash
python -m unittest discover -s edge/tests -v
```

`edge/tests/test_uploader.py` usa `botocore.stub.Stubber` para los parámetros exactos de
cada `put_object` (orden, claves, `ContentType`, `IfNoneMatch`) y para el manejo de 412 y
403. Un S3 en memoria con la semántica de `If-None-Match` prueba el doble envío, la caída
antes de la imagen, la caída entre imagen y evento, y la falta de credenciales.

## 8. Comandos de solo lectura para el evaluador

```bash
export AWS_PROFILE=mlops-p3
export BUCKET=mlops-p4-edge-captures-222629887955

aws s3api list-objects-v2 --bucket "$BUCKET" --prefix edge-captures/v1/events/
aws s3api head-object --bucket "$BUCKET" --key edge-captures/v1/events/<capture_id>.json
aws s3api head-object --bucket "$BUCKET" --key edge-captures/v1/images/<capture_id>.jpg
```

## 9. Configuración en `ml-api`

`ml-api` (P4-07, PR #18) lee el bucket desde `EDGE_CAPTURES_BUCKET`, con el perfil
`mlops-p3`. Su valor por defecto en `docker-compose.yml` es este bucket. No publicar
URLs prefirmadas completas en la evidencia.
