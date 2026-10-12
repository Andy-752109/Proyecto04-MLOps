# Portal en AWS (P4-14 y P4-15)

El profesor pidió que Capturas Edge esté **desplegada en AWS** (rúbrica 5.1, requisito M3).
Este documento describe dónde corre el portal: la **infraestructura** (P4-14) y el
**despliegue** del portal sobre ella (P4-15, ver [Despliegue](#despliegue-p4-15)).

## Infraestructura

Una sola EC2 en la **VPC por defecto** de `us-east-1`, en la misma cuenta que los
buckets de P3 y P4. El Terraform de P2 (`terraform/`) nunca se aplicó, así que no hay
VPC propia. Todo se crea con la AWS CLI, sin Terraform, con
[`scripts/aws/portal-infra.sh`](../../scripts/aws/portal-infra.sh) (idempotente: lo que
ya existe se reutiliza).

| Propiedad | Valor |
|---|---|
| Cuenta / región | `222629887955` / `us-east-1` |
| Instancia | `i-0c333de1158e247e9` (`Name=mlops-p4-portal`) |
| Tipo | `t3.medium` (2 vCPU, 4 GB; la `t3.micro` no alcanza para construir y correr el portal) |
| AMI | Amazon Linux 2023 (`al2023-ami-kernel-default-x86_64`, la vigente al crearla) |
| VPC / subred | `vpc-0cbd5153e723c37d7` (por defecto) / `subnet-0e651975c82b3a66d` (pública) |
| Disco | 30 GB gp3, cifrado (clave de EBS por defecto), se borra con la instancia |
| Metadatos | IMDSv2 obligatorio (`HttpTokens=required`), salto 2 (los contenedores alcanzan el rol) |
| IP | pública, con una Elastic IP (`Name=mlops-p4-portal`) para que la URL no cambie |
| Rol de instancia | `mlops-p4-portal-role` |
| Security group | `mlops-p4-portal` (`sg-04a2b1b39342ca35e`) |
| Software | Docker 25, Docker Compose v2.29.7, buildx v0.17.1 y Git, instalados por [`portal-user-data.sh`](../../scripts/aws/portal-user-data.sh) (log en `/var/log/portal-bootstrap.log`) |

No hay claves SSH ni par de llaves en la instancia: se administra con **Session Manager**.

### Rol de instancia: solo lectura a S3

El rol `mlops-p4-portal-role` confía en `ec2.amazonaws.com` y tiene dos políticas:

- `AmazonSSMManagedInstanceCore` (gestionada por AWS), solo para Session Manager.
- `s3-solo-lectura` (en línea), de
  [`scripts/aws/portal-role-policy.json`](../../scripts/aws/portal-role-policy.json):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListCapturasEdge",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::mlops-p4-edge-captures-222629887955"
    },
    {
      "Sid": "ReadCapturasEdge",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::mlops-p4-edge-captures-222629887955/*"
    },
    {
      "Sid": "ListModelosP3",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::mlops-p3-models-222629887955"
    },
    {
      "Sid": "ReadModelosP3",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:GetObjectVersion"],
      "Resource": "arn:aws:s3:::mlops-p3-models-222629887955/*"
    }
  ]
}
```

Sin permisos de escritura sobre S3 ni de IAM. El acceso al bucket de capturas funciona
sin tocar su política: al ser la misma cuenta basta con la política de identidad del rol
(la política del bucket solo tiene el `Deny` de HTTP sin TLS y los permisos de
`MLOpsP3`). `s3:GetObjectVersion` está en el bucket de modelos porque el registro de
P3 lee cada paquete por su `VersionId` exacto (ver `app/storage/model_store.py`).

> Las URL prefirmadas que genere el portal con este rol caducan junto con las
> credenciales temporales del rol (hasta unas 6 h), aunque se pida un plazo mayor.

### Security group

**Puerto 80 abierto, acceso con basic auth (decisión de PM, ver #36).** Para que el
profesor no tenga que pasar su IP, el portal queda abierto a cualquier IP y protegido con
usuario y contraseña en nginx (ver [Basic auth](#basic-auth)). El puerto se abrió solo
después de comprobar que sin contraseña responde 401.

| Dirección | Puerto | Origen |
|---|---|---|
| Entrada | TCP 80 | `0.0.0.0/0` (`publico-basic-auth`), con basic auth obligatorio |
| Entrada | TCP 80 | reglas `/32` del equipo (opcionales; redundantes con la pública) |
| Entrada | 22 (SSH) | **cerrado**: no hay regla; se usa Session Manager |
| Salida | todo | (necesario para jalar imágenes de Docker, S3 y SSM) |

Los demás puertos del portal (3100, 3306, 5050, 8080, 8100, 9000, 9001) no tienen regla y
además Compose no los publica en AWS. La regla pública se agregó con la CLI, porque
`portal-allow-ip.sh` rechaza `0.0.0.0/0` a propósito:

```bash
aws ec2 authorize-security-group-ingress --group-id sg-04a2b1b39342ca35e   --ip-permissions 'IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description=publico-basic-auth}]'
```

Para volver a cerrarlo a IP concretas: `aws ec2 revoke-security-group-ingress --group-id
sg-04a2b1b39342ca35e --protocol tcp --port 80 --cidr 0.0.0.0/0`, y dar de alta las IP con
el script. Las IP no se guardan en el repo: cada regla `/32` lleva como descripción el nombre
de su dueño (`karen`, `profe`, …) y se administran con
[`scripts/aws/portal-allow-ip.sh`](../../scripts/aws/portal-allow-ip.sh).

#### Las IP no son fijas

Solo aplica si se vuelve a cerrar el puerto a IP concretas. Las IP de casa, de datos móviles y de muchas redes cambian al reiniciar el módem, al cambiar
de red o con el tiempo. Si el portal deja de responder a alguien, casi siempre es que su IP
cambió. Cada quien consulta la suya con `curl -s https://checkip.amazonaws.com` y quien
tenga el permission set `MLOpsP4Portal` (ver [Acceso del equipo](#acceso-del-equipo)) la
actualiza; el rol `MLOpsP3` no puede modificar el security group.

- **Equipo:** al cambiar la IP, se reemplaza la del dueño; el script quita la anterior.
- **Profesor:** si entra desde la red del ITESO, abrir su IP o rango de salida (pedirlo a
  TI; mínimo `/24`). Si entra desde otra red, agregar su IP el día de la revisión y
  quitarla después.

Se ejecuta con un perfil con `MLOpsP4Portal` (o administrador); `MLOpsP3` no sirve:

```bash
export AWS_PROFILE=<perfil-admin>
bash scripts/aws/portal-allow-ip.sh karen                    # tu IP actual
bash scripts/aws/portal-allow-ip.sh profe 203.0.113.10/32    # IP o rango dado (ejemplo)
bash scripts/aws/portal-allow-ip.sh --remove profe           # quita todas las de ese dueño
bash scripts/aws/portal-allow-ip.sh --list                   # dueño y CIDR vigentes
```

Agregar una IP a un dueño que ya tenía otra **reemplaza** la anterior. Un dueño con varias
IP a la vez (por ejemplo, el profesor desde dos redes) no cabe en ese esquema: usa un
dueño distinto para cada una (`profe-iteso`, `profe-casa`).

### Costos y presupuesto

Precios de lista de `us-east-1` (confirmar en la consola de facturación):

| Concepto | Encendida 24/7 | Apagada |
|---|---|---|
| `t3.medium` (USD 0.0416/h) | ~USD 30/mes | 0 |
| EBS gp3 30 GB | ~USD 2.4/mes | ~USD 2.4/mes |
| IPv4 pública / Elastic IP (USD 0.005/h) | ~USD 3.7/mes | ~USD 3.7/mes |
| **Total** | **~USD 36/mes** | **~USD 6/mes** |

Por eso **la máquina se apaga cuando no se usa**. Cada hora encendida cuesta ~USD 0.04:
con los USD 10 del presupuesto alcanzan unas 95 h de uso al mes además de lo fijo.

AWS Budgets: `mlops-p4-portal-mensual`, USD 10 al mes, con aviso por correo al 80 % del
gasto real y cuando el pronóstico supere el 100 %. El aviso llega solo a la dueña de la
cuenta (y a quien ella añada del equipo); el profesor **no** recibe correos de facturación.
Se crea con `BUDGET_EMAIL` (varios correos separados por coma).

### Acceso del equipo

El acceso a la EC2 se da con un permission set propio de IAM Identity Center, para no ampliar
los permisos de todo el equipo (`MLOpsP3` no se modifica).

| Campo | Valor |
|---|---|
| Permission set | `MLOpsP4Portal` |
| Cuenta | `222629887955` |
| Asignado a | Ale (revisión y despliegue del portal, ticket #37) y Karen (uso diario, en lugar del usuario administrador) |
| Alcance | solo la instancia `i-0c333de1158e247e9` y el security group `sg-04a2b1b39342ca35e` |

Con él se puede entrar a la EC2 por Session Manager, ejecutar comandos, prenderla, apagarla y
reiniciarla, y agregar o quitar IP con `portal-allow-ip.sh`. No permite nada de IAM, no escribe
en S3 ni crea recursos. La política en línea vive en
[`scripts/aws/portal-permission-set-policy.json`](../../scripts/aws/portal-permission-set-policy.json):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Consultar",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances", "ec2:DescribeInstanceStatus",
        "ec2:DescribeSecurityGroups", "ec2:DescribeSecurityGroupRules",
        "ec2:DescribeAddresses",
        "ssm:DescribeInstanceInformation", "ssm:DescribeSessions",
        "ssm:GetConnectionStatus", "ssm:ListCommands",
        "ssm:ListCommandInvocations", "ssm:GetCommandInvocation"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PrenderApagarPortal",
      "Effect": "Allow",
      "Action": ["ec2:StartInstances", "ec2:StopInstances", "ec2:RebootInstances"],
      "Resource": "arn:aws:ec2:us-east-1:222629887955:instance/i-0c333de1158e247e9"
    },
    {
      "Sid": "SessionManagerYComandos",
      "Effect": "Allow",
      "Action": ["ssm:StartSession", "ssm:SendCommand"],
      "Resource": [
        "arn:aws:ec2:us-east-1:222629887955:instance/i-0c333de1158e247e9",
        "arn:aws:ssm:us-east-1:222629887955:document/SSM-SessionManagerRunShell",
        "arn:aws:ssm:us-east-1::document/AWS-RunShellScript"
      ]
    },
    {
      "Sid": "CerrarSesiones",
      "Effect": "Allow",
      "Action": ["ssm:TerminateSession", "ssm:ResumeSession"],
      "Resource": "*"
    },
    {
      "Sid": "IPsDelPortal",
      "Effect": "Allow",
      "Action": [
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:UpdateSecurityGroupRuleDescriptionsIngress"
      ],
      "Resource": "arn:aws:ec2:us-east-1:222629887955:security-group/sg-04a2b1b39342ca35e"
    }
  ]
}
```

### Encender y apagar

```bash
export AWS_REGION=us-east-1
ID=$(aws ec2 describe-instances \
  --filters Name=tag:Name,Values=mlops-p4-portal Name=instance-state-name,Values=running,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Encender (la Elastic IP se conserva; Docker y los contenedores del portal arrancan solos)
aws ec2 start-instances --instance-ids "$ID" && aws ec2 wait instance-running --instance-ids "$ID"
aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text

# Apagar (deja de cobrar cómputo; el disco y la IP siguen)
aws ec2 stop-instances --instance-ids "$ID" && aws ec2 wait instance-stopped --instance-ids "$ID"
```

Para entrar a la máquina (requiere el
[plugin de Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)):

```bash
aws ssm start-session --target "$ID"
```

Para ejecutar un comando suelto sin el plugin:

```bash
aws ssm send-command --instance-ids "$ID" --document-name AWS-RunShellScript \
  --parameters 'commands=["docker ps"]' --query Command.CommandId --output text
aws ssm get-command-invocation --command-id <CommandId> --instance-id "$ID" \
  --query StandardOutputContent --output text
```

Para eliminar todo (rol, security group, Elastic IP, instancia y presupuesto) hay que
hacerlo a mano; el script solo crea.

### Verificación (evidencia de P4-14)

Ejecutada **dentro de la instancia** vía Session Manager, sin SSO ni claves:

```text
$ aws sts get-caller-identity
{
    "UserId": "AROATHVOGP7J7OLAKNEHY:i-0c333de1158e247e9",
    "Account": "222629887955",
    "Arn": "arn:aws:sts::222629887955:assumed-role/mlops-p4-portal-role/i-0c333de1158e247e9"
}

$ aws s3 ls s3://mlops-p4-edge-captures-222629887955/edge-captures/v1/events/ | head -3
2026-10-09 01:04:42        653 012f9c78-3f17-4a68-9491-4f5b99df62f6.json
2026-10-09 00:55:32        651 03b63de6-3cd3-4a62-934f-07b8c9c33d13.json
2026-10-09 01:05:31        651 06ffb09b-eb85-4c3d-bf58-91defc18273d.json

$ aws s3 ls s3://mlops-p3-models-222629887955/models/
                           PRE edge/
                           PRE releases/

$ aws s3 cp /tmp/x.txt s3://mlops-p4-edge-captures-222629887955/edge-captures/v1/acceso-ec2.txt
upload failed: ... An error occurred (AccessDenied) when calling the PutObject operation:
User: arn:aws:sts::222629887955:assumed-role/mlops-p4-portal-role/i-0c333de1158e247e9 is not
authorized to perform: s3:PutObject on resource: ".../acceso-ec2.txt" because no
identity-based policy allows the s3:PutObject action

$ aws s3 cp /tmp/x.txt s3://mlops-p3-models-222629887955/acceso-ec2.txt     # AccessDenied (igual)
$ aws iam list-roles --max-items 1                                          # AccessDenied
$ curl http://169.254.169.254/latest/meta-data/   # sin token (IMDSv1): HTTP 401
```

Tras el arranque: `docker --version` → 25.0.14, `docker compose version` → v2.29.7,
`docker buildx version` → v0.17.1, `git --version` → 2.50.1; disco raíz de 30 GB.

Configuración comprobada con la CLI: el security group tiene una única regla de entrada
(TCP 80, un `/32`), `HttpTokens=required`, `HttpPutResponseHopLimit=2`, instancia
`running` con la Elastic IP asociada y el agente de SSM en línea.

## Despliegue (P4-15)

El portal corre en la EC2 con
[`docker-compose.aws.yml`](../../docker-compose.aws.yml), un override del compose local
(que no cambia):

| Qué | Local (`docker-compose.yml`) | AWS (con el override) |
|---|---|---|
| Servicios | todos | solo `frontend`, `backend`, `mariadb`, `minio` y `ml-api`; el resto queda en el perfil `local` |
| Puertos publicados | 8080, 3100, 3306, 9000/9001, 5050 | solo **80** (frontend) |
| Credenciales de AWS en `ml-api` | perfil SSO (`~/.aws` + `AWS_PROFILE`) | rol de instancia `mlops-p4-portal-role` por IMDSv2; sin `~/.aws` ni `AWS_PROFILE` |
| Acceso | libre | **basic auth** obligatorio en nginx para todo el sitio |
| Reinicio | `on-failure` | `unless-stopped`: vuelve solo tras reiniciar la EC2 |
| MinIO | `minio/minio` | `cgr.dev/chainguard/minio` (ver nota) |

Sin MLflow en AWS: las páginas que lo usan (Experiments, Training) muestran error;
Capturas Edge y Models no lo necesitan. En `ml-api`, `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` y
`MLFLOW_HTTP_REQUEST_TIMEOUT=5` hacen que ese error sea inmediato: con los reintentos por
defecto (~4 min) esas rutas congelaban todo `ml-api` y daban 504 también en Capturas Edge
(diagnóstico en el PR #40). `ml-api` tarda unos 2–3 minutos en arrancar
(importa torch); mientras tanto `/ml-api/` responde 502.

> **MinIO.** Desde octubre de 2026 `minio/minio` ya no se puede jalar de Docker Hub
> (`pull access denied`) y `quay.io/minio/minio` pide login. El override usa la imagen de
> Chainguard, con la misma versión de MinIO.

### Basic auth

[`deploy/aws/basic-auth.conf`](../../deploy/aws/basic-auth.conf) se monta en el contexto
`http` de nginx, así que protege todo: páginas, `/api/`, `/ml-api/` y `/reports/`. Al
arrancar, [`deploy/aws/basic-auth-init.sh`](../../deploy/aws/basic-auth-init.sh) genera el
`htpasswd` (SHA-512) con `PORTAL_USER` y `PORTAL_PASSWORD` del `.env`. Si alguna falta,
Compose no levanta el frontend: el portal nunca queda abierto sin contraseña.

La contraseña vive solo en el `.env` de la EC2. Nunca va en Git; a la PM y al profesor se
les comparte por privado.

### Primer despliegue

Desde tu equipo, con el perfil de `MLOpsP4Portal`, entra por Session Manager:

```bash
aws sso login --profile mlops-p4-portal
aws ssm start-session --profile mlops-p4-portal --region us-east-1 --target i-0c333de1158e247e9
```

Dentro de la EC2, primero 2 GB de swap (la `t3.medium` tiene 4 GB y la imagen de
`ml-api` instala torch):

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Después, como `ec2-user`:

```bash
sudo -iu ec2-user
git clone https://github.com/Andy-752109/Proyecto04-MLOps.git portal && cd portal

cat > .env <<EOF
MARIADB_ROOT_PASSWORD=$(openssl rand -hex 32)
MINIO_ROOT_USER=portal
MINIO_ROOT_PASSWORD=$(openssl rand -hex 32)
# Solo para que Compose no falle al interpolar trainer-worker, que aquí no corre;
# ml-api lo ignora y usa el rol de la instancia.
AWS_PROFILE=sin-sso
PORTAL_USER=evaluador
PORTAL_PASSWORD=$(openssl rand -base64 18)
EOF
chmod 600 .env

docker compose -f docker-compose.yml -f docker-compose.aws.yml up -d --build
```

La primera construcción tarda varios minutos. `grep PORTAL_PASSWORD .env` muestra la contraseña para compartirla por privado.

### Operación diaria

```bash
cd ~/portal
C="docker compose -f docker-compose.yml -f docker-compose.aws.yml"

$C ps                                   # estado
$C logs -f --tail 100 ml-api            # logs (o frontend, backend…)
git pull && $C up -d --build            # actualizar a lo último de main
$C down                                 # detener (los volúmenes se conservan)
```

Para apagar la máquina y dejar de pagar cómputo, ver [Encender y apagar](#encender-y-apagar):
al volver a encenderla, Docker arranca solo y los contenedores también (`unless-stopped`).

### Verificación (evidencia de P4-15)

Despliegue del 11 de octubre de 2026 (commit `5cd4ee0`), en `http://18.213.105.30`.

**Basic auth y puertos, desde fuera de AWS:**

```text
$ curl -s -o /dev/null -w '%{http_code}' http://18.213.105.30/                         # 401
$ curl -s -o /dev/null -w '%{http_code}' -u evaluador:incorrecta http://18.213.105.30/  # 401
$ curl -sI http://18.213.105.30/ | grep -i www-auth
WWW-Authenticate: Basic realm="Capturas Edge"
```

Lo mismo (401) para `/api/health`, `/ml-api/health` y `/ml-api/edge/captures`. Los puertos
22, 443, 3100, 3306, 5050, 8080, 8100, 9000 y 9001 no responden.

**Con usuario y contraseña** (dentro de la EC2, leyendo el `.env` sin imprimirlo):

```text
/ 200
/api/health 200
/ml-api/health 200            {"status":"ok"}
ml-api: INFO:botocore.credentials:Found credentials from IAM Role: mlops-p4-portal-role
```

`docker compose ps`: `frontend`, `backend`, `ml-api`, `mariadb` (healthy) y `minio`, todos
`Up`. Sin errores de credenciales ni `ProfileNotFound` en los logs de `ml-api`.

**Capturas reales y captura nueva.** Capturas Edge lista las capturas del bucket de la más
reciente a la más antigua, con la imagen firmada por el rol de la instancia (HTTP 200). Cuatro
capturas nuevas de `edge-laptop-01` (11 oct, modelo `1.0.0-int8`) aparecieron arriba al
pulsar Actualizar, con los mismos datos que imprimió la app edge:

| capture_id | captured_at (UTC) | Clase | Confianza | Envío |
|---|---|---|---|---|
| `015513c6-2ef3-4000-9e1a-45cdf4358fd5` | 23:39:38 | dog | 0.9987 | sent (1461 ms) |
| `c5a3b470-dc0e-4cb5-a161-e4805aa0d6c5` | 23:39:20 | dog | 0.9054 | sent (2549 ms) |
| `48289c2d-f2c6-4d09-899a-104a2004d2f6` | 23:31:15 | dog | 0.9054 | sent (1291 ms) |
| `085a6b5a-ad05-4133-b0b1-ac9a1bc03dc7` | 23:30:46 | cat | 0.8109 | sent (3427 ms) |

Las cuatro fotos eran del mismo perro, mostrado en pantalla; `085a6b5a…` es un error real del
clasificador y se deja visible.

**Reinicio.** `aws ec2 reboot-instances` a las 22:11:48 UTC: sin intervención, los cinco
contenedores volvieron (`unless-stopped`), el portal respondió 401 desde fuera en ~20 s y,
con contraseña, `/`, `/api/health` y `/ml-api/health` en 200 con las capturas de S3; el swap
siguió activo.

**Security group final:** TCP 80 `0.0.0.0/0` (`publico-basic-auth`) y el `/32` de `karen`;
sin SSH.
