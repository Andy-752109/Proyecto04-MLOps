# Portal en AWS (P4-14)

El profesor pidió que Capturas Edge esté **desplegada en AWS** (rúbrica 5.1, requisito M3).
Este documento describe dónde corre el portal. Esta primera parte cubre la
**infraestructura** (P4-14); el despliegue del portal sobre ella llega en los tickets
siguientes.

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

| Dirección | Puerto | Origen |
|---|---|---|
| Entrada | TCP 80 | solo las IP del equipo y del profesor (`/32`), una regla por IP |
| Entrada | 22 (SSH) | **cerrado**: no hay regla; se usa Session Manager |
| Salida | todo | (necesario para jalar imágenes de Docker, S3 y SSM) |

Nada está abierto a `0.0.0.0/0`; los scripts rechazan ese CIDR y cualquier rango menor a `/24`.
Las IP no se guardan en el repo: cada regla lleva como descripción el nombre de su dueño
(`karen`, `profe`, …) y se administran con
[`scripts/aws/portal-allow-ip.sh`](../../scripts/aws/portal-allow-ip.sh).

#### Las IP no son fijas

Las IP de casa, de datos móviles y de muchas redes cambian al reiniciar el módem, al cambiar
de red o con el tiempo. Si el portal deja de responder a alguien, casi siempre es que su IP
cambió. Cada quien consulta la suya con `curl -s https://checkip.amazonaws.com` y quien
administre la cuenta la actualiza (el rol `MLOpsP3` no puede modificar el security group).

- **Equipo:** al cambiar la IP, se reemplaza la del dueño; el script quita la anterior.
- **Profesor:** si entra desde la red del ITESO, abrir su IP o rango de salida (pedirlo a
  TI; mínimo `/24`). Si entra desde otra red, agregar su IP el día de la revisión y
  quitarla después.

Se ejecuta con un perfil con permisos sobre EC2 (no `MLOpsP3`):

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
gasto real y cuando el pronóstico supere el 100 %. Además de la dueña de la cuenta,
recibe el aviso el profesor. Se crea con `BUDGET_EMAIL` (varios correos separados por coma).

### Encender y apagar

```bash
export AWS_REGION=us-east-1
ID=$(aws ec2 describe-instances \
  --filters Name=tag:Name,Values=mlops-p4-portal Name=instance-state-name,Values=running,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Encender (la Elastic IP se conserva; Docker arranca solo, los contenedores hay que levantarlos)
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
