#!/usr/bin/env bash
# Infraestructura del portal en AWS (P4-14): EC2 + rol de solo lectura a S3 +
# security group + alerta de presupuesto. Todo con la AWS CLI, sin Terraform.
# Ver docs/p4/portal-aws.md.
#
# Requiere un perfil con permisos de administración en la cuenta (no MLOpsP3).
# Las IP NUNCA van en el repo: se pasan por variable de entorno.
#
#   export AWS_PROFILE=<perfil-admin>
#   export ALLOWED_CIDRS="203.0.113.10/32,198.51.100.7/32"   # equipo + profesor
#   export BUDGET_EMAIL="<correo1>,<correo2>"   # destinatarios de la alerta
#   bash scripts/aws/portal-infra.sh
#
# Opcionales: USE_EIP=0 (sin Elastic IP), BUDGET_USD=10, INSTANCE_TYPE=t3.medium.
# Se puede repetir: lo que ya existe se reutiliza.
set -euo pipefail
cd "$(dirname "$0")"

: "${ALLOWED_CIDRS:?Define ALLOWED_CIDRS (CIDR separados por coma, p. ej. x.x.x.x/32)}"
: "${BUDGET_EMAIL:?Define BUDGET_EMAIL (correos separados por coma que reciben la alerta de presupuesto)}"
USE_EIP="${USE_EIP:-1}"
BUDGET_USD="${BUDGET_USD:-10}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"

export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
NAME=mlops-p4-portal
ROLE=mlops-p4-portal-role
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
if [ "$ACCOUNT_ID" != "222629887955" ]; then
  echo "[portal] Cuenta inesperada ($ACCOUNT_ID); deben vivir junto a los buckets (222629887955)." >&2
  exit 1
fi

IFS=',' read -ra CIDRS <<< "$ALLOWED_CIDRS"
for cidr in "${CIDRS[@]}"; do
  if ! [[ "$cidr" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}/[0-9]{1,2}$ ]] || [[ "$cidr" == */0 ]] || [[ "${cidr#*/}" -lt 24 ]]; then
    echo "[portal] CIDR no permitido: '$cidr' (usa IP/32 o, como mínimo, /24; nada abierto a 0.0.0.0/0)." >&2
    exit 1
  fi
done

# --- 1. Rol de instancia (solo lectura a S3) --------------------------------
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" \
    --description "Portal P4: lectura de S3 y Session Manager" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' --query Role.RoleName --output text
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name s3-solo-lectura \
  --policy-document "file://portal-role-policy.json"
aws iam attach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" --query InstanceProfile.InstanceProfileName --output text
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 15   # propagación de IAM
fi

# --- 2. Red: VPC por defecto y una subred con t3.medium ---------------------
VPC_ID="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
[ "$VPC_ID" != "None" ] || { echo "[portal] No hay VPC por defecto en us-east-1." >&2; exit 1; }
AZS="$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values="$INSTANCE_TYPE" --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' ',')"
SUBNET_ID="$(aws ec2 describe-subnets \
  --filters Name=vpc-id,Values="$VPC_ID" Name=default-for-az,Values=true Name=availability-zone,Values=$AZS \
  --query 'Subnets[0].SubnetId' --output text)"
[ "$SUBNET_ID" != "None" ] || { echo "[portal] Sin subred por defecto con $INSTANCE_TYPE." >&2; exit 1; }

# --- 3. Security group: solo el puerto 80 a las IP indicadas, sin SSH -------
SG_ID="$(aws ec2 describe-security-groups --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values="$NAME" \
  --query 'SecurityGroups[0].GroupId' --output text)"
if [ "$SG_ID" = "None" ]; then
  SG_ID="$(aws ec2 create-security-group --group-name "$NAME" --vpc-id "$VPC_ID" \
    --description "Portal P4: HTTP solo para el equipo y el profesor" --query GroupId --output text)"
fi
for cidr in "${CIDRS[@]}"; do
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr "$cidr" \
    >/dev/null 2>&1 || true   # ya existía
done

# --- 4. Instancia -----------------------------------------------------------
INSTANCE_ID="$(aws ec2 describe-instances \
  --filters Name=tag:Name,Values="$NAME" Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)"
if [ "$INSTANCE_ID" = "None" ]; then
  AMI_ID="$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query Parameter.Value --output text)"
  INSTANCE_ID="$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
    --subnet-id "$SUBNET_ID" --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$ROLE" \
    --associate-public-ip-address \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2 \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' \
    --user-data "file://portal-user-data.sh" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=mlops-p4}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=mlops-p4}]" \
    --query 'Instances[0].InstanceId' --output text)"
fi
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# --- 5. Elastic IP (opcional) -----------------------------------------------
if [ "$USE_EIP" = "1" ]; then
  EXISTING="$(aws ec2 describe-addresses --filters Name=tag:Name,Values="$NAME" --query 'Addresses[0].AllocationId' --output text)"
  if [ "$EXISTING" = "None" ]; then
    ALLOC="$(aws ec2 allocate-address --domain vpc \
      --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=mlops-p4}]" \
      --query AllocationId --output text)"
    aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC" >/dev/null
  fi
fi

# --- 6. Alerta de presupuesto ------------------------------------------------
SUBSCRIBERS=""
IFS=',' read -ra MAILS <<< "$BUDGET_EMAIL"
for mail in "${MAILS[@]}"; do
  SUBSCRIBERS+="${SUBSCRIBERS:+,}{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$mail\"}"
done
if ! aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name "$NAME-mensual" >/dev/null 2>&1; then
  aws budgets create-budget --account-id "$ACCOUNT_ID" \
    --budget "{\"BudgetName\":\"$NAME-mensual\",\"BudgetLimit\":{\"Amount\":\"$BUDGET_USD\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" \
    --notifications-with-subscribers "[
      {\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":80,\"ThresholdType\":\"PERCENTAGE\"},
       \"Subscribers\":[$SUBSCRIBERS]},
      {\"Notification\":{\"NotificationType\":\"FORECASTED\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":100,\"ThresholdType\":\"PERCENTAGE\"},
       \"Subscribers\":[$SUBSCRIBERS]}
    ]"
fi

echo "[portal] Listo."
echo "  instancia : $INSTANCE_ID ($INSTANCE_TYPE) en $VPC_ID / $SUBNET_ID"
echo "  rol       : $ROLE"
echo "  sg        : $SG_ID"
echo "  presupuesto: $NAME-mensual (USD $BUDGET_USD)"
