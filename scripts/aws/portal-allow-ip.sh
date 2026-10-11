#!/usr/bin/env bash
# Administra quién entra al puerto 80 del portal (P4-14). Cada regla del security group
# `mlops-p4-portal` lleva como descripción el nombre de su dueño, así que cambiar la IP de
# alguien es una sola línea: agrega la nueva y quita las anteriores de ese dueño.
# Las IP nunca van en el repo. Requiere un perfil con permisos sobre EC2 (no MLOpsP3).
#
#   bash scripts/aws/portal-allow-ip.sh karen                      # tu IP actual (checkip)
#   bash scripts/aws/portal-allow-ip.sh profe 203.0.113.10/32      # una IP o un rango (mín. /24)
#   bash scripts/aws/portal-allow-ip.sh --remove profe             # quita todas las de ese dueño
#   bash scripts/aws/portal-allow-ip.sh --list                     # reglas vigentes (dueño y CIDR)
set -euo pipefail

export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
SG_NAME=mlops-p4-portal
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text)"
[ "$SG_ID" != "None" ] || { echo "[allow-ip] No existe el security group $SG_NAME." >&2; exit 1; }

rules() {  # "CIDR<TAB>dueño" de cada regla de entrada del puerto 80
  aws ec2 describe-security-groups --group-ids "$SG_ID" \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`80\`].IpRanges[].[CidrIp,Description]" --output text
}
revoke() { aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr "$1" >/dev/null; }
valid_owner() { [[ "$1" =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]]; }

[ $# -ge 1 ] || usage
case "$1" in
  --list)
    rules | awk -F'\t' '{printf "%-20s %s\n", ($2=="None"||$2==""?"(sin dueño)":$2), $1}'
    exit 0 ;;
  --remove)
    [ $# -eq 2 ] && valid_owner "$2" || usage
    found=0
    while IFS=$'\t' read -r cidr owner; do
      [ "$owner" = "$2" ] || continue
      revoke "$cidr"; found=1; echo "[allow-ip] Quitada la regla de '$2'."
    done < <(rules)
    [ "$found" = 1 ] || echo "[allow-ip] '$2' no tenía reglas."
    exit 0 ;;
  -*) usage ;;
esac

OWNER="$1"
valid_owner "$OWNER" || { echo "[allow-ip] Dueño inválido: usa minúsculas, números y guiones." >&2; exit 1; }
CIDR="${2:-$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')/32}"
if ! [[ "$CIDR" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}/[0-9]{1,2}$ ]] || [[ "${CIDR#*/}" -lt 24 ]]; then
  echo "[allow-ip] CIDR no permitido: '$CIDR' (usa IP/32 o, como mínimo, /24; nada abierto a 0.0.0.0/0)." >&2
  exit 1
fi

PERMISSION="IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=$CIDR,Description=$OWNER}]"
if ! aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions "$PERMISSION" >/dev/null 2>&1; then
  # La regla ya existía (quizá sin dueño): solo se actualiza su descripción.
  aws ec2 update-security-group-rule-descriptions-ingress --group-id "$SG_ID" --ip-permissions "$PERMISSION" >/dev/null
fi
echo "[allow-ip] '$OWNER' puede entrar al puerto 80 desde $CIDR."

while IFS=$'\t' read -r cidr owner; do   # quita las IP anteriores del mismo dueño
  [ "$owner" = "$OWNER" ] && [ "$cidr" != "$CIDR" ] || continue
  revoke "$cidr"; echo "[allow-ip] Quitada la IP anterior de '$OWNER'."
done < <(rules)
