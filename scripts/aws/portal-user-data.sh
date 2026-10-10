#!/bin/bash
# User data de la EC2 del portal (P4-14): Docker, Docker Compose, buildx y Git en
# Amazon Linux 2023. El AWS CLI y el agente de SSM ya vienen en la AMI.
# Registro: /var/log/portal-bootstrap.log
set -euxo pipefail
exec > >(tee -a /var/log/portal-bootstrap.log) 2>&1

COMPOSE_VERSION=v2.29.7
BUILDX_VERSION=v0.17.1

dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

PLUGINS=/usr/local/lib/docker/cli-plugins
mkdir -p "$PLUGINS"
curl -fsSL -o "$PLUGINS/docker-compose" \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64"
curl -fsSL -o "$PLUGINS/docker-buildx" \
  "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64"
chmod +x "$PLUGINS/docker-compose" "$PLUGINS/docker-buildx"

docker --version
docker compose version
git --version
touch /var/log/portal-bootstrap.done
