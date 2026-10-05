#!/usr/bin/env bash
# Revisión de Uriel sobre P3-09: trainer-worker/ml-api hornean GIT_COMMIT y
# GIT_DIRTY en la imagen (Dockerfile.ml -- ARG) en vez de leerlos en vivo del
# host al arrancar. `get_git_info()` (app/training/tracking.py) ya prioriza
# estas variables si llegan por el entorno, pero alguien tiene que calcularlas
# y exportarlas antes de `docker compose build` -- eso es lo que hace este
# script. Sin esto, la imagen queda con GIT_COMMIT=unknown (default del ARG).
#
# Uso (desde la raíz del repo):
#   bash scripts/build_trainer.sh [servicios adicionales de "docker compose build"...]
#
# Ejemplos:
#   bash scripts/build_trainer.sh                       # trainer-worker y ml-api
#   bash scripts/build_trainer.sh trainer-worker ml-api  # explícito, mismo resultado
set -euo pipefail
cd "$(dirname "$0")/.."

export GIT_COMMIT
GIT_COMMIT="$(git rev-parse HEAD)"
export GIT_DIRTY
if [ -n "$(git status --porcelain)" ]; then
  GIT_DIRTY=true
else
  GIT_DIRTY=false
fi

SERVICES=("$@")
if [ ${#SERVICES[@]} -eq 0 ]; then
  SERVICES=(trainer-worker ml-api)
fi

echo "[build_trainer] GIT_COMMIT=$GIT_COMMIT GIT_DIRTY=$GIT_DIRTY"
echo "[build_trainer] Reconstruyendo (secuencial): ${SERVICES[*]}"

# P3-10: `docker compose build a b` (buildx bake) construye los targets en
# paralelo, y todos comparten el mismo cache mount de uv (`/root/.cache/uv`,
# Dockerfile.ml). Al descargar los paquetes pesados de `--group ml`, dos
# `uv sync` concurrentes se pelean el lock de la distribution cache y uno
# revienta con "Timeout (300s) when waiting for lock" (visto real: nvidia-cufile
# en Mac M3, con trainer-worker y ml-api a la vez). Se construye uno por uno
# para que solo haya un `uv sync` tocando el cache; el cache se reaprovecha
# entre servicios, así que el único costo extra es la serialización.
for service in "${SERVICES[@]}"; do
  docker compose build "$service"
done
