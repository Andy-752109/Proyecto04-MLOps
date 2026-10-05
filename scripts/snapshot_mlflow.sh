#!/usr/bin/env bash
# Snapshot de MLflow (P3-03): detiene el servicio, vuelca la base "mlflow" de
# MariaDB y copia el bucket "mlflow-artifacts" de MinIO a ./mlflow-store/,
# lo versiona con DVC, y reinicia MLflow.
#
# Todo vía los contenedores que ya define docker-compose.yml -- no asume que
# mariadb-dump esté instalado en el host. La copia de artefactos usa
# `mlflow_ops.artifacts_sync` (SDK `minio`), no `mc`: `minio/mc` (Docker Hub)
# y `quay.io/minio/mc` (Quay) dejaron de poderse jalar sin autenticarse,
# probado al armar este ticket (ver docker-compose.yml).
#
# Uso (desde la raíz del repo, con .env ya completo):
#   bash scripts/snapshot_mlflow.sh
#
# Restaurar en un clon limpio:
#   1. dvc pull mlflow-store.dvc
#   2. docker compose up -d mariadb minio mariadb-mlflow-db minio-mlflow-bucket
#   3. docker compose exec -T mariadb mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" \
#        mlflow < mlflow-store/mlflow.sql
#   4. docker compose run --rm -v "$(pwd)/mlflow-store/artifacts:/restore" \
#        --entrypoint python ml-api -m mlflow_ops.artifacts_sync restore /restore
#   5. docker compose up -d mlflow
#   6. Confirmar en http://localhost:5050 que el run_id de la corrida de
#      prueba sigue ahí, con su artefacto.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "[snapshot] Falta .env (copia .env.example y complétalo)." >&2
  exit 1
fi
if ! command -v dvc >/dev/null 2>&1; then
  echo "[snapshot] 'dvc' no está en el PATH -- activa .venv-dvc primero:" >&2
  echo "  source .venv-dvc/bin/activate" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

STORE_DIR="mlflow-store"
mkdir -p "$STORE_DIR/artifacts"

# Si algo falla a partir de aquí, MLflow no debe quedar detenido -- un
# snapshot fallido no debería tumbar el servicio para el resto del equipo.
trap 'docker compose up -d mlflow' EXIT

echo "[snapshot] Deteniendo mlflow..."
docker compose stop mlflow

echo "[snapshot] Volcando la base 'mlflow' de MariaDB..."
docker compose exec -T mariadb \
  mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" mlflow >"$STORE_DIR/mlflow.sql"

echo "[snapshot] Descargando el bucket mlflow-artifacts de MinIO..."
docker compose run --rm --no-deps \
  -v "$(pwd)/$STORE_DIR/artifacts:/backup" \
  --entrypoint python \
  ml-api -m mlflow_ops.artifacts_sync backup /backup

echo "[snapshot] Versionando $STORE_DIR con DVC..."
dvc add "$STORE_DIR"
dvc push

echo "[snapshot] Listo (mlflow se reinicia solo, ver el trap de arriba). Revisa 'git status' y commitea ${STORE_DIR}.dvc."
