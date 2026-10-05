"""Crea el bucket de artefactos de MLflow si no existe (P3-03).

Reemplaza a `mc mb --ignore-existing`: la imagen `quay.io/minio/mc` pasó a
requerir autenticación (probado: 401 al resolver `mc:latest`), así que en
vez de depender de una imagen de terceros que puede volver a cambiar, se usa
el SDK `minio` que `app/` ya trae como dependencia -- mismo patrón que
`ensureMinioBucket()` en `backend/src/data/storage/minio.storage.ts`.

    uv run python -m mlflow_ops.ensure_bucket
"""

import logging

from storage.object_store import get_minio_client

logger = logging.getLogger("mlflow-ops")

MLFLOW_ARTIFACTS_BUCKET = "mlflow-artifacts"


def ensure_bucket(bucket: str = MLFLOW_ARTIFACTS_BUCKET) -> None:
    client = get_minio_client()
    if client.bucket_exists(bucket):
        logger.info("Bucket '%s' ya existe.", bucket)
        return
    client.make_bucket(bucket)
    logger.info("Bucket '%s' creado.", bucket)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_bucket()


if __name__ == "__main__":
    main()
