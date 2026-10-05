"""Copia el bucket de artefactos de MLflow hacia/desde un directorio local (P3-03).

Reemplaza a `mc mirror` por la misma razón que `ensure_bucket.py` reemplaza a
`mc mb`: la imagen `quay.io/minio/mc` ya no se puede jalar sin autenticarse.
Lo usa `scripts/snapshot_mlflow.sh` (backup, antes de `dvc add`+`dvc push`) y
el procedimiento de restauración documentado ahí (restore, después de
`dvc pull`).

    uv run python -m mlflow_ops.artifacts_sync backup ./mlflow-store/artifacts
    uv run python -m mlflow_ops.artifacts_sync restore ./mlflow-store/artifacts
"""

import argparse
import logging
from pathlib import Path

from mlflow_ops.ensure_bucket import MLFLOW_ARTIFACTS_BUCKET, ensure_bucket
from storage.object_store import get_minio_client

logger = logging.getLogger("mlflow-ops")


def backup(local_dir: Path, *, bucket: str = MLFLOW_ARTIFACTS_BUCKET) -> int:
    """Descarga cada objeto del bucket a `local_dir`, conservando su ruta."""
    client = get_minio_client()
    # NOSONAR (pythonsecurity:S8707): `local_dir` llega de `main()`/argparse, no de
    # un request HTTP -- este módulo no tiene ningún endpoint, solo se invoca por
    # `docker compose run` (ver scripts/snapshot_mlflow.sh). El nombre de objeto del
    # bucket sí se valida abajo, que es el dato con origen externo real aquí.
    local_dir.mkdir(parents=True, exist_ok=True)  # NOSONAR

    local_dir_resolved = local_dir.resolve()
    count = 0
    for obj in client.list_objects(bucket, recursive=True):
        destination = (local_dir / obj.object_name).resolve()
        if not destination.is_relative_to(local_dir_resolved):
            # No debería pasar (nosotros mismos escribimos los nombres de objeto
            # en `restore()`), pero el nombre de objeto lo decide el bucket, no
            # esta función -- no confiar en que nunca traiga un "../".
            raise ValueError(f"nombre de objeto fuera de {local_dir}: {obj.object_name!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(destination))
        count += 1
    logger.info("backup: %s objetos descargados de '%s' a %s", count, bucket, local_dir)
    return count


def restore(local_dir: Path, *, bucket: str = MLFLOW_ARTIFACTS_BUCKET) -> int:
    """Sube cada archivo de `local_dir` al bucket, con la misma ruta relativa."""
    ensure_bucket(bucket)
    client = get_minio_client()

    count = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        object_name = str(path.relative_to(local_dir))
        client.fput_object(bucket, object_name, str(path))
        count += 1
    logger.info("restore: %s archivos subidos de %s a '%s'", count, local_dir, bucket)
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=["backup", "restore"])
    parser.add_argument("local_dir", type=Path)
    args = parser.parse_args()

    if args.direction == "backup":
        backup(args.local_dir)
    else:
        restore(args.local_dir)


if __name__ == "__main__":
    main()
