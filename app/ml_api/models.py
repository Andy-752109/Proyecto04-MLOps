"""Lectura y estado de los modelos publicados (P3-15).

El catálogo (`models/registry.json`) lo produce P3-14 (`app/publish.py`); aquí
solo se lee y se le agrega el estado **en vivo** de S3 (`head-object`), la URL
prefirmada de descarga, la tarjeta (que viaja dentro del paquete) y la elección
de la versión activa para inferencia.

Las dependencias de S3 y del paquete entran como callables (`status_of`,
`download_url_of`, `package_info_of`), no como llamadas directas: así esta capa
es pura y las pruebas no tocan la red (mismo criterio que `create_app` inyectando
`list_jobs`).
"""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from ml_api.contracts import (
    ModelDetail,
    ModelEntry,
    ModelList,
    ModelS3Status,
    ModelSummary,
)

logger = logging.getLogger("ml-api.models")

# El estado del registry puede no existir todavía (P3-14 aún no publicó).
Registry = dict[str, ModelEntry]
StatusOf = Callable[[ModelEntry], dict | None]
DownloadUrlOf = Callable[[ModelEntry], str | None]
PackageInfoOf = Callable[[ModelEntry], dict | None]


class ModelVersionNotFound(LookupError):
    """La versión no está en `models/registry.json`."""


class ModelObjectMissing(RuntimeError):
    """Se pidió marcar como activa una versión cuyo objeto no existe en S3."""


def version_key(version: str) -> tuple[int, ...]:
    """Ordena 0.10.0 después de 0.9.0 (un sort de strings lo haría al revés)."""
    parts = []
    for piece in version.split("."):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)


def split_s3_path(s3_path: str) -> tuple[str, str]:
    """`s3://bucket/key` → `(bucket, key)`."""
    without_scheme = s3_path.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def load_registry(path: Path) -> Registry | None:
    """`None` mientras P3-14 no haya publicado el catálogo."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    # `model_validate_json` (no `model_validate`): el contrato es `strict=True`
    # y las fechas del registry son strings ISO; en modo JSON sí se convierten.
    return {
        version: ModelEntry.model_validate_json(json.dumps(entry))
        for version, entry in payload.items()
    }


def read_selected_run_id(selection_path: Path) -> str | None:
    """`run_id` del candidato sellado en P3-11; el registry de P3-14 no lo marca."""
    if not selection_path.is_file():
        return None
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def read_active_version(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    version = payload.get("active_version")
    return version if isinstance(version, str) and version else None


def write_active_version(path: Path, version: str) -> None:
    """Escritura atómica: nunca deja el archivo a medias entre dos requests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"active_version": version}), encoding="utf-8")
    os.replace(tmp, path)


def _status(raw: dict | None) -> ModelS3Status:
    if raw is None:
        return ModelS3Status(exists=False)
    return ModelS3Status(
        exists=True,
        version_id=raw.get("version_id"),
        size_bytes=raw.get("size_bytes"),
        last_modified=raw.get("last_modified"),
    )


def _status_for(version: str, entry: ModelEntry, status_of: StatusOf) -> ModelS3Status:
    """Estado de una versión sin tumbar la lista entera (P3-15).

    Un `head-object` que falla por red/SSO/permisos no es un 500 de `/models`:
    esa versión queda marcada con `error` y las demás se sirven igual.
    """
    try:
        return _status(status_of(entry))
    except Exception:
        logger.exception("no se pudo verificar la versión %s en S3", version)
        return ModelS3Status(exists=False, error="no se pudo verificar el objeto en S3")


def summarize(
    version: str,
    entry: ModelEntry,
    *,
    active_version: str | None,
    selected_run_id: str | None,
    status: ModelS3Status,
) -> ModelSummary:
    return ModelSummary(
        version=version,
        dataset_version=entry.data_release,
        run_id=entry.run_id,
        published_at=entry.published_at,
        package_sha256=entry.sha256,
        registered_version_id=entry.VersionId,
        selected=selected_run_id is not None and entry.run_id == selected_run_id,
        active=version == active_version,
        s3_status=status,
    )


def build_list(
    registry: Registry,
    *,
    active_version: str | None,
    selected_run_id: str | None,
    status_of: StatusOf,
) -> ModelList:
    versions = [
        summarize(
            version,
            entry,
            active_version=active_version,
            selected_run_id=selected_run_id,
            status=_status_for(version, entry, status_of),
        )
        for version, entry in sorted(registry.items(), key=lambda item: version_key(item[0]))
    ]
    return ModelList(active_version=active_version, versions=versions)


def find_entry(registry: Registry, version: str) -> ModelEntry:
    if version not in registry:
        raise ModelVersionNotFound(version)
    return registry[version]


def build_detail(
    registry: Registry,
    version: str,
    *,
    active_version: str | None,
    selected_run_id: str | None,
    status_of: StatusOf,
    download_url_of: DownloadUrlOf,
    package_info_of: PackageInfoOf,
) -> ModelDetail:
    entry = find_entry(registry, version)
    bucket, key = split_s3_path(entry.s3_path)
    raw_status = status_of(entry)
    # El paquete solo se inspecciona si de verdad existe en S3 (evita descargar
    # un objeto ausente y da una tarjeta/trazabilidad honestas).
    package = package_info_of(entry) if raw_status is not None else None
    return ModelDetail(
        version=version,
        dataset_version=entry.data_release,
        run_id=entry.run_id,
        published_at=entry.published_at,
        package_sha256=entry.sha256,
        registered_version_id=entry.VersionId,
        checkpoint_sha256=entry.checkpoint_sha256,
        run_kind=package.get("run_kind") if package else None,
        manifest_id=package.get("manifest_id") if package else None,
        selected=selected_run_id is not None and entry.run_id == selected_run_id,
        active=version == active_version,
        s3_bucket=bucket,
        s3_key=key,
        card=package.get("card") if package else None,
        download_url=download_url_of(entry) if raw_status is not None else None,
        s3_status=_status(raw_status),
    )


def activate_version(registry: Registry, version: str, *, status_of: StatusOf) -> ModelEntry:
    """Marca `version` como activa **solo si su objeto existe en S3**.

    El rechazo explícito de un objeto inexistente es parte del #23: no se puede
    dejar el sistema apuntando a un paquete que no está.
    """
    entry = find_entry(registry, version)
    if status_of(entry) is None:
        raise ModelObjectMissing(version)
    return entry
