"""Enriquece eventos edge v1 con metadatos S3 para la API de lectura."""

import logging
from datetime import datetime

from botocore.exceptions import ClientError
from pydantic import ValidationError

from ml_api.contracts import EdgeCaptureItem, EdgeCaptureList, EdgeEventV1
from storage.edge_capture_store import list_event_objects, read_event_json
from storage.model_store import presigned_get_url

logger = logging.getLogger("ml-api.edge_captures")


def _is_no_such_key(error: ClientError) -> bool:
    return error.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}


def build_capture_list(client, *, bucket: str, limit: int) -> EdgeCaptureList:
    valid: list[EdgeCaptureItem] = []
    for object_info in list_event_objects(client, bucket=bucket):
        key = object_info.get("Key")
        if not isinstance(key, str) or not key.endswith(".json"):
            logger.warning("Se omite objeto sin clave JSON de evento: %s", key)
            continue
        try:
            raw = read_event_json(client, bucket=bucket, key=key)
        except ClientError as error:
            if _is_no_such_key(error):
                logger.warning("Evento desapareció entre list y get: %s", key)
                continue
            raise
        try:
            event = EdgeEventV1.model_validate_json(raw)
            received_at = object_info["LastModified"]
            if not isinstance(received_at, datetime) or received_at.utcoffset() is None:
                raise ValueError("LastModified ausente o sin zona horaria")
        except (ValidationError, ValueError, KeyError) as error:
            logger.warning("Evento inválido omitido (%s): %s", key, type(error).__name__)
            continue
        image_url = presigned_get_url(client, bucket=bucket, key=event.image_key)
        valid.append(
            EdgeCaptureItem(**event.model_dump(), received_at=received_at, image_url=image_url)
        )

    valid.sort(
        key=lambda item: (
            datetime.fromisoformat(item.captured_at.replace("Z", "+00:00")),
            item.capture_id,
        ),
        reverse=True,
    )
    return EdgeCaptureList(items=valid[:limit], total=len(valid), bucket=bucket)
