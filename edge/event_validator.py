"""Valida eventos edge v1 sin dependencias de cámara, inferencia o AWS."""

import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from uuid import RFC_4122, UUID

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "contracts" / "edge-event.v1.schema.json"
IMAGE_PREFIX = "edge-captures/v1/images/"
IMAGE_SUFFIX = ".jpg"


class EventValidationError(ValueError):
    """El evento no cumple el contrato edge v1."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    with SCHEMA_PATH.open(encoding="utf-8") as source:
        schema = json.load(source)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_event(event: Mapping[str, Any]) -> None:
    """Lanza EventValidationError si el evento no cumple el contrato."""
    errors = sorted(_validator().iter_errors(event), key=lambda error: list(map(str, error.path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.path)) or "evento"
        raise EventValidationError(f"{location}: {error.message}")

    capture_id = event["capture_id"]
    try:
        parsed_id = UUID(capture_id)
    except ValueError as exc:
        raise EventValidationError("capture_id: debe ser un UUID v4 canónico") from exc
    if parsed_id.version != 4 or parsed_id.variant != RFC_4122:
        raise EventValidationError("capture_id: debe ser un UUID v4 RFC 4122")
    if str(parsed_id) != capture_id:
        raise EventValidationError("capture_id: debe usar representación canónica lowercase")

    try:
        captured_at = datetime.fromisoformat(event["captured_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventValidationError("captured_at: fecha y hora inválidas") from exc
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise EventValidationError("captured_at: debe incluir zona horaria")

    expected_image_key = f"{IMAGE_PREFIX}{capture_id}{IMAGE_SUFFIX}"
    if event["image_key"] != expected_image_key:
        raise EventValidationError("image_key: debe contener el mismo capture_id del evento")

    numeric_values = {
        "confidence": event["confidence"],
        "probabilities.cat": event["probabilities"]["cat"],
        "probabilities.dog": event["probabilities"]["dog"],
        "preprocess_ms": event["preprocess_ms"],
        "inference_ms": event["inference_ms"],
    }
    if event["crop"] is not None:
        numeric_values.update({f"crop.{key}": value for key, value in event["crop"].items()})
    for field, value in numeric_values.items():
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = isinstance(value, int)  # Python permite enteros finitos mayores que float.
        if not finite:
            raise EventValidationError(f"{field}: debe ser un número finito")

    crop = event["crop"]
    if crop is not None:
        if crop["x"] + crop["width"] > crop["frame_width"]:
            raise EventValidationError("crop: x + width excede frame_width")
        if crop["y"] + crop["height"] > crop["frame_height"]:
            raise EventValidationError("crop: y + height excede frame_height")
