"""Uploader idempotente de capturas edge a S3 (P4-05).

Sigue el contrato v1 (`docs/p4/contrato-evento-edge.md`):

1. sube la imagen a `edge-captures/v1/images/{capture_id}.jpg`;
2. después sube el evento a `edge-captures/v1/events/{capture_id}.json`.

Ambas escrituras son condicionales (`If-None-Match: *`): si el objeto ya existe, S3 responde
412 y no lo sobrescribe. Así un reintento nunca duplica ni cambia una captura. El evento va
al final porque su existencia indica que la captura está completa.

Credenciales: solo la cadena por defecto de boto3 (perfil SSO o credenciales temporales de
`aws configure export-credentials`). Nunca se registran ni se guardan.

Prueba manual contra S3 (dos veces seguidas = un objeto de cada tipo):

    python -m edge.uploader --bucket mlops-p4-edge-captures-222629887955 \\
        --profile mlops-p3 --event contracts/examples/valid-cat-no-crop.json \\
        --image contracts/examples/test-capture.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from edge.event_validator import EventValidationError, validate_event

EVENT_PREFIX = "edge-captures/v1/events/"
EVENT_SUFFIX = ".json"
Status = Literal["sent", "already_sent", "failed"]


@dataclass(frozen=True)
class UploadResult:
    status: Status
    capture_id: str
    upload_ms: float
    error: str | None = None


def event_key(capture_id: str) -> str:
    return f"{EVENT_PREFIX}{capture_id}{EVENT_SUFFIX}"


def event_body(event: Mapping[str, Any]) -> bytes:
    """JSON canónico del evento: el mismo evento produce siempre los mismos bytes."""
    return json.dumps(
        event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _is_precondition_failed(error: Exception) -> bool:
    response = getattr(error, "response", None) or {}
    code = response.get("Error", {}).get("Code")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code == "PreconditionFailed" or status == 412


def _describe(error: Exception) -> str:
    """Mensaje corto para el log local: código y texto de AWS, sin cabeceras ni firmas."""
    response = getattr(error, "response", None)
    if response and "Error" in response:
        err = response["Error"]
        return f"{err.get('Code', type(error).__name__)}: {err.get('Message', '')}".strip()
    return f"{type(error).__name__}: {error}"


class Uploader:
    def __init__(self, bucket: str, s3_client: Any) -> None:
        if not bucket:
            raise ValueError("falta el nombre del bucket")
        self.bucket = bucket
        self.s3 = s3_client

    def _put_if_absent(self, key: str, body: bytes, content_type: str) -> bool:
        """True si se creó el objeto; False si ya existía (412)."""
        try:
            self.s3.put_object(
                Bucket=self.bucket, Key=key, Body=body, ContentType=content_type, IfNoneMatch="*"
            )
        except Exception as error:
            if _is_precondition_failed(error):
                return False
            raise
        return True

    def upload(self, event: Mapping[str, Any], image_path: Path) -> UploadResult:
        """Sube imagen y evento. Nunca lanza: los errores vuelven como `failed`."""
        started = time.perf_counter()
        capture_id = str(event.get("capture_id", ""))

        def result(status: Status, error: str | None = None) -> UploadResult:
            elapsed = (time.perf_counter() - started) * 1000
            return UploadResult(status, capture_id, round(elapsed, 2), error)

        try:
            validate_event(event)
            image = Path(image_path).read_bytes()
        except (EventValidationError, OSError) as error:
            return result("failed", f"{type(error).__name__}: {error}")

        try:
            self._put_if_absent(event["image_key"], image, "image/jpeg")
            created = self._put_if_absent(
                event_key(capture_id), event_body(event), "application/json"
            )
        except Exception as error:  # red, credenciales, permisos, 409 de escritura concurrente
            return result("failed", _describe(error))
        return result("sent" if created else "already_sent")


def make_s3_client(profile: str | None = None, region: str | None = None) -> Any:
    """Cliente S3 con timeouts cortos: sin red, el envío falla rápido y se reintenta luego."""
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile, region_name=region)
    config = Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2})
    return session.client("s3", config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sube una captura (imagen + evento) a S3.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--event", type=Path, required=True, help="JSON del evento v1")
    parser.add_argument("--image", type=Path, required=True, help="JPEG de la captura")
    parser.add_argument("--profile", help="perfil SSO de AWS (o usa AWS_PROFILE)")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)

    event = json.loads(args.event.read_text(encoding="utf-8"))
    uploader = Uploader(args.bucket, make_s3_client(args.profile, args.region))
    outcome = uploader.upload(event, args.image)
    print(json.dumps(asdict(outcome), ensure_ascii=False))
    return 1 if outcome.status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
