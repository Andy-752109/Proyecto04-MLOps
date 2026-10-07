"""Envío de capturas a S3 desde la app edge (P4-09).

La inferencia nunca espera al envío: `run` clasifica, guarda la captura en `captures.jsonl`
y la pasa a un hilo de fondo que la sube con el uploader de P4-05. Cada intento queda en
`uploads.jsonl`, separado del log de inferencia:

    {"capture_id", "upload_status", "error", "upload_ms", "bucket", "at"}

`upload_status` es `pending` (en cola), `sent`, `already_sent` o `failed`. El estado de una
captura es el de su último registro; si no tiene ninguno (p. ej. sin bucket configurado),
está `pending`. Los reintentos (`edge retry`) leen el evento de `captures.jsonl` y reenvían
exactamente el mismo evento, con el mismo `capture_id`; nunca se vuelve a inferir.

No hay cola persistente ni reenvío automático al reconectar: lo que no se pudo enviar queda
`failed` (o `pending` si la app se cerró antes) y se reenvía con `edge retry --pending`.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge.config import AwsConfig, EdgeConfig
from edge.pipeline import read_log
from edge.uploader import Uploader, UploadResult, make_s3_client

log = logging.getLogger("edge")

STATUSES = ("pending", "sent", "already_sent", "failed")
DONE = ("sent", "already_sent")
ClientFactory = Callable[[str | None, str | None], Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class UploadLog:
    """`uploads.jsonl`: un registro por intento, solo se agregan líneas."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(
        self,
        capture_id: str,
        status: str,
        bucket: str,
        error: str | None = None,
        upload_ms: float | None = None,
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"upload_status inválido: {status!r}")
        record = {
            "capture_id": capture_id,
            "upload_status": status,
            "error": error,
            "upload_ms": upload_ms,
            "bucket": bucket,
            "at": _now(),
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        return record

    def append_result(self, result: UploadResult, bucket: str) -> dict[str, Any]:
        return self.append(result.capture_id, result.status, bucket, result.error, result.upload_ms)

    def read(self) -> list[dict[str, Any]]:
        return read_log(self.path)

    def latest(self) -> dict[str, dict[str, Any]]:
        """Último registro de cada `capture_id`."""
        return {record["capture_id"]: record for record in self.read()}


def status_of(latest: dict[str, dict[str, Any]], capture_id: str) -> str:
    return latest.get(capture_id, {}).get("upload_status", "pending")


def image_path(config: EdgeConfig, record: dict[str, Any]) -> Path:
    """Ruta absoluta de la imagen: `captures.jsonl` la guarda relativa a `data_dir`."""
    return config.data_dir / record["image_path"]


def make_uploader(aws: AwsConfig, client_factory: ClientFactory | None = None) -> Uploader:
    factory = client_factory or make_s3_client  # se busca al llamar (los tests lo sustituyen)
    return Uploader(aws.bucket, factory(aws.profile, aws.region))


def _report(result: UploadResult, bucket: str) -> None:
    if result.status == "failed":
        log.error(
            "envío %s FALLÓ (%s): %s; reintenta con: python -m edge retry %s",
            result.capture_id,
            bucket,
            result.error,
            result.capture_id,
        )
    else:
        log.info("envío %s %s (%.0f ms)", result.capture_id, result.status, result.upload_ms)


def send(
    uploader: Uploader, upload_log: UploadLog, event: dict[str, Any], image: Path
) -> UploadResult:
    """Un intento: sube, registra el resultado en `uploads.jsonl` y lo reporta en el log."""
    result = uploader.upload(event, image)
    upload_log.append_result(result, uploader.bucket)
    _report(result, uploader.bucket)
    return result


def _failed(capture_id: str, error: str) -> UploadResult:
    return UploadResult("failed", capture_id, 0.0, error)


class BackgroundSender:
    """Sube capturas en un hilo aparte, en orden de captura. `submit` no bloquea."""

    _STOP = object()

    def __init__(
        self,
        aws: AwsConfig,
        upload_log: UploadLog,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.aws = aws
        self.upload_log = upload_log
        self.client_factory = client_factory
        self.results: list[UploadResult] = []
        self.submitted = 0
        self._queue: queue.Queue[Any] = queue.Queue()
        self._uploader: Uploader | None = None
        self._thread = threading.Thread(target=self._work, name="edge-sender", daemon=True)

    def start(self) -> BackgroundSender:
        self._thread.start()
        return self

    def submit(self, event: dict[str, Any], image: Path) -> None:
        self.upload_log.append(event["capture_id"], "pending", self.aws.bucket)
        self.submitted += 1
        self._queue.put((event, image))

    def close(self, timeout: float | None = None) -> int:
        """Espera los envíos en cola; devuelve cuántos quedaron sin intentar (siguen `pending`)."""
        self._queue.put(self._STOP)
        self._thread.join(timeout)
        return self.submitted - len(self.results)

    def _get_uploader(self) -> Uploader:
        # Se crea al primer envío y se vuelve a intentar si falló (p. ej. perfil inexistente).
        if self._uploader is None:
            self._uploader = make_uploader(self.aws, self.client_factory)
        return self._uploader

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            event, image = item
            try:
                result = send(self._get_uploader(), self.upload_log, event, image)
            except Exception as error:  # el cliente no se pudo crear: el envío falla, la app sigue
                result = _failed(event["capture_id"], f"{type(error).__name__}: {error}")
                self.upload_log.append_result(result, self.aws.bucket)
                _report(result, self.aws.bucket)
            self.results.append(result)


class UnknownCaptureError(KeyError):
    """El `capture_id` no está en `captures.jsonl`."""


def retry_ids(config: EdgeConfig, pending: bool, capture_ids: Iterable[str]) -> list[str]:
    """IDs a reenviar: los pedidos (deben existir) o los que no están `sent`/`already_sent`."""
    records = {r["event"]["capture_id"]: r for r in read_log(config.log_path)}
    if pending:
        latest = UploadLog(config.upload_log_path).latest()
        return [cid for cid in records if status_of(latest, cid) not in DONE]
    ids = list(capture_ids)
    missing = [cid for cid in ids if cid not in records]
    if missing:
        raise UnknownCaptureError(f"no están en {config.log_path}: {', '.join(missing)}")
    return ids


def retry(
    config: EdgeConfig, capture_ids: Iterable[str], client_factory: ClientFactory | None = None
) -> list[UploadResult]:
    """Reenvía el mismo evento de cada captura (sin cola, uno tras otro)."""
    records = {r["event"]["capture_id"]: r for r in read_log(config.log_path)}
    upload_log = UploadLog(config.upload_log_path)
    ids = list(capture_ids)
    try:
        uploader = make_uploader(config.aws, client_factory)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        results = [_failed(cid, message) for cid in ids]
        for result in results:
            upload_log.append_result(result, config.aws.bucket)
            _report(result, config.aws.bucket)
        return results
    return [
        send(uploader, upload_log, records[cid]["event"], image_path(config, records[cid]))
        for cid in ids
    ]
