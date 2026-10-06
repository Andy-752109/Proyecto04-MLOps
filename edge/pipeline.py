"""Una captura: frame -> recorte -> preprocesamiento -> inferencia -> evento -> disco.

Nada aquí usa la red. Por captura se escriben, en este orden:

1. `images/{capture_id}.jpg`: el frame completo (lo que apunta `image_key`);
2. `crops/{capture_id}.jpg`: la región usada para inferir, si hay recorte;
3. una línea en `captures.jsonl` con el evento validado. La línea marca la captura como
   completa, igual que el objeto de evento en S3.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edge.backend import CLASSES, InferenceBackend, sha256_file
from edge.config import EdgeConfig
from edge.event_validator import IMAGE_PREFIX, IMAGE_SUFFIX, validate_event
from edge.preprocess import bgr_to_rgb, crop_frame, preprocess, resolve_crop

log = logging.getLogger("edge")


class ModelVerificationError(RuntimeError):
    """El modelo en caché no es el esperado."""


@dataclass(frozen=True)
class Capture:
    event: dict[str, Any]
    image_path: Path
    crop_path: Path | None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def expected_sha256(config: EdgeConfig) -> str:
    """SHA esperado: el de config y, si hay registro edge, también debe coincidir con él."""
    expected = config.model.sha256
    registry = config.model.registry
    if registry is not None:
        entries = json.loads(registry.read_text(encoding="utf-8"))
        entry = entries.get(config.model.version)
        if entry is None:
            raise ModelVerificationError(f"{config.model.version} no está en {registry}")
        if entry["sha256"].lower() != expected:
            raise ModelVerificationError(
                f"model.sha256 de config ({expected}) no coincide con {registry}"
                f" ({entry['sha256']})"
            )
    return expected


def verify_model(config: EdgeConfig) -> str:
    """Verifica el modelo local contra el SHA esperado; aborta si no coincide."""
    path = config.model.path
    if not path.is_file():
        raise ModelVerificationError(f"no hay modelo en caché en {path}")
    expected = expected_sha256(config)
    actual = sha256_file(path)
    if actual != expected:
        raise ModelVerificationError(
            f"SHA-256 del modelo no coincide: esperado={expected} archivo={actual}"
        )
    return actual


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jpeg(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    Image.fromarray(rgb).save(tmp, format="JPEG", quality=95)
    os.replace(tmp, path)


class EdgeApp:
    def __init__(self, config: EdgeConfig, backend: InferenceBackend) -> None:
        self.config = config
        self.backend = backend
        self.model_sha256 = ""
        self._seen_ids: set[str] = set()

    def start(self) -> None:
        """Verifica y carga el modelo desde la caché local. No usa la red."""
        self.model_sha256 = verify_model(self.config)
        log.info("modelo %s verificado (sha256=%s)", self.config.model.path.name, self.model_sha256)
        self.backend.load(self.config.model.path)
        log.info("runtime %s listo, modelo %s", self.backend.name, self.config.model.version)
        previous = read_log(self.config.log_path)
        self._seen_ids = {line["event"]["capture_id"] for line in previous}
        log.info("log %s: %d capturas previas", self.config.log_path, len(previous))

    def _new_capture_id(self) -> str:
        while True:
            capture_id = str(uuid.uuid4())
            if capture_id not in self._seen_ids:
                return capture_id

    def process(self, frame_bgr: np.ndarray, captured_at: datetime) -> Capture:
        """Clasifica un frame ya capturado y lo persiste en disco local."""
        if not self.model_sha256:
            raise RuntimeError("start() no se ha llamado")
        capture_id = self._new_capture_id()
        rgb = bgr_to_rgb(frame_bgr)
        frame_height, frame_width = rgb.shape[:2]

        crop_px = resolve_crop(self.config.crop, frame_width, frame_height)

        started = time.perf_counter()
        region = crop_frame(rgb, crop_px)
        tensor = preprocess(region, self.config.model.image_size)
        preprocessed = time.perf_counter()
        probs = self.backend.predict(tensor)
        inferred = time.perf_counter()

        probabilities = {name: float(p) for name, p in zip(CLASSES, probs, strict=True)}
        predicted = CLASSES[int(np.argmax(probs))]
        crop = None
        if crop_px is not None:
            crop = {**crop_px, "frame_width": frame_width, "frame_height": frame_height}
        event = {
            "schema_version": "1",
            "capture_id": capture_id,
            "captured_at": captured_at.isoformat(),
            "device_id": self.config.device_id,
            "model_version": self.config.model.version,
            "model_sha256": self.model_sha256,
            "runtime": self.backend.name,
            "predicted_class": predicted,
            "confidence": probabilities[predicted],
            "probabilities": probabilities,
            "crop": crop,
            "preprocess_ms": (preprocessed - started) * 1000,
            "inference_ms": (inferred - preprocessed) * 1000,
            "image_key": f"{IMAGE_PREFIX}{capture_id}{IMAGE_SUFFIX}",
        }
        validate_event(event)

        data_dir = self.config.data_dir
        image_path = data_dir / "images" / f"{capture_id}.jpg"
        _write_jpeg(rgb, image_path)
        crop_path = None
        if crop is not None:
            crop_path = data_dir / "crops" / f"{capture_id}.jpg"
            _write_jpeg(region, crop_path)

        record = {
            "event": event,
            "image_path": image_path.relative_to(data_dir).as_posix(),
            "crop_path": crop_path.relative_to(data_dir).as_posix() if crop_path else None,
        }
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seen_ids.add(capture_id)
        return Capture(event=event, image_path=image_path, crop_path=crop_path)
