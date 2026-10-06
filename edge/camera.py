"""Cámara con OpenCV: igual para webcam integrada o USB, solo cambia `camera.index`."""

from __future__ import annotations

import numpy as np

from edge.config import CameraConfig


class CameraError(RuntimeError):
    """No se pudo abrir o leer la cámara."""


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        import cv2

        self._cv2 = cv2
        self._config = config
        self._capture = cv2.VideoCapture(config.index)
        if not self._capture.isOpened():
            raise CameraError(f"no se pudo abrir la cámara {config.index}")
        if config.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        if config.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._discard(config.warmup_frames)

    def _discard(self, frames: int) -> None:
        # Vacía el búfer: en modo manual el siguiente read() devolvería un frame viejo.
        for _ in range(frames):
            self._capture.grab()

    def read(self) -> np.ndarray:
        """Devuelve un frame BGR uint8 HxWx3 recién capturado."""
        self._discard(self._config.warmup_frames)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError(f"la cámara {self._config.index} no devolvió un frame")
        return frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
