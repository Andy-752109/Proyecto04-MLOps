"""Cámara con OpenCV: igual para webcam integrada o USB.

Se elige con `camera.index` o, mejor, con `camera.name`: en Windows el orden de los índices
cambia entre arranques (la USB fue 1 y luego 0), pero el nombre del dispositivo no.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from edge.config import CameraConfig


class CameraError(RuntimeError):
    """No se pudo abrir o leer la cámara."""


def _api_preference(cv2: Any, api: str) -> int:
    if api == "auto":
        api = "dshow" if sys.platform == "win32" else "any"
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "any": cv2.CAP_ANY,
    }[api]


def list_camera_names() -> list[str]:
    """Nombres DirectShow en el mismo orden que los índices de `cv2.CAP_DSHOW` (solo Windows)."""
    if sys.platform != "win32":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError as exc:
        raise CameraError("camera.name necesita pygrabber: pip install pygrabber") from exc
    return list(FilterGraph().get_input_devices())


def resolve_camera_index(config: CameraConfig, list_names: Callable[[], list[str]]) -> int:
    """`camera.index`, o el índice del primer dispositivo cuyo nombre contiene `camera.name`."""
    if config.name is None:
        return config.index
    names = list_names()
    if not names:
        raise CameraError(
            f"no se pudo listar las cámaras para buscar {config.name!r}; usa camera.index"
        )
    wanted = config.name.casefold()
    for index, name in enumerate(names):
        if wanted in name.casefold():
            return index
    available = ", ".join(f"{i}: {n}" for i, n in enumerate(names))
    raise CameraError(f"no hay una cámara llamada {config.name!r}; disponibles: {available}")


class Camera:
    def __init__(
        self, config: CameraConfig, list_names: Callable[[], list[str]] | None = None
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._config = config
        self._index = resolve_camera_index(config, list_names or list_camera_names)
        self._capture = cv2.VideoCapture(self._index, _api_preference(cv2, config.api))
        if not self._capture.isOpened():
            raise CameraError(
                f"no se pudo abrir la cámara {self._index} (api {config.api});"
                " prueba otro camera.index o camera.name"
            )
        if config.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        if config.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._discard(config.warmup_frames)

    def _discard(self, frames: int) -> None:
        # Vacía el búfer: en modo manual el siguiente read() devolvería un frame viejo.
        for _ in range(frames):
            self._capture.grab()

    @property
    def index(self) -> int:
        """Índice real con el que se abrió (puede diferir de `camera.index` si hay `name`)."""
        return self._index

    def read(self) -> np.ndarray:
        """Devuelve un frame BGR uint8 HxWx3 recién capturado."""
        self._discard(self._config.warmup_frames)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError(f"la cámara {self._index} no devolvió un frame")
        return frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
