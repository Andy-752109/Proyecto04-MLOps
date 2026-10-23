"""Cámara con OpenCV: igual para webcam integrada o USB.

Se elige con `camera.index` o, mejor, con `camera.name`: en Windows el orden de los índices
cambia entre arranques (la USB fue 1 y luego 0), pero el nombre del dispositivo no.

`ImageSource` tiene la misma interfaz y lee un archivo de imagen: sirve para recorrer la app
sin cámara.

`read()` es para el modo por consola (una captura a la vez, con vaciado previo); `read_live()`
es para la interfaz gráfica, que mantiene el flujo abierto y leyéndose.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
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
        self._discard()

    def _discard(self) -> None:
        """Descarta cuadros hasta tener uno reciente.

        `grab()` de DirectShow no espera un cuadro nuevo: devuelve al instante el último
        guardado, que tras un rato sin leer (modo manual) es de la escena anterior. Se lee
        (`read()`) al menos `warmup_frames` veces y durante `settle_seconds` para que el
        controlador entregue cuadros actuales.
        """
        config = self._config
        deadline = time.monotonic() + config.settle_seconds
        reads = 0
        while reads < config.warmup_frames or time.monotonic() < deadline:
            self._capture.read()
            reads += 1

    @property
    def index(self) -> int:
        """Índice real con el que se abrió (puede diferir de `camera.index` si hay `name`)."""
        return self._index

    def read(self) -> np.ndarray:
        """Devuelve un frame BGR uint8 HxWx3 recién capturado."""
        self._discard()
        return self.read_live()

    def read_live(self) -> np.ndarray:
        """Siguiente cuadro del flujo, sin vaciar nada: para la vista en vivo (`edge gui`).

        Si el flujo se lee sin parar, el último cuadro siempre es reciente y no hace falta el
        vaciado de `read()`.
        """
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


class ImageSource:
    """Un JPG/PNG en lugar de la cámara (`edge run --image`), con la interfaz de `Camera`.

    Cada `read()` devuelve el mismo frame BGR uint8 HxWx3, así que el recorte, el
    preprocesamiento y el evento son los de una captura real. Se decodifica al abrir: un
    archivo que falta o no es imagen falla antes de la primera captura.
    """

    def __init__(self, path: Path) -> None:
        import cv2

        self._path = path
        if path.is_dir():
            raise CameraError(f"{path} es una carpeta, no una imagen")
        try:
            # imread() no abre rutas con tildes en Windows; fromfile + imdecode sí.
            data = np.fromfile(path, dtype=np.uint8)
        except FileNotFoundError as exc:
            raise CameraError(f"no existe la imagen {path}") from exc
        except OSError as exc:
            raise CameraError(f"no se pudo leer la imagen {path}: {exc.strerror or exc}") from exc
        try:
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except cv2.error:
            frame = None
        if frame is None:
            raise CameraError(f"{path} no es una imagen legible (se espera JPG o PNG)")
        self._frame = frame

    def read(self) -> np.ndarray:
        """Devuelve una copia del frame BGR uint8 HxWx3."""
        return self._frame.copy()

    def read_live(self) -> np.ndarray:
        """Como `read()`: la vista en vivo de una imagen es siempre el mismo cuadro."""
        return self.read()

    def close(self) -> None:
        pass

    def __enter__(self) -> ImageSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
