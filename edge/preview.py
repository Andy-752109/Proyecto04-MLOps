"""Vista en vivo de la cámara para la interfaz gráfica, sin depender de Tkinter.

`FrameGrabber` lee la fuente (cámara o imagen) en un hilo y guarda el último cuadro; la
ventana lo dibuja y el botón de captura usa ese mismo cuadro, así que la foto es lo que se ve.
`render_preview` ajusta el cuadro al tamaño de la ventana y marca la zona que recorta la
config, que es la que ve el modelo.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from edge.preprocess import bgr_to_rgb, resolve_crop

log = logging.getLogger("edge")

DIM_OUTSIDE_CROP = 0.45  # brillo de lo que queda fuera de la zona clasificada
CROP_OUTLINE = (255, 255, 255)


class Source(Protocol):
    """Lo que el hilo necesita de `Camera` e `ImageSource`."""

    def read_live(self) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8 HxWx3; quien lo recibe no debe modificarlo
    at: float  # time.monotonic() al leerlo
    number: int  # consecutivo: la ventana solo vuelve a dibujar cuando cambia


class FrameGrabber:
    """Un hilo dueño de la fuente: la abre, la lee sin parar y la cierra al terminar.

    Abrir, leer y cerrar ocurren en el mismo hilo (OpenCV no asegura otra cosa). Un fallo de
    lectura suelto no corta la vista: solo si pasan `patience_s` segundos sin un cuadro bueno
    se da por perdida la cámara (p. ej. USB desconectada) y queda el mensaje en `error`.
    """

    def __init__(
        self,
        open_source: Callable[[], Source],
        max_fps: float = 30.0,
        patience_s: float = 2.0,
    ) -> None:
        self._open_source = open_source
        self._min_interval = 1.0 / max_fps
        self._patience_s = patience_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: Frame | None = None
        self._error: str | None = None
        self._fps = 0.0
        self._thread = threading.Thread(target=self._work, name="edge-preview", daemon=True)

    def start(self) -> FrameGrabber:
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    def _fail(self, error: Exception) -> None:
        # CameraError ya trae un mensaje legible; de cualquier otra excepción se muestra el tipo.
        named = isinstance(error, RuntimeError)
        message = str(error) if named else f"{type(error).__name__}: {error}"
        log.error("vista en vivo detenida: %s", message)
        with self._lock:
            self._error = message

    def _work(self) -> None:
        try:
            source = self._open_source()
        except Exception as error:  # el mensaje llega a la ventana, que no tiene consola
            self._fail(error)
            return
        try:
            self._read_forever(source)
        finally:
            source.close()

    def _read_forever(self, source: Source) -> None:
        number = 0
        last_ok = last_frame_at = time.monotonic()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                image = source.read_live()
            except Exception as error:
                if started - last_ok > self._patience_s:
                    self._fail(error)
                    return
                self._stop.wait(0.05)
                continue
            number += 1
            last_ok = time.monotonic()
            elapsed = last_ok - last_frame_at
            last_frame_at = last_ok
            with self._lock:
                self._frame = Frame(image, last_ok, number)
                if number > 1 and elapsed > 0:
                    # Promedio móvil: el número no salta de un cuadro a otro.
                    instant = 1 / elapsed
                    self._fps = instant if not self._fps else 0.9 * self._fps + 0.1 * instant
            # Una imagen fija se lee al instante: sin esto la copia correría a miles de veces/s.
            self._stop.wait(max(0.0, self._min_interval - (time.monotonic() - started)))


def fit_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """Tamaño de `width`x`height` ajustado a `max_width`x`max_height` sin deformarlo."""
    scale = min(max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render_preview(
    frame_bgr: np.ndarray,
    max_size: tuple[int, int],
    crop: Mapping[str, Any] | None,
    show_crop: bool = True,
) -> Image.Image:
    """Cuadro BGR -> imagen RGB ajustada a `max_size`.

    Con `show_crop` y un recorte en la config, lo que queda fuera de la zona que ve el modelo
    se oscurece y la zona lleva un borde blanco.
    """
    height, width = frame_bgr.shape[:2]
    size = fit_size(width, height, *max_size)
    image = Image.fromarray(bgr_to_rgb(frame_bgr)).resize(size, Image.Resampling.BILINEAR)
    box = resolve_crop(crop, width, height) if show_crop else None
    if box is None:
        return image
    scale = size[0] / width
    # Un recorte en píxeles puede salirse del cuadro (la captura lo rechazaría): se dibuja lo
    # que cabe.
    left = min(max(round(box["x"] * scale), 0), size[0])
    top = min(max(round(box["y"] * scale), 0), size[1])
    right = min(max(round((box["x"] + box["width"]) * scale), 0), size[0])
    bottom = min(max(round((box["y"] + box["height"]) * scale), 0), size[1])
    if right - left < 2 or bottom - top < 2:
        return image
    pixels = np.asarray(image)
    dimmed = (pixels * DIM_OUTSIDE_CROP).astype(np.uint8)
    dimmed[top:bottom, left:right] = pixels[top:bottom, left:right]
    framed = Image.fromarray(dimmed)
    ImageDraw.Draw(framed).rectangle(
        (left, top, right - 1, bottom - 1), outline=CROP_OUTLINE, width=2
    )
    return framed
