"""Preprocesamiento de inferencia equivalente al de P3, sin torch ni torchvision.

P3 (`app/training/preprocess.py`, rama no-`train`, y `app/edge_model/baseline.py`):
RGB -> Resize((128, 128)) bilineal con antialias -> ToTensor (/255, CHW) -> Normalize
ImageNet. Con una imagen PIL, torchvision hace el resize con `PIL.Image.resize(...,
BILINEAR)`, así que aquí se usa Pillow para ese paso y numpy para lo demás.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image

IMAGE_SIZE = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """OpenCV entrega BGR; el modelo espera RGB."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"se esperaba un frame HxWx3, llegó {frame_bgr.shape}")
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])


def crop_frame(frame: np.ndarray, crop: Mapping[str, int] | None) -> np.ndarray:
    """Recorta `frame` (HxWxC) con `crop` = {x, y, width, height}; None devuelve el frame."""
    if crop is None:
        return frame
    height, width = frame.shape[:2]
    x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError(f"el recorte {dict(crop)} no cabe en un frame de {width}x{height}")
    return frame[y : y + h, x : x + w]


def preprocess(rgb: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Imagen RGB uint8 HxWx3 -> tensor float32 [1, 3, S, S] normalizado (ImageNet)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"se esperaba RGB uint8 HxWx3, llegó {rgb.dtype} {rgb.shape}")
    resized = Image.fromarray(rgb).resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[np.newaxis])
