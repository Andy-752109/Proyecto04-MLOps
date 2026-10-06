"""Lectura y validación de `config.yaml` (sin secretos).

Las rutas relativas se resuelven contra la carpeta del archivo de configuración, así la app
funciona igual sin importar desde dónde se lance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODES = ("manual", "interval")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """La configuración no es válida."""


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int | None = None
    height: int | None = None
    warmup_frames: int = 5


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    version: str
    sha256: str
    runtime: str = "onnxruntime"
    threads: int | None = None
    image_size: int = 128
    registry: Path | None = None


@dataclass(frozen=True)
class EdgeConfig:
    device_id: str
    camera: CameraConfig
    model: ModelConfig
    crop: dict[str, int] | None
    mode: str
    interval_seconds: float
    data_dir: Path
    bucket: str

    @property
    def log_path(self) -> Path:
        return self.data_dir / "captures.jsonl"


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if section.get(key) in (None, ""):
        raise ConfigError(f"falta {where}{key}")
    return section[key]


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_crop(raw: Any) -> dict[str, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("crop debe ser null o {x, y, width, height}")
    keys = ("x", "y", "width", "height")
    if set(raw) != set(keys):
        raise ConfigError(f"crop debe tener exactamente {keys}, tiene {sorted(raw)}")
    if not all(isinstance(raw[k], int) and not isinstance(raw[k], bool) for k in keys):
        raise ConfigError("los valores de crop deben ser enteros (píxeles)")
    if raw["x"] < 0 or raw["y"] < 0 or raw["width"] <= 0 or raw["height"] <= 0:
        raise ConfigError("crop: x, y >= 0 y width, height > 0")
    return {k: raw[k] for k in keys}


def load_config(path: Path) -> EdgeConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no existe {path}; copia edge/config.example.yaml a edge/config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent

    model_raw = _require(raw, "model", "")
    sha = str(_require(model_raw, "sha256", "model.")).lower()
    if not SHA256_RE.match(sha):
        raise ConfigError("model.sha256 debe tener 64 caracteres hexadecimales")
    model = ModelConfig(
        path=_resolve(base, _require(model_raw, "path", "model.")),
        version=str(_require(model_raw, "version", "model.")),
        sha256=sha,
        runtime=model_raw.get("runtime", "onnxruntime"),
        threads=model_raw.get("threads"),
        image_size=int(model_raw.get("image_size", 128)),
        registry=_resolve(base, model_raw.get("registry")),
    )

    camera = CameraConfig(**(raw.get("camera") or {}))

    capture = raw.get("capture") or {}
    mode = capture.get("mode", "manual")
    if mode not in MODES:
        raise ConfigError(f"capture.mode debe ser uno de {MODES}, es {mode!r}")
    interval = float(capture.get("interval_seconds", 5))
    if interval <= 0:
        raise ConfigError("capture.interval_seconds debe ser > 0")

    return EdgeConfig(
        device_id=str(_require(raw, "device_id", "")),
        camera=camera,
        model=model,
        crop=_parse_crop(raw.get("crop")),
        mode=mode,
        interval_seconds=interval,
        data_dir=_resolve(base, raw.get("data_dir", "data")),
        bucket=str((raw.get("aws") or {}).get("bucket", "")),
    )
