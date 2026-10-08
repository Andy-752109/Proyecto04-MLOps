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
    warmup_frames: int = 10
    # Tiempo mínimo (s) leyendo cuadros antes de cada captura. `grab()` de DirectShow no espera
    # un cuadro nuevo, así que el cuadro guardado puede ser de la escena anterior.
    settle_seconds: float = 0.5
    api: str = "auto"  # auto = DirectShow en Windows, el de OpenCV en otro SO
    # Si se define, tiene prioridad sobre `index`: se abre la cámara cuyo nombre DirectShow
    # contiene este texto. El orden de los índices cambia entre arranques en Windows.
    name: str | None = None


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
class AwsConfig:
    """Destino del envío (P4-09). Sin `bucket` la app solo clasifica y guarda en local."""

    bucket: str = ""
    profile: str | None = None  # None = cadena por defecto de boto3 (AWS_PROFILE, etc.)
    region: str = "us-east-1"

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)


@dataclass(frozen=True)
class EdgeConfig:
    device_id: str
    camera: CameraConfig
    model: ModelConfig
    crop: dict[str, Any] | None
    mode: str
    interval_seconds: float
    data_dir: Path
    aws: AwsConfig

    @property
    def log_path(self) -> Path:
        return self.data_dir / "captures.jsonl"

    @property
    def upload_log_path(self) -> Path:
        """Estados de envío, separados del log de inferencia (`captures.jsonl`)."""
        return self.data_dir / "uploads.jsonl"


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if section.get(key) in (None, ""):
        raise ConfigError(f"falta {where}{key}")
    return section[key]


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_crop(raw: Any) -> dict[str, Any] | None:
    """null, {center_square: fracción} o {x, y, width, height} en píxeles."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("crop debe ser null, {center_square} o {x, y, width, height}")
    if set(raw) == {"center_square"}:
        fraction = raw["center_square"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ConfigError("crop.center_square debe ser un número")
        if not 0 < fraction <= 1:
            raise ConfigError("crop.center_square debe estar en (0, 1]")
        return {"center_square": float(fraction)}
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
    if camera.api not in ("auto", "dshow", "msmf", "v4l2", "any"):
        raise ConfigError("camera.api debe ser auto, dshow, msmf, v4l2 o any")
    if camera.settle_seconds < 0:
        raise ConfigError("camera.settle_seconds no puede ser negativo")
    if camera.name is not None and not camera.name.strip():
        raise ConfigError("camera.name no puede estar vacío (usa null para elegir por índice)")

    capture = raw.get("capture") or {}
    mode = capture.get("mode", "manual")
    if mode not in MODES:
        raise ConfigError(f"capture.mode debe ser uno de {MODES}, es {mode!r}")
    interval = float(capture.get("interval_seconds", 10))
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
        aws=_parse_aws(raw.get("aws")),
    )


def _parse_aws(raw: Any) -> AwsConfig:
    if raw is None:
        return AwsConfig()
    if not isinstance(raw, dict):
        raise ConfigError("aws debe ser un objeto con bucket, profile y region")
    unknown = set(raw) - {"bucket", "profile", "region"}
    if unknown:
        raise ConfigError(f"aws tiene claves desconocidas: {sorted(unknown)}")
    profile = raw.get("profile")
    if profile is not None and not str(profile).strip():
        raise ConfigError("aws.profile no puede estar vacío (usa null para la cadena por defecto)")
    return AwsConfig(
        bucket=str(raw.get("bucket") or "").strip(),
        profile=None if profile is None else str(profile),
        region=str(raw.get("region") or "us-east-1"),
    )
