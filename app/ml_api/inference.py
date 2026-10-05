"""`POST /predict` (P3-16, #24): clasifica una imagen con el modelo activo.

`load_active_model` se conecta a lo que publica P3-14 (#21, PR #62):
- `models/registry.json` (montado en `settings.models_dir`, no horneado en
  la imagen -- una publicación nueva no debe esperar a reconstruir ml-api);
  la versión activa es la de semver más alto (hoy 1.0.0, el candidato
  seleccionado; 0.1.0 es el ensayo).
- el paquete (`s3_path`/`VersionId`/`sha256` de esa entrada) se descarga de
  S3 real con `boto3` (perfil `AWS_PROFILE`, mismo mecanismo que ya usa
  `trainer-worker` -- ver docker-compose.yml), se verifica contra el SHA-256
  del registro, y se cachea en disco por versión+hash: una corrida de
  `/predict` no debe volver a bajar ~decenas de MB por cada request.
- `config.json` adentro del paquete ya es un `TrainingConfig` completo
  (`publish.py` lo saca de `typed_params(run["params"])`), así que
  `training.model.build_model` lo reconstruye igual que `recarga_limpia.py`
  (P3-14) y `final.py`/`smoke.py` (P3-08/P3-10/P3-13).
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sqlalchemy import Engine, text
from starlette.requests import Request
from starlette.responses import JSONResponse

from crops.build import pixel_bounds
from evaluation.contracts import CLASSES
from ml_api.contracts import PredictionResponse
from storage.db import get_engine
from storage.object_store import get_bucket_name, get_minio_client
from storage.settings import Settings
from training.config import TrainingConfig
from training.model import build_model
from training.preprocess import get_preprocessing_transforms

logger = logging.getLogger("ml-api")

# Mismo límite que `MAX_UPLOAD_SIZE_BYTES` del backend (backend/src/config/env.ts) --
# el ticket lo deja "por definir, propuesta 5 MB"; ya es el default real del otro lado.
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024

# Firmas reales de archivo (no el Content-Type del upload, que cualquiera puede
# falsear) -- el dataset de entrenamiento son recortes JPEG; PNG se acepta porque
# es el formato más común para una foto nueva subida desde el portal.
_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)


@dataclass(frozen=True)
class ModelBundle:
    """Un modelo ya cargado y listo para inferir -- lo que `load_active_model`
    (o su inyección de prueba) debe producir."""

    model: nn.Module
    image_size: int
    version: str
    checkpoint_sha256: str


LoadModel = Callable[[], ModelBundle]
FetchCrop = Callable[[int, int], Image.Image | None]


def _active_version(registry: dict, models_dir: Path) -> str:
    """La de `models/active_version.json` (la marca POST /models/active, P3-15);
    si no existe, la de semver más alto (hoy 1.0.0 sobre 0.1.0)."""
    marker = models_dir / "active_version.json"
    if marker.is_file():
        try:
            chosen = json.loads(marker.read_text(encoding="utf-8")).get("active_version")
        except (json.JSONDecodeError, AttributeError):
            chosen = None
        if isinstance(chosen, str) and chosen:
            if chosen not in registry:
                raise RuntimeError(f"La versión activa {chosen} no está en registry.json")
            return chosen
    return max(registry, key=lambda version: tuple(int(part) for part in version.split(".")))


def _split_s3_path(s3_path: str) -> tuple[str, str]:
    bucket, _, key = s3_path.removeprefix("s3://").partition("/")
    return bucket, key


def _package_cache_dir(version: str, sha256: str) -> Path:
    return Path(tempfile.gettempdir()) / "ml-api-model-cache" / f"{version}-{sha256}"


_bundle_cache: dict[str, ModelBundle] = {}
_load_lock = threading.Lock()


def load_active_model(*, models_dir: Path | None = None, s3_client=None) -> ModelBundle:
    models_dir = models_dir or Settings().models_dir
    registry = json.loads((models_dir / "registry.json").read_text(encoding="utf-8"))
    version = _active_version(registry, models_dir)

    cached = _bundle_cache.get(version)
    if cached is not None:
        return cached
    with _load_lock:
        cached = _bundle_cache.get(version)
        if cached is not None:
            return cached
        return _load_bundle(version, registry[version], s3_client)


def _load_bundle(version: str, entry: dict, s3_client) -> ModelBundle:
    extracted = _package_cache_dir(version, entry["sha256"]) / "extracted"
    if not extracted.is_dir():
        extracted.parent.mkdir(parents=True, exist_ok=True)
        tar_path = extracted.parent / "package.tar.gz"

        import boto3

        s3 = s3_client or boto3.client("s3")
        bucket, key = _split_s3_path(entry["s3_path"])
        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=str(tar_path),
            ExtraArgs={"VersionId": entry["VersionId"]},
        )

        actual_sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if actual_sha256 != entry["sha256"]:
            tar_path.unlink()
            raise RuntimeError(
                f"SHA-256 del paquete {version} no coincide: "
                f"registro={entry['sha256']} descargado={actual_sha256}"
            )

        # Se extrae aparte y se publica con un rename: un fallo a medias nunca deja
        # `extracted/` incompleto.
        staging = Path(tempfile.mkdtemp(dir=extracted.parent, prefix="extracting-"))
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(staging, filter="data")
            os.replace(staging, extracted)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            tar_path.unlink(missing_ok=True)

    config = TrainingConfig.model_validate(
        json.loads((extracted / "config.json").read_text(encoding="utf-8"))
    )
    model = build_model(config)
    checkpoint = extracted / "artifacts" / "checkpoint" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    bundle = ModelBundle(
        model=model,
        image_size=config.image_size,
        version=version,
        checkpoint_sha256=entry["checkpoint_sha256"],
    )
    # Una sola versión activa a la vez: si cambió, no hay razón para seguir
    # cargando la anterior en memoria.
    _bundle_cache.clear()
    _bundle_cache[version] = bundle
    return bundle


def fetch_crop_from_annotation(
    image_id: int, annotation_id: int, *, engine: Engine | None = None
) -> Image.Image | None:
    """Recorta al vuelo una anotación ya existente (imagen + bbox en MariaDB,
    archivo en MinIO) -- `None` si la pareja `image_id`/`annotation_id` no
    existe. Usa `crops.build.pixel_bounds`, el mismo criterio con el que
    `crops/build.py` materializa `crops/images/*.jpg` en disco (P3-04): aquí
    no se guarda nada, solo se recorta en memoria para esta predicción.
    """
    engine = engine or get_engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT i.storage_key, i.width, i.height, "
                "a.bbox_x, a.bbox_y, a.bbox_width, a.bbox_height "
                "FROM `annotations` a JOIN `images` i ON i.id = a.image_id "
                "WHERE a.id = :annotation_id AND a.image_id = :image_id"
            ),
            {"annotation_id": annotation_id, "image_id": image_id},
        ).fetchone()
    if row is None:
        return None
    storage_key, width, height, bbox_x, bbox_y, bbox_w, bbox_h = row

    response = get_minio_client().get_object(get_bucket_name(), storage_key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()

    image = Image.open(io.BytesIO(data)).convert("RGB")
    x0, y0, x1, y1 = pixel_bounds([bbox_x, bbox_y, bbox_w, bbox_h], width, height)
    return image.crop((x0, y0, x1, y1))


def _sniff_image_type(data: bytes) -> str | None:
    for signature, mime in _MAGIC_BYTES:
        if data.startswith(signature):
            return mime
    return None


def _decode_image(data: bytes) -> Image.Image | None:
    """`None` si no es una imagen real, sin importar lo que diga su extensión
    o Content-Type -- mismo criterio que `sharp(buffer).metadata()` en
    `image-upload.service.ts` (backend), pero en Python."""
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        # `verify()` deja el objeto inutilizable para lo que sigue -- reabrir.
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (OSError, ValueError):
        return None


def predict_image(bundle: ModelBundle, image: Image.Image) -> PredictionResponse:
    transform = get_preprocessing_transforms("val", bundle.image_size)
    tensor = transform(image).unsqueeze(0)
    bundle.model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(bundle.model(tensor), dim=1)[0].tolist()
    probabilities_by_class = dict(zip(CLASSES, probabilities, strict=True))
    predicted_label = max(probabilities_by_class, key=probabilities_by_class.get)
    return PredictionResponse(
        predicted_label=predicted_label,
        probabilities=probabilities_by_class,
        model_version=bundle.version,
        checkpoint_sha256=bundle.checkpoint_sha256,
    )


class _PredictRequestError(Exception):
    """Trae su propio status HTTP -- 400 para un request mal formado, 404
    cuando `image_id`/`annotation_id` no corresponden a nada."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _parse_recrop_ids(form) -> tuple[int, int] | None:
    """`None` si no vienen los dos campos; error si vienen pero no son
    enteros -- distinto de "no se mandó nada", para un 400 más claro."""
    image_id_raw, annotation_id_raw = form.get("image_id"), form.get("annotation_id")
    if image_id_raw is None and annotation_id_raw is None:
        return None
    if image_id_raw is None or annotation_id_raw is None:
        raise _PredictRequestError("deben venir juntos image_id y annotation_id")
    try:
        return int(image_id_raw), int(annotation_id_raw)
    except (TypeError, ValueError) as error:
        raise _PredictRequestError("image_id/annotation_id deben ser enteros") from error


async def _resolve_image(form, fetch_crop: FetchCrop) -> Image.Image:
    """Dos flujos posibles (checklist de P3-16, #24): un archivo nuevo en el
    campo `image`, o `{image_id, annotation_id}` para recortar una anotación
    ya existente -- nunca los dos a la vez, nunca ninguno. Levanta
    `_PredictRequestError` (con su propio status) si no se puede resolver.
    """
    upload = form.get("image")
    has_upload = upload is not None and hasattr(upload, "read")
    recrop_ids = _parse_recrop_ids(form)

    if has_upload and recrop_ids is not None:
        raise _PredictRequestError("manda 'image' o {'image_id','annotation_id'}, no ambos")
    if not has_upload and recrop_ids is None:
        raise _PredictRequestError("falta el campo 'image' o {'image_id','annotation_id'}")

    if has_upload:
        data = await upload.read(MAX_UPLOAD_SIZE_BYTES + 1)
        if len(data) > MAX_UPLOAD_SIZE_BYTES:
            raise _PredictRequestError(
                f"la imagen supera el máximo de {MAX_UPLOAD_SIZE_BYTES} bytes"
            )
        if _sniff_image_type(data) is None:
            raise _PredictRequestError("tipo de archivo no soportado")
        image = _decode_image(data)
        if image is None:
            raise _PredictRequestError("no se pudo decodificar la imagen")
        return image

    image_id, annotation_id = recrop_ids
    image = await asyncio.to_thread(fetch_crop, image_id, annotation_id)
    if image is None:
        raise _PredictRequestError(
            f"no existe la anotación {annotation_id} de la imagen {image_id}", status_code=404
        )
    return image


def predict_route(load_model: LoadModel, fetch_crop: FetchCrop = fetch_crop_from_annotation):
    async def handler(request: Request) -> JSONResponse:
        form = await request.form()
        try:
            image = await _resolve_image(form, fetch_crop)
        except _PredictRequestError as error:
            return JSONResponse({"error": str(error)}, status_code=error.status_code)

        try:
            bundle = await asyncio.to_thread(load_model)
        except Exception:
            # Registro ausente, SHA-256 que no coincide, S3/SSO caído -- lo que
            # sea, es "el modelo no está disponible ahora". El detalle va al log,
            # no al cliente.
            logger.exception("No se pudo cargar el modelo activo")
            return JSONResponse(
                {"error": "El modelo no está disponible en este momento"}, status_code=503
            )

        result = await asyncio.to_thread(predict_image, bundle, image)
        return JSONResponse(result.model_dump(mode="json"), status_code=200)

    return handler
