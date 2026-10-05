"""P3-16 (#24): `POST /predict`, los dos flujos de imagen.

`load_model`/`fetch_crop` se inyectan -- igual que `create_job`/`cancel_job`
en `ml_api.server` -- así la mayoría de estas pruebas nunca tocan MariaDB/
MinIO/S3 reales. Los `ModelBundle` de prueba usan un modelo real de
`training.model.build_model` (sin entrenar): softmax es real, no un mock, así
que "las probabilidades suman 1" prueba algo de verdad.

La sección final SÍ prueba `load_active_model` (P3-14/#21, PR #62 ya
mergeado) de punta a punta -- con un paquete real (armado aquí mismo, mismo
formato que `publish.py`) y un `s3_client` de prueba que nunca sale a la red,
pero ejercitando la función real: selección de versión, verificación de
SHA-256, extracción y reconstrucción del modelo.

"Enviar a cola de anotación" queda fuera de este archivo -- ver seguimiento
en el PR.
"""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient

pytest.importorskip("torch")

from ml_api import inference as inference_module
from ml_api.inference import (
    MAX_UPLOAD_SIZE_BYTES,
    ModelBundle,
    _active_version,
    _decode_image,
    _sniff_image_type,
    load_active_model,
    predict_image,
)
from ml_api.server import create_app
from tests._mcp_fixtures import mcp_settings
from training.config import TrainingConfig
from training.grid import FROZEN_MIN_DELTA, FROZEN_SEEDS, GRID
from training.model import build_model

CONFIG = TrainingConfig(
    **{**GRID["r01"], **FROZEN_SEEDS, "patience": 3, "min_delta": FROZEN_MIN_DELTA}
)


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """`load_active_model` cachea el bundle en memoria entre llamadas (para
    no re-descargar en cada /predict) -- sin esto, una prueba vería el bundle
    que dejó la anterior."""
    inference_module._bundle_cache.clear()
    yield
    inference_module._bundle_cache.clear()


def a_bundle(*, seed_model: int = 45, version: str = "v1.0.0", sha: str = "a" * 64) -> ModelBundle:
    config = CONFIG.model_copy(update={"seed_model": seed_model})
    return ModelBundle(
        model=build_model(config),
        image_size=config.image_size,
        version=version,
        checkpoint_sha256=sha,
    )


def a_jpeg(color=(190, 70, 70)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def a_png(color=(70, 70, 190)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def client_for(monkeypatch, tmp_path, *, load_model=None, fetch_crop=None) -> TestClient:
    settings = mcp_settings(
        monkeypatch, tmp_path / "dataset", tmp_path / "reports", tmp_path / "derived"
    )
    return TestClient(create_app(settings, load_model=load_model, fetch_crop=fetch_crop))


# --- predict_image (puro, sin HTTP) ---------------------------------------------


def test_predict_image_probabilities_sum_to_one():
    bundle = a_bundle()
    result = predict_image(bundle, Image.open(io.BytesIO(a_jpeg())).convert("RGB"))

    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(result.probabilities) == {"cat", "dog"}
    assert result.predicted_label in result.probabilities


def test_predict_image_passes_through_the_bundles_version_and_sha256():
    bundle = a_bundle(version="v0.1.0", sha="b" * 64)
    result = predict_image(bundle, Image.open(io.BytesIO(a_jpeg())).convert("RGB"))

    assert result.model_version == "v0.1.0"
    assert result.checkpoint_sha256 == "b" * 64


def test_predict_image_output_changes_with_different_weights():
    """ "Con pesos alterados cambia la salida" (checklist del ticket) -- sin
    reglas fijas sobre cuánto, solo que no sea casualidad que sea igual."""
    image = Image.open(io.BytesIO(a_jpeg())).convert("RGB")
    probs_a = predict_image(a_bundle(seed_model=45), image).probabilities
    probs_b = predict_image(a_bundle(seed_model=99), image).probabilities

    assert probs_a != probs_b


# --- decodificación / validación (puras) ----------------------------------------


def test_sniff_image_type_recognizes_jpeg_and_png():
    assert _sniff_image_type(a_jpeg()) == "image/jpeg"
    assert _sniff_image_type(a_png()) == "image/png"


def test_sniff_image_type_rejects_anything_else():
    assert _sniff_image_type(b"no soy una imagen, solo texto") is None


def test_decode_image_rejects_bytes_with_a_valid_signature_but_corrupt_body():
    # Encabezado JPEG real, pero el resto es basura -- Pillow debe tronar al decodificar.
    corrupt = b"\xff\xd8\xff" + b"\x00" * 50
    assert _decode_image(corrupt) is None


def test_decode_image_accepts_a_real_jpeg():
    assert _decode_image(a_jpeg()) is not None


# --- POST /predict (HTTP, con load_model inyectado) -----------------------------


def test_predict_route_returns_200_with_the_prediction(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle)
    response = client.post("/predict", files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == {"cat", "dog"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["model_version"] == "v1.0.0"
    assert body["checkpoint_sha256"] == "a" * 64


def test_predict_route_rejects_a_non_image_file_with_400(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle)
    response = client.post(
        "/predict", files={"image": ("nota.txt", b"esto no es una imagen", "text/plain")}
    )

    assert response.status_code == 400


def test_predict_route_rejects_a_corrupt_image_with_400(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle)
    corrupt = b"\xff\xd8\xff" + b"\x00" * 50
    response = client.post("/predict", files={"image": ("cat.jpg", corrupt, "image/jpeg")})

    assert response.status_code == 400


def test_predict_route_rejects_an_oversized_upload(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle)
    oversized = a_jpeg() + bytes(MAX_UPLOAD_SIZE_BYTES)
    response = client.post("/predict", files={"image": ("cat.jpg", oversized, "image/jpeg")})

    assert response.status_code == 400


def test_predict_route_requires_the_image_field(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle)
    response = client.post("/predict", data={"not_image": "x"})

    assert response.status_code == 400


def test_predict_route_changing_the_active_version_changes_the_sha256(monkeypatch, tmp_path):
    """ "Cambiar de versión cambia el SHA-256" (checklist): se simula aquí con
    dos `load_model` distintos -- la ruta siempre refleja lo que devuelva
    `load_model()`, sin cachear el bundle entre llamadas."""
    client_v1 = client_for(
        monkeypatch, tmp_path, load_model=lambda: a_bundle(version="v0.1.0", sha="c" * 64)
    )
    client_v2 = client_for(
        monkeypatch, tmp_path, load_model=lambda: a_bundle(version="v1.0.0", sha="d" * 64)
    )

    response_v1 = client_v1.post("/predict", files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")})
    response_v2 = client_v2.post("/predict", files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")})

    assert response_v1.json()["checkpoint_sha256"] == "c" * 64
    assert response_v2.json()["checkpoint_sha256"] == "d" * 64


def test_predict_route_is_503_when_the_model_cannot_be_loaded(monkeypatch, tmp_path):
    """Cualquier falla de `load_model()` (registro ausente, S3/SSO caído,
    SHA-256 que no coincide) es "no disponible ahora", no un 400 ni un 500
    sin explicación."""
    empty_models_dir = tmp_path / "no-hay-modelos-aqui"
    client = client_for(
        monkeypatch,
        tmp_path,
        load_model=lambda: load_active_model(models_dir=empty_models_dir),
    )
    response = client.post("/predict", files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")})

    assert response.status_code == 503


# --- POST /predict: flujo de recorte existente ({image_id, annotation_id}) -----


def test_predict_route_accepts_image_id_and_annotation_id(monkeypatch, tmp_path):
    crop = Image.open(io.BytesIO(a_jpeg())).convert("RGB")
    calls = []

    def fetch_crop(image_id, annotation_id):
        calls.append((image_id, annotation_id))
        return crop

    client = client_for(monkeypatch, tmp_path, load_model=a_bundle, fetch_crop=fetch_crop)
    response = client.post("/predict", data={"image_id": "7", "annotation_id": "42"})

    assert response.status_code == 200
    assert calls == [(7, 42)]
    body = response.json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_route_is_404_when_the_annotation_does_not_exist(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle, fetch_crop=lambda *_: None)
    response = client.post("/predict", data={"image_id": "7", "annotation_id": "42"})

    assert response.status_code == 404


def test_predict_route_rejects_non_integer_ids(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle, fetch_crop=lambda *_: None)
    response = client.post("/predict", data={"image_id": "abc", "annotation_id": "42"})

    assert response.status_code == 400


def test_predict_route_rejects_an_incomplete_recrop_pair(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle, fetch_crop=lambda *_: None)
    response = client.post("/predict", data={"image_id": "7"})

    assert response.status_code == 400


def test_predict_route_rejects_sending_both_image_and_ids(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, load_model=a_bundle, fetch_crop=lambda *_: None)
    response = client.post(
        "/predict",
        data={"image_id": "7", "annotation_id": "42"},
        files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 400


# --- load_active_model: conexión real con P3-14 (#21, PR #62 ya mergeado) ------


class _FakeS3:
    """Nunca sale a la red: copia un paquete ya armado en disco, como si lo
    hubiera bajado de S3 -- `download_file` es la única llamada que hace
    `load_active_model`, así que es la única que hay que falsear."""

    def __init__(self, tar_path: Path):
        self._tar_path = tar_path
        self.calls: list[tuple[str, str, dict]] = []

    def download_file(self, *, Bucket, Key, Filename, ExtraArgs=None):
        self.calls.append((Bucket, Key, ExtraArgs or {}))
        shutil.copy(self._tar_path, Filename)


def _build_fake_package(tmp_path: Path) -> tuple[Path, str]:
    """Un paquete real y mínimo -- `config.json` + checkpoint real entrenable
    -- con el mismo layout que arma `publish.py` (P3-14): no un mock del
    formato, el `tarfile`/`config.json` que lee `load_active_model` es real."""
    staging = tmp_path / "staging"
    checkpoint_dir = staging / "artifacts" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (staging / "config.json").write_text(json.dumps(CONFIG.model_dump()), encoding="utf-8")

    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(build_model(CONFIG).state_dict(), checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    tar_path = tmp_path / "package.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(staging / "config.json", arcname="config.json")
        tar.add(checkpoint_path, arcname="artifacts/checkpoint/best.pt")
    return tar_path, checkpoint_sha256


def _registry_entry(tar_path: Path, checkpoint_sha256: str, *, version: str) -> dict:
    return {
        "s3_path": f"s3://fake-bucket/models/releases/v{version}/model_release_v{version}.tar.gz",
        "sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        "VersionId": f"fake-version-id-{version}",
        "run_id": "fake-run",
        "checkpoint_sha256": checkpoint_sha256,
        "data_release": "v0.1.1",
        "published_at": "2026-10-03T00:00:00+00:00",
    }


def test_load_active_model_picks_the_highest_semver(tmp_path):
    """0.1.0 es el ensayo, 1.0.0 el candidato seleccionado (PR #62) -- sin un
    campo "activa" explícito en el registro, gana el semver más alto."""
    tar_path, checkpoint_sha256 = _build_fake_package(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    old = _registry_entry(tar_path, checkpoint_sha256, version="0.1.0")
    new = _registry_entry(tar_path, checkpoint_sha256, version="1.0.0")
    (models_dir / "registry.json").write_text(
        json.dumps({"0.1.0": old, "1.0.0": new}), encoding="utf-8"
    )

    bundle = load_active_model(models_dir=models_dir, s3_client=_FakeS3(tar_path))

    assert bundle.version == "1.0.0"
    assert bundle.checkpoint_sha256 == checkpoint_sha256
    assert bundle.image_size == CONFIG.image_size


def test_load_active_model_downloads_by_version_id_and_builds_a_real_model(tmp_path):
    tar_path, checkpoint_sha256 = _build_fake_package(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    entry = _registry_entry(tar_path, checkpoint_sha256, version="9.9.9")
    (models_dir / "registry.json").write_text(json.dumps({"9.9.9": entry}), encoding="utf-8")
    fake_s3 = _FakeS3(tar_path)

    bundle = load_active_model(models_dir=models_dir, s3_client=fake_s3)

    assert fake_s3.calls == [
        (
            "fake-bucket",
            "models/releases/v9.9.9/model_release_v9.9.9.tar.gz",
            {"VersionId": "fake-version-id-9.9.9"},
        )
    ]
    # El modelo reconstruido predice como cualquier otro -- no es un stub.
    result = predict_image(bundle, Image.open(io.BytesIO(a_jpeg())).convert("RGB"))
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_load_active_model_caches_and_does_not_redownload(tmp_path):
    tar_path, checkpoint_sha256 = _build_fake_package(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    entry = _registry_entry(tar_path, checkpoint_sha256, version="9.9.9")
    (models_dir / "registry.json").write_text(json.dumps({"9.9.9": entry}), encoding="utf-8")
    fake_s3 = _FakeS3(tar_path)

    first = load_active_model(models_dir=models_dir, s3_client=fake_s3)
    second = load_active_model(models_dir=models_dir, s3_client=fake_s3)

    assert len(fake_s3.calls) == 1
    assert first is second


def test_load_active_model_rejects_a_sha256_mismatch(tmp_path):
    tar_path, checkpoint_sha256 = _build_fake_package(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    entry = _registry_entry(tar_path, checkpoint_sha256, version="9.9.9")
    entry["sha256"] = "0" * 64  # no coincide con el paquete real
    (models_dir / "registry.json").write_text(json.dumps({"9.9.9": entry}), encoding="utf-8")

    fake_s3 = _FakeS3(tar_path)

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_active_model(models_dir=models_dir, s3_client=fake_s3)


def test_load_active_model_fails_clearly_without_a_registry(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_active_model(models_dir=tmp_path / "no-existe")


def test_active_version_follows_the_marker_and_falls_back_to_highest_semver(tmp_path):
    registry = {"0.1.0": {}, "1.0.0": {}}
    assert _active_version(registry, tmp_path) == "1.0.0"

    (tmp_path / "active_version.json").write_text('{"active_version": "0.1.0"}', encoding="utf-8")
    assert _active_version(registry, tmp_path) == "0.1.0"

    (tmp_path / "active_version.json").write_text('{"active_version": "9.9.9"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"no está en registry\.json"):
        _active_version(registry, tmp_path)
