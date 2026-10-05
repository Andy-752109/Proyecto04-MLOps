"""`/models` (P3-15): catálogo real de P3-14 (`models/registry.json`) + estado S3
en vivo + versión activa.

El catálogo real ya está en `models/registry.json` (P3-14, #62); estas pruebas
usan un registry de datos de prueba en `tmp_path` y `head-object`/prefirmado/
inspección de paquete falsos inyectados -- sin tocar AWS ni MinIO (mismo criterio
que `list_jobs`/`create_job` en `test_ml_api_server.py`).
"""

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from starlette.testclient import TestClient

from ml_api import inference
from ml_api.contracts import PredictionResponse
from ml_api.server import create_app
from storage.model_store import head_object, presigned_get_url, read_package_info
from tests._mcp_fixtures import mcp_settings

LAST_MODIFIED = datetime(2026, 10, 3, 1, 26, 45, tzinfo=UTC)
SELECTED_RUN_ID = "7e7b4a4b35464cfebb6b41714a3ad931"


def a_registry() -> dict:
    return {
        "0.1.0": {
            "s3_path": (
                "s3://mlops-p3-models-222629887955/models/releases/v0.1.0/"
                "model_release_v0.1.0.tar.gz"
            ),
            "sha256": "a" * 64,
            "VersionId": "v-old",
            "run_id": "b828e032156e423097e90d797b643faa",
            "checkpoint_sha256": "b" * 64,
            "data_release": "v0.1.1",
            "published_at": "2026-10-03T01:26:45.239074+00:00",
        },
        "1.0.0": {
            "s3_path": (
                "s3://mlops-p3-models-222629887955/models/releases/v1.0.0/"
                "model_release_v1.0.0.tar.gz"
            ),
            "sha256": "c" * 64,
            "VersionId": "v-final",
            "run_id": SELECTED_RUN_ID,
            "checkpoint_sha256": "f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9",
            "data_release": "v0.1.1",
            "published_at": "2026-10-03T01:27:42.973813+00:00",
        },
    }


def write_registry(models_dir: Path, registry: dict) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def write_selection(reports_dir: Path, run_id: str = SELECTED_RUN_ID) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "selection.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


PACKAGE_INFO = {
    "card": "Modelo final, candidato r02.",
    "run_kind": "campaign",
    "manifest_id": "v0.1.1-53fc84fdaa07",
}


def client_for(monkeypatch, tmp_path, *, existing: set[str] | None = None) -> TestClient:
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    settings = mcp_settings(
        monkeypatch, tmp_path / "dataset", reports_dir, tmp_path / "derived", models_dir
    )
    existing = existing if existing is not None else {"0.1.0", "1.0.0"}

    def status_of(entry):
        if not any(entry.s3_path.endswith(f"v{v}/model_release_v{v}.tar.gz") for v in existing):
            return None
        version_id = entry.s3_path.split("/releases/")[1].split("/")[0]
        return {"version_id": version_id, "size_bytes": 1024, "last_modified": LAST_MODIFIED}

    return TestClient(
        create_app(
            settings,
            model_status_of=status_of,
            model_download_url_of=lambda entry: f"https://signed.example/{entry.s3_path}",
            model_package_info_of=lambda entry: PACKAGE_INFO,
        )
    )


def test_models_is_pending_until_p3_14_publishes_the_registry(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["ticket"] == "P3-14"


def test_models_lists_versions_with_live_s3_status_and_dataset_version(monkeypatch, tmp_path):
    write_registry(tmp_path / "models", a_registry())
    write_selection(tmp_path / "reports")
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["active_version"] is None
    # Orden ascendente por versión.
    assert [entry["version"] for entry in body["versions"]] == ["0.1.0", "1.0.0"]
    first, second = body["versions"]
    assert first["dataset_version"] == "v0.1.1"
    assert first["s3_status"] == {
        "exists": True,
        "version_id": "v0.1.0",
        "size_bytes": 1024,
        "last_modified": LAST_MODIFIED.isoformat().replace("+00:00", "Z"),
        "error": None,
    }
    # `selected` se deriva de reports/selection.json (el registry no lo trae).
    assert first["selected"] is False
    assert second["selected"] is True


def test_models_marks_a_missing_object_as_not_existing(monkeypatch, tmp_path):
    write_registry(tmp_path / "models", a_registry())
    client = client_for(monkeypatch, tmp_path, existing={"1.0.0"})

    body = client.get("/models").json()

    statuses = {entry["version"]: entry["s3_status"]["exists"] for entry in body["versions"]}
    assert statuses == {"0.1.0": False, "1.0.0": True}


def test_models_survives_a_head_object_error_on_one_version(monkeypatch, tmp_path):
    # Un fallo que NO es "no existe" (red, SSO expirado) no debe tumbar la lista:
    # esa versión queda con `error` y las demás se sirven igual.
    write_registry(tmp_path / "models", a_registry())
    settings = mcp_settings(
        monkeypatch,
        tmp_path / "dataset",
        tmp_path / "reports",
        tmp_path / "derived",
        tmp_path / "models",
    )

    def status_of(entry):
        if entry.s3_path.endswith("v0.1.0/model_release_v0.1.0.tar.gz"):
            raise RuntimeError("SSO expirado")
        return {"version_id": "v-final", "size_bytes": 10, "last_modified": LAST_MODIFIED}

    client = TestClient(
        create_app(
            settings,
            model_status_of=status_of,
            model_download_url_of=lambda entry: None,
            model_package_info_of=lambda entry: PACKAGE_INFO,
        )
    )

    response = client.get("/models")

    assert response.status_code == 200
    by_version = {entry["version"]: entry["s3_status"] for entry in response.json()["versions"]}
    assert by_version["0.1.0"]["exists"] is False
    assert by_version["0.1.0"]["error"] == "no se pudo verificar el objeto en S3"
    assert by_version["1.0.0"]["exists"] is True
    assert by_version["1.0.0"]["error"] is None


def test_model_detail_derives_bucket_key_card_and_presigned_url(monkeypatch, tmp_path):
    write_registry(tmp_path / "models", a_registry())
    write_selection(tmp_path / "reports")
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models/1.0.0")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "1.0.0"
    assert body["dataset_version"] == "v0.1.1"
    assert body["run_id"] == SELECTED_RUN_ID
    assert body["selected"] is True
    assert body["s3_bucket"] == "mlops-p3-models-222629887955"
    assert body["s3_key"] == "models/releases/v1.0.0/model_release_v1.0.0.tar.gz"
    assert body["card"] == "Modelo final, candidato r02."
    assert body["manifest_id"] == "v0.1.1-53fc84fdaa07"
    assert body["run_kind"] == "campaign"
    assert body["download_url"] == (
        "https://signed.example/s3://mlops-p3-models-222629887955/models/releases/"
        "v1.0.0/model_release_v1.0.0.tar.gz"
    )


def test_model_detail_is_404_for_an_unknown_version(monkeypatch, tmp_path):
    write_registry(tmp_path / "models", a_registry())
    client = client_for(monkeypatch, tmp_path)

    assert client.get("/models/9.9.9").status_code == 404


def test_model_detail_for_the_smoke_version_keeps_dataset_and_model_apart(monkeypatch, tmp_path):
    # Checklist del ticket: por cada versión (0.1.0 y 1.0.0) la tarjeta, run,
    # release de datos, manifiesto, bucket/key, VersionId y SHA-256. La versión
    # del DATASET (data_release = v0.1.1) no es la del modelo (0.1.0).
    write_registry(tmp_path / "models", a_registry())
    write_selection(tmp_path / "reports")
    client = client_for(monkeypatch, tmp_path)

    body = client.get("/models/0.1.0").json()

    assert body["version"] == "0.1.0"
    assert body["dataset_version"] == "v0.1.1"
    assert body["run_id"] == "b828e032156e423097e90d797b643faa"
    assert body["selected"] is False
    assert body["registered_version_id"] == "v-old"
    assert body["checkpoint_sha256"] == "b" * 64
    assert body["s3_bucket"] == "mlops-p3-models-222629887955"
    assert body["s3_key"] == "models/releases/v0.1.0/model_release_v0.1.0.tar.gz"
    assert body["card"] == PACKAGE_INFO["card"]
    assert body["manifest_id"] == PACKAGE_INFO["manifest_id"]
    assert body["download_url"] is not None
    assert body["s3_status"]["exists"] is True


def test_setting_active_version_persists_and_rejects_a_missing_object(monkeypatch, tmp_path):
    write_registry(tmp_path / "models", a_registry())
    client = client_for(monkeypatch, tmp_path, existing={"1.0.0"})
    active_path = tmp_path / "models" / "active_version.json"

    rejected = client.post("/models/active", json={"version": "0.1.0"})
    assert rejected.status_code == 400
    assert "no existe en S3" in rejected.json()["error"]
    assert not active_path.exists()

    accepted = client.post("/models/active", json={"version": "1.0.0"})
    assert accepted.status_code == 200
    assert accepted.json()["active"] is True
    assert json.loads(active_path.read_text(encoding="utf-8")) == {"active_version": "1.0.0"}

    assert client.get("/models").json()["active_version"] == "1.0.0"


def test_setting_active_version_rejects_an_unknown_version_and_bad_body(monkeypatch, tmp_path):
    write_registry(tmp_path / "models", a_registry())
    client = client_for(monkeypatch, tmp_path)

    assert client.post("/models/active", json={"version": "9.9.9"}).status_code == 404
    assert client.post("/models/active", json={}).status_code == 400


def a_jpeg() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_changing_the_active_version_changes_what_predict_loads(monkeypatch, tmp_path):
    """ "Hecho cuando" del ticket: cambiar la versión activa (P3-15) cambia el
    `model_version` y el `checkpoint_sha256` que devuelve `/predict` (P3-16),
    en el mismo app/settings -- no solo el texto en pantalla.
    """
    write_registry(tmp_path / "models", a_registry())
    write_selection(tmp_path / "reports")
    settings = mcp_settings(
        monkeypatch,
        tmp_path / "dataset",
        tmp_path / "reports",
        tmp_path / "derived",
        tmp_path / "models",
    )

    def load_model():
        # Mismo contrato que `inference.load_active_model`, pero sin bajar de S3:
        # resuelve la versión desde el `active_version.json` que escribe
        # `POST /models/active`, usando la función real de selección de P3-16.
        registry = json.loads((tmp_path / "models" / "registry.json").read_text(encoding="utf-8"))
        version = inference._active_version(registry, tmp_path / "models")
        return inference.ModelBundle(
            model=None,  # `predict_image` va monkeypatcheado: no se usa el modelo.
            image_size=8,
            version=version,
            checkpoint_sha256=registry[version]["checkpoint_sha256"],
        )

    monkeypatch.setattr(
        inference,
        "predict_image",
        lambda bundle, image: PredictionResponse(
            predicted_label="cat",
            probabilities={"cat": 1.0, "dog": 0.0},
            model_version=bundle.version,
            checkpoint_sha256=bundle.checkpoint_sha256,
        ),
    )

    client = TestClient(
        create_app(
            settings,
            load_model=load_model,
            fetch_crop=lambda image_id, annotation_id: None,
            model_status_of=lambda entry: {
                "version_id": "x",
                "size_bytes": 1,
                "last_modified": LAST_MODIFIED,
            },
            model_download_url_of=lambda entry: "https://signed.example/package",
            model_package_info_of=lambda entry: PACKAGE_INFO,
        )
    )

    assert client.post("/models/active", json={"version": "0.1.0"}).status_code == 200
    first = client.post("/predict", files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")}).json()

    assert client.post("/models/active", json={"version": "1.0.0"}).status_code == 200
    second = client.post("/predict", files={"image": ("cat.jpg", a_jpeg(), "image/jpeg")}).json()

    assert first["model_version"] == "0.1.0"
    assert first["checkpoint_sha256"] == "b" * 64
    assert second["model_version"] == "1.0.0"
    assert second["checkpoint_sha256"] == (
        "f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9"
    )
    assert first["model_version"] != second["model_version"]
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]


# --- storage/model_store (P3-15) --------------------------------------------


class _FakeS3:
    """Cliente S3 falso que registra con qué `VersionId` se pidió cada objeto."""

    def __init__(self, *, stat=None, missing: bool = False, body: bytes = b"") -> None:
        self._stat = stat
        self._missing = missing
        self._body = body
        self.head_kwargs: dict | None = None
        self.get_kwargs: dict | None = None
        self.presign_params: dict | None = None

    def head_object(self, Bucket, Key, VersionId=None):
        self.head_kwargs = {"Bucket": Bucket, "Key": Key}
        if VersionId is not None:
            self.head_kwargs["VersionId"] = VersionId
        if self._missing:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return self._stat

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presign_params = dict(Params)
        return f"https://signed/{Params['Bucket']}/{Params['Key']}?op={operation}&exp={ExpiresIn}"

    def get_object(self, Bucket, Key, VersionId=None):
        self.get_kwargs = {"Bucket": Bucket, "Key": Key}
        if VersionId is not None:
            self.get_kwargs["VersionId"] = VersionId
        return {"Body": io.BytesIO(self._body)}


def a_package_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in (
            ("model_card.md", "# Tarjeta\n"),
            ("package.json", json.dumps({"run_kind": "smoke", "manifest_id": "m-1"})),
        ):
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_head_object_reads_live_status_and_pins_the_version_id():
    stat = {"VersionId": "v-1", "ContentLength": 2048, "LastModified": LAST_MODIFIED}
    client = _FakeS3(stat=stat)

    status = head_object(client, bucket="b", key="k", version_id="v-1")

    assert status == {"version_id": "v-1", "size_bytes": 2048, "last_modified": LAST_MODIFIED}
    assert client.head_kwargs == {"Bucket": "b", "Key": "k", "VersionId": "v-1"}


def test_head_object_omits_version_id_when_not_given():
    client = _FakeS3(stat={"VersionId": "v-1"})

    head_object(client, bucket="b", key="k")

    assert client.head_kwargs == {"Bucket": "b", "Key": "k"}


def test_head_object_returns_none_when_missing():
    assert head_object(_FakeS3(missing=True), bucket="b", key="k") is None


def test_presigned_get_url_pins_the_version_id():
    client = _FakeS3()

    url = presigned_get_url(client, bucket="b", key="k", version_id="v-1", expires_seconds=60)

    assert url == "https://signed/b/k?op=get_object&exp=60"
    assert client.presign_params == {"Bucket": "b", "Key": "k", "VersionId": "v-1"}


def test_read_package_info_extracts_card_and_package_json_in_streaming():
    client = _FakeS3(body=a_package_tar())

    info = read_package_info(client, bucket="b", key="k", version_id="v-1")

    assert info == {"card": "# Tarjeta\n", "run_kind": "smoke", "manifest_id": "m-1"}
    assert client.get_kwargs == {"Bucket": "b", "Key": "k", "VersionId": "v-1"}


def test_read_package_info_tolerates_a_package_without_optional_files():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz"):
        pass

    info = read_package_info(_FakeS3(body=buffer.getvalue()), bucket="b", key="k")

    assert info == {"card": None, "run_kind": None, "manifest_id": None}


def test_split_s3_path():
    from ml_api.models import split_s3_path

    assert split_s3_path("s3://bucket/a/b/c") == ("bucket", "a/b/c")


@pytest.mark.parametrize("a,b", [("0.9.0", "0.10.0"), ("1.0.0", "1.0.1")])
def test_version_key_orders_numerically(a, b):
    from ml_api.models import version_key

    assert version_key(a) < version_key(b)
