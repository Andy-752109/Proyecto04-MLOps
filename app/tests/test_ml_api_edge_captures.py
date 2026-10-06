"""P4-07: contrato y endpoint de capturas con S3 completamente falso."""

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ParamValidationError
from pydantic import ValidationError
from starlette.testclient import TestClient

from ml_api.contracts import EdgeEventV1
from ml_api.server import create_app
from tests._mcp_fixtures import mcp_settings

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"
CAT = json.loads((EXAMPLES / "valid-cat-no-crop.json").read_text(encoding="utf-8"))
DOG = json.loads((EXAMPLES / "valid-dog-with-crop.json").read_text(encoding="utf-8"))
LAST_MODIFIED = datetime(2026, 10, 6, 12, 30, tzinfo=timezone.utc)


def s3_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakePaginator:
    def __init__(self, pages, owner):
        self.pages = pages
        self.owner = owner

    def paginate(self, **kwargs):
        self.owner.list_kwargs = kwargs
        yield from self.pages


class FakeS3:
    def __init__(self, events=(), *, page_size=1000):
        self.objects = {}
        self.entries = []
        for index, (event, modified) in enumerate(events):
            key = f"edge-captures/v1/events/{index:04d}.json"
            self.entries.append({"Key": key, "LastModified": modified})
            self.objects[key] = (
                event if isinstance(event, bytes) else json.dumps(event).encode("utf-8")
            )
        self.page_size = page_size
        self.list_kwargs = None
        self.get_keys = []
        self.presign_params = []
        self.list_error = None
        self.get_error = None
        self.presign_error = None

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        if self.list_error:
            raise self.list_error
        pages = [
            {"Contents": self.entries[i : i + self.page_size]}
            for i in range(0, len(self.entries), self.page_size)
        ]
        return FakePaginator(pages or [{}], self)

    def get_object(self, *, Bucket, Key):
        self.get_keys.append(Key)
        error = self.get_error.get(Key) if isinstance(self.get_error, dict) else self.get_error
        if error:
            raise error
        return {"Body": io.BytesIO(self.objects[Key])}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        if self.presign_error:
            raise self.presign_error
        assert operation == "get_object"
        assert ExpiresIn == 3600
        self.presign_params.append(Params)
        return f"https://signed.example/{Params['Key']}"


def client_for(monkeypatch, tmp_path, fake, bucket="offline-edge-bucket"):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    settings = settings.model_copy(update={"edge_captures_bucket": bucket})

    def forbid_real_client(_settings):
        raise AssertionError("Se intentó crear un cliente AWS real")

    monkeypatch.setattr("ml_api.server.get_edge_s3_client", forbid_real_client)
    return TestClient(create_app(settings, edge_s3_client=lambda: fake))


def test_p4_02_examples_match_pydantic_model():
    for name in ("valid-cat-no-crop.json", "valid-dog-with-crop.json"):
        EdgeEventV1.model_validate_json((EXAMPLES / name).read_bytes())
    for path in EXAMPLES.glob("invalid-*.json"):
        with pytest.raises(ValidationError):
            EdgeEventV1.model_validate_json(path.read_bytes())


@pytest.mark.parametrize(
    ("sha", "valid"),
    [
        ("a" * 64, True),
        ("A" * 64, True),
        ("g" * 64, False),
        ("a" * 63, False),
        ("a" * 65, False),
    ],
)
def test_model_sha256_matches_p4_02_schema(sha, valid):
    event = {**CAT, "model_sha256": sha}
    if valid:
        assert EdgeEventV1.model_validate_json(json.dumps(event)).model_sha256 == sha
    else:
        with pytest.raises(ValidationError):
            EdgeEventV1.model_validate_json(json.dumps(event))


@pytest.mark.parametrize(
    ("field", "value"),
    [("confidence", float("nan")), ("preprocess_ms", -1), ("inference_ms", float("inf"))],
)
def test_nonfinite_or_negative_numbers_are_rejected(field, value):
    event = {**CAT, field: value}
    with pytest.raises(ValidationError):
        EdgeEventV1.model_validate_json(json.dumps(event))


def test_response_fields_are_not_persisted_event_fields():
    event = {**CAT, "received_at": LAST_MODIFIED.isoformat()}
    with pytest.raises(ValidationError):
        EdgeEventV1.model_validate_json(json.dumps(event))


def test_order_total_limit_received_at_url_and_response(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED), (DOG, LAST_MODIFIED)])
    response = client_for(monkeypatch, tmp_path, fake).get("/edge/captures?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total", "bucket"}
    assert body["bucket"] == "offline-edge-bucket"
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["capture_id"] == DOG["capture_id"]
    assert datetime.fromisoformat(body["items"][0]["received_at"]) == LAST_MODIFIED
    assert body["items"][0]["image_url"].endswith(DOG["image_key"])
    assert {params["Key"] for params in fake.presign_params} == {CAT["image_key"], DOG["image_key"]}
    assert fake.list_kwargs == {
        "Bucket": "offline-edge-bucket",
        "Prefix": "edge-captures/v1/events/",
    }


def test_default_limit_is_50(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED)] * 51)
    body = client_for(monkeypatch, tmp_path, fake).get("/edge/captures").json()
    assert len(body["items"]) == 50
    assert body["total"] == 51


@pytest.mark.parametrize("limit", ["0", "101", "abc", "", "-1", "1.5", "1&limit=2"])
def test_invalid_limit_is_400(monkeypatch, tmp_path, limit):
    fake = FakeS3()
    response = client_for(monkeypatch, tmp_path, fake).get(f"/edge/captures?limit={limit}")
    assert response.status_code == 400
    assert fake.list_kwargs is None


def test_invalid_and_corrupt_events_are_logged_and_omitted(monkeypatch, tmp_path, caplog):
    invalid = json.loads((EXAMPLES / "invalid-crop-out-of-frame.json").read_text())
    fake = FakeS3([(CAT, LAST_MODIFIED), (invalid, LAST_MODIFIED), (b"{broken", LAST_MODIFIED)])
    with caplog.at_level("WARNING"):
        response = client_for(monkeypatch, tmp_path, fake).get("/edge/captures")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["items"]) == 1
    assert "Evento inválido omitido" in caplog.text


def test_empty_list(monkeypatch, tmp_path):
    response = client_for(monkeypatch, tmp_path, FakeS3()).get("/edge/captures")
    assert response.json() == {"items": [], "total": 0, "bucket": "offline-edge-bucket"}


def test_all_pages_are_read(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED)] * 1001)
    body = client_for(monkeypatch, tmp_path, fake).get("/edge/captures?limit=100").json()
    assert body["total"] == 1001
    assert len(body["items"]) == 100
    assert len(fake.get_keys) == 1001


def test_nosuchkey_is_an_individual_race(monkeypatch, tmp_path, caplog):
    fake = FakeS3([(CAT, LAST_MODIFIED), (DOG, LAST_MODIFIED)])
    fake.get_error = {fake.entries[0]["Key"]: s3_error("NoSuchKey", "GetObject")}
    with caplog.at_level("WARNING"):
        response = client_for(monkeypatch, tmp_path, fake).get("/edge/captures")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["capture_id"] == DOG["capture_id"]
    assert "desapareció" in caplog.text


@pytest.mark.parametrize("code", ["AccessDenied", "NoSuchBucket"])
def test_s3_errors_are_503(monkeypatch, tmp_path, code):
    fake = FakeS3([(CAT, LAST_MODIFIED)])
    fake.list_error = s3_error(code, "ListObjectsV2")
    response = client_for(monkeypatch, tmp_path, fake).get("/edge/captures")
    assert response.status_code == 503
    assert "S3" in response.json()["error"]


def test_botocore_credentials_error_is_503(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED)])
    fake.list_error = NoCredentialsError()
    response = client_for(monkeypatch, tmp_path, fake).get("/edge/captures")
    assert response.status_code == 503
    assert response.json() == {
        "error": "Capturas no disponibles: verifica S3, permisos y sesión SSO"
    }


def test_get_object_failure_is_503(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED)])
    fake.get_error = s3_error("AccessDenied", "GetObject")
    assert client_for(monkeypatch, tmp_path, fake).get("/edge/captures").status_code == 503


def test_presign_failure_is_503(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED)])
    fake.presign_error = ParamValidationError(report="firma no disponible")
    assert client_for(monkeypatch, tmp_path, fake).get("/edge/captures").status_code == 503


def test_unexpected_programming_error_is_not_misreported_as_s3(monkeypatch, tmp_path):
    fake = FakeS3([(CAT, LAST_MODIFIED)])
    fake.presign_error = TypeError("error interno")
    with pytest.raises(TypeError, match="error interno"):
        client_for(monkeypatch, tmp_path, fake).get("/edge/captures")


def test_unconfigured_bucket_is_503_without_s3(monkeypatch, tmp_path):
    fake = FakeS3()
    response = client_for(monkeypatch, tmp_path, fake, bucket="").get("/edge/captures")
    assert response.status_code == 503
    assert fake.list_kwargs is None
