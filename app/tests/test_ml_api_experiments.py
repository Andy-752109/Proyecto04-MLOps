"""Contratos HTTP y concurrencia de Experiments sin MLflow ni AWS reales."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ml_api.server import create_app
from tests._mcp_fixtures import mcp_settings

CASES = (
    ("runs", "GET", "/experiments/exp-1/runs", None),
    ("metrics", "GET", "/experiments/runs/run-1/metrics/accuracy", None),
    ("tags", "POST", "/experiments/runs/run-1/tags", {"key": "estado", "value": "listo"}),
)


class FakeMlflowClient:
    def __init__(self, *, slow_operation=None, failing_operation=None):
        self.slow_operation = slow_operation
        self.failing_operation = failing_operation
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.tags_written = []

    def _before(self, operation):
        if self.slow_operation == operation:
            self.entered.set()
            if not self.release.wait(10):
                raise AssertionError("La operación falsa no fue liberada")
        if self.failing_operation == operation:
            raise RuntimeError("MLflow no disponible")

    def search_runs(self, *, experiment_ids):
        self.calls.append(("runs", experiment_ids))
        self._before("runs")
        return [SimpleNamespace(to_dictionary=lambda: {"run_id": "run-1"})]

    def get_metric_history(self, run_id, metric_key):
        self.calls.append(("metrics", run_id, metric_key))
        self._before("metrics")
        return [SimpleNamespace(step=2, value=0.75, timestamp=1234)]

    def set_tag(self, run_id, key, value):
        self.calls.append(("tags", run_id, key, value))
        self._before("tags")
        self.tags_written.append((run_id, key, value))


class EmptyS3:
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == "offline-edge-bucket"
        assert Prefix == "edge-captures/v1/events/"
        yield {}


def app_for(monkeypatch, tmp_path, get_client):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    settings = settings.model_copy(update={"edge_captures_bucket": "offline-edge-bucket"})
    monkeypatch.setattr("ml_api.experiments.get_client", get_client)
    return create_app(settings, edge_s3_client=lambda: EmptyS3())


async def asgi_request(app, method, path, payload=None):
    """Enviar solicitudes al mismo objeto ASGI y al mismo event loop."""
    body = json.dumps(payload).encode() if payload is not None else b""
    received = False
    messages = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")] if body else [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
    }

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Future()

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    status = next(
        message["status"] for message in messages if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


@pytest.mark.parametrize("operation,method,path,payload", CASES)
def test_experiments_does_not_block_health_or_edge(
    monkeypatch, tmp_path, operation, method, path, payload
):
    fake = FakeMlflowClient(slow_operation=operation)
    creation_threads = []

    def get_client():
        creation_threads.append(threading.get_ident())
        return fake

    app = app_for(monkeypatch, tmp_path, get_client)

    async def exercise():
        loop_thread = threading.get_ident()
        slow_request = asyncio.create_task(asgi_request(app, method, path, payload))
        # Solo libera la llamada si el código antiguo congela el loop o la prueba falla.
        watchdog = threading.Timer(8, fake.release.set)
        watchdog.daemon = True
        watchdog.start()
        try:
            assert await asyncio.to_thread(fake.entered.wait, 2), "Experiments no entró en MLflow"
            assert not fake.release.is_set(), "MLflow terminó antes de probar concurrencia"

            health = await asyncio.wait_for(asgi_request(app, "GET", "/health"), 2)
            assert health == (200, {"status": "ok"})
            assert not fake.release.is_set()
            assert not slow_request.done()

            edge = await asyncio.wait_for(asgi_request(app, "GET", "/edge/captures"), 2)
            assert edge == (200, {"items": [], "total": 0, "bucket": "offline-edge-bucket"})
            assert not fake.release.is_set()
            assert not slow_request.done()

            fake.release.set()
            status, _ = await asyncio.wait_for(slow_request, 2)
            assert status == 200
            assert creation_threads and all(thread != loop_thread for thread in creation_threads)
        finally:
            fake.release.set()
            watchdog.cancel()
            await asyncio.gather(slow_request, return_exceptions=True)

    asyncio.run(exercise())
    if operation == "tags":
        assert fake.tags_written == [("run-1", "estado", "listo")]


@pytest.mark.parametrize("operation,method,path,payload", CASES)
def test_experiments_success_contract(monkeypatch, tmp_path, operation, method, path, payload):
    fake = FakeMlflowClient()
    client = TestClient(app_for(monkeypatch, tmp_path, lambda: fake))
    response = client.request(method, path, json=payload)

    assert response.status_code == 200
    expected = {
        "runs": [{"run_id": "run-1"}],
        "metrics": [{"step": 2, "value": 0.75, "timestamp": 1234}],
        "tags": {"status": "success", "run_id": "run-1", "tag": "estado", "value": "listo"},
    }
    assert response.json() == expected[operation]
    if operation == "tags":
        assert fake.tags_written == [("run-1", "estado", "listo")]


@pytest.mark.parametrize("operation,method,path,payload", CASES)
def test_experiments_error_contract(monkeypatch, tmp_path, operation, method, path, payload):
    fake = FakeMlflowClient(failing_operation=operation)
    client = TestClient(app_for(monkeypatch, tmp_path, lambda: fake))
    response = client.request(method, path, json=payload)

    assert response.status_code == (400 if operation == "tags" else 404)
    assert response.json() == {"error": "MLflow no disponible"}
    assert fake.tags_written == []


def test_invalid_tag_payload_never_calls_mlflow(monkeypatch, tmp_path):
    def forbidden_client():
        raise AssertionError("No debe construirse MlflowClient para un payload inválido")

    client = TestClient(app_for(monkeypatch, tmp_path, forbidden_client))
    response = client.post("/experiments/runs/run-1/tags", json={"key": "estado"})

    assert response.status_code == 400
    assert response.json()["error"].startswith("Payload inválido:")
