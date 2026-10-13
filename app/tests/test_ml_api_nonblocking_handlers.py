"""Concurrencia de Models y Training sin S3 ni base de datos reales."""

import asyncio
import json
import threading

from ml_api import models
from ml_api.server import create_app
from tests._mcp_fixtures import mcp_settings
from tests._training_job_fixtures import VALID_TRAINING_CONFIG, write_manifest
from tests.test_ml_api_experiments import asgi_request
from tests.test_ml_api_models import LAST_MODIFIED, PACKAGE_INFO, a_registry, write_registry
from tests.test_ml_api_server import a_job


class BlockingCall:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self):
        self.entered.set()
        if not self.release.wait(12):
            raise AssertionError("La operación falsa no fue liberada")


async def assert_health_while_waiting(app, path, payload, blocker, expected_status):
    slow_request = asyncio.create_task(asgi_request(app, "POST", path, payload))
    watchdog = threading.Timer(10, blocker.release.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert await asyncio.to_thread(blocker.entered.wait, 3), "El I/O lento no comenzó"
        assert not blocker.release.is_set(), "El I/O terminó antes de probar concurrencia"

        health = await asyncio.wait_for(asgi_request(app, "GET", "/health"), 3)
        assert health == (200, {"status": "ok"})
        assert not blocker.release.is_set(), "/health esperó a que terminara el I/O"
        assert not slow_request.done()

        blocker.release.set()
        status, body = await asyncio.wait_for(slow_request, 3)
        assert status == expected_status
        return body
    finally:
        blocker.release.set()
        watchdog.cancel()
        await asyncio.gather(slow_request, return_exceptions=True)


def test_set_active_model_keeps_health_responsive(monkeypatch, tmp_path):
    settings = mcp_settings(
        monkeypatch,
        tmp_path / "dataset",
        tmp_path / "reports",
        tmp_path / "derived",
        tmp_path / "models",
    )
    write_registry(settings.models_dir, a_registry())
    blocker = BlockingCall()
    calls = []

    def status_of(entry):
        calls.append(entry.VersionId)
        if len(calls) == 1:
            blocker()
        return {"version_id": entry.VersionId, "size_bytes": 1, "last_modified": LAST_MODIFIED}

    app = create_app(
        settings,
        model_status_of=status_of,
        model_download_url_of=lambda entry: "https://signed.example/package",
        model_package_info_of=lambda entry: PACKAGE_INFO,
    )
    body = asyncio.run(
        assert_health_while_waiting(app, "/models/active", {"version": "1.0.0"}, blocker, 200)
    )

    assert body["version"] == "1.0.0"
    assert body["active"] is True
    assert json.loads((settings.models_dir / "active_version.json").read_text()) == {
        "active_version": "1.0.0"
    }
    assert calls == ["v-final", "v-final"]


def test_create_training_job_keeps_health_responsive(monkeypatch, tmp_path):
    settings = mcp_settings(
        monkeypatch, tmp_path / "dataset", tmp_path / "reports", tmp_path / "derived"
    )
    write_manifest(reports_dir=settings.reports_dir, derived_dir=settings.derived_dir)
    blocker = BlockingCall()
    calls = []

    def create_job(dataset_release, manifest_id, config, run_kind, grid_row):
        blocker()
        calls.append((dataset_release, manifest_id, config, run_kind, grid_row))
        return a_job(id="new-job", status="queued", config=config)

    app = create_app(settings, create_job=create_job)
    body = asyncio.run(
        assert_health_while_waiting(
            app,
            "/training/jobs",
            {"dataset_release": "v0.1.1", "config": VALID_TRAINING_CONFIG, "run_kind": "smoke"},
            blocker,
            201,
        )
    )

    assert body["id"] == "new-job"
    assert body["status"] == "queued"
    assert len(calls) == 1
    assert calls[0][0] == "v0.1.1"


def test_cancel_training_job_keeps_health_responsive(monkeypatch, tmp_path):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    blocker = BlockingCall()
    calls = []

    def cancel_job(job_id):
        blocker()
        calls.append(job_id)
        return a_job(id=job_id, status="cancelled")

    app = create_app(settings, cancel_job=cancel_job)
    body = asyncio.run(
        assert_health_while_waiting(app, "/training/jobs/r01/cancel", None, blocker, 200)
    )

    assert body["id"] == "r01"
    assert body["status"] == "cancelled"
    assert calls == ["r01"]


def test_active_model_writes_are_serialized(monkeypatch, tmp_path):
    settings = mcp_settings(
        monkeypatch,
        tmp_path / "dataset",
        tmp_path / "reports",
        tmp_path / "derived",
        tmp_path / "models",
    )
    write_registry(settings.models_dir, a_registry())
    first_write = BlockingCall()
    second_loaded = threading.Event()
    second_write = threading.Event()
    original_load = models.load_registry
    original_write = models.write_active_version
    load_count = 0

    def load_registry(path):
        nonlocal load_count
        registry = original_load(path)
        load_count += 1
        if load_count == 2:
            second_loaded.set()
        return registry

    def write_active_version(path, version):
        if version == "0.1.0":
            first_write()
        else:
            second_write.set()
        original_write(path, version)

    monkeypatch.setattr("ml_api.server.load_registry", load_registry)
    monkeypatch.setattr("ml_api.server.write_active_version", write_active_version)
    app = create_app(
        settings,
        model_status_of=lambda entry: {
            "version_id": entry.VersionId,
            "size_bytes": 1,
            "last_modified": LAST_MODIFIED,
        },
        model_download_url_of=lambda entry: "https://signed.example/package",
        model_package_info_of=lambda entry: PACKAGE_INFO,
    )

    async def exercise():
        first = asyncio.create_task(
            asgi_request(app, "POST", "/models/active", {"version": "0.1.0"})
        )
        watchdog = threading.Timer(10, first_write.release.set)
        watchdog.daemon = True
        watchdog.start()
        second = None
        try:
            assert await asyncio.to_thread(first_write.entered.wait, 3)
            second = asyncio.create_task(
                asgi_request(app, "POST", "/models/active", {"version": "1.0.0"})
            )
            assert await asyncio.to_thread(second_loaded.wait, 3)
            assert not await asyncio.to_thread(second_write.wait, 0.5)
            assert not first_write.release.is_set()

            first_write.release.set()
            first_status, _ = await asyncio.wait_for(first, 3)
            second_status, _ = await asyncio.wait_for(second, 3)
            assert (first_status, second_status) == (200, 200)
            assert second_write.is_set()
        finally:
            first_write.release.set()
            watchdog.cancel()
            tasks = [task for task in (first, second) if task is not None]
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())
    assert json.loads((settings.models_dir / "active_version.json").read_text()) == {
        "active_version": "1.0.0"
    }
