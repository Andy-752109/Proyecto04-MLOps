"""`ml_worker.__main__.process_one_job` (P3-09).

`spawn` se inyecta -- igual que `list_jobs`/`create_job` en `ml_api.server`
-- así esta prueba nunca lanza un subproceso ni toca una MariaDB real: solo
prueba que, dada una secuencia de mensajes de la cola, `process_one_job`
llama a la función correcta de `ml_worker.repository` con los argumentos
correctos, en el orden correcto. Las funciones de `ml_worker.repository`
se parchan directo (son las que sí tocan la base).
"""

from datetime import datetime
from queue import Empty

import pytest

from ml_api.contracts import TrainingJob
from ml_worker.__main__ import process_one_job


def a_job(**overrides) -> TrainingJob:
    defaults = {
        "id": "job-1",
        "status": "running",
        "progress": 0.0,
        "config": {"optimizer": "adam"},
        "dataset_release": "v0.1.1",
        "manifest_id": "manifest-abc",
        "run_kind": "smoke",
        "grid_row": None,
        "mlflow_run_id": None,
        "error": None,
        "logs": [],
        "heartbeat_at": datetime(2026, 9, 30, 12, 0, 0),
    }
    return TrainingJob(**{**defaults, **overrides})


class FakeQueue:
    def __init__(self, messages):
        self._messages = list(messages)

    def get(self, timeout):
        if not self._messages:
            raise Empty
        return self._messages.pop(0)


class FakeProcess:
    def __init__(self, *, dies_silently=False, exitcode=1):
        self.joined = False
        self.terminated = False
        self.killed = False
        self.exitcode = exitcode
        self._dies_silently = dies_silently

    def is_alive(self):
        return not self._dies_silently

    def terminate(self):
        self.terminated = True
        self._dies_silently = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        self.joined = True


def never_cancelled(_engine, _job_id):
    return False


@pytest.fixture
def recorder(monkeypatch):
    calls = []
    for name in (
        "update_progress",
        "mark_completed",
        "mark_failed",
        "mark_cancelled",
        "set_mlflow_run_id",
    ):

        def make(name):
            def fn(engine, job_id, **kwargs):
                calls.append((name, job_id, kwargs))

            return fn

        monkeypatch.setattr(f"ml_worker.__main__.{name}", make(name))
    return calls


def test_applies_each_progress_message_and_then_completed(recorder, tmp_path):
    job = a_job()
    messages = [
        {"type": "progress", "progress": 0.5, "log_line": "época 1/2"},
        {"type": "progress", "progress": 1.0, "log_line": "época 2/2"},
        {"type": "completed", "mlflow_run_id": "run-42"},
    ]

    def spawn(_job, _manifest_path, _crops_root, _meta_path):
        return FakeProcess(), FakeQueue(messages)

    process_one_job(
        engine=object(),
        job=job,
        derived_dir=tmp_path,
        reports_dir=tmp_path,
        spawn=spawn,
        is_cancel_requested=never_cancelled,
    )

    assert recorder == [
        ("update_progress", "job-1", {"progress": 0.5, "log_line": "época 1/2"}),
        ("update_progress", "job-1", {"progress": 1.0, "log_line": "época 2/2"}),
        ("mark_completed", "job-1", {"mlflow_run_id": "run-42"}),
    ]


def test_applies_failed_message_and_stops(recorder, tmp_path):
    job = a_job()
    messages = [
        {"type": "progress", "progress": 0.3, "log_line": "época 1/3"},
        {"type": "failed", "error": "CUDA out of memory"},
    ]

    def spawn(_job, _manifest_path, _crops_root, _meta_path):
        return FakeProcess(), FakeQueue(messages)

    process_one_job(
        engine=object(),
        job=job,
        derived_dir=tmp_path,
        reports_dir=tmp_path,
        spawn=spawn,
        is_cancel_requested=never_cancelled,
    )

    assert recorder == [
        ("update_progress", "job-1", {"progress": 0.3, "log_line": "época 1/3"}),
        ("mark_failed", "job-1", {"error": "CUDA out of memory"}),
    ]


def test_marks_failed_when_the_subprocess_dies_without_a_final_message(recorder, tmp_path):
    job = a_job()

    def spawn(_job, _manifest_path, _crops_root, _meta_path):
        return FakeProcess(dies_silently=True, exitcode=137), FakeQueue([])

    process_one_job(
        engine=object(),
        job=job,
        derived_dir=tmp_path,
        reports_dir=tmp_path,
        spawn=spawn,
        is_cancel_requested=never_cancelled,
    )

    assert len(recorder) == 1
    name, job_id, kwargs = recorder[0]
    assert (name, job_id) == ("mark_failed", "job-1")
    assert "137" in kwargs["error"]


def test_closes_the_mlflow_run_as_failed_if_it_had_started_and_the_subprocess_dies_silently(
    recorder, tmp_path
):
    job = a_job()
    messages = [{"type": "run_started", "mlflow_run_id": "run-orphan"}]

    def spawn(_job, _manifest_path, _crops_root, _meta_path):
        return FakeProcess(dies_silently=True, exitcode=137), FakeQueue(messages)

    closed = []
    process_one_job(
        engine=object(),
        job=job,
        derived_dir=tmp_path,
        reports_dir=tmp_path,
        spawn=spawn,
        is_cancel_requested=never_cancelled,
        close_mlflow_run=lambda run_id, status: closed.append((run_id, status)),
    )

    assert ("set_mlflow_run_id", "job-1", {"mlflow_run_id": "run-orphan"}) in recorder
    mark_failed_calls = [call for call in recorder if call[0] == "mark_failed"]
    assert len(mark_failed_calls) == 1
    assert "137" in mark_failed_calls[0][2]["error"]
    assert closed == [("run-orphan", "FAILED")]


def test_spawn_receives_the_dataset_and_meta_paths_built_from_release(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_worker.__main__.update_progress", lambda *a, **k: None)
    monkeypatch.setattr("ml_worker.__main__.mark_completed", lambda *a, **k: None)
    job = a_job(dataset_release="v0.1.1")
    received = {}

    def spawn(_job, manifest_path, crops_root, meta_path):
        received["manifest_path"] = manifest_path
        received["crops_root"] = crops_root
        received["meta_path"] = meta_path
        return FakeProcess(), FakeQueue([{"type": "completed", "mlflow_run_id": "run-1"}])

    derived_dir = tmp_path / "derived"
    reports_dir = tmp_path / "reports"
    process_one_job(
        engine=object(),
        job=job,
        derived_dir=derived_dir,
        reports_dir=reports_dir,
        spawn=spawn,
        is_cancel_requested=never_cancelled,
    )

    assert received["manifest_path"] == derived_dir / "manifests" / "v0.1.1" / "manifest.csv"
    assert received["crops_root"] == derived_dir / "crops"
    assert received["meta_path"] == reports_dir / "manifests" / "v0.1.1" / "manifest_meta.json"


def test_cancels_a_running_job_when_a_cancellation_is_requested(recorder, tmp_path):
    """Revisión de Uriel sobre P3-09: "no hay forma de cancelar"."""
    job = a_job()
    messages = [
        {"type": "run_started", "mlflow_run_id": "run-cancel-me"},
        {"type": "progress", "progress": 0.1, "log_line": "época 1/10"},
    ]
    process_box = {}

    def spawn(_job, _manifest_path, _crops_root, _meta_path):
        process = FakeProcess()
        process_box["process"] = process
        return process, FakeQueue(messages)

    # Pide cancelar justo después de ver el run_started -- antes de que
    # llegue cualquier otro mensaje de progreso.
    calls = {"n": 0}

    def cancel_after_run_started(_engine, _job_id):
        calls["n"] += 1
        return calls["n"] > 1

    closed = []
    process_one_job(
        engine=object(),
        job=job,
        derived_dir=tmp_path,
        reports_dir=tmp_path,
        spawn=spawn,
        is_cancel_requested=cancel_after_run_started,
        close_mlflow_run=lambda run_id, status: closed.append((run_id, status)),
    )

    assert process_box["process"].terminated
    assert ("mark_cancelled", "job-1", {}) in recorder
    assert closed == [("run-cancel-me", "KILLED")]
    assert not any(call[0] == "mark_failed" for call in recorder)


def test_cancellation_forces_a_kill_if_terminate_does_not_stop_it_in_time(recorder, tmp_path):
    job = a_job()

    class StubbornProcess(FakeProcess):
        def terminate(self):
            self.terminated = True
            # A diferencia de FakeProcess, sigue "vivo" tras el SIGTERM.

    process_box = {}

    def spawn(_job, _manifest_path, _crops_root, _meta_path):
        process = StubbornProcess()
        process_box["process"] = process
        return process, FakeQueue([])

    process_one_job(
        engine=object(),
        job=job,
        derived_dir=tmp_path,
        reports_dir=tmp_path,
        spawn=spawn,
        is_cancel_requested=lambda *_: True,
        close_mlflow_run=lambda *_: None,
    )

    assert process_box["process"].terminated
    assert process_box["process"].killed
    assert ("mark_cancelled", "job-1", {}) in recorder
