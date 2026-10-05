"""`ml_worker.run_training` (P3-09): arma el entrenamiento real y llama a
`train_with_mlflow` (P3-08) de verdad, contra el fixture compartido de
`conftest.py` -- mismo dataset mínimo que usan las pruebas de P3-07/P3-08,
no uno inventado aparte.
"""

import json
from datetime import datetime
from pathlib import Path
from queue import Queue

import pytest

pytest.importorskip("torch")
pytest.importorskip("mlflow")

from ml_api.contracts import TrainingJob
from ml_worker.run_training import (
    dataset_paths,
    manifest_meta_path,
    run_training,
    run_training_subprocess,
)

VALID_CONFIG = {
    "optimizer": "adam",
    "batch_size": 16,
    "max_epochs": 15,
    "learning_rate": 0.001,
    "image_size": 128,
    "hidden_layers": 0,
    "dropout": 0.0,
    "seed_split": 42,
    "seed_train": 43,
    "seed_aug": 44,
    "seed_model": 45,
    "patience": 5,
    "min_delta": 0.01,
}


def a_job(**overrides) -> TrainingJob:
    defaults = {
        "id": "job-1",
        "status": "running",
        "progress": 0.0,
        "config": VALID_CONFIG,
        "dataset_release": "vtest",
        "manifest_id": "vtest-abc",
        "run_kind": "smoke",
        "grid_row": None,
        "mlflow_run_id": None,
        "error": None,
        "logs": [],
        "heartbeat_at": None,
    }
    return TrainingJob(**{**defaults, **overrides})


@pytest.fixture(autouse=True)
def mlflow_local_tracking(tmp_path, monkeypatch):
    """Apunta MLflow a un tracking store local (sqlite) para no pegarle a un
    servidor real -- mismo criterio que usa test_p3_08.py."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")


@pytest.fixture
def meta_path(tmp_path) -> Path:
    """`manifest_meta.json` real (P3-08 lo exige vía MANIFEST_META_PATH)."""
    path = manifest_meta_path(tmp_path, "vtest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "manifest_id": "vtest-abc",
                "manifest_sha256": "deadbeef",
                "classes": {"0": "cat", "1": "dog"},
                "release": {"name": "vtest"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dataset_paths_join_derived_dir_with_the_release():
    manifest_path, crops_root = dataset_paths(Path("/data/derived"), "v0.1.1")
    assert manifest_path == Path("/data/derived/manifests/v0.1.1/manifest.csv")
    assert crops_root == Path("/data/derived/crops")


def test_run_training_returns_a_real_run_id_and_reports_every_epoch(mock_dataset_env, meta_path):
    manifest_path, crops_root = mock_dataset_env
    job = a_job()
    epochs_seen = []

    def on_epoch(epoch, max_epochs, train_loss, val_loss, val_accuracy):
        epochs_seen.append((epoch, max_epochs))

    run_id = run_training(
        job,
        manifest_path=Path(manifest_path),
        crops_root=Path(crops_root),
        meta_path=meta_path,
        on_epoch=on_epoch,
    )

    assert isinstance(run_id, str) and run_id
    # El patience de la config puede parar antes de las 15 épocas -- lo que
    # importa es que sí avisó, en orden, desde la 1.
    assert epochs_seen
    assert [epoch for epoch, _max in epochs_seen] == list(range(1, len(epochs_seen) + 1))
    assert all(max_epochs == 15 for _epoch, max_epochs in epochs_seen)


def test_run_training_calls_on_run_started_with_the_real_run_id_before_any_epoch(
    mock_dataset_env, meta_path
):
    """Revisión de Uriel sobre P3-09 ("la corrida queda en RUNNING para
    siempre"): el worker necesita el run_id apenas existe, no solo al
    terminar -- esto es lo que se lo entrega."""
    manifest_path, crops_root = mock_dataset_env
    job = a_job(config={**VALID_CONFIG, "patience": 0})
    seen = []

    def on_run_started(run_id):
        seen.append(run_id)

    run_id = run_training(
        job,
        manifest_path=Path(manifest_path),
        crops_root=Path(crops_root),
        meta_path=meta_path,
        on_run_started=on_run_started,
    )

    assert seen == [run_id]


def test_run_training_works_without_an_on_epoch_callback(mock_dataset_env, meta_path):
    manifest_path, crops_root = mock_dataset_env
    job = a_job(config={**VALID_CONFIG, "patience": 0})

    run_id = run_training(
        job, manifest_path=Path(manifest_path), crops_root=Path(crops_root), meta_path=meta_path
    )

    assert isinstance(run_id, str) and run_id


def test_run_training_rejects_campaign_without_a_grid_row(mock_dataset_env, meta_path):
    manifest_path, crops_root = mock_dataset_env
    job = a_job(config={**VALID_CONFIG, "patience": 0}, run_kind="campaign", grid_row=None)

    with pytest.raises(ValueError, match="grid_row"):
        run_training(
            job, manifest_path=Path(manifest_path), crops_root=Path(crops_root), meta_path=meta_path
        )


def test_run_training_subprocess_accepts_the_payload_spawn_sends_it(mock_dataset_env, meta_path):
    """Regresión: `_spawn_training_process` le pasa `job.model_dump()` (no
    `mode="json"`) a este subproceso -- con `mode="json"` `heartbeat_at`
    llega como texto, y `TrainingJob` (`strict=True`) lo rechaza al
    reconstruirlo. Encontrado de verdad corriendo el worker contra Docker,
    no solo hipotético."""
    manifest_path, crops_root = mock_dataset_env
    job = a_job(
        config={**VALID_CONFIG, "patience": 0}, heartbeat_at=datetime(2026, 9, 30, 12, 0, 0)
    )
    payload = job.model_dump()
    queue = Queue()

    run_training_subprocess(payload, manifest_path, crops_root, str(meta_path), queue)

    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())

    assert messages[-1]["type"] == "completed"
    assert messages[-1]["mlflow_run_id"]
    assert messages[0]["type"] == "run_started"
    assert messages[0]["mlflow_run_id"] == messages[-1]["mlflow_run_id"]
