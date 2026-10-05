"""`ml_api.server` (P3-03/P3-09): /health, /training/jobs (GET+POST) y los 4
endpoints pendientes.

`list_jobs`/`create_job` se inyectan -- igual que `create_message` en
`copilot/server.py` -- así esta prueba nunca toca una MariaDB real; eso lo
cubre `test_ml_api_repository.py` con datos tomados de una corrida real. La
validación de `POST /training/jobs` (release/manifiesto/fuga) sí corre de
verdad, contra los archivos que escribe `write_manifest` -- es la pieza que
este ticket (P3-09) tiene que probar, no un detalle de infraestructura.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from starlette.testclient import TestClient

from ml_api.contracts import TrainingJob
from ml_api.repository import TrainingJobNotCancellable
from ml_api.server import create_app
from selection.lock import test_ids_sha256 as _test_ids_sha256
from tests._mcp_fixtures import mcp_settings
from tests._training_job_fixtures import VALID_TRAINING_CONFIG, write_manifest


def client_for(
    monkeypatch,
    tmp_path,
    *,
    jobs: list[TrainingJob] | None = None,
    create_job=None,
    cancel_job=None,
) -> TestClient:
    derived_dir = tmp_path / "derived"
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports", derived_dir)
    return TestClient(
        create_app(
            settings, list_jobs=lambda: jobs or [], create_job=create_job, cancel_job=cancel_job
        )
    )


def a_job(**overrides) -> TrainingJob:
    defaults = {
        "id": "r01",
        "status": "running",
        "progress": 0.4,
        "config": {"optimizer": "adam"},
        "dataset_release": "v0.1.1",
        "manifest_id": "manifest-abc",
        "run_kind": "smoke",
        "grid_row": None,
        "mlflow_run_id": None,
        "error": None,
        "logs": ["epoch 1 done"],
        "heartbeat_at": datetime(2026, 9, 29, 21, 23, 26),
    }
    return TrainingJob(**{**defaults, **overrides})


def test_health_says_ok(monkeypatch, tmp_path):
    response = client_for(monkeypatch, tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_training_jobs_returns_the_injected_jobs(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, jobs=[a_job(), a_job(id="r02", status="queued")])
    response = client.get("/training/jobs")

    assert response.status_code == 200
    body = response.json()
    assert [job["id"] for job in body["jobs"]] == ["r01", "r02"]
    assert body["jobs"][0]["config"] == {"optimizer": "adam"}


def test_training_jobs_is_an_empty_list_before_any_run(monkeypatch, tmp_path):
    response = client_for(monkeypatch, tmp_path).get("/training/jobs")

    assert response.status_code == 200
    assert response.json() == {"jobs": []}


def test_the_three_pending_endpoints_name_their_own_ticket(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)
    expected_tickets = {
        "/models": "P3-14",
        "/inference": "P3-16",
    }

    for path, ticket in expected_tickets.items():
        response = client.get(path)
        assert response.status_code == 200, path
        body = response.json()
        assert body["status"] == "pending"
        assert body["ticket"] == ticket


def test_evaluation_is_locked_until_selection_is_closed(monkeypatch, tmp_path):
    # P3-11 (#18): sin reports/selection.json, /evaluation se niega a servir datos.
    client = client_for(monkeypatch, tmp_path)
    response = client.get("/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "selection_not_closed"
    assert body["ticket"] == "P3-11"
    assert "selection" in body["message"]


class _RecordingCreateJob:
    """Espía: prueba que ninguna fila se crea cuando la solicitud se rechaza."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict, str, str | None]] = []

    def __call__(self, dataset_release, manifest_id, config, run_kind, grid_row):
        self.calls.append((dataset_release, manifest_id, config, run_kind, grid_row))
        return a_job(
            id="new-job",
            status="queued",
            progress=0.0,
            dataset_release=dataset_release,
            manifest_id=manifest_id,
            config=config,
            run_kind=run_kind,
            grid_row=grid_row,
            logs=[],
        )


def test_create_job_rejects_invalid_config_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")

    bad_config = {**VALID_TRAINING_CONFIG, "batch_size": 7}  # 7 no es 16 ni 32
    response = client.post(
        "/training/jobs",
        json={"dataset_release": "v0.1.1", "config": bad_config, "run_kind": "smoke"},
    )

    assert response.status_code == 400
    assert recorder.calls == []


def test_create_job_rejects_missing_run_kind_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")

    response = client.post(
        "/training/jobs", json={"dataset_release": "v0.1.1", "config": VALID_TRAINING_CONFIG}
    )

    assert response.status_code == 400
    assert recorder.calls == []


def test_create_job_rejects_a_non_frozen_patience_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")

    response = client.post(
        "/training/jobs",
        json={
            "dataset_release": "v0.1.1",
            "config": {**VALID_TRAINING_CONFIG, "patience": 5},
            "run_kind": "smoke",
        },
    )

    assert response.status_code == 400
    assert "congelad" in response.json()["error"]
    assert recorder.calls == []


def test_create_job_rejects_a_non_frozen_seed_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")

    response = client.post(
        "/training/jobs",
        json={
            "dataset_release": "v0.1.1",
            "config": {**VALID_TRAINING_CONFIG, "seed_aug": 99},
            "run_kind": "smoke",
        },
    )

    assert response.status_code == 400
    assert "congelad" in response.json()["error"]
    assert recorder.calls == []


def test_create_job_rejects_campaign_with_config_that_does_not_match_the_grid_row(
    monkeypatch, tmp_path
):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(
        reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived", quality_status="warning"
    )

    response = client.post(
        "/training/jobs",
        json={
            "dataset_release": "v0.1.1",
            # r01 real es batch_size=32; esto manda 16 -- no coincide con la fila.
            "config": {**VALID_TRAINING_CONFIG, "batch_size": 16},
            "run_kind": "campaign",
            "grid_row": "r01",
        },
    )

    assert response.status_code == 400
    assert "no coincide con la rejilla" in response.json()["error"]
    assert recorder.calls == []


def test_create_job_rejects_campaign_without_a_grid_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")

    response = client.post(
        "/training/jobs",
        json={
            "dataset_release": "v0.1.1",
            "config": VALID_TRAINING_CONFIG,
            "run_kind": "campaign",
        },
    )

    assert response.status_code == 400
    assert "grid_row" in response.json()["error"]
    assert recorder.calls == []


def test_create_job_rejects_unknown_release_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    # Ningún manifest_meta.json escrito para "v9.9.9": no existe el release.

    response = client.post(
        "/training/jobs",
        json={"dataset_release": "v9.9.9", "config": VALID_TRAINING_CONFIG, "run_kind": "smoke"},
    )

    assert response.status_code == 400
    assert recorder.calls == []


def test_create_job_rejects_a_failed_release_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(
        reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived", quality_status="failed"
    )

    response = client.post(
        "/training/jobs",
        json={"dataset_release": "v0.1.1", "config": VALID_TRAINING_CONFIG, "run_kind": "smoke"},
    )

    assert response.status_code == 400
    assert "failed" in response.json()["error"]
    assert recorder.calls == []


def test_create_job_rejects_a_leaking_manifest_without_creating_a_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived", leaking=True)

    response = client.post(
        "/training/jobs",
        json={"dataset_release": "v0.1.1", "config": VALID_TRAINING_CONFIG, "run_kind": "smoke"},
    )

    assert response.status_code == 400
    assert "fuga" in response.json()["error"]
    assert recorder.calls == []


def test_create_job_accepts_a_valid_request_and_creates_exactly_one_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    manifest_id = write_manifest(
        reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived", quality_status="warning"
    )

    response = client.post(
        "/training/jobs",
        json={"dataset_release": "v0.1.1", "config": VALID_TRAINING_CONFIG, "run_kind": "smoke"},
    )

    assert response.status_code == 201
    assert len(recorder.calls) == 1
    called_release, called_manifest_id, called_config, called_run_kind, called_grid_row = (
        recorder.calls[0]
    )
    assert called_release == "v0.1.1"
    assert called_manifest_id == manifest_id
    assert called_config == VALID_TRAINING_CONFIG
    assert called_run_kind == "smoke"
    assert called_grid_row is None
    assert response.json()["status"] == "queued"


def test_create_job_accepts_a_valid_campaign_request_with_grid_row(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)
    write_manifest(
        reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived", quality_status="warning"
    )

    response = client.post(
        "/training/jobs",
        json={
            "dataset_release": "v0.1.1",
            "config": VALID_TRAINING_CONFIG,
            "run_kind": "campaign",
            "grid_row": "r01",
        },
    )

    assert response.status_code == 201
    assert len(recorder.calls) == 1
    assert recorder.calls[0][3:] == ("campaign", "r01")


def test_create_job_rejects_a_non_object_body(monkeypatch, tmp_path):
    recorder = _RecordingCreateJob()
    client = client_for(monkeypatch, tmp_path, create_job=recorder)

    response = client.post("/training/jobs", json=[1, 2, 3])

    assert response.status_code == 400
    assert recorder.calls == []


def test_cancel_job_returns_the_updated_job(monkeypatch, tmp_path):
    calls = []

    def cancel_job(job_id):
        calls.append(job_id)
        return a_job(id=job_id, status="cancelled")

    client = client_for(monkeypatch, tmp_path, cancel_job=cancel_job)

    response = client.post("/training/jobs/r01/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert calls == ["r01"]


def test_cancel_job_is_404_when_it_does_not_exist(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, cancel_job=lambda job_id: None)

    response = client.post("/training/jobs/ghost/cancel")

    assert response.status_code == 404


def test_cancel_job_is_409_when_it_already_finished(monkeypatch, tmp_path):
    def cancel_job(job_id):
        raise TrainingJobNotCancellable("completed")

    client = client_for(monkeypatch, tmp_path, cancel_job=cancel_job)

    response = client.post("/training/jobs/r01/cancel")

    assert response.status_code == 409
    assert "completed" in response.json()["error"]


# --- P3-15: /evaluation y /crops ---------------------------------------------


def write_closed_selection(
    *, reports_dir: Path, derived_dir: Path, release: str = "v0.1.1"
) -> None:
    """Cierra el candado de P3-11: manifiesto con split `test` + selection.json
    cuyo `test_ids_sha256` coincide de verdad (lo calcula el mismo código que
    usa `lock_reason`)."""
    manifest_dir = derived_dir / "manifests" / release
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"crop_id": "000001_000001", "split": "train", "label": 0, "source_image_id": 1},
        {"crop_id": "000002_000001", "split": "val", "label": 1, "source_image_id": 2},
        {"crop_id": "000003_000001", "split": "test", "label": 0, "source_image_id": 3},
        {"crop_id": "000004_000001", "split": "test", "label": 1, "source_image_id": 4},
    ]
    with (manifest_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selection = {
        "run_id": "7e7b4a4b35464cfebb6b41714a3ad931",
        "release": release,
        "manifest_id": f"{release}-53fc84fdaa07",
        "manifest_sha256": "5" * 64,
        "checkpoint_sha256": "f" * 64,
        "test_ids_sha256": _test_ids_sha256(manifest_dir / "manifest.csv"),
        "selected_at": "2026-10-02T05:43:20.705345Z",
        "selection_metric": "best_val_accuracy",
        "candidate": {
            "grid_row": "r02",
            "best_val_accuracy": 0.9618320610687023,
            "best_val_macro_f1": 0.9618231625575566,
            "best_val_loss": 0.12332657724618912,
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")


def write_evaluation_reports(evaluation_dir: Path) -> None:
    """Los tres archivos que deja `app/final.py` (P3-13), en miniatura."""
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "classes": ["cat", "dog"],
        "accuracy": 0.75,
        "macro_f1": 0.7333333333333334,
        "confusion_matrix": [[2, 0], [1, 1]],
        "per_class": {
            "cat": {"precision": 0.6666666666666666, "recall": 1.0, "f1": 0.8, "support": 2},
            "dog": {"precision": 1.0, "recall": 0.5, "f1": 0.6666666666666666, "support": 2},
        },
        "total": 4,
    }
    analysis = {
        "baseline_majority_accuracy": 0.5,
        "most_confused_class": "dog",
        "recall_per_class": {"cat": 1.0, "dog": 0.5},
        "accuracy_hides_low_recall": False,
        "successes": [
            {
                "crop_id": "000003_000001",
                "source_image_id": "3",
                "true_class": "cat",
                "predicted_class": "cat",
                "probability": 0.95,
            }
        ],
        "errors": [
            {
                "crop_id": "000004_000001",
                "source_image_id": "4",
                "true_class": "dog",
                "predicted_class": "cat",
                "probability": 0.55,
            }
        ],
    }
    (evaluation_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (evaluation_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")


def test_evaluation_is_pending_when_selection_is_closed_but_p3_13_has_not_run(
    monkeypatch, tmp_path
):
    write_closed_selection(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["ticket"] == "P3-13"


def test_evaluation_returns_the_real_metrics_once_p3_13_exists(monkeypatch, tmp_path):
    write_closed_selection(reports_dir=tmp_path / "reports", derived_dir=tmp_path / "derived")
    write_evaluation_reports(tmp_path / "reports" / "evaluation")
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["grid_row"] == "r02"
    assert body["run_id"] == "7e7b4a4b35464cfebb6b41714a3ad931"
    assert body["accuracy"] == 0.75
    assert body["per_class"]["dog"]["recall"] == 0.5
    assert body["baseline_majority_accuracy"] == 0.5
    assert body["most_confused_class"] == "dog"
    assert [example["crop_id"] for example in body["errors"]] == ["000004_000001"]


def test_crops_reject_a_malformed_crop_id(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/crops/no-es-un-crop")

    assert response.status_code == 400


def test_crops_are_404_when_the_file_does_not_exist(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/crops/000003_000001")

    assert response.status_code == 404


def test_crops_serve_the_real_jpeg(monkeypatch, tmp_path):
    images_dir = tmp_path / "derived" / "crops" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "000003_000001.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/crops/000003_000001")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8\xff")
