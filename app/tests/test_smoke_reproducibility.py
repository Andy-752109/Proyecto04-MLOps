"""Smoke test automatizado de reproducibilidad (P3-10, issue #17, modo SMOKE).

Corre el MISMO camino que el worker real: `ml_worker.run_training.run_training`
(`set_reproducibility` + `create_dataloader(batch_size=config.batch_size,
seed_aug=config.seed_aug)` + `train_with_mlflow` con `run_kind=smoke`), no una
copia de esa lógica. Datos: recortes JPEG pequeños escritos en `tmp_path` con
un manifiesto real (mismas columnas que `manifest.csv`); MLflow en un store
temporal (SQLite + artefactos en `tmp_path`). Nada se escribe en `app/` ni en `mlruns/` del repo.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("mlflow")

import mlflow
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from PIL import Image, ImageDraw

from ml_api.contracts import TrainingJob
from ml_worker.run_training import run_training
from training.grid import FROZEN_MIN_DELTA, FROZEN_SEEDS, GRID
from training.smoke import (
    EPOCH_METRICS,
    compare_runs,
    predict_from_run,
    predict_in_new_process,
    verify_checkpoint_sha256,
)

TOLERANCE = 1e-6
GIT_COMMIT = "a" * 40  # valor de fixture, no el commit real
TRAIN_PER_CLASS = 12
VAL_PER_CLASS = 4

# r01 de la rejilla con batch_size=16 (distinto de r01 y del 2 que usaba la
# versión anterior del smoke test) y early stopping corto para acotar el tiempo.
SMOKE_CONFIG = {
    **GRID["r01"],
    **FROZEN_SEEDS,
    "batch_size": 16,
    "patience": 1,
    "min_delta": FROZEN_MIN_DELTA,
}


def _write_crop(path: Path, label: int, index: int) -> None:
    """Recorte sintético separable por clase (gato rojizo, perro azulado)."""
    import random

    rng = random.Random(1000 * label + index)
    base = (190, 70, 70) if label == 0 else (70, 70, 190)
    image = Image.new("RGB", (64, 64), color=base)
    draw = ImageDraw.Draw(image)
    x0, y0 = rng.randint(0, 36), rng.randint(0, 36)
    draw.rectangle([x0, y0, x0 + 20, y0 + 20], fill=(rng.randint(0, 255),) * 3)
    image.save(path, format="JPEG")


def _build_dataset(root: Path) -> SimpleNamespace:
    crops_root = root / "crops"
    crops_root.mkdir(parents=True)
    rows = []
    for split, per_class in (("train", TRAIN_PER_CLASS), ("val", VAL_PER_CLASS)):
        for label in (0, 1):
            for index in range(per_class):
                name = f"{split}_{label}_{index}.jpg"
                _write_crop(crops_root / name, label, index)
                rows.append(
                    {
                        "crop_id": f"{split}-{label}-{index}",
                        "path": name,
                        "split": split,
                        "label": label,
                    }
                )
    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    meta_path = root / "manifest_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "manifest_id": "vsmoke-fixture",
                "manifest_sha256": manifest_sha,
                "classes": {"0": "cat", "1": "dog"},
                "release": {"name": "vsmoke"},
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        manifest_path=manifest_path,
        crops_root=crops_root,
        meta_path=meta_path,
        manifest_sha256=manifest_sha,
    )


def _smoke_job() -> TrainingJob:
    return TrainingJob(
        id="smoke-job",
        status="running",
        progress=0.0,
        config=SMOKE_CONFIG,
        dataset_release="vsmoke",
        manifest_id="vsmoke-fixture",
        run_kind="smoke",
        grid_row=None,
        mlflow_run_id=None,
        error=None,
        logs=[],
        heartbeat_at=None,
    )


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """Dos corridas smoke con la misma config y semillas, por el camino del worker."""
    root = tmp_path_factory.mktemp("smoke")
    data = _build_dataset(root)
    # SQLite temporal: MLflow 3.16 deja el file store en modo mantenimiento
    # (lanza error salvo MLFLOW_ALLOW_FILE_STORE). Los artefactos van a tmp.
    tracking_uri = f"sqlite:///{root / 'mlflow.db'}"

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_uri = mlflow.get_tracking_uri()
    with pytest.MonkeyPatch.context() as patch:
        # `run_training` exporta MANIFEST_META_PATH y `set_reproducibility`
        # CUBLAS_WORKSPACE_CONFIG: setearlas antes con monkeypatch las restaura.
        patch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        patch.setenv("MANIFEST_META_PATH", str(data.meta_path))
        patch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        patch.setenv("GIT_COMMIT", GIT_COMMIT)
        patch.setenv("GIT_DIRTY", "false")
        mlflow.set_tracking_uri(tracking_uri)
        # Sin esto el experimento nace con artefactos en ./mlruns (cwd = app/).
        mlflow.create_experiment(
            "clasificador-perro-gato", artifact_location=(root / "artifacts").as_uri()
        )
        try:
            run_ids = [
                run_training(
                    _smoke_job(),
                    manifest_path=data.manifest_path,
                    crops_root=data.crops_root,
                    meta_path=data.meta_path,
                )
                for _ in range(2)
            ]
            yield SimpleNamespace(
                data=data, tracking_uri=tracking_uri, run_ids=run_ids, client=MlflowClient()
            )
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
            mlflow.set_tracking_uri(previous_uri)


def _history(client, run_id, metric):
    return [
        m.value for m in sorted(client.get_metric_history(run_id, metric), key=lambda m: m.step)
    ]


def test_runs_use_the_worker_config_and_real_provenance(smoke):
    run = mlflow.get_run(smoke.run_ids[0])
    assert run.data.params["batch_size"] == "16"
    assert run.data.params["seed_aug"] == "44"
    assert run.data.tags["run_kind"] == "smoke"
    assert run.data.tags["grid_row"] == ""
    assert run.data.tags["git_commit"] == GIT_COMMIT
    assert run.data.tags["git_dirty"] == "false"
    # El hash registrado es el del manifiesto con el que se entrenó de verdad.
    assert run.data.tags["manifest_sha256"] == smoke.data.manifest_sha256
    assert run.info.status == "FINISHED"


def test_two_runs_give_the_same_metrics_per_epoch_and_best_epoch(smoke):
    first, second = smoke.run_ids
    for metric in EPOCH_METRICS:
        a = _history(smoke.client, first, metric)
        b = _history(smoke.client, second, metric)
        assert a, f"{metric} no tiene historial"
        assert len(a) == len(b), f"{metric}: distinto número de épocas"
        assert a == pytest.approx(b, abs=TOLERANCE), metric

    metrics_a = mlflow.get_run(first).data.metrics
    metrics_b = mlflow.get_run(second).data.metrics
    assert metrics_a["best_epoch"] == metrics_b["best_epoch"]
    assert metrics_a["stopped_epoch"] == metrics_b["stopped_epoch"]
    assert metrics_a["duration_s"] > 0 and metrics_b["duration_s"] > 0


def test_compare_runs_reports_no_differences_for_identical_runs(smoke):
    assert compare_runs(smoke.client, *smoke.run_ids, tolerance=TOLERANCE) == []


def test_compare_runs_reports_a_difference(smoke):
    other = smoke.client.create_run(
        smoke.client.get_run(smoke.run_ids[0]).info.experiment_id, run_name="alterada"
    )
    smoke.client.log_metric(other.info.run_id, "val_loss", 123.0, step=1)
    try:
        diffs = compare_runs(smoke.client, smoke.run_ids[0], other.info.run_id, tolerance=TOLERANCE)
    finally:
        smoke.client.delete_run(other.info.run_id)
    assert any("val_loss" in d for d in diffs)


def test_checkpoint_in_a_new_process_reproduces_predictions(smoke):
    run_id = smoke.run_ids[0]
    tag_sha = mlflow.get_run(run_id).data.tags["checkpoint_sha256"]

    here = predict_from_run(
        smoke.tracking_uri, run_id, SMOKE_CONFIG, smoke.data.manifest_path, smoke.data.crops_root
    )
    there = predict_in_new_process(
        smoke.tracking_uri, run_id, SMOKE_CONFIG, smoke.data.manifest_path, smoke.data.crops_root
    )

    assert there["checkpoint_sha256"] == tag_sha
    assert there["pid"] != here["pid"]
    assert len(there["crop_ids"]) == 2 * VAL_PER_CLASS
    assert there["crop_ids"] == here["crop_ids"]
    assert there["probabilities"] == pytest.approx(here["probabilities"], abs=TOLERANCE)


def test_second_run_checkpoint_predicts_like_the_first(smoke):
    first, second = (
        predict_from_run(
            smoke.tracking_uri,
            run_id,
            SMOKE_CONFIG,
            smoke.data.manifest_path,
            smoke.data.crops_root,
        )
        for run_id in smoke.run_ids
    )
    assert second["probabilities"] == pytest.approx(first["probabilities"], abs=TOLERANCE)


def test_verify_checkpoint_sha256_accepts_a_match_and_rejects_a_mismatch(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"pesos")
    expected = hashlib.sha256(b"pesos").hexdigest()

    assert verify_checkpoint_sha256(checkpoint, expected) == expected
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        verify_checkpoint_sha256(checkpoint, "0" * 64)
