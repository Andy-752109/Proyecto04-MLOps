"""P3-17 (#27): pipeline de punta a punta, con IDs trazables, <5 min en CPU.

Dos tramos, cada uno real en su propio terreno (ninguno es un mock):

1. **En vivo** -- manifiesto -> trabajo corto (`run_training`, P3-09) ->
   corrida real en MLflow -> selección (`validate_runs`/`select_candidate`,
   P3-11) -> evaluación (`final.main`, P3-13, que registra las métricas
   `test_*` dentro de la MISMA corrida, no en una aparte). El `run_id` es el
   mismo de principio a fin -- trazable de verdad.
2. **Publicación + predicción** -- un mundo de fixture autoconsistente
   (mismo patrón que `test_publish_minio.py`, pero con un checkpoint real
   cargable, no bytes cualquiera): `publish.publish` sube un paquete a MinIO
   real, y `ml_api.inference.load_active_model` + `predict_route` (vía
   `TestClient`, una predicción real por API) lo bajan, verifican su
   SHA-256 y predicen con él.

Por qué dos mundos y no uno solo: el pipeline real de publicación
(`publish.py`/`card.py`) lee de un volcado SQL de MLflow (`mlflow-store/
mlflow.sql`, el snapshot de DVC), no de un MLflow en vivo -- son dos
representaciones de datos distintas a propósito (P3-14 no depende de que el
servidor de MLflow de la campaña siga corriendo al publicar). Traducir una
corrida en vivo a ese formato es trabajo real pero aparte; aquí se opta por
un segundo mundo de fixture, consistente en sí mismo, en vez de esa
traducción -- la trazabilidad del tramo 1 ya demuestra que esa mitad de la
cadena conecta de verdad, y el tramo 2 demuestra lo mismo para la otra
mitad, cada regla de negocio en el medio ya tiene su propia prueba en
`test_p3_11.py`, `test_p3_13.py`, `test_publish_minio.py` y `test_p3_16.py`.

Necesita MinIO con versionado para el tramo 2 (igual que `test_publish_minio
.py`); se salta si no responde.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("boto3")

import boto3
import mlflow
import pandas as pd
import publish
import torch
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from card import build_card
from PIL import Image
from starlette.testclient import TestClient

from final import main as final_main
from ml_api.contracts import TrainingJob
from ml_api.inference import load_active_model
from ml_api.server import create_app
from ml_worker.run_training import run_training
from selection.contracts import Candidate, Selection
from selection.lock import test_ids_sha256 as compute_test_ids_sha256
from selection.mlflow_reader import EXPERIMENT, read_runs
from selection.select import select_candidate
from selection.validate import validate_runs
from tests._mcp_fixtures import mcp_settings
from training.config import TrainingConfig
from training.grid import FROZEN_MIN_DELTA, FROZEN_PATIENCE, FROZEN_SEEDS, GRID
from training.model import build_model

ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
GIT_COMMIT = "c" * 40

CONFIG = {
    **GRID["r01"],
    **FROZEN_SEEDS,
    "batch_size": 16,
    # patience/min_delta congelados (sección 4) -- validate_runs los exige
    # para considerar la corrida válida; con un dataset de 16 recortes el
    # early stopping de todas formas corta rápido.
    "patience": FROZEN_PATIENCE,
    "min_delta": FROZEN_MIN_DELTA,
}


# --- tramo 1: dataset + entrenamiento en vivo -----------------------------------


def _write_crop(path: Path, label: int) -> None:
    color = (190, 70, 70) if label == 0 else (70, 70, 190)
    Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


def _build_dataset(root: Path) -> dict:
    crops_root = root / "crops"
    crops_root.mkdir(parents=True)
    rows = []
    for split, per_class in (("train", 4), ("val", 2), ("test", 2)):
        for label in (0, 1):
            for index in range(per_class):
                name = f"{split}_{label}_{index}.jpg"
                _write_crop(crops_root / name, label)
                rows.append(
                    {
                        "crop_id": f"{split}-{label}-{index}",
                        "path": name,
                        "split": split,
                        "label": label,
                        "source_image_id": f"img-{split}-{label}-{index}",
                    }
                )
    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    manifest_sha256 = "a" * 64
    meta_path = root / "manifest_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "manifest_id": "ve2e-fixture",
                "manifest_sha256": manifest_sha256,
                "classes": {"0": "cat", "1": "dog"},
                "release": {"name": "ve2e"},
            }
        ),
        encoding="utf-8",
    )
    return {"manifest_path": manifest_path, "crops_root": crops_root, "meta_path": meta_path}


@pytest.fixture
def live_chain(tmp_path, monkeypatch):
    """Entrena, selecciona y evalúa en vivo; devuelve el `run_id` -- el
    mismo de principio a fin -- y las rutas del manifiesto."""
    dataset = _build_dataset(tmp_path / "dataset")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    monkeypatch.setenv("GIT_COMMIT", GIT_COMMIT)
    monkeypatch.setenv("GIT_DIRTY", "false")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    try:
        job = TrainingJob(
            id="e2e-job",
            status="running",
            progress=0.0,
            config=CONFIG,
            dataset_release="ve2e",
            manifest_id="ve2e-fixture",
            run_kind="campaign",
            grid_row="r01",
            mlflow_run_id=None,
            error=None,
            logs=[],
            heartbeat_at=None,
        )
        run_id = run_training(
            job,
            manifest_path=dataset["manifest_path"],
            crops_root=dataset["crops_root"],
            meta_path=dataset["meta_path"],
        )

        # Selección: con una sola corrida no tiene sentido pedir cobertura de
        # rejilla (eso ya lo prueba test_p3_11.py por separado) -- aquí
        # importa que la corrida real sea válida y se pueda seleccionar.
        [run] = read_runs(tracking_uri, run_kind="campaign")
        validity = validate_runs(
            [run], manifest_sha256="a" * 64, classes={"0": "cat", "1": "dog"}, min_required=1
        )
        assert run_id in validity.valid, validity.invalid

        winner = select_candidate([run])
        assert winner.run_id == run_id

        now = datetime.now(timezone.utc)
        selection = Selection(
            run_id=run_id,
            release="ve2e",
            manifest_id="ve2e-fixture",
            manifest_sha256="a" * 64,
            checkpoint_sha256=winner.tags["checkpoint_sha256"],
            test_ids_sha256=compute_test_ids_sha256(dataset["manifest_path"]),
            selected_at=now - timedelta(seconds=1),
            candidate=Candidate(
                grid_row="r01",
                best_val_accuracy=winner.metrics["best_val_accuracy"],
                best_val_macro_f1=winner.metrics["best_val_macro_f1"],
                best_val_loss=winner.metrics["best_val_loss"],
            ),
        )
        selection_path = tmp_path / "selection.json"
        selection_path.write_text(selection.model_dump_json(), encoding="utf-8")

        output_dir = tmp_path / "evaluation"
        rc = final_main(
            [
                "--split",
                "test",
                "--tracking-uri",
                tracking_uri,
                "--selection",
                str(selection_path),
                "--manifest-csv",
                str(dataset["manifest_path"]),
                "--crops",
                str(dataset["crops_root"]),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert rc == 0

        yield {
            "run_id": run_id,
            "tracking_uri": tracking_uri,
            "checkpoint_sha256": winner.tags["checkpoint_sha256"],
            "manifest_id": "ve2e-fixture",
        }
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_live_chain_keeps_the_same_run_id_and_registers_test_metrics_in_it(live_chain):
    """Manifiesto -> entrenamiento -> selección -> evaluación, con el MISMO
    run_id de principio a fin (criterio del checklist: "IDs trazables")."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=live_chain["tracking_uri"])
    run = client.get_run(live_chain["run_id"])

    assert run.data.tags["manifest_id"] == live_chain["manifest_id"]
    assert run.data.tags["checkpoint_sha256"] == live_chain["checkpoint_sha256"]
    # Las métricas de test quedan en ESTA corrida, no en una aparte (P3-13).
    assert "test_accuracy" in run.data.metrics
    assert run.data.tags["eval_split"] == "test"

    experiment = client.get_experiment_by_name(EXPERIMENT)
    all_runs = client.search_runs([experiment.experiment_id])
    assert [r.info.run_id for r in all_runs] == [live_chain["run_id"]]


# --- tramo 2: publicación en MinIO + predicción por API -------------------------


@pytest.fixture(scope="module")
def minio():
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            connect_timeout=2,
            read_timeout=10,
            retries={"max_attempts": 1},
        ),
    )
    try:
        client.list_buckets()
    except (BotoCoreError, ClientError) as error:
        pytest.skip(f"MinIO no responde en {ENDPOINT}: {error.__class__.__name__}")
    return client


def _empty_bucket(client, bucket: str) -> None:
    for page in client.get_paginator("list_object_versions").paginate(Bucket=bucket):
        for item in page.get("Versions", []) + page.get("DeleteMarkers", []):
            client.delete_object(Bucket=bucket, Key=item["Key"], VersionId=item["VersionId"])


@pytest.fixture
def bucket(minio):
    name = f"p3-e2e-{uuid.uuid4().hex[:12]}"
    minio.create_bucket(Bucket=name)
    try:
        minio.put_bucket_versioning(Bucket=name, VersioningConfiguration={"Status": "Enabled"})
        if minio.get_bucket_versioning(Bucket=name).get("Status") != "Enabled":
            pytest.skip("MinIO sin versionado (necesita varios discos: server /data{1...4})")
        yield name
    finally:
        _empty_bucket(minio, name)
        minio.delete_bucket(Bucket=name)


def _sql_text(value) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "''").replace('"', '\\"')


def _insert(table: str, rows: list[tuple]) -> str:
    lines = [
        "(" + ",".join(f"'{_sql_text(v)}'" if isinstance(v, str) else str(v) for v in row) + ")"
        for row in rows
    ]
    return f"INSERT INTO {table} VALUES\n" + ",\n".join(lines) + ";\n"


@pytest.fixture
def published_world(tmp_path, monkeypatch):
    """Mundo de fixture autoconsistente (mismo patrón que `test_publish_minio
    .py`), pero con un checkpoint real y cargable: la predicción del tramo 2
    no es un mock, de verdad reconstruye el modelo y corre `forward()`."""
    run_id = "e" * 32
    config = {**CONFIG}
    typed_config = TrainingConfig(**config)
    real_model = build_model(typed_config)
    best_pt = tmp_path / "best.pt"
    torch.save(real_model.state_dict(), best_pt)
    checkpoint_sha256 = hashlib.sha256(best_pt.read_bytes()).hexdigest()

    snapshot = tmp_path / "mlflow-store"
    tags = {
        "release": "ve2e",
        "manifest_id": "ve2e-fixture",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": checkpoint_sha256,
        "run_kind": "campaign",
        "classes": json.dumps({"0": "cat", "1": "dog"}),
        "mlflow.runName": "e2e-fixture",
        "device": "cpu",
        "platform": "Linux-fixture",
        "python_version": "3.12.0",
        "torch_version": torch.__version__,
        "torchvision_version": "0.29.0",
        "git_commit": "f" * 40,
        "git_dirty": "false",
    }
    metrics = {
        "best_val_accuracy": 0.9,
        "best_val_macro_f1": 0.88,
        "best_val_loss": 0.31,
        "best_epoch": 1,
        "stopped_epoch": 1,
    }
    sql = (
        _insert("params", [(k, str(v), run_id) for k, v in config.items()])
        + _insert("tags", [(k, v, run_id) for k, v in tags.items()])
        + _insert("latest_metrics", [(k, float(v), 1, 0, 0, run_id) for k, v in metrics.items()])
    )
    (snapshot).mkdir(parents=True, exist_ok=True)
    (snapshot / "mlflow.sql").write_text(sql, encoding="utf-8")

    artifacts = snapshot / "artifacts" / "1" / run_id / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "checkpoint").mkdir()
    (artifacts / "checkpoint" / "best.pt").write_bytes(best_pt.read_bytes())
    (artifacts / "curves.csv").write_text("epoch,val_loss\n1,0.31\n", encoding="utf-8")

    reports = tmp_path / "reports"
    (reports / "manifests" / "ve2e").mkdir(parents=True)
    (reports / "manifests" / "ve2e" / "counts.json").write_text(
        json.dumps(
            {
                "release": "ve2e",
                "totals": {
                    "crops": 20,
                    "originals": 12,
                    "classes": {"cat": {"crops": 10}, "dog": {"crops": 10}},
                },
                "splits": {
                    "train": {
                        "crops": 8,
                        "originals": 5,
                        "classes": {"cat": {"crops": 4}, "dog": {"crops": 4}},
                    },
                    "val": {
                        "crops": 4,
                        "originals": 3,
                        "classes": {"cat": {"crops": 2}, "dog": {"crops": 2}},
                    },
                    "test": {
                        "crops": 4,
                        "originals": 3,
                        "classes": {"cat": {"crops": 2}, "dog": {"crops": 2}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "selection.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_id": "ve2e-fixture",
                "selection_metric": "best_val_accuracy",
                "selected_at": "2026-01-01T00:00:00Z",
                "candidate": {"grid_row": "r01"},
            }
        ),
        encoding="utf-8",
    )
    evaluation = artifacts / "evaluation"
    (evaluation).mkdir()
    (evaluation / "metrics.json").write_text(
        json.dumps(
            {
                "total": 4,
                "accuracy": 0.9,
                "macro_f1": 0.89,
                "classes": ["cat", "dog"],
                "per_class": {
                    "cat": {"precision": 0.8, "recall": 1.0, "f1": 0.89, "support": 2},
                    "dog": {"precision": 1.0, "recall": 0.83, "f1": 0.91, "support": 2},
                },
                "confusion_matrix": [[2, 0], [1, 1]],
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "analysis.json").write_text(
        json.dumps(
            {
                "most_confused_class": "dog",
                "recall_per_class": {"cat": 1.0, "dog": 0.83},
                "baseline_majority_accuracy": 0.5,
                "accuracy_hides_low_recall": False,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MLFLOW_SNAPSHOT", str(snapshot))
    return {"run_id": run_id, "config": typed_config, "checkpoint_sha256": checkpoint_sha256}


def test_publish_to_minio_and_predict_through_the_real_api(
    minio, bucket, published_world, tmp_path, monkeypatch
):
    """Publicación (P3-14) + predicción real por `POST /predict` (P3-16),
    sobre un paquete recién publicado en MinIO -- no un bundle inyectado."""
    models_dir = tmp_path / "models"
    card_path = tmp_path / "model_card_1.0.0.md"
    card_path.write_text(
        build_card(published_world["run_id"], "1.0.0", reports=tmp_path / "reports"),
        encoding="utf-8",
    )

    status = publish.publish(
        published_world["run_id"],
        "1.0.0",
        minio,
        bucket,
        models_dir / "registry.json",
        card_path,
        tmp_path / "work",
    )
    assert status == 0

    settings = mcp_settings(
        monkeypatch, tmp_path / "dataset", tmp_path / "reports2", tmp_path / "derived"
    )
    app = create_app(
        settings, load_model=lambda: load_active_model(models_dir=models_dir, s3_client=minio)
    )
    client = TestClient(app)

    image_bytes = tmp_path / "cat.jpg"
    Image.new("RGB", (32, 32), color=(190, 70, 70)).save(image_bytes, format="JPEG")
    response = client.post(
        "/predict", files={"image": ("cat.jpg", image_bytes.read_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1.0.0"
    assert body["checkpoint_sha256"] == published_world["checkpoint_sha256"]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
