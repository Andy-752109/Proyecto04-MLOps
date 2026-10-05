"""Pruebas de publicación de modelos (P3-14) contra MinIO con datos simulados.

Los datos son falsos y viven en `tmp_path`: un `mlflow.sql` con el mismo formato de INSERT
multilínea del snapshot real, `best.pt` con bytes cualquiera (nunca se cargan) y reportes
mínimos. No dependen de `dvc pull` ni tocan el S3 real ni `models/registry.json`.

Las pruebas que suben paquetes necesitan MinIO con versionado (varios discos):

    docker run -d --name minio-p3 -p 9000:9000 `
        -v /data1 -v /data2 -v /data3 -v /data4 `
        minio/minio server "/data{1...4}"

(en PowerShell; credenciales por defecto minioadmin/minioadmin. Sin los `-v /dataN` MinIO se queda
esperando discos y no responde. Para quitarlo: `docker rm -fv minio-p3`.)

Si MinIO no responde, o no da VersionId, esas pruebas se marcan como skip.
"""

import hashlib
import json
import os
import re
import tarfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("boto3")

import boto3
import publish
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from card import build_card

ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

SMOKE_RUN = "a" * 32
SELECTED_RUN = "b" * 32
RELEASE = "vfixture"
MANIFEST_ID = "vfixture-000000000000"
CLASSES = {"0": "cat", "1": "dog"}
BEST_BYTES = {SMOKE_RUN: b"pesos-smoke", SELECTED_RUN: b"pesos-seleccionados"}

SECTIONS = (
    "## Identificación",
    "## Propósito",
    "## Clases",
    "## Datos y release",
    "## Split",
    "## Métricas de validación (MLflow)",
    "## Métricas de test",
    "## Hiperparámetros",
    "## Limitaciones",
    "## Origen de los pesos",
    "## Modo de carga",
)
EMPTY_INLINE_CODE = re.compile(r"(?<!`)``(?!`)")


def _sql_text(value) -> str:
    """Escapa como el volcado real: comillas simples dobles y `\\"` para las dobles."""
    return str(value).replace("\\", "\\\\").replace("'", "''").replace('"', '\\"')


def _insert(table: str, rows: list[tuple]) -> str:
    lines = []
    for row in rows:
        cells = [f"'{_sql_text(v)}'" if isinstance(v, str) else str(v) for v in row]
        lines.append("(" + ",".join(cells) + ")")
    return f"INSERT INTO {table} VALUES\n" + ",\n".join(lines) + ";\n"


def _run_rows(run_id: str, kind: str, name: str, drop_tag: str | None):
    best = BEST_BYTES[run_id]
    tags = {
        "release": RELEASE,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": "d" * 64,
        "checkpoint_sha256": hashlib.sha256(best).hexdigest(),
        "run_kind": kind,
        "classes": json.dumps(CLASSES),
        "mlflow.runName": name,
        "device": "cpu",
        "platform": "Linux-fixture",
        "python_version": "3.12.0",
        "torch_version": "2.0.0",
        "torchvision_version": "0.15.0",
        "git_commit": "e" * 40,
        "git_dirty": "false",
    }
    if drop_tag:
        tags.pop(drop_tag)
    params = {"image_size": 64, "hidden_layers": 256, "dropout": 0.2, "batch_size": 16}
    metrics = {
        "best_val_accuracy": 0.9,
        "best_val_macro_f1": 0.88,
        "best_val_loss": 0.31,
        "best_epoch": 3,
        "stopped_epoch": 5,
    }
    return (
        [(k, str(v), run_id) for k, v in params.items()],
        [(k, v, run_id) for k, v in tags.items()],
        [(k, float(v), 1, 0, 0, run_id) for k, v in metrics.items()],
    )


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _build_world(root: Path, drop_tag: str | None = None) -> SimpleNamespace:
    """Snapshot de MLflow y reportes falsos: un run smoke, uno seleccionado y uno ajeno."""
    snapshot, reports = root / "mlflow-store", root / "reports"
    sql = ""
    for run_id, kind, name in (
        (SMOKE_RUN, "smoke", "smoke-fixture"),
        (SELECTED_RUN, "grid", "r02-fixture"),
    ):
        params, tags, metrics = _run_rows(run_id, kind, name, drop_tag)
        sql += _insert("params", params) + _insert("tags", tags)
        sql += _insert("latest_metrics", metrics)
        art = snapshot / "artifacts" / "1" / run_id / "artifacts"
        _write(art / "checkpoint" / "best.pt", BEST_BYTES[run_id])
        _write(art / "curves.csv", "epoch,val_loss\n1,0.5\n")
    _write(snapshot / "mlflow.sql", sql)

    def split(crops: int, originals: int) -> dict:
        half = crops // 2
        return {
            "crops": crops,
            "originals": originals,
            "classes": {"cat": {"crops": half}, "dog": {"crops": crops - half}},
        }

    _write(
        reports / "manifests" / RELEASE / "counts.json",
        json.dumps(
            {
                "release": RELEASE,
                "totals": {
                    "crops": 100,
                    "originals": 60,
                    "classes": {"cat": {"crops": 50}, "dog": {"crops": 50}},
                },
                "splits": {
                    "train": split(70, 40),
                    "val": split(20, 12),
                    "test": split(10, 8),
                },
            }
        ),
    )
    _write(
        reports / "selection.json",
        json.dumps(
            {
                "run_id": SELECTED_RUN,
                "checkpoint_sha256": hashlib.sha256(BEST_BYTES[SELECTED_RUN]).hexdigest(),
                "manifest_id": MANIFEST_ID,
                "selection_metric": "best_val_accuracy",
                "selected_at": "2026-01-01T00:00:00Z",
                "candidate": {"grid_row": "r02"},
            }
        ),
    )
    evaluation = snapshot / "artifacts" / "1" / SELECTED_RUN / "artifacts" / "evaluation"
    _write(
        evaluation / "metrics.json",
        json.dumps(
            {
                "total": 10,
                "accuracy": 0.9,
                "macro_f1": 0.89,
                "classes": ["cat", "dog"],
                "per_class": {
                    "cat": {"precision": 0.8, "recall": 1.0, "f1": 0.89, "support": 4},
                    "dog": {"precision": 1.0, "recall": 0.83, "f1": 0.91, "support": 6},
                },
                "confusion_matrix": [[4, 0], [1, 5]],
            }
        ),
    )
    _write(
        evaluation / "analysis.json",
        json.dumps(
            {
                "most_confused_class": "dog",
                "recall_per_class": {"cat": 1.0, "dog": 0.83},
                "baseline_majority_accuracy": 0.6,
                "accuracy_hides_low_recall": False,
                "errors": [
                    {
                        "crop_id": "test-1-0",
                        "true_class": "dog",
                        "predicted_class": "cat",
                        "probability": 0.62,
                    }
                ],
            }
        ),
    )
    return SimpleNamespace(root=root, reports=reports, snapshot=snapshot)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_SNAPSHOT", str(tmp_path / "mlflow-store"))
    return _build_world(tmp_path)


def _make_card(world, run_id: str, version: str) -> Path:
    path = world.root / "cards" / f"model_card_{version}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(build_card(run_id, version, reports=world.reports), encoding="utf-8")
    return path


# --- Tarjeta (sin MinIO) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("run_id", "version", "label"),
    [(SMOKE_RUN, "0.1.0", "Ensayo, no seleccionada"), (SELECTED_RUN, "1.0.0", "Seleccionada")],
)
def test_card_has_every_section_and_no_empty_fields(world, run_id, version, label):
    card = build_card(run_id, version, reports=world.reports)

    for section in SECTIONS:
        assert section in card, section
    assert label in card
    assert run_id in card
    # La tarjeta de un run no seleccionado nombra cuál es el seleccionado.
    assert f"el run `{SELECTED_RUN}`" in card or run_id == SELECTED_RUN
    assert "None" not in card
    assert "N/A" not in card
    assert not EMPTY_INLINE_CODE.search(card)


def test_selected_card_has_test_metrics_per_class_and_smoke_card_does_not(world):
    selected = build_card(SELECTED_RUN, "1.0.0", reports=world.reports)
    smoke = build_card(SMOKE_RUN, "0.1.0", reports=world.reports)

    assert "| cat | 0.8000 | 1.0000 | 0.8900 | 4 |" in selected
    assert "| dog | 1.0000 | 0.8300 | 0.9100 | 6 |" in selected
    assert "No evaluada en test" in smoke
    assert "Precision" not in smoke


@pytest.mark.parametrize(
    "tag", ["release", "manifest_id", "checkpoint_sha256", "run_kind", "classes", "git_commit"]
)
def test_card_fails_when_a_run_tag_is_missing(tmp_path, monkeypatch, tag):
    monkeypatch.setenv("MLFLOW_SNAPSHOT", str(tmp_path / "mlflow-store"))
    broken = _build_world(tmp_path, drop_tag=tag)

    with pytest.raises(SystemExit) as error:
        build_card(SMOKE_RUN, "0.1.0", reports=broken.reports)
    assert tag in str(error.value) or "vacíos" in str(error.value)


def test_card_fails_when_a_1x_version_is_not_the_selected_run(world):
    with pytest.raises(SystemExit, match="seleccionado"):
        build_card(SMOKE_RUN, "1.0.0", reports=world.reports)


# --- Paquete (sin MinIO) ---------------------------------------------------------------


def test_package_fails_when_best_checkpoint_is_missing(world, tmp_path):
    card = _make_card(world, SELECTED_RUN, "1.0.0")
    art = world.snapshot / "artifacts" / "1" / SELECTED_RUN / "artifacts"
    (art / "checkpoint" / "best.pt").unlink()

    run = publish.load_run(SELECTED_RUN)

    with pytest.raises(SystemExit, match=r"best\.pt"):
        publish.build_package(run, "1.0.0", card, tmp_path / "work")


# --- MinIO -----------------------------------------------------------------------------


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
    name = f"p3-test-{uuid.uuid4().hex[:12]}"
    minio.create_bucket(Bucket=name)
    try:
        minio.put_bucket_versioning(Bucket=name, VersioningConfiguration={"Status": "Enabled"})
        if minio.get_bucket_versioning(Bucket=name).get("Status") != "Enabled":
            pytest.skip("MinIO sin versionado (necesita varios discos: server /data{1...4})")
        yield name
    finally:
        _empty_bucket(minio, name)
        minio.delete_bucket(Bucket=name)


def _publish(world, minio, bucket, run_id: str, version: str) -> tuple[int, Path]:
    registry = world.root / "registry.json"
    status = publish.publish(
        run_id,
        version,
        minio,
        bucket,
        registry,
        _make_card(world, run_id, version),
        world.root / "work",
    )
    return status, registry


def _download(minio, bucket: str, entry: dict, dest: Path) -> Path:
    key = entry["s3_path"].removeprefix(f"s3://{bucket}/")
    body = minio.get_object(Bucket=bucket, Key=key, VersionId=entry["VersionId"])["Body"].read()
    dest.write_bytes(body)
    return dest


def test_published_package_is_complete_and_its_file_hashes_match(world, minio, bucket, tmp_path):
    status, registry = _publish(world, minio, bucket, SELECTED_RUN, "1.0.0")
    assert status == 0
    entry = json.loads(registry.read_text(encoding="utf-8"))["1.0.0"]
    assert entry["VersionId"] not in ("", "null")

    tar_path = _download(minio, bucket, entry, tmp_path / "paquete.tar.gz")
    extracted = tmp_path / "extraido"
    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
        tar.extractall(extracted, filter="data")

    assert set(publish.REQUIRED) <= names
    package = json.loads((extracted / "package.json").read_text(encoding="utf-8"))
    assert package["version"] == "1.0.0"
    assert package["run_id"] == SELECTED_RUN
    assert package["files"], "package.json no lista archivos"
    for name, expected in package["files"].items():
        assert publish.sha256(extracted / name) == expected, name
    assert package["checkpoint_sha256"] == publish.sha256(
        extracted / "artifacts" / "checkpoint" / "best.pt"
    )


def test_downloaded_package_hash_matches_the_registry(world, minio, bucket, tmp_path):
    _, registry = _publish(world, minio, bucket, SMOKE_RUN, "0.1.0")
    entry = json.loads(registry.read_text(encoding="utf-8"))["0.1.0"]

    downloaded = _download(minio, bucket, entry, tmp_path / "paquete.tar.gz")

    assert publish.sha256(downloaded) == entry["sha256"]
    assert entry["run_id"] == SMOKE_RUN
    assert entry["checkpoint_sha256"] == hashlib.sha256(BEST_BYTES[SMOKE_RUN]).hexdigest()


def test_publishing_an_existing_version_fails_and_changes_nothing(world, minio, bucket):
    first, registry = _publish(world, minio, bucket, SELECTED_RUN, "1.0.0")
    assert first == 0
    registry_before = registry.read_bytes()
    key = "models/releases/v1.0.0/model_release_v1.0.0.tar.gz"
    before = minio.list_object_versions(Bucket=bucket, Prefix=key)["Versions"]

    second, _ = _publish(world, minio, bucket, SELECTED_RUN, "1.0.0")

    after = minio.list_object_versions(Bucket=bucket, Prefix=key)
    assert second == 1
    assert len(before) == 1
    assert [v["VersionId"] for v in after["Versions"]] == [before[0]["VersionId"]]
    assert not after.get("DeleteMarkers")
    assert registry.read_bytes() == registry_before


def test_two_versions_stay_recoverable_and_differ(world, minio, bucket, tmp_path):
    _publish(world, minio, bucket, SMOKE_RUN, "0.1.0")
    _, registry = _publish(world, minio, bucket, SELECTED_RUN, "1.0.0")
    entries = json.loads(registry.read_text(encoding="utf-8"))

    old = _download(minio, bucket, entries["0.1.0"], tmp_path / "v010.tar.gz")
    new = _download(minio, bucket, entries["1.0.0"], tmp_path / "v100.tar.gz")

    assert publish.sha256(old) == entries["0.1.0"]["sha256"]
    assert publish.sha256(new) == entries["1.0.0"]["sha256"]
    assert entries["0.1.0"]["sha256"] != entries["1.0.0"]["sha256"]
    assert entries["0.1.0"]["VersionId"] != entries["1.0.0"]["VersionId"]
