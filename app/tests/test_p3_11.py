"""P3-11 (#18): validador de corridas, selección por validación y candado.

Las pruebas puras no tocan MLflow; solo `test_read_runs_roundtrip` usa un MLflow
temporal (SQLite en `tmp_path`), igual patrón que `test_p3_08.py`.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pytest

from select_candidate import main as select_main
from selection.contracts import (
    Candidate,
    ExperimentsValidity,
    RunSummary,
    Selection,
    SelectionLocked,
)
from selection.lock import (
    load_selection,
    lock_reason,
    require_closed,
    sha256_file,
)
from selection.lock import (
    test_ids_sha256 as compute_test_ids_sha256,
)
from selection.mlflow_reader import EXPERIMENT, read_runs
from selection.select import rank_runs, select_candidate
from selection.validate import REQUIRED_ARTIFACTS, validate_runs

MANIFEST_SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64
CLASSES = {"0": "cat", "1": "dog"}

# Semillas, patience y min_delta congelados (training.grid); toda corrida válida
# los trae con estos valores.
FROZEN_PARAMS = {
    "seed_split": "42",
    "seed_train": "43",
    "seed_aug": "44",
    "seed_model": "45",
    "patience": "3",
    "min_delta": "0.001",
}

GRID_R01 = {
    "optimizer": "adam",
    "batch_size": "32",
    "max_epochs": "15",
    "learning_rate": "0.001",
    "image_size": "128",
    "hidden_layers": "0",
    "dropout": "0.0",
}
GRID_R02 = {
    "optimizer": "sgd",
    "batch_size": "16",
    "max_epochs": "30",
    "learning_rate": "0.01",
    "image_size": "160",
    "hidden_layers": "1",
    "dropout": "0.3",
    "momentum": "0.9",
}


def a_run(
    run_id: str = "run-1",
    grid_row: str = "r01",
    *,
    status: str = "FINISHED",
    stopped_epoch: float = 5.0,
    git_dirty: str = "false",
    manifest_sha256: str = MANIFEST_SHA,
    classes: dict | None = None,
    params: dict | None = None,
    extra_metrics: dict | None = None,
    tags: dict | None = None,
    artifacts: list[str] | None = None,
    drop_params: tuple[str, ...] = (),
    drop_tags: tuple[str, ...] = (),
    drop_metrics: tuple[str, ...] = (),
) -> RunSummary:
    base_params = {
        **FROZEN_PARAMS,
        "optimizer": "adam",
        "batch_size": "32",
        "max_epochs": "15",
        "learning_rate": "0.001",
        "image_size": "128",
        "hidden_layers": "0",
        "dropout": "0.0",
    }
    base_params.update(params or {})
    metrics = {
        "best_val_accuracy": 0.9,
        "best_val_macro_f1": 0.9,
        "best_val_loss": 0.2,
        "best_epoch": 4.0,
        "stopped_epoch": stopped_epoch,
        "duration_s": 12.0,
    }
    metrics.update(extra_metrics or {})
    base_tags = {
        "grid_row": grid_row,
        "run_kind": "campaign",
        "git_commit": "e082d9c8e06e0d243a1ab2f839857f5cb2a2cd0a",
        "release": "v0.1.1",
        "manifest_id": "v0.1.1-test",
        "manifest_sha256": manifest_sha256,
        "classes": json.dumps(classes or CLASSES),
        "git_dirty": git_dirty,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "python_version": "3.12.0",
        "torch_version": "2.14.0",
        "torchvision_version": "0.29.0",
        "platform": "macOS-15",
        "device": "cpu",
    }
    base_tags.update(tags or {})
    for key in drop_params:
        base_params.pop(key, None)
    for key in drop_tags:
        base_tags.pop(key, None)
    for key in drop_metrics:
        metrics.pop(key, None)
    return RunSummary(
        run_id=run_id,
        status=status,
        params=base_params,
        tags=base_tags,
        metrics=metrics,
        artifacts=artifacts if artifacts is not None else list(REQUIRED_ARTIFACTS),
    )


# --- validación -----------------------------------------------------------------


def test_validate_accepts_a_valid_run():
    result = validate_runs([a_run()], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert result.valid == ["run-1"]
    assert result.invalid == []


def test_validate_rejects_test_metrics_without_selected_candidate():
    run = a_run(extra_metrics={"test_accuracy": 0.99})
    result = validate_runs([run], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert result.valid == []
    assert any("test_" in reason for reason in result.invalid[0].reasons)


def test_validate_accepts_test_metrics_on_the_already_selected_candidate():
    """P3-13, revisión: una vez cerrada la selección, final.py registra test_*
    en la propia corrida del candidato -- validate_runs no debe rechazarla por
    eso si se vuelve a correr después."""
    run = a_run(
        extra_metrics={"test_accuracy": 0.99},
        tags={"selected_candidate": "true", "selected_at": "2026-10-02T00:00:00+00:00"},
    )
    result = validate_runs([run], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert result.valid == ["run-1"]
    assert result.invalid == []


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(a_run(status="FAILED"), id="no-finish"),
        pytest.param(a_run(manifest_sha256="c" * 64), id="otro-manifiesto"),
        pytest.param(a_run(classes={"0": "dog", "1": "cat"}), id="otras-clases"),
        pytest.param(a_run(stopped_epoch=1.0), id="una-epoca"),
        pytest.param(a_run(git_dirty="true"), id="dirty"),
    ],
)
def test_validate_rejects_each_reason(run):
    result = validate_runs([run], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert not result.passed
    assert result.valid == []
    assert result.invalid[0].reasons


def test_validate_rejects_duplicated_grid_params():
    runs = [a_run("run-1", "r01"), a_run("run-2", "r02")]  # mismos params de rejilla
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert sorted(result.valid) == []
    assert all("duplicad" in reason for entry in result.invalid for reason in entry.reasons)


def test_validate_requires_minimum_valid_runs():
    runs = [a_run("run-1", "r01")]
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=10)
    assert not result.passed
    assert result.valid == ["run-1"]


def test_validate_requires_two_values_per_param():
    runs = [a_run("run-1", "r01", params={"optimizer": "adam"})]
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert not result.passed
    assert result.per_param_values["optimizer"] == 1


def test_validate_passes_full_grid():
    runs = [
        a_run("run-r01", "r01", params=GRID_R01),
        a_run("run-r02", "r02", params=GRID_R02),
    ]
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=2)
    assert result.passed
    assert all(count == 2 for count in result.per_param_values.values())


# --- selección ------------------------------------------------------------------


def test_select_rejects_any_test_metric():
    """Sin selected_candidate/selected_at: test_* sigue prohibida, antes y después
    del candado (P3-13, revisión)."""
    run = a_run(extra_metrics={"test_accuracy": 0.99})
    with pytest.raises(ValueError, match="test_"):
        rank_runs([run])


def test_select_accepts_test_metrics_on_the_already_selected_candidate():
    run = a_run(
        extra_metrics={"test_accuracy": 0.99},
        tags={"selected_candidate": "true", "selected_at": "2026-10-02T00:00:00+00:00"},
    )
    assert rank_runs([run]) == [run]


def test_select_orders_by_accuracy_then_tiebreakers():
    low = a_run("low", "r01", extra_metrics={"best_val_accuracy": 0.8})
    mid = a_run(
        "mid",
        "r02",
        extra_metrics={"best_val_accuracy": 0.9, "best_val_macro_f1": 0.5, "best_val_loss": 0.1},
    )
    high = a_run(
        "high",
        "r03",
        extra_metrics={"best_val_accuracy": 0.9, "best_val_macro_f1": 0.9, "best_val_loss": 0.3},
    )
    ranked = rank_runs([low, mid, high])
    assert [run.run_id for run in ranked] == ["high", "mid", "low"]
    assert select_candidate([low, high, mid]).run_id == "high"


# --- candado --------------------------------------------------------------------


def write_manifest(path: Path, test_ids: list[str]) -> None:
    lines = ["crop_id,split,label,source_image_id,duplicate_group_id"]
    for index, crop_id in enumerate(test_ids):
        lines.append(f"{crop_id},test,0,{index},g{index:06d}")
    lines.append("000001_000001,train,0,99,g000099")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def a_selection(test_ids_sha256: str, release: str = "v0.1.1") -> Selection:
    return Selection(
        run_id="run-1",
        release=release,
        manifest_id="v0.1.1-test",
        manifest_sha256=MANIFEST_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        test_ids_sha256=test_ids_sha256,
        selected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        candidate=Candidate(
            grid_row="r02",
            best_val_accuracy=0.96,
            best_val_macro_f1=0.96,
            best_val_loss=0.12,
        ),
    )


def test_test_ids_sha256_is_deterministic(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["b", "a"])
    first = compute_test_ids_sha256(manifest)
    assert first == compute_test_ids_sha256(manifest)
    assert len(first) == 64


def test_require_closed_refuses_without_selection(tmp_path):
    selection_path = tmp_path / "selection.json"
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    with pytest.raises(SelectionLocked):
        require_closed(split="test", selection_path=selection_path, manifest_csv=manifest)


def test_require_closed_refuses_on_hash_mismatch(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(a_selection("0" * 64).model_dump_json(), encoding="utf-8")
    with pytest.raises(SelectionLocked, match="hash del test"):
        require_closed(split="test", selection_path=selection_path, manifest_csv=manifest)


def test_require_closed_accepts_a_matching_selection(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a", "b"])
    selection = a_selection(compute_test_ids_sha256(manifest))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(selection.model_dump_json(), encoding="utf-8")
    assert (
        require_closed(split="test", selection_path=selection_path, manifest_csv=manifest)
        == selection
    )


def test_require_closed_exempts_validation_split(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    assert (
        require_closed(
            split="validation", selection_path=tmp_path / "nope.json", manifest_csv=manifest
        )
        is None
    )


def test_lock_reason_reports_not_configured_and_open(tmp_path):
    manifest = tmp_path / "derived" / "manifests" / "v0.1.1" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    write_manifest(manifest, ["a"])
    selection_path = tmp_path / "reports" / "selection.json"
    assert lock_reason(selection_path=selection_path, derived_dir=tmp_path / "derived")

    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        a_selection(compute_test_ids_sha256(manifest)).model_dump_json(), encoding="utf-8"
    )
    assert lock_reason(selection_path=selection_path, derived_dir=tmp_path / "derived") is None


# --- lectura de MLflow ----------------------------------------------------------


def test_read_runs_roundtrip(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("clasificador-perro-gato")
        with mlflow.start_run():
            mlflow.log_params({"optimizer": "adam"})
            mlflow.set_tags(
                {
                    "run_kind": "campaign",
                    "grid_row": "r01",
                    "manifest_sha256": MANIFEST_SHA,
                    "classes": json.dumps(CLASSES),
                    "git_dirty": "false",
                }
            )
            mlflow.log_metric("best_val_accuracy", 0.9)
            mlflow.log_metric("stopped_epoch", 5)

        runs = read_runs(uri)
        assert len(runs) == 1
        assert runs[0].grid_row == "r01"
        assert runs[0].status == "FINISHED"
        assert runs[0].metrics["best_val_accuracy"] == pytest.approx(0.9)
        assert runs[0].classes == CLASSES
    finally:
        mlflow.set_tracking_uri(previous)


def test_read_runs_filters_by_run_kind(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("clasificador-perro-gato")
        with mlflow.start_run():
            mlflow.set_tags({"run_kind": "smoke"})
        with mlflow.start_run():
            mlflow.set_tags({"run_kind": "campaign", "grid_row": "r01"})
        assert len(read_runs(uri, run_kind="campaign")) == 1
        assert len(read_runs(uri, run_kind="smoke")) == 1
    finally:
        mlflow.set_tracking_uri(previous)


def test_experiments_validity_contract_roundtrip():
    result = validate_runs([a_run()], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    again = ExperimentsValidity.model_validate_json(result.model_dump_json())
    assert again == result


def test_load_selection_validates_contract(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    path = tmp_path / "selection.json"
    path.write_text(
        a_selection(compute_test_ids_sha256(manifest)).model_dump_json(), encoding="utf-8"
    )
    loaded = load_selection(path)
    assert loaded.selection_metric == "best_val_accuracy"
    assert loaded.candidate.grid_row == "r02"


# --- retro PR #53: valores congelados y campaign con grid_row -------------------


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"seed_train": "99"}, id="semilla"),
        pytest.param({"patience": "5"}, id="patience"),
        pytest.param({"min_delta": "0.01"}, id="min-delta"),
    ],
)
def test_validate_rejects_unfrozen_hyperparameters(bad):
    result = validate_runs(
        [a_run(params=bad)], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1
    )
    assert result.valid == []
    assert any("congelad" in reason for entry in result.invalid for reason in entry.reasons)


def test_validate_rejects_campaign_without_grid_row():
    result = validate_runs(
        [a_run(grid_row="")], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1
    )
    assert result.valid == []
    assert any("grid_row" in reason for entry in result.invalid for reason in entry.reasons)


# --- sección 8: una corrida a la que le falte algo no cuenta como válida ---------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"drop_tags": ["checkpoint_sha256"]}, id="tag-checkpoint"),
        pytest.param({"drop_tags": ["device"]}, id="tag-device"),
        pytest.param({"drop_tags": ["torch_version"]}, id="tag-torch"),
        pytest.param({"drop_tags": ["platform"]}, id="tag-platform"),
        pytest.param({"drop_tags": ["release"]}, id="tag-release"),
        pytest.param({"drop_tags": ["manifest_id"]}, id="tag-manifest-id"),
        pytest.param({"drop_tags": ["git_commit"]}, id="tag-git-commit"),
        pytest.param({"drop_tags": ["python_version"]}, id="tag-python"),
        pytest.param({"drop_params": ["learning_rate"]}, id="param-rejilla"),
        pytest.param({"drop_params": ["seed_aug"]}, id="param-semilla"),
        pytest.param({"drop_params": ["patience"]}, id="param-patience"),
        pytest.param({"drop_params": ["min_delta"]}, id="param-min-delta"),
        pytest.param({"drop_metrics": ["best_val_macro_f1"]}, id="metrica-final"),
        pytest.param({"drop_metrics": ["duration_s"]}, id="metrica-duracion"),
        pytest.param({"artifacts": ["curves.csv"]}, id="artefacto-checkpoint"),
        pytest.param({"artifacts": ["checkpoint/best.pt"]}, id="artefacto-curvas"),
        pytest.param({"extra_metrics": {"test_accuracy": 0.9}}, id="metrica-test"),
    ],
)
def test_validate_rejects_missing_contract_pieces(kwargs):
    result = validate_runs(
        [a_run(**kwargs)], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1
    )
    assert result.valid == []
    assert result.invalid[0].reasons


def test_validate_requires_momentum_for_sgd():
    params = {key: value for key, value in GRID_R02.items() if key != "momentum"}
    result = validate_runs(
        [a_run(params=params)], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1
    )
    assert result.valid == []
    assert any("momentum" in reason for entry in result.invalid for reason in entry.reasons)


# --- retro PR #53: candado de select_candidate.py (cruce de entradas y atomicidad) ---


@pytest.fixture
def mlflow_uri(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield uri
    finally:
        mlflow.set_tracking_uri(previous)


def log_campaign_run(*, grid_row: str, params: dict, manifest_sha256: str) -> str:
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "run_kind": "campaign",
                "grid_row": grid_row,
                "manifest_sha256": manifest_sha256,
                "classes": json.dumps(CLASSES),
                "git_dirty": "false",
                "checkpoint_sha256": CHECKPOINT_SHA,
            }
        )
        mlflow.log_metric("best_val_accuracy", 0.9)
        mlflow.log_metric("best_val_macro_f1", 0.9)
        mlflow.log_metric("best_val_loss", 0.2)
        mlflow.log_metric("stopped_epoch", 5)
        return run.info.run_id


def write_meta(path: Path, *, manifest_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "manifest_id": "v0.1.1-test",
                "manifest_sha256": manifest_sha256,
                "classes": CLASSES,
                "release": {"name": "v0.1.1"},
            }
        ),
        encoding="utf-8",
    )


def write_validity(path: Path, runs: list[RunSummary], *, manifest_sha256: str) -> None:
    validity = validate_runs(
        runs, manifest_sha256=manifest_sha256, classes=CLASSES, min_required=len(runs)
    )
    path.write_text(validity.model_dump_json(), encoding="utf-8")


def select_argv(tracking_uri, *, meta, manifest, validity, out) -> list[str]:
    return [
        "--tracking-uri",
        tracking_uri,
        "--manifest-meta",
        str(meta),
        "--manifest-csv",
        str(manifest),
        "--validity",
        str(validity),
        "--out",
        str(out),
    ]


def test_select_refuses_when_validity_manifest_differs(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a", "b"])
    sha = sha256_file(manifest)
    meta = tmp_path / "meta.json"
    write_meta(meta, manifest_sha256=sha)

    other = "b" * 64
    validity = tmp_path / "validity.json"
    write_validity(
        validity,
        [
            a_run("r1", "r01", manifest_sha256=other, params=GRID_R01),
            a_run("r2", "r02", manifest_sha256=other, params=GRID_R02),
        ],
        manifest_sha256=other,
    )
    out = tmp_path / "selection.json"
    rc = select_main(
        select_argv("sqlite:///unused.db", meta=meta, manifest=manifest, validity=validity, out=out)
    )
    assert rc == 3
    assert not out.exists()


def test_select_refuses_when_csv_is_not_the_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a", "b"])
    sha = sha256_file(manifest)
    meta = tmp_path / "meta.json"
    write_meta(meta, manifest_sha256=sha)
    validity = tmp_path / "validity.json"
    write_validity(
        validity,
        [
            a_run("r1", "r01", manifest_sha256=sha, params=GRID_R01),
            a_run("r2", "r02", manifest_sha256=sha, params=GRID_R02),
        ],
        manifest_sha256=sha,
    )
    wrong_csv = tmp_path / "wrong.csv"
    write_manifest(wrong_csv, ["x", "y", "z"])
    out = tmp_path / "selection.json"
    rc = select_main(
        select_argv(
            "sqlite:///unused.db", meta=meta, manifest=wrong_csv, validity=validity, out=out
        )
    )
    assert rc == 3
    assert not out.exists()


def test_select_writes_file_then_tags_and_is_idempotent(tmp_path, mlflow_uri):
    mlflow.set_experiment(EXPERIMENT)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a", "b"])
    sha = sha256_file(manifest)
    meta = tmp_path / "meta.json"
    write_meta(meta, manifest_sha256=sha)

    first = log_campaign_run(grid_row="r01", params=GRID_R01, manifest_sha256=sha)
    second = log_campaign_run(grid_row="r02", params=GRID_R02, manifest_sha256=sha)
    validity = tmp_path / "validity.json"
    write_validity(
        validity,
        [
            a_run(first, "r01", manifest_sha256=sha, params=GRID_R01),
            a_run(second, "r02", manifest_sha256=sha, params=GRID_R02),
        ],
        manifest_sha256=sha,
    )
    out = tmp_path / "selection.json"
    argv = select_argv(mlflow_uri, meta=meta, manifest=manifest, validity=validity, out=out)

    assert select_main(argv) == 0
    selection = Selection.model_validate_json(out.read_text(encoding="utf-8"))
    assert selection.run_id in {first, second}
    assert mlflow.get_run(selection.run_id).data.tags["selected_candidate"] == "true"

    # Re-ejecutar con la misma selección es idempotente y conserva el archivo.
    before = out.read_text(encoding="utf-8")
    assert select_main(argv) == 0
    assert out.read_text(encoding="utf-8") == before


def test_select_refuses_when_another_run_is_already_selected(tmp_path, mlflow_uri):
    mlflow.set_experiment(EXPERIMENT)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a", "b"])
    sha = sha256_file(manifest)
    meta = tmp_path / "meta.json"
    write_meta(meta, manifest_sha256=sha)

    first = log_campaign_run(grid_row="r01", params=GRID_R01, manifest_sha256=sha)
    second = log_campaign_run(grid_row="r02", params=GRID_R02, manifest_sha256=sha)
    with mlflow.start_run():
        mlflow.set_tag("selected_candidate", "true")  # otra corrida ya seleccionada

    validity = tmp_path / "validity.json"
    write_validity(
        validity,
        [
            a_run(first, "r01", manifest_sha256=sha, params=GRID_R01),
            a_run(second, "r02", manifest_sha256=sha, params=GRID_R02),
        ],
        manifest_sha256=sha,
    )
    out = tmp_path / "selection.json"
    rc = select_main(
        select_argv(mlflow_uri, meta=meta, manifest=manifest, validity=validity, out=out)
    )
    assert rc == 4
    assert not out.exists()


def test_no_root_module_shadows_a_stdlib_module():
    """Un .py en la raíz de app/ no puede llamarse igual que un módulo de la stdlib.

    `select.py` (antes) chocaba con `select` de la stdlib; por eso el CLI se llama
    `select_candidate.py`.
    """
    root = Path(__file__).resolve().parents[1]
    stdlib = set(sys.stdlib_module_names)
    offenders = sorted(path.stem for path in root.glob("*.py") if path.stem in stdlib)
    assert offenders == [], f"Módulos de la raíz que chocan con la stdlib: {offenders}"
