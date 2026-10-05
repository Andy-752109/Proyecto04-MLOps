"""P3-13 (#20): métricas, análisis de errores y candado de la evaluación final.

Las pruebas de métricas/análisis son puras (fixture calculado a mano); la
mayoría de las de `final.py` no entrenan ni tocan MLflow (fallan en el candado
antes). La excepción es la sección "revisión: métricas dentro de r02", que sí
entrena una corrida real y corre `final.py --split test` de verdad -- es la
única forma honesta de probar que las métricas quedan en esa misma corrida y
no en una aparte.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from PIL import Image

from evaluation.analysis import analyze
from evaluation.contracts import CLASSES, Prediction
from evaluation.metrics import evaluate
from final import MANIFEST_SPLITS, _source_images
from final import main as final_main
from ml_api.contracts import TrainingJob
from ml_worker.run_training import run_training
from recompute import sklearn_metrics
from selection.contracts import Candidate, Selection
from selection.lock import test_ids_sha256 as compute_test_ids_sha256
from selection.mlflow_reader import EXPERIMENT
from training.grid import FROZEN_MIN_DELTA, FROZEN_SEEDS, GRID


def a_prediction(
    true_label: int,
    predicted_label: int,
    crop_id: str = "c",
    source_image_id: str = "1",
) -> Prediction:
    probabilities = [0.9, 0.1] if predicted_label == 0 else [0.1, 0.9]
    return Prediction(
        crop_id=crop_id,
        source_image_id=source_image_id,
        true_label=true_label,
        predicted_label=predicted_label,
        probabilities=probabilities,
    )


def fixture_predictions() -> list[Prediction]:
    # Confusión real x predicha: [[3, 1], [2, 4]] (cat=0, dog=1).
    pairs = [(0, 0)] * 3 + [(0, 1)] + [(1, 1)] * 4 + [(1, 0)] * 2
    return [
        a_prediction(true, pred, crop_id=f"c{index:02d}")
        for index, (true, pred) in enumerate(pairs)
    ]


# --- métricas -------------------------------------------------------------------


def test_evaluate_hand_computed_metrics():
    metrics = evaluate(fixture_predictions(), CLASSES)
    assert metrics.confusion_matrix == [[3, 1], [2, 4]]
    assert metrics.total == 10
    assert metrics.accuracy == pytest.approx(0.7)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 8 / 11) / 2)

    cat = metrics.per_class["cat"]
    dog = metrics.per_class["dog"]
    assert cat.precision == pytest.approx(0.6)
    assert cat.recall == pytest.approx(0.75)
    assert cat.support == 4
    assert dog.precision == pytest.approx(0.8)
    assert dog.recall == pytest.approx(4 / 6)
    assert dog.support == 6


def test_changing_a_prediction_changes_the_metrics():
    predictions = fixture_predictions()
    before = evaluate(predictions, CLASSES)
    predictions[0] = a_prediction(0, 1, crop_id="c00")
    after = evaluate(predictions, CLASSES)
    assert after.accuracy != before.accuracy
    assert after.confusion_matrix != before.confusion_matrix


def test_confusion_matrix_sums_to_total():
    metrics = evaluate(fixture_predictions(), CLASSES)
    assert sum(sum(row) for row in metrics.confusion_matrix) == metrics.total


# --- análisis -------------------------------------------------------------------


def test_analysis_reports_baseline_most_confused_and_recall():
    predictions = fixture_predictions()
    metrics = evaluate(predictions, CLASSES)
    analysis = analyze(predictions, metrics)
    assert analysis.baseline_majority_accuracy == pytest.approx(0.6)
    assert analysis.most_confused_class == "dog"
    assert analysis.recall_per_class == pytest.approx({"cat": 0.75, "dog": 4 / 6})
    assert analysis.accuracy_hides_low_recall is False
    assert len(analysis.successes) == 6  # tope de 6
    assert len(analysis.errors) == 3


def test_analysis_flags_a_low_recall_hidden_by_accuracy():
    predictions = [a_prediction(0, 0, crop_id=f"c{i}") for i in range(18)]
    predictions.append(a_prediction(0, 1, crop_id="c18"))
    predictions.append(a_prediction(1, 0, crop_id="c19"))  # dog nunca acierta
    metrics = evaluate(predictions, CLASSES)
    analysis = analyze(predictions, metrics)
    assert metrics.accuracy == pytest.approx(0.9)
    assert metrics.per_class["dog"].recall == 0.0
    assert analysis.accuracy_hides_low_recall is True


# --- recompute (scikit-learn) ---------------------------------------------------


def test_recompute_matches_the_pure_metrics():
    predictions = fixture_predictions()
    metrics = evaluate(predictions, CLASSES)
    recomputed = sklearn_metrics(
        [p.true_label for p in predictions], [p.predicted_label for p in predictions]
    )
    assert recomputed["accuracy"] == pytest.approx(metrics.accuracy)
    assert recomputed["macro_f1"] == pytest.approx(metrics.macro_f1)
    assert recomputed["confusion_matrix"] == metrics.confusion_matrix
    assert recomputed["total"] == metrics.total
    for name in CLASSES:
        assert recomputed["per_class"][name]["recall"] == pytest.approx(
            metrics.per_class[name].recall
        )
        assert recomputed["per_class"][name]["support"] == metrics.per_class[name].support


# --- candado (sin tocar MLflow ni el modelo) ------------------------------------


def test_final_test_split_refuses_without_selection(tmp_path):
    rc = final_main(
        [
            "--split",
            "test",
            "--selection",
            str(tmp_path / "no-existe.json"),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 3
    assert not (tmp_path / "out").exists()


def test_final_validation_requires_run_id_without_selection(tmp_path):
    rc = final_main(
        [
            "--split",
            "validation",
            "--selection",
            str(tmp_path / "no-existe.json"),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out").exists()


def test_validation_split_reads_only_val_never_test(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "crop_id,split,source_image_id\nv1,val,1\nv2,val,2\nt1,test,3\n",
        encoding="utf-8",
    )
    assert MANIFEST_SPLITS["validation"] == "val"
    assert _source_images(manifest, MANIFEST_SPLITS["validation"]) == {"v1": "1", "v2": "2"}
    assert _source_images(manifest, MANIFEST_SPLITS["test"]) == {"t1": "3"}


# --- retro PR #56: "una sola vez" y cronología antes de crear nada --------------


@pytest.fixture
def mlflow_uri(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield uri
    finally:
        mlflow.set_tracking_uri(previous)


def write_selection_and_manifest(tmp_path, *, selected_at, run_id="rid-1"):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "crop_id,split,label,source_image_id,duplicate_group_id\n"
        "000001_000001,test,0,1,g000001\n"
        "000002_000002,train,1,2,g000002\n",
        encoding="utf-8",
    )
    selection = Selection(
        run_id=run_id,
        release="v0.1.1",
        manifest_id="v0.1.1-test",
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        test_ids_sha256=compute_test_ids_sha256(manifest),
        selected_at=selected_at,
        candidate=Candidate(
            grid_row="r02",
            best_val_accuracy=0.9,
            best_val_macro_f1=0.9,
            best_val_loss=0.1,
        ),
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(selection.model_dump_json(), encoding="utf-8")
    return selection_path, manifest


def test_final_test_refuses_when_metrics_already_exist(tmp_path):
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    rc = final_main(
        [
            "--split",
            "test",
            "--selection",
            str(tmp_path / "no-existe.json"),
            "--manifest-csv",
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 5
    assert (output_dir / "metrics.json").read_text(encoding="utf-8") == "{}"


def test_final_test_refuses_before_selected_at(tmp_path):
    selection_path, manifest = write_selection_and_manifest(
        tmp_path, selected_at=datetime.now(timezone.utc) + timedelta(days=1)
    )
    output_dir = tmp_path / "out"
    rc = final_main(
        [
            "--split",
            "test",
            "--selection",
            str(selection_path),
            "--manifest-csv",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 6
    assert not output_dir.exists()


def test_final_test_refuses_when_the_candidate_already_has_test_metrics(tmp_path, mlflow_uri):
    """Reemplaza la guarda por `evaluation_of` (P3-13, revisión): ahora que las
    métricas test_* viven en la propia corrida del candidato, la unicidad se
    revisa ahí directamente -- por eso esta vez el run_id de la selección debe
    ser una corrida real (antes bastaba "rid-1" porque la guarda vieja nunca
    llegaba a pedirle la corrida a MLflow)."""
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        candidate_run_id = run.info.run_id
        mlflow.log_metric("test_accuracy", 0.9)

    selection_path, manifest = write_selection_and_manifest(
        tmp_path,
        selected_at=datetime.now(timezone.utc) - timedelta(days=1),
        run_id=candidate_run_id,
    )
    output_dir = tmp_path / "out"
    rc = final_main(
        [
            "--split",
            "test",
            "--tracking-uri",
            mlflow_uri,
            "--selection",
            str(selection_path),
            "--manifest-csv",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 5
    assert not output_dir.exists()


# --- revisión: métricas dentro de r02, no en una corrida aparte -----------------
#
# A diferencia de las pruebas de arriba, esta sí entrena (patience=1, dataset
# sintético diminuto) y corre `final.py --split test` de verdad: es la única
# forma honesta de confirmar que las métricas test_* terminan en la MISMA
# corrida del candidato, no en una "evaluacion-final" aparte.

CANDIDATE_CONFIG = {
    **GRID["r01"],
    **FROZEN_SEEDS,
    "batch_size": 16,
    "patience": 1,
    "min_delta": FROZEN_MIN_DELTA,
}


def _write_crop(path: Path, label: int) -> None:
    color = (190, 70, 70) if label == 0 else (70, 70, 190)
    Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


def _train_real_candidate(tmp_path, tracking_uri, monkeypatch):
    """Entrena por el mismo camino que el worker real (`ml_worker.run_training
    .run_training`), para tener una corrida real -- no un `RunSummary` de
    fixture -- sobre la que correr `final.py --split test` de verdad."""
    monkeypatch.setenv("GIT_COMMIT", "c" * 40)
    monkeypatch.setenv("GIT_DIRTY", "false")

    crops_root = tmp_path / "crops"
    crops_root.mkdir()
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
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    meta_path = tmp_path / "manifest_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "manifest_id": "vcandidate-fixture",
                "manifest_sha256": "a" * 64,
                "classes": {"0": "cat", "1": "dog"},
                "release": {"name": "vcandidate"},
            }
        ),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())

    job = TrainingJob(
        id="candidate-job",
        status="running",
        progress=0.0,
        config=CANDIDATE_CONFIG,
        dataset_release="vcandidate",
        manifest_id="vcandidate-fixture",
        run_kind="smoke",
        grid_row=None,
        mlflow_run_id=None,
        error=None,
        logs=[],
        heartbeat_at=None,
    )
    run_id = run_training(
        job, manifest_path=manifest_path, crops_root=crops_root, meta_path=meta_path
    )
    return SimpleNamespace(run_id=run_id, manifest_path=manifest_path, crops_root=crops_root)


def test_final_test_run_registers_metrics_inside_the_candidate_run_not_a_new_one(
    tmp_path, mlflow_uri, monkeypatch
):
    trained = _train_real_candidate(tmp_path, mlflow_uri, monkeypatch)
    client = MlflowClient(tracking_uri=mlflow_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert len(client.search_runs([experiment.experiment_id])) == 1  # solo el entrenamiento

    checkpoint_sha256 = client.get_run(trained.run_id).data.tags["checkpoint_sha256"]
    selection = Selection(
        run_id=trained.run_id,
        release="vcandidate",
        manifest_id="vcandidate-fixture",
        manifest_sha256="a" * 64,
        checkpoint_sha256=checkpoint_sha256,
        test_ids_sha256=compute_test_ids_sha256(trained.manifest_path),
        selected_at=datetime.now(timezone.utc) - timedelta(days=1),
        candidate=Candidate(
            grid_row="r01", best_val_accuracy=0.5, best_val_macro_f1=0.5, best_val_loss=0.5
        ),
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(selection.model_dump_json(), encoding="utf-8")

    output_dir = tmp_path / "out"
    args = [
        "--split",
        "test",
        "--tracking-uri",
        mlflow_uri,
        "--selection",
        str(selection_path),
        "--manifest-csv",
        str(trained.manifest_path),
        "--crops",
        str(trained.crops_root),
        "--output-dir",
        str(output_dir),
    ]
    rc = final_main(args)
    assert rc == 0

    runs_after = client.search_runs([experiment.experiment_id])
    assert len(runs_after) == 1  # sigue siendo UNA sola corrida -- no "evaluacion-final" aparte
    assert runs_after[0].info.run_id == trained.run_id

    candidate_run = client.get_run(trained.run_id)
    metrics = candidate_run.data.metrics
    assert {"test_accuracy", "test_macro_f1", "test_baseline_accuracy", "test_num_crops"} <= set(
        metrics
    )
    assert "test_duration_crops" not in metrics  # renombrada a test_num_crops, no duplicada
    assert candidate_run.data.tags["eval_split"] == "test"
    assert "evaluation_started_at" in candidate_run.data.tags
    assert "evaluation_of" not in candidate_run.data.tags  # ya no aplica (P3-13, revisión)

    artifact_paths = {f.path for f in client.list_artifacts(trained.run_id, "evaluation")}
    assert artifact_paths == {
        "evaluation/predictions.csv",
        "evaluation/metrics.json",
        "evaluation/analysis.json",
    }

    # Segunda ejecución (otro --output-dir, para no tropezar con la guarda de
    # metrics.json en disco y probar específicamente la de test_* en MLflow).
    rc_again = final_main([*args[:-1], str(tmp_path / "out2")])
    assert rc_again == 5
    assert not (tmp_path / "out2").exists()
