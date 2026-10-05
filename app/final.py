"""Evaluación final única del candidato (P3-13, #20).

Pasa por el candado (`selection.lock`), verifica el `checkpoint_sha256`, usa
`training/preprocess.py` y escribe `predictions.csv`, `metrics.json` y
`analysis.json`. Con `--split test` verifica que `now > selected_at` **antes** de
crear nada, se niega a repetirse (si ya hay `metrics.json` o si la corrida del
candidato ya tiene alguna métrica `test_*`), y luego registra las métricas
`test_*` **dentro de la misma corrida del candidato** (no crea una corrida
aparte: el profe pidió que las métricas de test vivan en r02, no en
`evaluacion-final`); con `--split validation` ensaya (exento del candado) y
escribe en `reports/evaluation/validation/`.

Uso (desde la raíz del repo, con el venv de `app/`):

    app/.venv/bin/python final.py --split validation --run-id <candidato>
    app/.venv/bin/python final.py --split test   # solo Karen (custodia), una vez
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from evaluation.analysis import analyze
from evaluation.contracts import CLASSES, Prediction
from evaluation.metrics import evaluate
from selection.contracts import SelectionLocked
from selection.lock import load_selection, require_closed
from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model
from training.smoke import config_from_run_params, verify_checkpoint_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT = "test"
# El CLI usa "validation" (semántica del ticket), pero en manifest.csv la
# partición se llama "val". Nunca se mapea a "test" salvo con --split test.
MANIFEST_SPLITS = {"validation": "val", "test": "test"}


def _source_images(manifest_csv: Path, split: str) -> dict[str, str]:
    frame = pd.read_csv(manifest_csv)
    rows = frame[frame["split"] == split]
    return {str(row["crop_id"]): str(row["source_image_id"]) for _, row in rows.iterrows()}


def predict_split(model, loader, source_map: dict[str, str]) -> list[Prediction]:
    predictions: list[Prediction] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets, crop_ids in loader:
            probabilities = torch.softmax(model(inputs), dim=1)
            for index, crop_id in enumerate(crop_ids):
                scores = [float(value) for value in probabilities[index].tolist()]
                predicted = max(range(len(scores)), key=scores.__getitem__)
                key = str(crop_id)
                predictions.append(
                    Prediction(
                        crop_id=key,
                        source_image_id=source_map[key],
                        true_label=int(targets[index].item()),
                        predicted_label=predicted,
                        probabilities=scores,
                    )
                )
    return predictions


def write_predictions_csv(predictions: list[Prediction], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["crop_id", "source_image_id", "true_class", "predicted_class", "prob_cat", "prob_dog"]
        )
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.crop_id,
                    prediction.source_image_id,
                    CLASSES[prediction.true_label],
                    CLASSES[prediction.predicted_label],
                    *prediction.probabilities,
                ]
            )


def _log_evaluation_run(*, tracking_uri: str, selection, metrics, analysis, output_dir) -> str:
    """Registra las métricas `test_*` dentro de la corrida del candidato (`r02`),
    no en una corrida aparte (revisión P3-13: el profe pidió que vivan ahí).

    `mlflow.start_run(run_id=...)` **reanuda** esa corrida en vez de crear una
    nueva -- no copia los tags del candidato (r02 ya los tiene todos) y usa
    `datetime.now()` para `evaluation_started_at`, no `run.info.start_time`
    (que sería cuándo empezó el *entrenamiento*, no esta evaluación).
    """
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=selection.run_id) as run:
        started = datetime.now(timezone.utc)
        mlflow.set_tags({"eval_split": TEST_SPLIT, "evaluation_started_at": started.isoformat()})
        mlflow.log_metrics(
            {
                "test_accuracy": metrics.accuracy,
                "test_macro_f1": metrics.macro_f1,
                "test_baseline_accuracy": analysis.baseline_majority_accuracy,
                "test_num_crops": float(metrics.total),
            }
        )
        for name in ("predictions.csv", "metrics.json", "analysis.json"):
            mlflow.log_artifact(str(output_dir / name), artifact_path="evaluation")
        return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["validation", TEST_SPLIT], required=True)
    parser.add_argument("--release", default="v0.1.1")
    parser.add_argument(
        "--run-id", help="Candidato a evaluar (obligatorio en validation sin selection.json)."
    )
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    )
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Ruta local de checkpoint/best.pt; si no, se descarga de MLflow.",
    )
    args = parser.parse_args(argv)

    manifest_csv = args.manifest_csv or (
        REPO_ROOT / "data" / "derived" / "manifests" / args.release / "manifest.csv"
    )
    selection_path = args.selection or (REPO_ROOT / "reports" / "selection.json")
    crops = args.crops or (REPO_ROOT / "data" / "derived" / "crops")
    if args.output_dir:
        output_dir = args.output_dir
    elif args.split == TEST_SPLIT:
        output_dir = REPO_ROOT / "reports" / "evaluation"
    else:
        output_dir = REPO_ROOT / "reports" / "evaluation" / "validation"

    # La evaluación final es única (sección 11): no repetirla.
    if args.split == TEST_SPLIT and (output_dir / "metrics.json").exists():
        print("final: ya existe una evaluación final (metrics.json); es única", file=sys.stderr)
        return 5

    try:
        selection = require_closed(
            split=args.split, selection_path=selection_path, manifest_csv=manifest_csv
        )
    except SelectionLocked as error:
        print(f"final: {error}", file=sys.stderr)
        return 3

    if selection is not None:
        run_id = selection.run_id
        if args.run_id and args.run_id != selection.run_id:
            print("final: --run-id no es el candidato sellado", file=sys.stderr)
            return 3
    elif args.run_id:
        run_id = args.run_id
    elif selection_path.is_file():
        run_id = load_selection(selection_path).run_id
    else:
        print(
            "final: --split validation requiere --run-id (no hay selection.json)", file=sys.stderr
        )
        return 2

    # Cronología: la evaluación es posterior a selected_at. Se comprueba ANTES de
    # crear nada (ni corrida de MLflow ni archivos).
    if selection is not None and datetime.now(timezone.utc) <= selection.selected_at:
        print(
            "final: la evaluación no puede correr antes de selected_at "
            f"({selection.selected_at.isoformat()})",
            file=sys.stderr,
        )
        return 6

    client = MlflowClient(tracking_uri=args.tracking_uri)
    candidate = client.get_run(run_id)
    if args.split == TEST_SPLIT and any(key.startswith("test_") for key in candidate.data.metrics):
        print(f"final: {run_id} ya tiene métricas test_*; la evaluación es única", file=sys.stderr)
        return 5
    config = TrainingConfig.model_validate(config_from_run_params(candidate.data.params))
    expected_sha = candidate.data.tags.get("checkpoint_sha256")
    if selection is not None and selection.checkpoint_sha256 != expected_sha:
        print(
            "final: checkpoint_sha256 del candado no coincide con el de la corrida", file=sys.stderr
        )
        return 3

    if args.checkpoint:
        checkpoint = str(args.checkpoint)
    else:
        checkpoint = download_artifacts(
            run_id=run_id, artifact_path="checkpoint/best.pt", tracking_uri=args.tracking_uri
        )
    verify_checkpoint_sha256(checkpoint, expected_sha)

    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    loader = create_dataloader(
        str(manifest_csv),
        str(crops),
        MANIFEST_SPLITS[args.split],
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed_train=config.seed_train,
        seed_aug=config.seed_aug,
        num_workers=0,
    )
    predictions = predict_split(
        model, loader, _source_images(manifest_csv, MANIFEST_SPLITS[args.split])
    )
    metrics = evaluate(predictions, CLASSES)
    analysis = analyze(predictions, metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(predictions, output_dir / "predictions.csv")
    (output_dir / "metrics.json").write_text(
        metrics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        analysis.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "split": args.split,
        "run_id": run_id,
        "checkpoint_sha256": expected_sha,
        "total": metrics.total,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "output_dir": str(output_dir),
    }
    if args.split == TEST_SPLIT:
        summary["mlflow_evaluation_run_id"] = _log_evaluation_run(
            tracking_uri=args.tracking_uri,
            selection=selection,
            metrics=metrics,
            analysis=analysis,
            output_dir=output_dir,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
