"""Comprobaciones del smoke test de reproducibilidad (P3-10, issue #17).

No entrena: el entrenamiento lo hace `ml_worker.run_training.run_training`, el
mismo camino que usa el worker real (tanto la prueba automatizada como las
corridas `run_kind=smoke` lanzadas desde Training). Aquí solo se *comparan*
corridas ya registradas en MLflow y se comprueba que `checkpoint/best.pt`
reproduce predicciones en un proceso nuevo.

Uso contra un MLflow real (después de lanzar dos corridas smoke desde Training):

    uv run python -m training.smoke verify --tracking-uri http://localhost:5050 \\
        --run-a RUN_ID_1 --run-b RUN_ID_2 \\
        --manifest <manifest.csv del release> --crops <raíz de recortes>
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import mlflow
import torch
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model

# Sección 8 de docs/decisiones-proyecto3.md: las cinco métricas por época.
EPOCH_METRICS = ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1")
FINAL_METRICS = ("best_epoch", "stopped_epoch")
_SPAWN_TIMEOUT_S = 300


def _history(client: MlflowClient, run_id: str, metric: str) -> list[float]:
    history = client.get_metric_history(run_id, metric)
    return [m.value for m in sorted(history, key=lambda m: m.step)]


def compare_runs(
    client: MlflowClient, run_a: str, run_b: str, tolerance: float = 1e-6
) -> list[str]:
    """Diferencias entre dos corridas; lista vacía si coinciden dentro de `tolerance`.

    Compara las métricas por época paso a paso y `best_epoch`/`stopped_epoch`
    (exactos). `duration_s` no se compara: depende de la máquina, no del modelo.
    """
    differences = []
    for metric in EPOCH_METRICS:
        a, b = _history(client, run_a, metric), _history(client, run_b, metric)
        if len(a) != len(b):
            differences.append(f"{metric}: {len(a)} épocas vs {len(b)}")
            continue
        for step, (x, y) in enumerate(zip(a, b, strict=True), start=1):
            if abs(x - y) > tolerance:
                differences.append(f"{metric} época {step}: {x!r} vs {y!r}")
    metrics_a = client.get_run(run_a).data.metrics
    metrics_b = client.get_run(run_b).data.metrics
    for metric in FINAL_METRICS:
        if metrics_a.get(metric) != metrics_b.get(metric):
            differences.append(f"{metric}: {metrics_a.get(metric)} vs {metrics_b.get(metric)}")
    return differences


def verify_checkpoint_sha256(path: str | Path, expected: str) -> str:
    """SHA-256 de `path`; falla si no coincide con el tag `checkpoint_sha256`."""
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"checkpoint_sha256 no coincide: tag={expected} archivo={actual}")
    return actual


def config_from_run_params(params: dict[str, str]) -> dict:
    """Reconstruye la config de una corrida desde sus parámetros de MLflow (texto)."""
    config = {}
    for name in TrainingConfig.model_fields:
        raw = params[name]
        try:
            config[name] = json.loads(raw)
        except json.JSONDecodeError:
            config[name] = raw
    return config


def predict_from_run(
    tracking_uri: str,
    run_id: str,
    config: dict,
    manifest_path: str | Path,
    crops_root: str | Path,
    split: str = "val",
) -> dict:
    """Descarga `checkpoint/best.pt`, verifica su SHA-256, lo carga y predice.

    Devuelve `pid`, `checkpoint_sha256`, `crop_ids` y `probabilities`
    (softmax, aplanadas fila por fila: 2 por recorte, en el orden de `crop_ids`).
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    expected = client.get_run(run_id).data.tags["checkpoint_sha256"]
    checkpoint = download_artifacts(
        run_id=run_id, artifact_path="checkpoint/best.pt", tracking_uri=tracking_uri
    )
    sha = verify_checkpoint_sha256(checkpoint, expected)

    cfg = TrainingConfig.model_validate(config)
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    loader = create_dataloader(
        str(manifest_path),
        str(crops_root),
        split,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        seed_train=cfg.seed_train,
        seed_aug=cfg.seed_aug,
        num_workers=0,
    )
    crop_ids: list[str] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for inputs, _targets, ids in loader:
            probs = torch.softmax(model(inputs), dim=1)
            crop_ids.extend(str(i) for i in ids)
            probabilities.extend(probs.flatten().tolist())
    return {
        "pid": os.getpid(),
        "checkpoint_sha256": sha,
        "crop_ids": crop_ids,
        "probabilities": probabilities,
    }


def _predict_worker(queue, args: tuple) -> None:
    try:
        queue.put({"ok": predict_from_run(*args)})
    except Exception as error:  # el padre la relanza con el texto
        queue.put({"error": f"{type(error).__name__}: {error}"})


def predict_in_new_process(
    tracking_uri: str,
    run_id: str,
    config: dict,
    manifest_path: str | Path,
    crops_root: str | Path,
    split: str = "val",
) -> dict:
    """`predict_from_run` en un intérprete nuevo (`spawn`: nada heredado del padre)."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    args = (tracking_uri, run_id, config, str(manifest_path), str(crops_root), split)
    process = ctx.Process(target=_predict_worker, args=(queue, args))
    process.start()
    try:
        message = queue.get(timeout=_SPAWN_TIMEOUT_S)
    finally:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
    if "error" in message:
        raise RuntimeError(f"El proceso nuevo falló: {message['error']}")
    return message["ok"]


def _verify(args: argparse.Namespace) -> int:
    client = MlflowClient(tracking_uri=args.tracking_uri)
    mlflow.set_tracking_uri(args.tracking_uri)
    runs = {name: client.get_run(rid) for name, rid in (("A", args.run_a), ("B", args.run_b))}
    for name, run in runs.items():
        tags, metrics = run.data.tags, run.data.metrics
        print(
            f"Corrida {name}: {run.info.run_id} estado={run.info.status} "
            f"run_kind={tags.get('run_kind')} commit={tags.get('git_commit')} "
            f"dirty={tags.get('git_dirty')} manifest_sha256={tags.get('manifest_sha256')} "
            f"best_epoch={metrics.get('best_epoch')} duration_s={metrics.get('duration_s')}"
        )

    failures = []
    differences = compare_runs(client, args.run_a, args.run_b, args.tolerance)
    print(f"Métricas por época y best_epoch: {'IGUALES' if not differences else 'DIFIEREN'}")
    failures += differences

    config = config_from_run_params(runs["A"].data.params)
    here = predict_from_run(args.tracking_uri, args.run_a, config, args.manifest, args.crops)
    there = predict_in_new_process(args.tracking_uri, args.run_a, config, args.manifest, args.crops)
    expected_sha = runs["A"].data.tags["checkpoint_sha256"]
    print(f"checkpoint_sha256 del tag: {expected_sha}")
    print(f"checkpoint_sha256 recalculado en el proceso nuevo (pid {there['pid']}): ", end="")
    print(there["checkpoint_sha256"])
    if there["checkpoint_sha256"] != expected_sha:
        failures.append("checkpoint_sha256 del proceso nuevo no coincide con el tag")
    if there["crop_ids"] != here["crop_ids"]:
        failures.append("los recortes de val difieren entre procesos")
    worst = max(
        (abs(x - y) for x, y in zip(here["probabilities"], there["probabilities"], strict=True)),
        default=0.0,
    )
    print(f"Recortes de val: {len(there['crop_ids'])}; máx |Δ probabilidad| = {worst:.3e}")
    if worst > args.tolerance:
        failures.append(f"probabilidades difieren {worst:.3e} > {args.tolerance}")

    for failure in failures:
        print(f"FALLO: {failure}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="compara dos corridas smoke y el checkpoint de A")
    verify.add_argument("--tracking-uri", required=True)
    verify.add_argument("--run-a", required=True)
    verify.add_argument("--run-b", required=True)
    verify.add_argument("--manifest", required=True, help="manifest.csv del release")
    verify.add_argument("--crops", required=True, help="raíz de los recortes")
    verify.add_argument("--tolerance", type=float, default=1e-6)
    return _verify(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
