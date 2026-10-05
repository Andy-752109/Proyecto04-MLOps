"""Elige el candidato con la métrica de selección y sella el test (P3-11, #18).

Escribe `reports/selection.json` y etiqueta la corrida ganadora en MLflow
(`selected_candidate`, `selected_at`, `selection_metric`). Solo selecciona si
`reports/experiments_validity.json` pasó.

Varias comprobaciones antes de sellar, porque `test_ids_sha256` (el candado)
depende de que las entradas sean coherentes:

- la validez debe ser del mismo manifiesto que el `manifest_meta` (mismo
  `manifest_sha256` y mismas `classes`);
- el `manifest.csv` recibido debe ser ese manifiesto (su SHA-256 = `manifest_sha256`);
- no debe haber otra corrida ya etiquetada `selected_candidate`.

La escritura de `selection.json` es atómica (`open(..., "x")`) y re-ejecutar con
la misma selección es idempotente (conserva el archivo y re-aplica los tags).

Uso (desde la raíz del repo, con el venv de `app/`):

    app/.venv/bin/python validate_runs.py --release v0.1.1
    app/.venv/bin/python select_candidate.py --release v0.1.1

Después de correrlo, Karen declara **MODEL SELECTION CLOSED**. No toca el
split `test`.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mlflow.tracking import MlflowClient

from selection.contracts import Candidate, ExperimentsValidity, Selection
from selection.lock import sha256_file, test_ids_sha256
from selection.mlflow_reader import EXPERIMENT, read_runs
from selection.select import select_candidate

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_METRIC = "best_val_accuracy"


def _same_selection(existing: Selection, wanted: Selection) -> bool:
    """Igual salvo `selected_at` (para que re-ejecutar sea idempotente)."""
    return (
        existing.run_id == wanted.run_id
        and existing.manifest_sha256 == wanted.manifest_sha256
        and existing.checkpoint_sha256 == wanted.checkpoint_sha256
        and existing.test_ids_sha256 == wanted.test_ids_sha256
    )


def _already_selected_run(client: MlflowClient, winner_run_id: str) -> str | None:
    """Otra corrida ya marcada como seleccionada, si existe."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="tags.selected_candidate = 'true'"
    )
    others = [run.info.run_id for run in runs if run.info.run_id != winner_run_id]
    return others[0] if others else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="v0.1.1", help="Release del manifiesto.")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"),
    )
    parser.add_argument("--manifest-meta", type=Path)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--validity", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    manifest_meta = args.manifest_meta or (
        REPO_ROOT / "reports" / "manifests" / args.release / "manifest_meta.json"
    )
    manifest_csv = args.manifest_csv or (
        REPO_ROOT / "data" / "derived" / "manifests" / args.release / "manifest.csv"
    )
    validity_path = args.validity or (REPO_ROOT / "reports" / "experiments_validity.json")
    out = args.out or (REPO_ROOT / "reports" / "selection.json")

    meta = json.loads(manifest_meta.read_text(encoding="utf-8"))
    validity = ExperimentsValidity.model_validate_json(validity_path.read_text(encoding="utf-8"))
    if not validity.passed:
        print("select: la validación de corridas no pasó; no se selecciona", file=sys.stderr)
        return 1

    # Las entradas deben describir el mismo manifiesto; si no, el candado sellaría
    # un test que no es el que se validó.
    if validity.manifest_sha256 != meta["manifest_sha256"]:
        print(
            "select: la validez no corresponde al manifiesto (manifest_sha256 distinto)",
            file=sys.stderr,
        )
        return 3
    if validity.classes != meta["classes"]:
        print("select: las clases de la validez no coinciden con el manifiesto", file=sys.stderr)
        return 3
    if sha256_file(manifest_csv) != meta["manifest_sha256"]:
        print("select: --manifest-csv no es el CSV de este manifiesto", file=sys.stderr)
        return 3

    runs = read_runs(args.tracking_uri, run_kind="campaign")
    by_id = {run.run_id: run for run in runs}
    missing = [run_id for run_id in validity.valid if run_id not in by_id]
    if missing:
        print(f"select: faltan en MLflow corridas válidas: {missing}", file=sys.stderr)
        return 2

    try:
        winner = select_candidate([by_id[run_id] for run_id in validity.valid])
    except ValueError as error:
        print(f"select: {error}", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    release = meta["release"]["name"] if isinstance(meta.get("release"), dict) else args.release
    wanted = Selection(
        run_id=winner.run_id,
        release=release,
        manifest_id=meta["manifest_id"],
        manifest_sha256=meta["manifest_sha256"],
        checkpoint_sha256=winner.tags["checkpoint_sha256"],
        test_ids_sha256=test_ids_sha256(manifest_csv),
        selected_at=now,
        selection_metric=SELECTION_METRIC,
        candidate=Candidate(
            grid_row=winner.grid_row,
            best_val_accuracy=winner.metrics["best_val_accuracy"],
            best_val_macro_f1=winner.metrics["best_val_macro_f1"],
            best_val_loss=winner.metrics["best_val_loss"],
        ),
    )

    client = MlflowClient(tracking_uri=args.tracking_uri)
    other = _already_selected_run(client, winner.run_id)
    if other is not None:
        print(
            f"select: ya hay otra corrida seleccionada ({other}); no se re-selecciona",
            file=sys.stderr,
        )
        return 4

    try:
        with out.open("x", encoding="utf-8") as handle:
            handle.write(wanted.model_dump_json(indent=2) + "\n")
        selection = wanted
    except FileExistsError:
        existing = Selection.model_validate_json(out.read_text(encoding="utf-8"))
        if not _same_selection(existing, wanted):
            print("select: ya existe reports/selection.json con otra selección", file=sys.stderr)
            return 4
        selection = existing
        print("select: selection.json ya existía (idéntico); solo se re-aplican los tags")

    # El archivo ya existe; los tags son idempotentes.
    client.set_tag(winner.run_id, "selected_candidate", "true")
    client.set_tag(winner.run_id, "selected_at", selection.selected_at.isoformat())
    client.set_tag(winner.run_id, "selection_metric", selection.selection_metric)

    print(json.dumps(selection.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
