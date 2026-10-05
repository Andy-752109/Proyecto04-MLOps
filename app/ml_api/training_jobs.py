"""Valida una solicitud de `training_job` nueva, antes de crear ninguna fila (P3-09).

No confía en que quien construyó el manifiesto (P3-06, `dvc_manifest_stage.py`)
ya garantizó que el release pasó la compuerta y que el manifiesto no tiene
fuga -- los vuelve a comprobar aquí, sobre los archivos reales, porque esta
es la última puerta antes de lanzar un entrenamiento real. Mismo criterio que
`presentation/release.cut_release` y `dvc_manifest_stage.quality_reference`,
que también revisan `status == "failed"` cada una por su cuenta en vez de
asumir que la otra ya lo hizo.
"""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from manifest.check_leakage import find_leakage, read_manifest
from training.config import TrainingConfig
from training.grid import FROZEN_MIN_DELTA, FROZEN_PATIENCE, FROZEN_SEEDS, GRID

# Mismas reglas que `training.tracking.train_with_mlflow` (P3-08, sección 8 de
# docs/decisiones-proyecto3.md) -- repetidas aquí a propósito: rechazar un
# run_kind/grid_row inválido antes de crear la fila es más barato que dejar
# que el worker lo descubra a medio entrenar.
GRID_ROW_PATTERN = re.compile(r"^r(0[1-9]|1[0-2])$")


class TrainingJobRejected(Exception):
    """La solicitud se rechaza antes de crear ninguna fila en `training_jobs`."""


def validate_new_training_job(
    *,
    dataset_release: str,
    config: dict,
    run_kind: str,
    grid_row: str | None,
    reports_dir: Path,
    derived_dir: Path,
) -> tuple[TrainingConfig, str]:
    """Valida `config`, `run_kind`/`grid_row` y el release; devuelve (config
    validado, manifest_id).

    Levanta `TrainingJobRejected` si `config` no cumple `TrainingConfig`, si
    `run_kind`/`grid_row` no cumplen el contrato de MLflow, si no hay un
    manifiesto para `dataset_release`, si su compuerta de calidad está
    `failed`, o si el manifiesto tiene fuga entre particiones.
    """
    try:
        validated_config = TrainingConfig.model_validate(config)
    except ValidationError as error:
        raise TrainingJobRejected(f"config inválido: {error}") from error

    # Secciones 4 y 6: congeladas para cualquier corrida, no solo para la
    # rejilla -- un formulario con otro default (o alguien pegando un JSON a
    # mano) no debe poder lanzar un early stopping ni unas semillas distintas
    # a las que el resto del proyecto asume.
    patience_ok = validated_config.patience == FROZEN_PATIENCE
    min_delta_ok = validated_config.min_delta == FROZEN_MIN_DELTA
    if not (patience_ok and min_delta_ok):
        raise TrainingJobRejected(
            f"patience/min_delta están congelados en {FROZEN_PATIENCE}/{FROZEN_MIN_DELTA} "
            "(docs/decisiones-proyecto3.md sección 4)"
        )
    for seed_field, frozen_value in FROZEN_SEEDS.items():
        if getattr(validated_config, seed_field) != frozen_value:
            raise TrainingJobRejected(
                f"{seed_field} está congelada en {frozen_value} "
                "(docs/decisiones-proyecto3.md sección 6)"
            )

    if run_kind not in ("smoke", "campaign"):
        raise TrainingJobRejected(
            f"run_kind debe ser 'smoke' o 'campaign', se recibió {run_kind!r}"
        )
    if run_kind == "smoke" and grid_row:
        raise TrainingJobRejected("grid_row debe estar vacío para run_kind 'smoke'")
    if run_kind == "campaign" and not (grid_row and GRID_ROW_PATTERN.fullmatch(grid_row)):
        raise TrainingJobRejected(f"grid_row inválido para campaign: {grid_row!r}")

    # Sección 7: una fila de la rejilla tiene un único combo válido de
    # parámetros -- sin esto, "r05" se podía registrar con cualquier
    # optimizador/batch/lr y nada lo notaba (r11/r12 quedan sin definir a
    # propósito, ver training/grid.py, así que no hay nada que validar ahí).
    if run_kind == "campaign" and grid_row in GRID:
        expected = GRID[grid_row]
        mismatches = {
            field: (getattr(validated_config, field), expected_value)
            for field, expected_value in expected.items()
            if getattr(validated_config, field) != expected_value
        }
        if mismatches:
            raise TrainingJobRejected(
                f"config no coincide con la rejilla de {grid_row!r}: {mismatches}"
            )

    meta_path = reports_dir / "manifests" / dataset_release / "manifest_meta.json"
    if not meta_path.exists():
        raise TrainingJobRejected(f"no existe un manifiesto para el release {dataset_release!r}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    status = meta["quality_reference"]["status"]
    if status == "failed":
        raise TrainingJobRejected(
            f"el release {dataset_release!r} está failed en la compuerta de calidad"
        )

    manifest_csv = derived_dir / "manifests" / dataset_release / "manifest.csv"
    if not manifest_csv.exists():
        raise TrainingJobRejected(f"no existe el manifiesto {manifest_csv}")
    leakage = find_leakage(read_manifest(manifest_csv))
    if any(leakage.values()):
        raise TrainingJobRejected(f"el manifiesto de {dataset_release!r} tiene fuga: {leakage}")

    return validated_config, meta["manifest_id"]
