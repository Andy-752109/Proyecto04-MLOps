"""Contratos de P3-11 (#18); solo validación, sin I/O.

- `RunSummary`: lo que se lee de una corrida de MLflow (parámetros, tags y
  métricas finales). Los módulos puros `validate`/`select` trabajan sobre esto.
- `ExperimentsValidity`: el reporte `reports/experiments_validity.json`.
- `Selection`: el reporte `reports/selection.json`, que leen P3-11, P3-13,
  P3-14 y P3-15.
"""

import json
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Los 7 parámetros que varían por fila de la rejilla (sección 7 de
# docs/decisiones-proyecto3.md). La validación exige ≥2 valores por parámetro.
GRID_PARAMS = (
    "optimizer",
    "batch_size",
    "max_epochs",
    "learning_rate",
    "image_size",
    "hidden_layers",
    "dropout",
)

# Métricas finales que deja toda corrida (sección 8). `best_*` son de la época
# restaurada; P3-11 selecciona con ellas, no con el último valor por época.
SELECTION_METRIC = "best_val_accuracy"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RunSummary(ContractModel):
    """Una corrida de MLflow, ya reducida a lo que la selección necesita."""

    run_id: str
    status: str
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]
    artifacts: list[str] = Field(default_factory=list)

    @property
    def grid_row(self) -> str:
        return self.tags.get("grid_row", "")

    @property
    def classes(self) -> dict | None:
        raw = self.tags.get("classes")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @property
    def is_selected_candidate(self) -> bool:
        """`True` solo para el candidato ya cerrado (`select_candidate.py` pone
        estos tags). Usado por `validate`/`select` para permitir métricas
        `test_*` únicamente en esa corrida -- en cualquier otra, antes o
        después del candado, siguen prohibidas (P3-13, revisión)."""
        return self.tags.get("selected_candidate") == "true" and bool(self.tags.get("selected_at"))


class RunValidity(ContractModel):
    run_id: str
    grid_row: str
    valid: bool
    reasons: list[str]


class ExperimentsValidity(ContractModel):
    """`reports/experiments_validity.json` (chequeo 1 del #18)."""

    manifest_sha256: str
    classes: dict[str, str]
    min_required: int
    valid: list[str]
    invalid: list[RunValidity]
    per_param_values: dict[str, int]
    passed: bool
    produced_at: datetime


class Candidate(ContractModel):
    grid_row: str
    best_val_accuracy: float
    best_val_macro_f1: float
    best_val_loss: float


class Selection(ContractModel):
    """`reports/selection.json` — el sello de la selección (chequeo 4 del #18)."""

    run_id: str
    release: str
    manifest_id: str
    manifest_sha256: str
    checkpoint_sha256: str
    test_ids_sha256: str
    selected_at: datetime
    selection_metric: Literal["best_val_accuracy"] = "best_val_accuracy"
    candidate: Candidate

    @model_validator(mode="after")
    def hex_hashes(self) -> Self:
        for field in ("manifest_sha256", "checkpoint_sha256", "test_ids_sha256"):
            value = getattr(self, field)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{field} no es un SHA-256 hexadecimal de 64 caracteres")
        return self


class SelectionLocked(RuntimeError):
    """El candado se niega a seguir: sin selección cerrada o con hash distinto."""


__all__ = [
    "GRID_PARAMS",
    "SELECTION_METRIC",
    "Candidate",
    "ContractModel",
    "ExperimentsValidity",
    "RunSummary",
    "RunValidity",
    "Selection",
    "SelectionLocked",
]
