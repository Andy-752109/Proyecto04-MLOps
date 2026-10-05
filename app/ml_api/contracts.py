"""Contratos JSON de `ml-api` (P3-03); solo validación, sin I/O.

`TrainingJob` refleja uno a uno la tabla `training_jobs` que migra el backend
(`backend/src/data/db/schema.ts`) -- misma fuente de verdad, dos lenguajes.
Los campos van en snake_case, igual que el resto de los contratos de `app/`
(`presentation/contracts.py`, `copilot/contracts.py`): este repo no mezcla
camelCase y snake_case entre reportes JSON.

Los otros cuatro endpoints (Experiments, Evaluation, Models, Inference) no
tienen todavía un contrato de datos real -- lo definen P3-12, P3-13, P3-14 y
P3-15/16 respectivamente, contra MLflow/el registro de modelos real. Por
ahora solo declaran que están pendientes, para que el frontend tenga algo
real que consumir (un 200 con forma conocida) en vez de un 404 sin explicar
por qué, mientras esas piezas no existen.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


TrainingJobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
# Contrato de MLflow (docs/decisiones-proyecto3.md sección 8, P3-08): 'smoke'
# es el smoke test de P3-10, 'campaign' una fila de la rejilla de P3-01.
RunKind = Literal["smoke", "campaign"]


class TrainingJob(ContractModel):
    id: str
    status: TrainingJobStatus
    progress: float = Field(ge=0, le=1)
    config: dict[str, JsonValue]
    dataset_release: str
    manifest_id: str
    run_kind: RunKind
    grid_row: str | None
    mlflow_run_id: str | None
    error: str | None
    logs: list[str]
    heartbeat_at: datetime | None


class TrainingJobList(ContractModel):
    jobs: list[TrainingJob]


class PendingEndpoint(ContractModel):
    """Respuesta de un área del producto que todavía no tiene datos reales."""

    status: Literal["pending"] = "pending"
    ticket: str
    message: str


class EvaluationLocked(ContractModel):
    """P3-11 (#18): la API de Evaluation se niega a servir datos hasta que la
    selección esté cerrada y el `test_ids_sha256` coincida con el manifiesto.

    Cuando la selección sí está cerrada, `/evaluation` sigue devolviendo
    `PendingEndpoint` hasta que P3-13/P3-15 construyan el payload real.
    """

    status: Literal["selection_not_closed"] = "selection_not_closed"
    ticket: str = "P3-11"
    message: str


class TagUpdate(ContractModel):
    key: str
    value: str
