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

    Cuando la selección sí está cerrada, `/evaluation` devuelve el
    `EvaluationReport` real (P3-15); si todavía no existen los reportes de la
    evaluación final (P3-13), sigue respondiendo `PendingEndpoint`.
    """

    status: Literal["selection_not_closed"] = "selection_not_closed"
    ticket: str = "P3-11"
    message: str


class EvaluationExample(ContractModel):
    """Un recorte de test con su resultado, para la galería de aciertos/errores
    (`reports/evaluation/analysis.json`, P3-13)."""

    crop_id: str
    source_image_id: str
    true_class: str
    predicted_class: str
    probability: float


class EvaluationClassMetrics(ContractModel):
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationReport(ContractModel):
    """`/evaluation` (P3-15) tras cerrar la selección: junta el sello de P3-11
    (`selection.json`) con los reportes de la evaluación final de P3-13
    (`metrics.json` + `analysis.json`). Nada se recalcula aquí: la API solo
    sirve lo que P3-13 ya midió una sola vez sobre el test."""

    status: Literal["ready"] = "ready"

    # Procedencia (reports/selection.json).
    run_id: str
    release: str
    manifest_id: str
    manifest_sha256: str
    checkpoint_sha256: str
    selected_at: datetime
    grid_row: str
    best_val_accuracy: float
    best_val_macro_f1: float
    best_val_loss: float

    # Métricas (reports/evaluation/metrics.json).
    classes: list[str]
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, EvaluationClassMetrics]
    total: int

    # Análisis de errores (reports/evaluation/analysis.json).
    baseline_majority_accuracy: float
    most_confused_class: str
    recall_per_class: dict[str, float]
    accuracy_hides_low_recall: bool
    successes: list[EvaluationExample]
    errors: list[EvaluationExample]


class TagUpdate(ContractModel):
    key: str
    value: str


# --- P3-15: Models -----------------------------------------------------------


# `models/registry.json` es lo que produce P3-14 (`publish.py`, #62): un dict
# `version -> entrada`. Los nombres de campo son los del registry real
# (`s3_path`, `sha256`, `VersionId`, `data_release`); no se inventan. El SHA-256
# del paquete y el `VersionId` los registró `publish.py` al publicar; el estado
# en vivo de S3 lo agrega P3-15 con `head-object`.
class ModelEntry(ContractModel):
    s3_path: str
    sha256: str
    VersionId: str
    run_id: str
    checkpoint_sha256: str
    data_release: str
    published_at: datetime


class ModelS3Status(ContractModel):
    exists: bool
    version_id: str | None = None
    size_bytes: int | None = None
    last_modified: datetime | None = None
    # P3-15: la verificación en vivo falló (red, SSO expirado, permisos), que
    # no es lo mismo que "el objeto no existe". GET /models sigue respondiendo
    # 200 con esta versión marcada, en vez de tumbar toda la lista.
    error: str | None = None


class ModelSummary(ContractModel):
    version: str
    dataset_version: str
    run_id: str
    published_at: datetime
    package_sha256: str
    registered_version_id: str
    selected: bool
    active: bool
    s3_status: ModelS3Status


class ModelList(ContractModel):
    status: Literal["ready"] = "ready"
    active_version: str | None
    versions: list[ModelSummary]


class ModelDetail(ContractModel):
    status: Literal["ready"] = "ready"
    version: str
    dataset_version: str
    run_id: str
    published_at: datetime
    package_sha256: str
    registered_version_id: str
    checkpoint_sha256: str
    run_kind: str | None
    manifest_id: str | None
    selected: bool
    active: bool
    s3_bucket: str
    s3_key: str
    card: str | None
    download_url: str | None
    s3_status: ModelS3Status


class SetActiveVersionRequest(ContractModel):
    version: str


class PredictionResponse(ContractModel):
    """`POST /predict` (P3-16, #24): resultado de clasificar una imagen con el
    modelo activo. `probabilities` va por nombre de clase (`{"cat": 0.9,
    "dog": 0.1}`), no por índice posicional -- es la respuesta HTTP que
    consume el frontend, no una fila de `evaluation.contracts.Prediction`."""

    predicted_label: str
    probabilities: dict[str, float]
    model_version: str
    checkpoint_sha256: str
