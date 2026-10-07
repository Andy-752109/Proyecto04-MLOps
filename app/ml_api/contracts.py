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

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

EDGE_PREDICTION_TOLERANCE = 1e-6


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class EdgeProbabilities(ContractModel):
    cat: float = Field(ge=0, le=1)
    dog: float = Field(ge=0, le=1)


class EdgeCrop(ContractModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)

    @model_validator(mode="after")
    def inside_frame(self) -> "EdgeCrop":
        if self.x + self.width > self.frame_width or self.y + self.height > self.frame_height:
            raise ValueError("crop fuera de las dimensiones del frame")
        return self


class EdgeEventV1(ContractModel):
    """JSON persistido; los datos derivados de S3 no pertenecen a este modelo."""

    schema_version: Literal["1"]
    capture_id: str
    captured_at: str
    device_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    runtime: str = Field(min_length=1)
    predicted_class: Literal["cat", "dog"]
    confidence: float = Field(ge=0, le=1)
    probabilities: EdgeProbabilities
    crop: EdgeCrop | None
    preprocess_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    image_key: str

    @field_validator("capture_id")
    @classmethod
    def canonical_uuid_v4(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("capture_id debe ser UUID v4 canónico lowercase") from exc
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("capture_id debe ser UUID v4 canónico lowercase")
        return value

    @field_validator("captured_at")
    @classmethod
    def aware_iso_datetime(cls, value: str) -> str:
        # ISO 8601 extendido con hora y zona; fromisoformat comprueba fecha/hora reales.
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
        if not re.fullmatch(pattern, value):
            raise ValueError("captured_at debe ser ISO 8601 con zona horaria")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("captured_at contiene fecha u hora inválida") from exc
        if parsed.utcoffset() is None:
            raise ValueError("captured_at debe incluir zona horaria")
        return value

    @model_validator(mode="after")
    def image_matches_capture(self) -> "EdgeEventV1":
        expected = f"edge-captures/v1/images/{self.capture_id}.jpg"
        if self.image_key != expected:
            raise ValueError("image_key debe apuntar al frame de capture_id")
        return self

    @model_validator(mode="after")
    def prediction_is_consistent(self) -> "EdgeEventV1":
        # Misma regla que edge/event_validator.py (P4-06): salen de la misma salida del modelo.
        probabilities = {"cat": self.probabilities.cat, "dog": self.probabilities.dog}
        predicted = probabilities[self.predicted_class]
        if predicted < max(probabilities.values()):
            raise ValueError("predicted_class debe ser la clase de mayor probabilidad")
        if abs(self.confidence - predicted) > EDGE_PREDICTION_TOLERANCE:
            raise ValueError("confidence debe ser igual a probabilities[predicted_class]")
        return self


class EdgeCaptureItem(EdgeEventV1):
    """Evento validado enriquecido para HTTP, nunca persistido en S3."""

    received_at: datetime
    image_url: str


class EdgeCaptureList(ContractModel):
    items: list[EdgeCaptureItem]
    total: int
    bucket: str


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
