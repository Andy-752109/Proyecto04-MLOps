"""Contratos de P3-13 (#20); solo validación, sin I/O.

- `Prediction`: una fila de `reports/evaluation/predictions.csv`.
- `EvaluationMetrics`: `reports/evaluation/metrics.json`.
- `Analysis`: `reports/evaluation/analysis.json`.
"""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


# `class_map.json` del modelo: 0 = cat, 1 = dog (sección 3 del doc de decisiones).
CLASSES = ("cat", "dog")


class Prediction(ContractModel):
    crop_id: str
    source_image_id: str
    true_label: int
    predicted_label: int
    probabilities: list[float]


class ClassMetrics(ContractModel):
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationMetrics(ContractModel):
    classes: list[str]
    accuracy: float
    macro_f1: float
    # Filas = clase real, columnas = clase predicha.
    confusion_matrix: list[list[int]]
    per_class: dict[str, ClassMetrics]
    total: int


class Example(ContractModel):
    crop_id: str
    source_image_id: str
    true_class: str
    predicted_class: str
    probability: float


class Analysis(ContractModel):
    baseline_majority_accuracy: float
    most_confused_class: str
    recall_per_class: dict[str, float]
    accuracy_hides_low_recall: bool
    successes: list[Example]
    errors: list[Example]
