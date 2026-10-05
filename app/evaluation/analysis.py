"""Análisis de errores de P3-13 (#20); puro.

- baseline de la clase mayoritaria sobre el mismo split;
- clase real más confundida (la que más veces se clasificó mal);
- recall por clase;
- si una accuracy ≥ 0.85 oculta un recall bajo (< 0.70);
- 6 aciertos y 6 errores, con su recorte (`crop_id`).
"""

from collections import Counter
from collections.abc import Sequence

from evaluation.contracts import Analysis, EvaluationMetrics, Example, Prediction

LOW_RECALL_THRESHOLD = 0.70
TARGET_ACCURACY = 0.85
SAMPLES = 6


def _example(prediction: Prediction, classes: Sequence[str]) -> Example:
    return Example(
        crop_id=prediction.crop_id,
        source_image_id=prediction.source_image_id,
        true_class=classes[prediction.true_label],
        predicted_class=classes[prediction.predicted_label],
        probability=prediction.probabilities[prediction.predicted_label],
    )


def analyze(
    predictions: Sequence[Prediction],
    metrics: EvaluationMetrics,
    *,
    low_recall_threshold: float = LOW_RECALL_THRESHOLD,
    target_accuracy: float = TARGET_ACCURACY,
    samples: int = SAMPLES,
) -> Analysis:
    classes = metrics.classes
    counts = Counter(prediction.true_label for prediction in predictions)
    total = len(predictions)
    baseline = max(counts.values()) / total if total else 0.0

    errors_by_true = Counter(
        prediction.true_label
        for prediction in predictions
        if prediction.true_label != prediction.predicted_label
    )
    most_confused = classes[max(errors_by_true, key=errors_by_true.get)] if errors_by_true else ""

    recall_per_class = {name: metrics.per_class[name].recall for name in classes}
    hides = metrics.accuracy >= target_accuracy and any(
        recall < low_recall_threshold for recall in recall_per_class.values()
    )

    successes = [
        _example(prediction, classes)
        for prediction in predictions
        if prediction.true_label == prediction.predicted_label
    ][:samples]
    errors = [
        _example(prediction, classes)
        for prediction in predictions
        if prediction.true_label != prediction.predicted_label
    ][:samples]

    return Analysis(
        baseline_majority_accuracy=baseline,
        most_confused_class=most_confused,
        recall_per_class=recall_per_class,
        accuracy_hides_low_recall=hides,
        successes=successes,
        errors=errors,
    )
