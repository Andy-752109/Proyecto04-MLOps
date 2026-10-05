"""Métricas de P3-13 (#20); puras, sin I/O ni sklearn.

Matriz de confusión con filas = clase real y columnas = clase predicha;
`accuracy` = diagonal ÷ total, sin redondear; `macro_f1` = promedio de los F1
por clase; y precisión/recall/support por clase.
"""

from collections.abc import Sequence

from evaluation.contracts import CLASSES, ClassMetrics, EvaluationMetrics, Prediction


def confusion_matrix(predictions: Sequence[Prediction], n_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for prediction in predictions:
        matrix[prediction.true_label][prediction.predicted_label] += 1
    return matrix


def evaluate(
    predictions: Sequence[Prediction], classes: Sequence[str] = CLASSES
) -> EvaluationMetrics:
    n_classes = len(classes)
    matrix = confusion_matrix(predictions, n_classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(n_classes))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, ClassMetrics] = {}
    f1_scores: list[float] = []
    for index, name in enumerate(classes):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(n_classes)) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / n_classes if n_classes else 0.0
    return EvaluationMetrics(
        classes=list(classes),
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=matrix,
        per_class=per_class,
        total=total,
    )
