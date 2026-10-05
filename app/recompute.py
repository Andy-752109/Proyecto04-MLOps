"""Recalcula las métricas de la evaluación con scikit-learn y las compara (P3-13, #20).

No usa `evaluation/metrics.py` a propósito: es un cálculo independiente para
comprobar que las cifras de `metrics.json` son correctas.

Uso (desde la raíz del repo, con el venv de `app/`):

    app/.venv/bin/python recompute.py   # usa reports/evaluation/{predictions.csv,metrics.json}
    app/.venv/bin/python recompute.py --predictions <csv> --metrics <json>
"""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from evaluation.contracts import CLASSES

TOLERANCE = 1e-9
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = REPO_ROOT / "reports" / "evaluation" / "predictions.csv"
DEFAULT_METRICS = REPO_ROOT / "reports" / "evaluation" / "metrics.json"


def sklearn_metrics(
    true_labels: Sequence[int], predicted_labels: Sequence[int], classes: Sequence[str] = CLASSES
) -> dict:
    labels = list(range(len(classes)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", labels=labels, zero_division=0)
        ),
        "confusion_matrix": matrix,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(classes)
        },
        "total": len(true_labels),
    }


def read_predictions(path: Path) -> tuple[list[int], list[int]]:
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    # Ruta de CLI local (la elige quien corre recompute); NOSONAR para S8707.
    with path.open(newline="", encoding="utf-8") as handle:  # NOSONAR
        for row in csv.DictReader(handle):
            true_labels.append(CLASSES.index(row["true_class"]))
            predicted_labels.append(CLASSES.index(row["predicted_class"]))
    return true_labels, predicted_labels


def compare(expected: dict, got: dict) -> list[str]:
    differences: list[str] = []
    for key in ("accuracy", "macro_f1"):
        if abs(expected[key] - got[key]) > TOLERANCE:
            differences.append(f"{key}: metrics={expected[key]} recompute={got[key]}")
    if expected["confusion_matrix"] != got["confusion_matrix"]:
        differences.append(
            f"confusion_matrix: metrics={expected['confusion_matrix']} "
            f"recompute={got['confusion_matrix']}"
        )
    if expected["total"] != got["total"]:
        differences.append(f"total: metrics={expected['total']} recompute={got['total']}")
    for name, per_class in expected["per_class"].items():
        for field in ("precision", "recall", "f1", "support"):
            if abs(per_class[field] - got["per_class"][name][field]) > TOLERANCE:
                differences.append(
                    f"{name}.{field}: metrics={per_class[field]} "
                    f"recompute={got['per_class'][name][field]}"
                )
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", type=Path, help=f"Por defecto, {DEFAULT_PREDICTIONS}.")
    parser.add_argument("--metrics", type=Path, help=f"Por defecto, {DEFAULT_METRICS}.")
    args = parser.parse_args(argv)

    predictions_path = args.predictions or DEFAULT_PREDICTIONS
    metrics_path = args.metrics or DEFAULT_METRICS
    true_labels, predicted_labels = read_predictions(predictions_path)
    got = sklearn_metrics(true_labels, predicted_labels)
    # NOSONAR para S8707 (path traversal): ruta de CLI local.
    expected = json.loads(metrics_path.read_text(encoding="utf-8"))  # NOSONAR
    differences = compare(expected, got)

    print(json.dumps({"accuracy": got["accuracy"], "macro_f1": got["macro_f1"]}, indent=2))
    for difference in differences:
        print(f"DIFIERE: {difference}", file=sys.stderr)
    print("recompute: coincide con metrics.json" if not differences else "recompute: NO coincide")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
