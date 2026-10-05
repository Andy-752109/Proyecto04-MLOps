"""Lectura de la evaluación final (P3-15): une el sello de P3-11 y los reportes
de P3-13 en un `EvaluationReport`.

No recalcula nada: `reports/evaluation/{metrics,analysis}.json` los produjo
`app/final.py` una sola vez sobre el test (P3-13) y `reports/selection.json` lo
selló P3-11. Si esos archivos todavía no existen, la API responde
`PendingEndpoint` (P3-13) en vez de inventar cifras -- mismo criterio que el
resto de los endpoints que esperan a otro ticket.
"""

import json
from pathlib import Path

from ml_api.contracts import EvaluationClassMetrics, EvaluationExample, EvaluationReport
from selection.contracts import Selection


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_evaluation_report(
    *, selection_path: Path, evaluation_dir: Path
) -> EvaluationReport | None:
    """`None` si la evaluación final (P3-13) todavía no dejó sus reportes."""
    metrics_path = evaluation_dir / "metrics.json"
    analysis_path = evaluation_dir / "analysis.json"
    if not metrics_path.is_file() or not analysis_path.is_file():
        return None

    selection = Selection.model_validate_json(selection_path.read_text(encoding="utf-8"))
    metrics = _load_json(metrics_path)
    analysis = _load_json(analysis_path)

    return EvaluationReport(
        run_id=selection.run_id,
        release=selection.release,
        manifest_id=selection.manifest_id,
        manifest_sha256=selection.manifest_sha256,
        checkpoint_sha256=selection.checkpoint_sha256,
        selected_at=selection.selected_at,
        grid_row=selection.candidate.grid_row,
        best_val_accuracy=selection.candidate.best_val_accuracy,
        best_val_macro_f1=selection.candidate.best_val_macro_f1,
        best_val_loss=selection.candidate.best_val_loss,
        classes=metrics["classes"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        confusion_matrix=metrics["confusion_matrix"],
        per_class={
            name: EvaluationClassMetrics(**values) for name, values in metrics["per_class"].items()
        },
        total=metrics["total"],
        baseline_majority_accuracy=analysis["baseline_majority_accuracy"],
        most_confused_class=analysis["most_confused_class"],
        recall_per_class=analysis["recall_per_class"],
        accuracy_hides_low_recall=analysis["accuracy_hides_low_recall"],
        successes=[EvaluationExample(**example) for example in analysis["successes"]],
        errors=[EvaluationExample(**example) for example in analysis["errors"]],
    )
