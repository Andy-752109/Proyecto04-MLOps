"""Selección por validación (chequeo 2 de #18); puro, recibe `RunSummary`.

Métrica de selección (sección 5 de docs/decisiones-proyecto3.md):
`best_val_accuracy` (mayor), desempate por `best_val_macro_f1` (mayor) y luego
por menor `best_val_loss`.

Se niega a correr si alguna corrida trae cualquier métrica `test_*`: el test no
se toca antes de cerrar la selección (secciones 5 y 11). Única excepción: el
candidato ya cerrado (`selected_candidate=true` + `selected_at`, ver
`RunSummary.is_selected_candidate`) -- re-ejecutar la selección después de que
P3-13 registró las métricas test_* ahí mismo no debe tropezar con sus propias
métricas.
"""

from selection.contracts import RunSummary


def _forbidden_metrics(run: RunSummary) -> list[str]:
    if run.is_selected_candidate:
        return []
    return sorted(key for key in run.metrics if key.startswith("test_"))


def rank_runs(runs: list[RunSummary]) -> list[RunSummary]:
    offenders = {run.run_id: keys for run in runs if (keys := _forbidden_metrics(run))}
    if offenders:
        raise ValueError(f"la selección prohíbe métricas test_*: {offenders}")
    return sorted(
        runs,
        key=lambda run: (
            -run.metrics.get("best_val_accuracy", float("-inf")),
            -run.metrics.get("best_val_macro_f1", float("-inf")),
            run.metrics.get("best_val_loss", float("inf")),
        ),
    )


def select_candidate(runs: list[RunSummary]) -> RunSummary:
    ranked = rank_runs(runs)
    if not ranked:
        raise ValueError("no hay corridas válidas para seleccionar")
    return ranked[0]
