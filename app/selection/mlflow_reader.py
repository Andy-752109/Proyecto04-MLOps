"""Lectura de corridas desde MLflow (I/O de P3-11).

Traduce cada corrida a `RunSummary` (contrato puro). El resto de `selection/`
no conoce MLflow.
"""

from mlflow.tracking import MlflowClient

from selection.contracts import RunSummary

EXPERIMENT = "clasificador-perro-gato"


def _artifact_paths(client: MlflowClient, run_id: str, path: str | None = None) -> list[str]:
    """Rutas de todos los archivos del run; `list_artifacts` no es recursivo aquí."""
    paths: list[str] = []
    for info in client.list_artifacts(run_id, path):
        if info.is_dir:
            paths.extend(_artifact_paths(client, run_id, info.path))
        else:
            paths.append(info.path)
    return paths


def read_runs(
    tracking_uri: str, *, run_kind: str = "campaign", experiment: str = EXPERIMENT
) -> list[RunSummary]:
    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow_experiment = client.get_experiment_by_name(experiment)
    if mlflow_experiment is None:
        return []
    filter_string = f"tags.run_kind = '{run_kind}'" if run_kind else ""
    runs = client.search_runs(
        [mlflow_experiment.experiment_id], filter_string=filter_string, max_results=1000
    )
    return [
        RunSummary(
            run_id=run.info.run_id,
            status=run.info.status,
            params=dict(run.data.params),
            tags=dict(run.data.tags),
            metrics=dict(run.data.metrics),
            artifacts=_artifact_paths(client, run.info.run_id),
        )
        for run in runs
    ]
