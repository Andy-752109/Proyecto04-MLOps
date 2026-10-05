"""Acceso a `training_jobs` del lado del worker (P3-09).

Separado de `ml_api.repository`: ese módulo sirve lecturas/creación para el
HTTP (`ml-api`); este reclama, actualiza y cierra trabajos para el proceso
de fondo (`trainer-worker`). Mismo criterio de SQL crudo con backticks que
`ml_api.repository` (ver ahí por qué: `dataset_release` es un nombre
renombrado para evitar la palabra reservada `release`, y un ORM no evita
ese tipo de error por sí solo).
"""

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, text

from ml_api.contracts import TrainingJob
from ml_api.repository import SELECT_COLUMNS, row_to_job


def claim_next_queued_job(engine: Engine) -> TrainingJob | None:
    """Reclama el `training_job` más antiguo en `queued`, o `None` si no hay.

    `FOR UPDATE SKIP LOCKED` hace el reclamo atómico entre workers
    concurrentes: bloquea la fila elegida hasta el commit, y si otro worker
    ya la tiene bloqueada (a medio reclamar), la salta en vez de esperarla o
    chocar con ella -- dos workers corriendo esto a la vez nunca reclaman la
    misma fila.
    """
    now = datetime.now(UTC).replace(tzinfo=None)
    with engine.begin() as connection:
        job_id = connection.execute(
            text(
                "SELECT `id` FROM `training_jobs` "
                "WHERE `status` = 'queued' "
                "ORDER BY `created_at` ASC LIMIT 1 "
                "FOR UPDATE SKIP LOCKED"
            )
        ).scalar_one_or_none()
        if job_id is None:
            return None
        connection.execute(
            text(
                "UPDATE `training_jobs` SET `status` = 'running', `heartbeat_at` = :now "
                "WHERE `id` = :id"
            ),
            {"now": now, "id": job_id},
        )
        row = connection.execute(
            text(f"SELECT {SELECT_COLUMNS} FROM `training_jobs` WHERE `id` = :id"), {"id": job_id}
        ).fetchone()
    return row_to_job(row) if row is not None else None


def update_progress(
    engine: Engine,
    job_id: str,
    *,
    progress: float,
    log_line: str,
    mlflow_run_id: str | None = None,
) -> None:
    """Agrega `log_line` al final de `logs`, actualiza `progress` y `heartbeat_at`."""
    with engine.begin() as connection:
        current_logs = connection.execute(
            text("SELECT `logs` FROM `training_jobs` WHERE `id` = :id"), {"id": job_id}
        ).scalar_one()
        logs = json.loads(current_logs)
        logs.append(log_line)
        connection.execute(
            text(
                "UPDATE `training_jobs` "
                "SET `progress` = :progress, `logs` = :logs, `heartbeat_at` = :now, "
                "    `mlflow_run_id` = COALESCE(:mlflow_run_id, `mlflow_run_id`) "
                "WHERE `id` = :id"
            ),
            {
                "progress": progress,
                "logs": json.dumps(logs),
                "now": datetime.now(UTC).replace(tzinfo=None),
                "mlflow_run_id": mlflow_run_id,
                "id": job_id,
            },
        )


def mark_completed(engine: Engine, job_id: str, *, mlflow_run_id: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE `training_jobs` "
                "SET `status` = 'completed', `progress` = 1, `mlflow_run_id` = :mlflow_run_id, "
                "    `heartbeat_at` = :now "
                "WHERE `id` = :id"
            ),
            {
                "mlflow_run_id": mlflow_run_id,
                "now": datetime.now(UTC).replace(tzinfo=None),
                "id": job_id,
            },
        )


def mark_failed(engine: Engine, job_id: str, *, error: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE `training_jobs` "
                "SET `status` = 'failed', `error` = :error, `heartbeat_at` = :now "
                "WHERE `id` = :id"
            ),
            {"error": error, "now": datetime.now(UTC).replace(tzinfo=None), "id": job_id},
        )


def mark_cancelled(engine: Engine, job_id: str) -> None:
    """Revisión de Uriel sobre P3-09: `process_one_job` llega aquí cuando ve
    `cancel_requested_at` puesto, mata el subproceso y cierra el run de
    MLflow como `KILLED` (ver `ml_worker.__main__._close_mlflow_run`)."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE `training_jobs` SET `status` = 'cancelled', `heartbeat_at` = :now "
                "WHERE `id` = :id"
            ),
            {"now": datetime.now(UTC).replace(tzinfo=None), "id": job_id},
        )


def is_cancel_requested(engine: Engine, job_id: str) -> bool:
    with engine.connect() as connection:
        value = connection.execute(
            text("SELECT `cancel_requested_at` FROM `training_jobs` WHERE `id` = :id"),
            {"id": job_id},
        ).scalar_one_or_none()
    return value is not None


def set_mlflow_run_id(engine: Engine, job_id: str, mlflow_run_id: str) -> None:
    """Guarda el `mlflow_run_id` en cuanto arranca el run, no solo al terminar
    (revisión de Uriel sobre P3-09, punto "corrida queda en RUNNING para
    siempre"): sin esto, un trabajo que muere a medio entrenar nunca tiene
    `mlflow_run_id` en la base, y nadie puede cerrar ese run huérfano."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE `training_jobs` "
                "SET `mlflow_run_id` = :mlflow_run_id, `heartbeat_at` = :now "
                "WHERE `id` = :id"
            ),
            {
                "mlflow_run_id": mlflow_run_id,
                "now": datetime.now(UTC).replace(tzinfo=None),
                "id": job_id,
            },
        )


def reclaim_stale_jobs(engine: Engine, *, stale_after_seconds: int) -> list[str]:
    """Marca `failed` los trabajos en `running` cuyo heartbeat lleva mudo demasiado.

    Cubre al worker que muere a medio entrenamiento (el proceso se cae, el
    trabajo se queda en `running` para siempre si nadie lo detecta). Se
    corre al inicio de cada vuelta del bucle, antes de reclamar uno nuevo.
    Devuelve los `mlflow_run_id` reclamados (sin los `None`) para que
    `ml_worker.__main__.main` cierre esos runs huérfanos en MLflow también
    -- marcar `failed` aquí no le avisa nada a MLflow, son sistemas aparte.
    """
    with engine.begin() as connection:
        threshold = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=stale_after_seconds)
        stale_runs = connection.execute(
            text(
                "SELECT `mlflow_run_id` FROM `training_jobs` "
                "WHERE `status` = 'running' AND `heartbeat_at` < :threshold"
            ),
            {"threshold": threshold},
        ).scalars()
        run_ids = [run_id for run_id in stale_runs if run_id is not None]
        connection.execute(
            text(
                "UPDATE `training_jobs` "
                "SET `status` = 'failed', "
                "    `error` = 'heartbeat vencido: el worker dejó de responder' "
                "WHERE `status` = 'running' "
                "AND `heartbeat_at` < :threshold"
            ),
            {"threshold": threshold},
        )
        return run_ids
