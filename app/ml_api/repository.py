"""Lectura de `training_jobs` (P3-03): SQL crudo, no un ORM duplicado.

La tabla la migra y posee el backend (Drizzle, `schema.ts`) -- `ml-api` solo
lee. Nombres de columna entre backticks a propósito: `dataset_release` se
renombró para evitar la palabra reservada `release`, pero cualquier otra
columna futura corre el mismo riesgo, y un SQL crudo no se defiende solo.
"""

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import Engine, text

from ml_api.contracts import TrainingJob


class TrainingJobNotCancellable(Exception):
    """El job existe pero ya terminó (`completed`/`failed`/`cancelled`)."""


SELECT_COLUMNS = """
    `id`, `status`, `progress`, `config`, `dataset_release`, `manifest_id`,
    `run_kind`, `grid_row`, `mlflow_run_id`, `error`, `logs`, `heartbeat_at`
"""


def row_to_job(row: Sequence) -> TrainingJob:
    (
        id_,
        status,
        progress,
        config,
        dataset_release,
        manifest_id,
        run_kind,
        grid_row,
        mlflow_run_id,
        error,
        logs,
        heartbeat_at,
    ) = row
    return TrainingJob(
        id=id_,
        status=status,
        progress=progress,
        # El driver (PyMySQL) entrega las columnas JSON como texto crudo, no
        # como dict/list ya parseado.
        config=json.loads(config),
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        run_kind=run_kind,
        grid_row=grid_row,
        mlflow_run_id=mlflow_run_id,
        error=error,
        logs=json.loads(logs),
        heartbeat_at=heartbeat_at,
    )


def list_training_jobs(engine: Engine) -> list[TrainingJob]:
    query = text(f"SELECT {SELECT_COLUMNS} FROM `training_jobs` ORDER BY `updated_at` DESC")
    with engine.connect() as connection:
        rows = connection.execute(query).fetchall()
    return [row_to_job(row) for row in rows]


def create_training_job(
    engine: Engine,
    *,
    dataset_release: str,
    manifest_id: str,
    config: dict,
    run_kind: str,
    grid_row: str | None,
) -> TrainingJob:
    """Inserta un `training_job` en `queued` y devuelve la fila creada.

    El `id` se genera aquí (no lo asigna MariaDB): a diferencia de un
    autoincremental, un uuid no revela cuántos entrenamientos se han lanzado.
    """
    job = TrainingJob(
        id=uuid.uuid4().hex,
        status="queued",
        progress=0.0,
        config=config,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        run_kind=run_kind,
        grid_row=grid_row,
        mlflow_run_id=None,
        error=None,
        logs=[],
        heartbeat_at=None,
    )
    query = text(
        "INSERT INTO `training_jobs` "
        "(`id`, `status`, `progress`, `config`, `dataset_release`, `manifest_id`, "
        " `run_kind`, `grid_row`, `logs`) "
        "VALUES (:id, :status, :progress, :config, :dataset_release, :manifest_id, "
        "        :run_kind, :grid_row, :logs)"
    )
    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "config": json.dumps(job.config),
                "dataset_release": job.dataset_release,
                "manifest_id": job.manifest_id,
                "run_kind": job.run_kind,
                "grid_row": job.grid_row,
                "logs": json.dumps(job.logs),
            },
        )
    return job


def request_training_job_cancellation(engine: Engine, job_id: str) -> TrainingJob | None:
    """Pide cancelar `job_id`; `None` si no existe (revisión de Uriel sobre P3-09).

    `queued` se cancela directo -- nadie lo reclamó todavía, no hay subproceso
    que avisar. `running` solo deja una marca (`cancel_requested_at`):
    trainer-worker la revisa entre mensajes de progreso (no hay forma de
    interrumpirlo al instante, ver `ml_worker.__main__.process_one_job`) y es
    quien de verdad mata el subproceso y cierra el run de MLflow. Terminal
    (`completed`/`failed`/`cancelled`) levanta `TrainingJobNotCancellable`.
    """
    with engine.begin() as connection:
        row = connection.execute(
            text(f"SELECT {SELECT_COLUMNS} FROM `training_jobs` WHERE `id` = :id FOR UPDATE"),
            {"id": job_id},
        ).fetchone()
        if row is None:
            return None
        job = row_to_job(row)
        if job.status == "queued":
            connection.execute(
                text("UPDATE `training_jobs` SET `status` = 'cancelled' WHERE `id` = :id"),
                {"id": job_id},
            )
        elif job.status == "running":
            connection.execute(
                text(
                    "UPDATE `training_jobs` SET `cancel_requested_at` = :now "
                    "WHERE `id` = :id AND `cancel_requested_at` IS NULL"
                ),
                {"now": datetime.now(UTC).replace(tzinfo=None), "id": job_id},
            )
        else:
            raise TrainingJobNotCancellable(job.status)
        row = connection.execute(
            text(f"SELECT {SELECT_COLUMNS} FROM `training_jobs` WHERE `id` = :id"), {"id": job_id}
        ).fetchone()
    return row_to_job(row)
