"""`trainer-worker` (P3-09): toma trabajos reales de `training_jobs`.

Bucle: reclama trabajos con heartbeat vencido como `failed` (el worker
anterior murió a medio entrenamiento), reclama el siguiente `queued` con
bloqueo atómico, lo corre en un subproceso aparte (`multiprocessing.Process`
-- si el entrenamiento se cuelga o revienta, este proceso sigue vivo y
puede reaccionar, no se cae con él), y va aplicando su progreso a la base
mientras corre.
"""

import logging
import multiprocessing
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty
from typing import Any

import torch
from sqlalchemy import Engine, text

from ml_api.contracts import TrainingJob
from ml_worker.repository import (
    claim_next_queued_job,
    is_cancel_requested,
    mark_cancelled,
    mark_completed,
    mark_failed,
    reclaim_stale_jobs,
    set_mlflow_run_id,
    update_progress,
)
from ml_worker.run_training import dataset_paths, manifest_meta_path, run_training_subprocess
from storage.db import get_engine
from storage.settings import Settings

logger = logging.getLogger("trainer-worker")

POLL_INTERVAL_SECONDS = 5.0
QUEUE_POLL_TIMEOUT_SECONDS = 2.0
# Cuánto puede callar el heartbeat de un trabajo `running` antes de asumir
# que el worker que lo tenía murió sin avisar. Una época real puede tardar
# más que esto con un dataset grande -- se documenta como simplificación
# conocida del worker único de hoy, no un límite pensado para varios workers.
HEARTBEAT_STALE_SECONDS = 600
# Cuánto espera `process.terminate()` (SIGTERM) a que el subproceso cierre
# solo (le da tiempo a que el `with mlflow.start_run()` de train_with_mlflow
# reaccione a la excepción y cierre el run) antes de forzar `process.kill()`.
CANCEL_GRACE_SECONDS = 5.0

SpawnFn = Callable[[TrainingJob, Path, Path, Path], tuple[Any, Any]]
IsCancelRequestedFn = Callable[[Engine, str], bool]
CloseMlflowRunFn = Callable[[str, str], None]


def _close_mlflow_run(run_id: str, status: str) -> None:
    """Cierra un run de MLflow desde fuera del proceso que lo abrió.

    Revisión de Uriel sobre P3-09: si el subproceso que entrena muere sin
    terminar su `with mlflow.start_run()` (lo mata una cancelación, o lo
    mata el sistema operativo), el run queda `RUNNING` en MLflow para
    siempre -- nadie más lo cierra. `MlflowClient.set_terminated` es la
    forma de cerrarlo desde el proceso padre, que sí sigue vivo.
    """
    from mlflow.tracking import MlflowClient

    try:
        MlflowClient().set_terminated(run_id, status=status)
    except Exception:  # el run ya pudo haberse cerrado solo -- no tumbar al worker por esto
        logger.warning(
            "No se pudo cerrar el run de MLflow %s como %s.", run_id, status, exc_info=True
        )


def _spawn_training_process(
    job: TrainingJob, manifest_path: Path, crops_root: Path, meta_path: Path
) -> tuple[multiprocessing.Process, multiprocessing.Queue]:
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_training_subprocess,
        # `model_dump()` corriente, no `mode="json"`: `TrainingJob` es `strict=True`,
        # así que `heartbeat_at` tiene que seguir siendo un `datetime` real para que
        # `TrainingJob(**job_payload)` lo acepte del otro lado -- pickle (lo que usa
        # `multiprocessing` para pasar los argumentos) ya sabe serializar `datetime`.
        args=(job.model_dump(), str(manifest_path), str(crops_root), str(meta_path), queue),
    )
    process.start()
    return process, queue


def process_one_job(
    engine: Engine,
    job: TrainingJob,
    *,
    derived_dir: Path,
    reports_dir: Path,
    spawn: SpawnFn = _spawn_training_process,
    is_cancel_requested: IsCancelRequestedFn = is_cancel_requested,
    close_mlflow_run: CloseMlflowRunFn = _close_mlflow_run,
) -> None:
    """Corre `job` hasta terminar (completed/failed/cancelled) y deja la base al día."""
    manifest_path, crops_root = dataset_paths(derived_dir, job.dataset_release)
    meta_path = manifest_meta_path(reports_dir, job.dataset_release)
    process, queue = spawn(job, manifest_path, crops_root, meta_path)

    mlflow_run_id: str | None = None
    finished = False
    try:
        while True:
            # No hay forma de interrumpir el subproceso al instante (revisión de
            # Uriel sobre P3-09, "no hay forma de cancelar"): se revisa la marca
            # entre mensajes, cada QUEUE_POLL_TIMEOUT_SECONDS como mucho.
            if is_cancel_requested(engine, job.id):
                process.terminate()
                process.join(timeout=CANCEL_GRACE_SECONDS)
                if process.is_alive():
                    process.kill()
                    process.join()
                if mlflow_run_id is not None:
                    close_mlflow_run(mlflow_run_id, "KILLED")
                mark_cancelled(engine, job.id)
                logger.info("%s: cancelado a pedido.", job.id)
                finished = True
                break

            try:
                message = queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except Empty:
                if not process.is_alive():
                    break
                continue

            if message["type"] == "run_started":
                # Guardarlo apenas existe, no solo al terminar (revisión de Uriel:
                # "la corrida queda en RUNNING en MLflow para siempre") -- si el
                # proceso muere después de esto, ya sabemos qué run cerrar.
                mlflow_run_id = message["mlflow_run_id"]
                set_mlflow_run_id(engine, job.id, mlflow_run_id=mlflow_run_id)
            elif message["type"] == "progress":
                update_progress(
                    engine, job.id, progress=message["progress"], log_line=message["log_line"]
                )
                logger.info("%s: %s", job.id, message["log_line"])
            elif message["type"] == "completed":
                mark_completed(engine, job.id, mlflow_run_id=message["mlflow_run_id"])
                logger.info("%s: completado (mlflow_run_id=%s).", job.id, message["mlflow_run_id"])
                finished = True
                break
            else:  # "failed"
                # El propio `with mlflow.start_run()` (training/tracking.py) ya
                # cierra el run como FAILED cuando la excepción se propaga ahí
                # dentro -- no hace falta cerrarlo otra vez aquí.
                mark_failed(engine, job.id, error=message["error"])
                logger.warning("%s: falló (%s).", job.id, message["error"])
                finished = True
                break
    finally:
        process.join()

    if not finished:
        # El subproceso terminó (se cayó, lo mataron) sin mandar un mensaje
        # final -- el padre no se queda esperando para siempre, lo cierra él.
        # A diferencia del caso "failed" de arriba, aquí el `with
        # mlflow.start_run()` nunca llegó a reaccionar (lo mataron de golpe),
        # así que el run sigue RUNNING en MLflow hasta que lo cerremos nosotros.
        error = f"el proceso de entrenamiento terminó inesperadamente (código {process.exitcode})"
        mark_failed(engine, job.id, error=error)
        logger.error("%s: %s", job.id, error)
        if mlflow_run_id is not None:
            close_mlflow_run(mlflow_run_id, "FAILED")


def _wait_for_dependencies(retries: int = 10, delay_seconds: float = 3.0) -> None:
    engine = get_engine()
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Conectado a MariaDB.")
            return
        except Exception as error:  # reintento deliberado en el arranque
            logger.warning("Intento %s/%s: MariaDB aún no lista (%s).", attempt, retries, error)
            time.sleep(delay_seconds)
    raise RuntimeError("No se pudo conectar a MariaDB tras varios intentos.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    logger.info(
        "trainer-worker: arrancando (MLflow en %s, torch %s, cuda=%s).",
        settings.mlflow_tracking_uri,
        torch.__version__,
        torch.cuda.is_available(),
    )
    _wait_for_dependencies()
    engine = get_engine()

    logger.info("trainer-worker listo, esperando training_jobs en 'queued'.")
    while True:
        reclaimed = reclaim_stale_jobs(engine, stale_after_seconds=HEARTBEAT_STALE_SECONDS)
        if reclaimed:
            logger.warning(
                "%s trabajo(s) con heartbeat vencido marcados como failed.", len(reclaimed)
            )
            for mlflow_run_id in reclaimed:
                _close_mlflow_run(mlflow_run_id, "FAILED")

        job = claim_next_queued_job(engine)
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        logger.info("Reclamado %s (release=%s).", job.id, job.dataset_release)
        process_one_job(
            engine, job, derived_dir=settings.derived_dir, reports_dir=settings.reports_dir
        )


if __name__ == "__main__":
    main()
