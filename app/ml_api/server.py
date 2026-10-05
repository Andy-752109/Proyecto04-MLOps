"""`ml-api` (P3-03): endpoints HTTP para las 5 pantallas de "Modelo".

Solo `/training/jobs` tiene datos reales hoy (lee `training_jobs`, que este
mismo ticket migra). Los otros cuatro responden `PendingEndpoint`: existen
como contrato -- para que el frontend los pueda consumir ya -- pero sin
datos reales todavía, porque eso lo construyen P3-12 (Experiments), P3-13
(Evaluation), P3-14 (Models) y P3-15/P3-16 (Inference).

Mismo patrón que `copilot/server.py`: `create_app(settings)` para pruebas,
`main()` para producción.
"""

import logging
from collections.abc import Callable

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ml_api.contracts import EvaluationLocked, PendingEndpoint, TrainingJob, TrainingJobList
from ml_api.experiments import get_metric_history, list_runs, update_run_tag
from ml_api.repository import (
    TrainingJobNotCancellable,
    create_training_job,
    list_training_jobs,
    request_training_job_cancellation,
)
from ml_api.training_jobs import TrainingJobRejected, validate_new_training_job
from selection.lock import lock_reason
from storage.db import get_engine
from storage.settings import Settings

logger = logging.getLogger("ml-api")

ListJobs = Callable[[], list[TrainingJob]]
CreateJob = Callable[[str, str, dict, str, str | None], TrainingJob]
CancelJob = Callable[[str], TrainingJob | None]

_PENDING = {
    "experiments": PendingEndpoint(
        ticket="P3-12", message="Experiments se conecta a MLflow en vivo en P3-12."
    ),
    "evaluation": PendingEndpoint(
        ticket="P3-13", message="Evaluation se llena con la evaluación final en P3-13."
    ),
    "models": PendingEndpoint(
        ticket="P3-14", message="Models se llena al publicar el paquete del modelo en P3-14."
    ),
    "inference": PendingEndpoint(
        ticket="P3-16", message="Inference se conecta a la inferencia real en P3-16."
    ),
}


def create_app(
    settings: Settings | None = None,
    *,
    list_jobs: ListJobs | None = None,
    create_job: CreateJob | None = None,
    cancel_job: CancelJob | None = None,
) -> Starlette:
    settings = settings if settings is not None else Settings()
    list_jobs = list_jobs or (lambda: list_training_jobs(get_engine()))
    create_job = create_job or (
        lambda dataset_release, manifest_id, config, run_kind, grid_row: create_training_job(
            get_engine(),
            dataset_release=dataset_release,
            manifest_id=manifest_id,
            config=config,
            run_kind=run_kind,
            grid_row=grid_row,
        )
    )
    cancel_job = cancel_job or (
        lambda job_id: request_training_job_cancellation(get_engine(), job_id)
    )

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    def training_jobs(_: Request) -> JSONResponse:
        jobs = list_jobs()
        return JSONResponse(TrainingJobList(jobs=jobs).model_dump(mode="json"))

    async def create_training_job_route(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "el cuerpo debe ser JSON"}, status_code=400)
        if not isinstance(payload, dict) or not isinstance(payload.get("dataset_release"), str):
            return JSONResponse({"error": "falta dataset_release (str)"}, status_code=400)
        if not isinstance(payload.get("config"), dict):
            return JSONResponse({"error": "falta config (obj)"}, status_code=400)
        if not isinstance(payload.get("run_kind"), str):
            return JSONResponse({"error": "falta run_kind (str)"}, status_code=400)
        grid_row = payload.get("grid_row")
        if grid_row is not None and not isinstance(grid_row, str):
            return JSONResponse({"error": "grid_row debe ser str o null"}, status_code=400)

        try:
            validated_config, manifest_id = validate_new_training_job(
                dataset_release=payload["dataset_release"],
                config=payload["config"],
                run_kind=payload["run_kind"],
                grid_row=grid_row,
                reports_dir=settings.reports_dir,
                derived_dir=settings.derived_dir,
            )
        except TrainingJobRejected as error:
            return JSONResponse({"error": str(error)}, status_code=400)

        job = create_job(
            payload["dataset_release"],
            manifest_id,
            validated_config.model_dump(),
            payload["run_kind"],
            grid_row,
        )
        return JSONResponse(job.model_dump(mode="json"), status_code=201)

    async def cancel_training_job_route(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        try:
            job = cancel_job(job_id)
        except TrainingJobNotCancellable as error:
            return JSONResponse(
                {"error": f"no se puede cancelar: ya está {error}"}, status_code=409
            )
        if job is None:
            return JSONResponse({"error": "no existe ese training_job"}, status_code=404)
        return JSONResponse(job.model_dump(mode="json"))

    def evaluation(_: Request) -> JSONResponse:
        reason = lock_reason(
            selection_path=settings.reports_dir / "selection.json",
            derived_dir=settings.derived_dir,
        )
        if reason:
            return JSONResponse(EvaluationLocked(message=reason).model_dump(mode="json"))
        return JSONResponse(_PENDING["evaluation"].model_dump(mode="json"))

    def pending(name: str):
        async def handler(_: Request) -> JSONResponse:
            return JSONResponse(_PENDING[name].model_dump(mode="json"))

        return handler

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/training/jobs", training_jobs, methods=["GET"]),
            Route("/training/jobs", create_training_job_route, methods=["POST"]),
            Route("/training/jobs/{job_id}/cancel", cancel_training_job_route, methods=["POST"]),
            Route("/experiments/{experiment_id}/runs", list_runs, methods=["GET"]),
            Route(
                "/experiments/runs/{run_id}/metrics/{metric_key}",
                get_metric_history,
                methods=["GET"],
            ),
            Route("/experiments/runs/{run_id}/tags", update_run_tag, methods=["POST"]),
            Route("/evaluation", evaluation, methods=["GET"]),
            Route("/models", pending("models"), methods=["GET"]),
            Route("/inference", pending("inference"), methods=["GET"]),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
    )


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=settings.ml_api_host, port=settings.ml_api_port)


if __name__ == "__main__":
    main()
