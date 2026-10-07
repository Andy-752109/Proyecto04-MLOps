"""`ml-api` (P3-03): endpoints HTTP para las 5 pantallas de "Modelo".

`/training/jobs` (P3-09), `/experiments/*`, `/evaluation` + `/crops` (P3-15),
`/models` (P3-15, sobre el catálogo de P3-14) y `/predict` (P3-16, contra el
paquete publicado en S3) tienen datos reales. El GET de `/inference` sigue
respondiendo `PendingEndpoint`.

Mismo patrón que `copilot/server.py`: `create_app(settings)` para pruebas,
`main()` para producción.
"""

import logging
import re
from collections.abc import Callable

import uvicorn
from botocore.exceptions import BotoCoreError, ClientError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from ml_api.contracts import (
    EvaluationLocked,
    ModelEntry,
    PendingEndpoint,
    TrainingJob,
    TrainingJobList,
)
from ml_api.edge_captures import build_capture_list
from ml_api.evaluation import build_evaluation_report
from ml_api.experiments import get_metric_history, list_runs, update_run_tag
from ml_api.inference import (
    FetchCrop,
    LoadModel,
    fetch_crop_from_annotation,
    load_active_model,
    predict_route,
)
from ml_api.models import (
    ModelObjectMissing,
    ModelVersionNotFound,
    activate_version,
    build_detail,
    build_list,
    load_registry,
    read_active_version,
    read_selected_run_id,
    split_s3_path,
    write_active_version,
)
from ml_api.repository import (
    TrainingJobNotCancellable,
    create_training_job,
    list_training_jobs,
    request_training_job_cancellation,
)
from ml_api.training_jobs import TrainingJobRejected, validate_new_training_job
from selection.lock import lock_reason
from storage.db import get_engine
from storage.edge_capture_store import get_edge_s3_client
from storage.model_store import (
    get_model_s3_client,
    head_object,
    presigned_get_url,
    read_package_info,
)
from storage.settings import Settings

logger = logging.getLogger("ml-api")

ListJobs = Callable[[], list[TrainingJob]]
CreateJob = Callable[[str, str, dict, str, str | None], TrainingJob]
CancelJob = Callable[[str], TrainingJob | None]
ModelStatusOf = Callable[[ModelEntry], dict | None]
ModelDownloadUrlOf = Callable[[ModelEntry], str | None]
ModelPackageInfoOf = Callable[[ModelEntry], dict | None]
EdgeS3Client = Callable[[], object]


def _default_model_status_of(entry: ModelEntry) -> dict | None:
    """`head-object` en vivo (P3-15) contra el bucket del `s3_path` de P3-14,
    pidiendo el `VersionId` registrado (no la última versión de la key)."""
    bucket, key = split_s3_path(entry.s3_path)
    return head_object(get_model_s3_client(), bucket=bucket, key=key, version_id=entry.VersionId)


def _default_model_download_url_of(entry: ModelEntry) -> str | None:
    bucket, key = split_s3_path(entry.s3_path)
    return presigned_get_url(
        get_model_s3_client(), bucket=bucket, key=key, version_id=entry.VersionId
    )


# La tarjeta y el package.json viajan dentro del `.tar.gz`; descargarlo en cada
# click del detalle sería un desperdicio, así que se cachea por VersionId.
_package_info_cache: dict[str, dict] = {}


def _default_model_package_info_of(entry: ModelEntry) -> dict | None:
    bucket, key = split_s3_path(entry.s3_path)
    cache_key = f"{bucket}/{key}#{entry.VersionId}"
    if cache_key not in _package_info_cache:
        _package_info_cache[cache_key] = read_package_info(
            get_model_s3_client(), bucket=bucket, key=key, version_id=entry.VersionId
        )
    return _package_info_cache[cache_key]


_PENDING = {
    "experiments": PendingEndpoint(
        ticket="P3-12", message="Experiments se conecta a MLflow en vivo en P3-12."
    ),
    "evaluation": PendingEndpoint(
        ticket="P3-13",
        message="La selección está cerrada, pero la evaluación final todavía no dejó "
        "reports/evaluation (P3-13).",
    ),
    "models": PendingEndpoint(
        ticket="P3-14",
        message="El catálogo de modelos (models/registry.json) todavía no existe; "
        "lo publica P3-14.",
    ),
    "inference": PendingEndpoint(
        ticket="P3-16", message="Inference se conecta a la inferencia real en P3-16."
    ),
}

# Los recortes viven en `data/derived/crops/images/<crop_id>.jpg` (P3-04). El
# formato lo fija `crops.build.make_crop_id`; se revalida aquí porque este
# endpoint sirve un archivo a partir de un valor de la URL (path traversal).
CROP_ID_PATTERN = re.compile(r"^\d{6}_\d{6}$")


def create_app(
    settings: Settings | None = None,
    *,
    list_jobs: ListJobs | None = None,
    create_job: CreateJob | None = None,
    cancel_job: CancelJob | None = None,
    model_status_of: ModelStatusOf | None = None,
    model_download_url_of: ModelDownloadUrlOf | None = None,
    model_package_info_of: ModelPackageInfoOf | None = None,
    load_model: LoadModel | None = None,
    fetch_crop: FetchCrop | None = None,
    edge_s3_client: EdgeS3Client | None = None,
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
    model_status_of = model_status_of or _default_model_status_of
    model_download_url_of = model_download_url_of or _default_model_download_url_of
    model_package_info_of = model_package_info_of or _default_model_package_info_of
    load_model = load_model or load_active_model
    fetch_crop = fetch_crop or fetch_crop_from_annotation
    edge_s3_client = edge_s3_client or (lambda: get_edge_s3_client(settings))

    registry_path = settings.models_dir / "registry.json"
    active_version_path = settings.models_dir / "active_version.json"
    selection_path = settings.reports_dir / "selection.json"

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    def edge_captures(request: Request) -> JSONResponse:
        values = request.query_params.getlist("limit")
        if len(values) > 1 or (values and not re.fullmatch(r"[0-9]+", values[0])):
            return JSONResponse(
                {"error": "limit debe ser un entero entre 1 y 100"}, status_code=400
            )
        limit = int(values[0]) if values else 50
        if not 1 <= limit <= 100:
            return JSONResponse(
                {"error": "limit debe ser un entero entre 1 y 100"}, status_code=400
            )
        bucket = settings.edge_captures_bucket.strip()
        if not bucket:
            return JSONResponse(
                {"error": "EDGE_CAPTURES_BUCKET no está configurado"}, status_code=503
            )
        try:
            payload = build_capture_list(edge_s3_client(), bucket=bucket, limit=limit)
        except (ClientError, BotoCoreError) as error:
            logger.error("Capturas edge no disponibles por fallo S3/SSO (%s)", type(error).__name__)
            return JSONResponse(
                {"error": "Capturas no disponibles: verifica S3, permisos y sesión SSO"},
                status_code=503,
            )
        return JSONResponse(payload.model_dump(mode="json"))

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
        reason = lock_reason(selection_path=selection_path, derived_dir=settings.derived_dir)
        if reason:
            return JSONResponse(EvaluationLocked(message=reason).model_dump(mode="json"))
        report = build_evaluation_report(
            selection_path=selection_path,
            evaluation_dir=settings.reports_dir / "evaluation",
        )
        if report is None:
            return JSONResponse(_PENDING["evaluation"].model_dump(mode="json"))
        return JSONResponse(report.model_dump(mode="json"))

    def crop(request: Request) -> FileResponse | JSONResponse:
        """Sirve un recorte real para la galería de Evaluation (P3-15)."""
        crop_id = request.path_params["crop_id"]
        if not CROP_ID_PATTERN.match(crop_id):
            return JSONResponse({"error": "crop_id inválido"}, status_code=400)
        path = settings.derived_dir / "crops" / "images" / f"{crop_id}.jpg"
        if not path.is_file():
            return JSONResponse({"error": "no existe ese recorte"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    def models(_: Request) -> JSONResponse:
        registry = load_registry(registry_path)
        if registry is None:
            return JSONResponse(_PENDING["models"].model_dump(mode="json"))
        payload = build_list(
            registry,
            active_version=read_active_version(active_version_path),
            selected_run_id=read_selected_run_id(selection_path),
            status_of=model_status_of,
        )
        return JSONResponse(payload.model_dump(mode="json"))

    def model_detail(request: Request) -> JSONResponse:
        registry = load_registry(registry_path)
        if registry is None:
            return JSONResponse(
                {"error": "Models todavía no está publicado (P3-14)"}, status_code=404
            )
        version = request.path_params["version"]
        try:
            detail = build_detail(
                registry,
                version,
                active_version=read_active_version(active_version_path),
                selected_run_id=read_selected_run_id(selection_path),
                status_of=model_status_of,
                download_url_of=model_download_url_of,
                package_info_of=model_package_info_of,
            )
        except ModelVersionNotFound:
            return JSONResponse({"error": f"no existe la versión {version}"}, status_code=404)
        return JSONResponse(detail.model_dump(mode="json"))

    async def set_active_model(request: Request) -> JSONResponse:
        registry = load_registry(registry_path)
        if registry is None:
            return JSONResponse(
                {"error": "Models todavía no está publicado (P3-14)"}, status_code=404
            )
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "el cuerpo debe ser JSON"}, status_code=400)
        version = payload.get("version") if isinstance(payload, dict) else None
        if not isinstance(version, str) or not version:
            return JSONResponse({"error": "falta version (str)"}, status_code=400)
        try:
            activate_version(registry, version, status_of=model_status_of)
        except ModelVersionNotFound:
            return JSONResponse({"error": f"no existe la versión {version}"}, status_code=404)
        except ModelObjectMissing:
            return JSONResponse(
                {
                    "error": f"el objeto de la versión {version} no existe en S3; "
                    "no se marca como activa"
                },
                status_code=400,
            )
        write_active_version(active_version_path, version)
        detail = build_detail(
            registry,
            version,
            active_version=version,
            selected_run_id=read_selected_run_id(selection_path),
            status_of=model_status_of,
            download_url_of=model_download_url_of,
            package_info_of=model_package_info_of,
        )
        return JSONResponse(detail.model_dump(mode="json"))

    def pending(name: str):
        async def handler(_: Request) -> JSONResponse:
            return JSONResponse(_PENDING[name].model_dump(mode="json"))

        return handler

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/edge/captures", edge_captures, methods=["GET"]),
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
            Route("/crops/{crop_id}", crop, methods=["GET"]),
            Route("/models", models, methods=["GET"]),
            Route("/models/active", set_active_model, methods=["POST"]),
            Route("/models/{version}", model_detail, methods=["GET"]),
            Route("/inference", pending("inference"), methods=["GET"]),
            Route("/predict", predict_route(load_model, fetch_crop), methods=["POST"]),
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
