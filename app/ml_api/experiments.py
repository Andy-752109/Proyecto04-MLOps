from mlflow.tracking import MlflowClient
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from ml_api.contracts import TagUpdate


def get_client() -> MlflowClient:
    return MlflowClient()


async def list_runs(request: Request) -> JSONResponse:
    experiment_id = request.path_params["experiment_id"]
    client = get_client()
    try:
        runs = client.search_runs(experiment_ids=[experiment_id])
        return JSONResponse([run.to_dictionary() for run in runs])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)


async def get_metric_history(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    metric_key = request.path_params["metric_key"]
    client = get_client()
    try:
        history = client.get_metric_history(run_id, metric_key)
        return JSONResponse(
            [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history]
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)


async def update_run_tag(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    try:
        payload = await request.json()
        tag = TagUpdate.model_validate(payload)
    except (ValueError, ValidationError) as e:
        return JSONResponse({"error": f"Payload inválido: {e}"}, status_code=400)

    client = get_client()
    try:
        client.set_tag(run_id, tag.key, tag.value)
        return JSONResponse(
            {"status": "success", "run_id": run_id, "tag": tag.key, "value": tag.value}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
