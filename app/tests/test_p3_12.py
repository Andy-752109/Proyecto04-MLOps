import mlflow
from mlflow.tracking import MlflowClient


def test_api_lee_tag_mlflow_en_vivo(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = MlflowClient()
    exp_id = client.create_experiment("test_vivo")
    run = client.create_run(exp_id)

    client.set_tag(run.info.run_id, "estado", "valida")

    client.set_tag(run.info.run_id, "estado", "invalida")

    run_actualizada = client.get_run(run.info.run_id)
    assert run_actualizada.data.tags["estado"] == "invalida"
