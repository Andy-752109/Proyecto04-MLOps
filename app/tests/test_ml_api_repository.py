"""`ml_api.repository.row_to_job` (P3-03).

No corre contra una MariaDB real -- CI no levanta una para el job de `app/`
(ver CONTRIBUTING.md) -- pero la forma de fila que usa está tomada de una
corrida real contra MariaDB (`docker run mariadb:11`, migración aplicada,
INSERT + SELECT con el driver PyMySQL), no inventada: las columnas JSON
(`config`, `logs`) llegan como texto crudo, no como dict/list ya parseado,
que es justo lo que esta prueba fija.
"""

from datetime import datetime

from ml_api.repository import row_to_job


def test_row_to_job_parses_json_columns_from_raw_text():
    row = (
        "r01",
        "running",
        0.4,
        '{"optimizer": "adam", "batch": 32}',
        "v0.1.1",
        "manifest-abc",
        "campaign",
        "r01",
        None,
        None,
        '["epoch 1 done"]',
        datetime(2026, 9, 29, 21, 23, 26),
    )

    job = row_to_job(row)

    assert job.id == "r01"
    assert job.status == "running"
    assert job.config == {"optimizer": "adam", "batch": 32}
    assert job.run_kind == "campaign"
    assert job.grid_row == "r01"
    assert job.logs == ["epoch 1 done"]
    assert job.mlflow_run_id is None
    assert job.error is None


def test_row_to_job_handles_a_finished_job_with_mlflow_run_and_no_logs():
    row = (
        "r02",
        "completed",
        1.0,
        "{}",
        "v0.1.1",
        "manifest-abc",
        "smoke",
        None,
        "mlflow-run-42",
        None,
        "[]",
        datetime(2026, 9, 29, 22, 0, 0),
    )

    job = row_to_job(row)

    assert job.status == "completed"
    assert job.progress == 1.0
    assert job.run_kind == "smoke"
    assert job.grid_row is None
    assert job.mlflow_run_id == "mlflow-run-42"
    assert job.logs == []
