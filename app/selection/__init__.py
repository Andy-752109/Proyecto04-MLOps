"""P3-11 (#18): validación de corridas, selección por validación y candado.

Lógica pura (sin MLflow ni disco) en `validate`/`select`/`lock`; el acceso a
MLflow vive en `mlflow_reader`. Los CLIs `app/validate_runs.py` y
`app/select_candidate.py` son la I/O.
"""
