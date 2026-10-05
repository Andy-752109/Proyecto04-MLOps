# Reproducibilidad y smoke test (P3-10)

**Objetivo:** comprobar, antes de la campaña (issue #17), que una corrida corta
repetida con la misma configuración y semillas da las mismas métricas, y que
`checkpoint/best.pt` cargado en un proceso nuevo reproduce las predicciones.
La corrida del smoke test es la versión `0.1.0` del modelo ("ensayo, no
seleccionada", sección 10 de `decisiones-proyecto3.md`).

> **Estado:** completa. La parte automatizada (sección 1) corre en el CI y la
> corrida real (secciones 2 y 3) se ejecutó contra el MLflow del stack completo,
> con el manifiesto `v0.1.1`. Todos los datos de abajo salen de MLflow, no están
> escritos a mano.

## 1. Prueba automatizada (modo SMOKE)

`app/tests/test_smoke_reproducibility.py` entrena con
`ml_worker.run_training.run_training`, la misma función que usa el worker real
(`set_reproducibility`, `create_dataloader` con `batch_size=config.batch_size` y
`seed_aug=config.seed_aug`, y `train_with_mlflow` con
`{"run_kind": "smoke", "grid_row": ""}`). Usa recortes JPEG pequeños escritos en
`tmp_path` con un `manifest.csv` real y un MLflow temporal (SQLite y artefactos en
`tmp_path`); no escribe en `app/` ni en `mlruns/`. `GIT_COMMIT` y
`GIT_DIRTY` se fijan con `monkeypatch` solo dentro de la prueba.

Comprueba:

1. Dos corridas con la misma config y semillas dan las mismas `train_loss`,
   `train_accuracy`, `val_loss`, `val_accuracy` y `val_macro_f1` por época
   (tolerancia 1e-6) y el mismo `best_epoch` y `stopped_epoch`.
2. Los parámetros y tags registrados corresponden a lo que se usó
   (`batch_size`, `run_kind`, `grid_row`, `manifest_sha256` del manifiesto
   entrenado).
3. En un proceso nuevo (`multiprocessing.get_context("spawn")`) se descarga
   `checkpoint/best.pt`, su SHA-256 coincide con el tag `checkpoint_sha256`, se
   carga y se predice sobre los recortes de `val`; las probabilidades coinciden
   con las del proceso original dentro de 1e-6.

## 2. Corrida real (manifiesto v0.1.1)

Dos corridas `run_kind=smoke` con el manifiesto v0.1.1, sus recortes reales, la
misma config y las mismas semillas. Se lanzaron por `POST /ml-api/training/jobs`
—el mismo endpoint que usa la página Training— y las ejecutó `trainer-worker`
con `git_commit=e082d9c8e06e0d243a1ab2f839857f5cb2a2cd0a` y `git_dirty=false`.

| Dato | Corrida A (candidata a 0.1.0) | Corrida B |
|---|---|---|
| `run_id` | `b828e032156e423097e90d797b643faa` | `596cf75497f6484d8d5fd102966ee38a` |
| Manifiesto (`manifest_id`) | `v0.1.1-53fc84fdaa07` | `v0.1.1-53fc84fdaa07` |
| `manifest_sha256` | `53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f` | `53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f` |
| `git_commit` / `git_dirty` | `e082d9c8…` / `false` | `e082d9c8…` / `false` |
| `best_epoch` / `stopped_epoch` | 4 / 7 | 4 / 7 |
| `best_val_loss` | 0.1571481739 | 0.1571481739 |
| `best_val_accuracy` | 0.9312977099 | 0.9312977099 |
| `best_val_macro_f1` | 0.9312816926 | 0.9312816926 |
| `duration_s` | 243.71 | 210.72 |

Etiquetas: la corrida A es la candidata a `0.1.0` (**ensayo, no seleccionada**).
La B solo confirma la reproducibilidad. Las cinco métricas por época y
`best_epoch`/`stopped_epoch` coinciden entre A y B.

**Tiempo por corrida (sección 7):** ~4 min (243.7 s y 210.7 s; CPU, Docker en Mac
M3, 4 hilos). Como queda muy por debajo de los 15 min, la campaña se lanza con
la rejilla original (imagen `{128, 160}`), sin la reducción a `{112, 128}`.

## 3. Checkpoint en un proceso nuevo (corrida real)

Se verifica con `python -m training.smoke verify` (ver el comando en el PR):
compara las dos corridas y, para la A, descarga `best.pt`, recalcula su SHA-256
y predice sobre los recortes reales de `val` en un proceso `spawn`.

| Comprobación | Resultado |
|---|---|
| Métricas por época y `best_epoch` A = B (1e-6) | **IGUALES** |
| `checkpoint_sha256` (tag) = SHA-256 recalculado | **COINCIDE** (`9875970463ed8f5977e8fccb5bb25044c7fd75d99268f8b6b4cb5845f77b2a1e`) |
| Recortes de `val` predichos y máx. \|Δ probabilidad\| | 131 recortes; **0.000e+00** |
