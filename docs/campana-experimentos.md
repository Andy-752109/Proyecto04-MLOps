# Campaña de experimentos (P3-10)

Diez corridas de la rejilla congelada (sección 7 de `docs/decisiones-proyecto3.md`) sobre el mismo manifiesto (`v0.1.1`), lanzadas por `POST /ml-api/training/jobs` (el endpoint que usa la página Training) y ejecutadas por `trainer-worker` en CPU. Los `run_id`, las métricas y los hashes salen de MLflow; nada está escrito a mano.

- **Manifiesto (`manifest_sha256`):** `53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f`
- **`git_commit`:** `e082d9c8e06e0d243a1ab2f839857f5cb2a2cd0a` · **`git_dirty`:** `false`
- **Corridas `FINISHED`:** 10 / 10

## Rejilla

| Corrida | Optimizador | Batch | Épocas | LR | Imagen | Ocultas | Dropout | `run_id` |
|---|---|---|---|---|---|---|---|---|
| r01 | adam | 32 | 15 | 0.001 | 128 | 0 | 0.0 | `5e9ae5ebd2e146528a15a4cb5286795d` |
| r02 | adam | 32 | 15 | 0.0003 | 128 | 1 | 0.3 | `7e7b4a4b35464cfebb6b41714a3ad931` |
| r03 | adam | 16 | 15 | 0.001 | 160 | 1 | 0.3 | `0430937ba09d49a78ae67abd62da26cb` |
| r04 | adam | 16 | 30 | 0.0003 | 160 | 0 | 0.3 | `f13d017045c24e8aa59320e6cbcb9126` |
| r05 | sgd | 32 | 30 | 0.01 | 128 | 1 | 0.0 | `9f0caa6305f1477795596e930ab7a28b` |
| r06 | sgd | 16 | 15 | 0.01 | 160 | 0 | 0.3 | `0d202fdb6fd849a59a1423eaea882165` |
| r07 | sgd | 32 | 30 | 0.003 | 160 | 1 | 0.3 | `e7bf686b020f4d02a3fc3e1462e9da33` |
| r08 | adam | 32 | 30 | 0.0001 | 160 | 1 | 0.0 | `e91087b8d30a43abb91f159acecfd886` |
| r09 | sgd | 16 | 30 | 0.01 | 128 | 1 | 0.3 | `00143fe56a564f9a8226c67004c13bdc` |
| r10 | adam | 16 | 15 | 0.001 | 128 | 1 | 0.5 | `7fe4e6afbe424331934017a11a0b2f17` |

## Resultados (mejor época, selección por validación)

| Corrida | Estado | `best_val_accuracy` | `best_val_macro_f1` | `best_val_loss` | `best_epoch` / `stopped_epoch` | `duration_s` |
|---|---|---|---|---|---|---|
| r01 | completed | 0.931298 | 0.931282 | 0.157148 | 4 / 7 | 4729.8 |
| r02 | completed | 0.961832 | 0.961823 | 0.123327 | 4 / 7 | 212.0 |
| r03 | completed | 0.946565 | 0.946515 | 0.083805 | 7 / 10 | 999.9 |
| r04 | completed | 0.931298 | 0.931153 | 0.112995 | 1 / 4 | 369.1 |
| r05 | completed | 0.938931 | 0.938928 | 0.109956 | 7 / 10 | 334.4 |
| r06 | completed | 0.954198 | 0.954174 | 0.212705 | 1 / 4 | 399.1 |
| r07 | completed | 0.931298 | 0.931298 | 0.107727 | 8 / 11 | 1002.1 |
| r08 | completed | 0.946565 | 0.946515 | 0.092334 | 4 / 7 | 707.2 |
| r09 | completed | 0.946565 | 0.946515 | 0.125432 | 3 / 6 | 253.4 |
| r10 | completed | 0.916031 | 0.916011 | 0.165013 | 3 / 6 | 281.7 |

> **Nota sobre `duration_s`:** el de r01 (4729.8 s) está inflado: la Mac entró
> en reposo ~1 h a mitad de esa corrida y el reloj de pared de MLflow siguió
> contando durante el sueño. El tiempo real de r01 fue de ~6 min. Los demás son
> reales: las corridas con imagen 160 y hasta 30 épocas (r03, r07) llegaron a
> ~16.7 min. El smoke midió ~4 min, muy por debajo del umbral de 15 min de la
> sección 7, por lo que **no aplicó** la reducción de imagen a `{112, 128}`.

## Decisión

La mejor `best_val_accuracy` de r01–r10 es **0.961832** (r02).
Como es **≥ 0.90**, no se agregan las corridas extra r11–r12 (sección 7); la rejilla queda cerrada con las 10 corridas.

El snapshot de MLflow (base MariaDB + artefactos MinIO) queda versionado con DVC en `mlflow-store.dvc`, para restaurar en un clon limpio.

