# Trazabilidad de punta a punta (P3-18)

Cadena que permite seguir una predicción hasta el dato que entrenó el modelo. Cada eslabón
tiene un identificador y un hash que se pueden recomprobar a mano. Los valores de abajo
salen de `reports/selection.json`, `reports/manifests/v0.1.1/manifest_meta.json`,
`models/registry.json` y `app/manifest/manifest.yaml`.

```
release de datos ──► manifiesto ──► run_id (MLflow) ──► checkpoint ──► versión del modelo ──► S3 ──► predicción
```

## Modelo seleccionado (versión activa 1.0.0)

| Eslabón | Identificador | Hash / evidencia | Fuente |
|---|---|---|---|
| Release de datos | `v0.1.1` | imágenes `951150dd4fb053f4665089fcb37a1c87.dir`, anotaciones `c7cb86ae7ece94ef7b853620e464a4d7.dir` (md5 de DVC) | `manifest_meta.json` → `release` |
| Reporte de calidad | `reports/releases/v0.1.1/quality.json` | SHA-256 `89c1ff7a7bb2a8843fe9da2b2d8bcb2b36afce0ae9f886e48b94caa165afaf6a` | `manifest_meta.json` → `quality_reference` |
| Manifiesto | `v0.1.1-53fc84fdaa07` | SHA-256 `53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f` (semilla 42, 70/20/10) | `manifest_meta.json` |
| Test sellado | — | `test_ids_sha256` `2ff781befc9d118ead9f9af49274fa541e63457d89be306033d22bd563eb4b22` | `selection.json` |
| Run en MLflow | `7e7b4a4b35464cfebb6b41714a3ad931` (fila `r02`) | `best_val_accuracy` 0.9618 | `selection.json` → `candidate` |
| Checkpoint | `artifacts/checkpoint/best.pt` | SHA-256 `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9` | `selection.json`, `registry.json` |
| Versión del modelo | `1.0.0` | publicada 2026-10-03T01:27:42Z | `models/registry.json` |
| Bucket | `mlops-p3-models-222629887955` (versionado activo) | — | `registry.json` → `s3_path` |
| Key | `models/releases/v1.0.0/model_release_v1.0.0.tar.gz` | — | `registry.json` |
| VersionId | `V8H6fouL5HiaUvrxAz1gUI5RbRD008VH` | — | `registry.json` |
| Paquete | `model_release_v1.0.0.tar.gz` | SHA-256 `bb93a41c8f83a2e42b5f3c337136804f9734687a8ba58360308945347cdfa270` | `registry.json` |
| Predicción | `POST /ml-api/predict` | respuesta con `model_version` y `checkpoint_sha256` (= el del checkpoint de arriba) | `PredictionResponse` |

## Versión de contraste (0.1.0, smoke, no seleccionada)

| Eslabón | Valor |
|---|---|
| Release de datos | `v0.1.1` |
| Run | `b828e032156e423097e90d797b643faa` |
| Checkpoint SHA-256 | `9875970463ed8f5977e8fccb5bb25044c7fd75d99268f8b6b4cb5845f77b2a1e` |
| Key | `models/releases/v0.1.0/model_release_v0.1.0.tar.gz` |
| VersionId | `VhbG4LZnDiYc_8hduR4kPUiVfrJzZFBj` |
| Paquete SHA-256 | `02d0d454a59dea842c7d89c4f950ee4231a5ae035c5140c32713d4b1c18aaa06` |

Cambiar de versión en Models cambia `model_version` y `checkpoint_sha256` en la respuesta.

## Cómo recomprobar cada eslabón

| Eslabón | Comando o lugar |
|---|---|
| Datos y DVC | `dvc status` limpio; `dvc.lock` fija los md5 de la release |
| Manifiesto | `sha256sum data/derived/manifests/v0.1.1/manifest.csv` = `manifest_sha256` |
| Run | `http://localhost:5050` → buscar el `run_id`; sus tags deben repetir release, manifiesto y checkpoint |
| Paquete en S3 | `cd app && uv run python verify_reload.py` descarga con el `VersionId` del registro y compara SHA-256 |
| Recarga | `app/recarga_limpia.py` (evidencia en `docs/evidencia-recarga.md`) |
| Predicción | la respuesta de `/ml-api/predict` trae `model_version` y `checkpoint_sha256` |

## Pendiente de confirmar en el clon limpio (viernes 9:00)

- [ ] Que el `run_id` abra en MLflow tras `dvc pull` de `mlflow-store`.
- [ ] Que `sha256sum` del manifiesto coincida tras regenerarlo/bajarlo.
- [ ] Que `verify_reload.py` termine con código 0 desde el clon.
- [ ] Captura de una predicción real con `model_version` 1.0.0 y otra con 0.1.0.
