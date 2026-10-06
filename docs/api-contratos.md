# Contratos de API — Frente "Modelo" (P3-03)

Congela la forma de los datos entre `ml-api` (Python/Starlette), el frontend
(React) y `trainer-worker` (Python), para que Uriel, Emilio y Juan Pablo
avancen en paralelo sin esperarse los unos a los otros. Ver
`docs/decisiones-proyecto3.md` para las decisiones de modelado/entrenamiento
que estos contratos exponen, no redefinen.

Convención: **snake_case** en todos los campos JSON, igual que el resto de
los reportes/contratos del repo (`presentation/contracts.py`,
`copilot/contracts.py`, `reports/*.json`) — este proyecto no mezcla
camelCase y snake_case entre payloads.

Fuente de verdad de cada contrato:

- Python (`ml-api`, `trainer-worker`): `app/ml_api/contracts.py` (Pydantic,
  `extra="forbid"`, `strict=True`).
- TypeScript (`frontend`): `frontend/src/model/contracts.ts` (Zod), espejo
  campo a campo del Pydantic — un campo que se agregue de un lado y no del
  otro es un bug, no una libertad de implementación.

## Rutas

Todas bajo `/ml-api/` (nginx, `frontend/docker/nginx.conf`), que reenvía al
servicio `ml-api` (`docker-compose.yml`). El backend Node (`/api/`) no las
expone: es un servicio Python aparte, no una ruta más del backend existente.

| Ruta | Método | Contrato | Estado |
|---|---|---|---|
| `/ml-api/health` | GET | `{"status": "ok"}` | Real |
| `/ml-api/training/jobs` | GET | → `TrainingJobList` | Real (lee `training_jobs`) |
| `/ml-api/training/jobs` | POST | `CreateTrainingJobRequest` → `TrainingJob` (201) | Real (P3-09) — valida y encola en `queued` |
| `/ml-api/experiments` | GET | `PendingEndpoint` | Pendiente — P3-12 |
| `/ml-api/evaluation` | GET | `EvaluationReport`, `EvaluationLocked` o `PendingEndpoint` | Real (P3-15): candado P3-11; `PendingEndpoint` (P3-13) solo si falta `reports/evaluation` |
| `/ml-api/crops/{crop_id}` | GET | `image/jpeg` | Real (P3-15): recorte real de `data/derived/crops/images/` |
| `/ml-api/models` | GET | `ModelList` o `PendingEndpoint` | P3-15: lee `models/registry.json` (P3-14); `PendingEndpoint` mientras no exista |
| `/ml-api/models/{version}` | GET | `ModelDetail` | Real (P3-15): tarjeta, procedencia y URL prefirmada |
| `/ml-api/models/active` | POST | `SetActiveVersionRequest` → `ModelDetail` | Real (P3-15): rechaza marcar una versión sin objeto en S3 |
| `/ml-api/inference` | GET | `PendingEndpoint` | Pendiente — P3-16 |
| `/ml-api/edge/captures` | GET | `EdgeCaptureList` | P4-07: eventos edge v1 leídos de S3 |

El progreso de una corrida (`status`/`progress`/`logs`/`heartbeat_at`) lo
escribe `trainer-worker` directo en MariaDB, no a través de `ml-api` — no hay
round-trip HTTP ahí, para no depender de que `ml-api` esté arriba mientras se
entrena. `ml-api` solo escribe la fila inicial, al encolar (`POST`).

La página Training (P3-09) también lee directo, sin pasar por `ml-api`, los
reportes estáticos que ya sirve nginx (`frontend/docker/nginx.conf`,
`./reports:/usr/share/nginx/html/reports:ro`): `reports/versions.json` (el
catálogo de releases, P2-45) y `reports/manifests/<release>/{manifest_meta,
counts}.json` (P3-06) para el selector de release y la procedencia — no hace
falta un endpoint nuevo para eso, ya existen como JSON versionado en Git.

## `TrainingJob`

Refleja uno a uno la tabla `training_jobs`
(`backend/src/data/db/schema.ts`) — misma fuente de verdad en dos lenguajes,
no dos modelos independientes que puedan desincronizarse.

| Campo | Tipo | Fuente | Notas |
|---|---|---|---|
| `id` | `string` | Asignado por quien encola la corrida (ej. `"r01"`, ver la rejilla en `docs/decisiones-proyecto3.md`) | Primary key, no autoincremental |
| `status` | `"queued" \| "running" \| "completed" \| "failed" \| "cancelled"` | `trainer-worker`, al cambiar de fase | — |
| `progress` | `float` (0-1) | `trainer-worker`, fracción de `epocas_maximas` completadas | — |
| `config` | `object` | `trainer-worker`, al encolar — los hiperparámetros de la corrida (optimizador, batch, lr, ...) | Forma exacta la define P3-07 |
| `dataset_release` | `string` | `trainer-worker`, al encolar (ej. `"v0.1.1"`) | Columna `dataset_release`, no `release` — palabra reservada en SQL, confirmado con MariaDB real |
| `manifest_id` | `string` | `ml-api`, al encolar — resuelto desde `manifest_meta.json` del release | — |
| `run_kind` | `"smoke" \| "campaign"` | Quien lanza la corrida, en el formulario de Training | Contrato de MLflow, `docs/decisiones-proyecto3.md` sección 8 |
| `grid_row` | `string \| null` | Quien lanza la corrida, solo si `run_kind = "campaign"` | `null` en `smoke`; `r01`–`r12` en `campaign` |
| `mlflow_run_id` | `string \| null` | `trainer-worker`, al crear el run en MLflow | `null` hasta que el run existe |
| `error` | `string \| null` | `trainer-worker`, solo si `status = "failed"` | — |
| `logs` | `string[]` | `trainer-worker`, líneas recientes (no el log completo — eso vive en MLflow) | — |
| `heartbeat_at` | `string \| null` (ISO 8601) | `trainer-worker`, cada vez que sigue vivo | `null` antes de arrancar; sin heartbeat reciente + `status="running"` = corrida colgada |

## `CreateTrainingJobRequest` (cuerpo del `POST`)

| Campo | Tipo | Notas |
|---|---|---|
| `dataset_release` | `string` | Debe tener un manifiesto (`reports/manifests/<release>/manifest_meta.json`); `ml-api` revalida su compuerta y su fuga antes de encolar, no confía en que P3-06 ya lo garantizó |
| `config` | `TrainingConfig` | Los 7 hiperparámetros de la rejilla, las 4 semillas, `patience` y `min_delta` (`app/training/config.py`) |
| `run_kind` | `"smoke" \| "campaign"` | — |
| `grid_row` | `string \| null` | `null`/vacío en `smoke`; `r01`–`r12` en `campaign` |

Un `POST` inválido (config fuera de rango, release sin manifiesto, compuerta
`failed`, manifiesto con fuga, o `run_kind`/`grid_row` que no cumplen la
regla de arriba) responde `400` con `{"error": "..."}` y no crea ninguna
fila.

## `PendingEndpoint`

Respuesta de un área que todavía no tiene datos reales — para que el
frontend reciba un `200` con forma conocida en vez de un `404` sin explicar
por qué, mientras esa pieza no existe.

| Campo | Tipo | Notas |
|---|---|---|
| `status` | `"pending"` | Constante |
| `ticket` | `string` | Qué ticket construye el contrato real (ej. `"P3-12"`) |
| `message` | `string` | Explicación corta para mostrar en la UI |

## Selección y candado (P3-11, #18)

P3-11 no expone endpoints nuevos: produce dos reportes versionados y un candado
sobre `/ml-api/evaluation`. La lógica vive en `app/selection/` (pura) y los CLIs
`app/validate_runs.py` / `app/select_candidate.py`.

### `reports/experiments_validity.json`

Marca cada corrida de la campaña como válida o no (misma `manifest_sha256` y
`classes`, `FINISHED`, ≥2 épocas, `git_dirty="false"`, parámetros de rejilla no
duplicados) y exige ≥10 válidas con ≥2 valores por parámetro.

| Campo | Tipo |
|---|---|
| `manifest_sha256` | `string` |
| `classes` | `object` (`{"0": "cat", "1": "dog"}`) |
| `min_required` | `int` |
| `valid` | `string[]` (`run_id`) |
| `invalid` | `{run_id, grid_row, reasons}`[] |
| `per_param_values` | `{param: int}` |
| `passed` | `bool` |
| `produced_at` | ISO 8601 |

### `reports/selection.json`

Lo leen P3-11, P3-13, P3-14 y P3-15. Su `test_ids_sha256` **sella el test**.

| Campo | Tipo | Notas |
|---|---|---|
| `run_id` | `string` | Corrida ganadora |
| `release` | `string` | Release del manifiesto (ej. `"v0.1.1"`) |
| `manifest_id` | `string` | — |
| `manifest_sha256` | `string` (64 hex) | — |
| `checkpoint_sha256` | `string` (64 hex) | Tag de la ganadora en MLflow |
| `test_ids_sha256` | `string` (64 hex) | SHA-256 de los `crop_id` del split `test`, ordenados |
| `selected_at` | ISO 8601 | Con zona |
| `selection_metric` | `"best_val_accuracy"` | Métrica de selección (sección 5) |
| `candidate` | `{grid_row, best_val_accuracy, best_val_macro_f1, best_val_loss}` | Métricas de la época restaurada |

Además, la ganadora se etiqueta en MLflow con `selected_candidate`, `selected_at`
y `selection_metric`.

### `EvaluationLocked`

`/ml-api/evaluation` responde esto (200) cuando **no** existe `selection.json` o
su `test_ids_sha256` no coincide con el manifiesto; cuando la selección está
cerrada, vuelve a responder `PendingEndpoint` (P3-13).

| Campo | Tipo | Notas |
|---|---|---|
| `status` | `"selection_not_closed"` | Constante |
| `ticket` | `"P3-11"` | Constante |
| `message` | `string` | Motivo: sin selección, `selection.json` inválido, o hash del test distinto |

El candado (`app/selection/lock.py`) lo usa también `final.py` (P3-13): con
`--split test` exige la selección cerrada y el hash coincidente; con
`--split validation` está exento (es el ensayo previo al cierre).

## Evaluation (P3-15)

`/ml-api/evaluation` responde, en este orden:

1. `EvaluationLocked` (200) si el candado de P3-11 está activo.
2. `PendingEndpoint` (200, ticket `P3-13`) si la selección está cerrada pero
   todavía no existen `reports/evaluation/{metrics,analysis}.json`.
3. `EvaluationReport` (200) cuando ambos existen. La API **no recalcula**: solo
   une el sello de P3-11 (`selection.json`) con lo que P3-13 midió una sola vez.

| Campo | Fuente | Notas |
|---|---|---|
| `run_id`, `release`, `manifest_id`, `manifest_sha256`, `checkpoint_sha256`, `selected_at` | `selection.json` | procedencia |
| `grid_row`, `best_val_accuracy`, `best_val_macro_f1`, `best_val_loss` | `selection.json.candidate` | métricas de validación del candidato |
| `classes`, `accuracy`, `macro_f1`, `confusion_matrix`, `per_class`, `total` | `evaluation/metrics.json` | accuracy = diagonal ÷ total, sin redondear |
| `baseline_majority_accuracy`, `most_confused_class`, `recall_per_class`, `accuracy_hides_low_recall`, `successes`, `errors` | `evaluation/analysis.json` | `successes`/`errors` son `EvaluationExample` |

`EvaluationExample`: `{crop_id, source_image_id, true_class, predicted_class,
probability}`. El frontend arma la imagen con `/ml-api/crops/<crop_id>`.

### `/ml-api/crops/{crop_id}` (P3-15)

Sirve `data/derived/crops/images/<crop_id>.jpg` (P3-04). `crop_id` se revalida
con `^\d{6}_\d{6}$` antes de tocar el disco (path traversal): un valor inválido
da `400`, un archivo ausente `404`.

## Models (P3-15)

El catálogo lo produce **P3-14** (`app/publish.py`, #62) en
`models/registry.json`: un dict `version → entrada`, con los nombres de campo
tal como los escribe `publish.py` (no se inventan).

```json
{
  "0.1.0": {
    "s3_path": "s3://mlops-p3-models-222629887955/models/releases/v0.1.0/model_release_v0.1.0.tar.gz",
    "sha256": "…",            // SHA-256 del tar.gz
    "VersionId": "…",         // VersionId de S3 al publicar
    "run_id": "b828e0…",
    "checkpoint_sha256": "…",
    "data_release": "v0.1.1",  // versión del DATASET
    "published_at": "2026-10-03T01:26:45+00:00"
  },
  "1.0.0": { "…": "run_id 7e7b4a4b… (r02, seleccionada)" }
}
```

`dataset_version` (que expone la API desde `data_release`) va separada de
`version`: la versión del dataset no es la del modelo (requisito del #23). El
registry **no** trae `manifest_id`, `run_kind`, `selected` ni la tarjeta:
- `selected` se deriva comparando `run_id` con `reports/selection.json` (P3-11);
- `manifest_id` y `run_kind` se leen del `package.json`, y la tarjeta de
  `model_card.md`, ambos **dentro del `.tar.gz`** en S3 (`storage/model_store.py`,
  cacheado por `VersionId`).

| Ruta | Contrato | Notas |
|---|---|---|
| `GET /ml-api/models` | `ModelList` o `PendingEndpoint` (P3-14) | agrega `head-object` en vivo (`s3_status`) |
| `GET /ml-api/models/{version}` | `ModelDetail` | tarjeta + trazabilidad + URL prefirmada |
| `POST /ml-api/models/active` | `{version}` → `ModelDetail` | 400 si el objeto no existe en S3, 404 si la versión no está en el registry |

`/models` responde `PendingEndpoint` (ticket `P3-14`) solo si todavía no existe
`models/registry.json`.

La **versión activa para inferencia** (P3-16) vive en
`models/active_version.json` (`{"active_version": "1.0.0"}`), escrito
atómicamente por `ml-api`. `ml-api` monta `./models` (lectura/escritura) y
`~/.aws` (con escritura para renovar la caché SSO) con `AWS_PROFILE=mlops-p3`; el `.gitignore` excluye
`models/active_version.json*`.

### Nota de verificación (P3-16)

El criterio del #23 «cambiar la versión activa cambia el SHA-256 que carga
`ml-api`» se cierra con P3-16 (Inference), que todavía no existe: aquí se
implementa el selector y el rechazo del objeto inexistente, y el resto se
verifica cuando Inference consuma `active_version.json`.

### Decisión abierta (P3-16)

`active_version.json` vs una tabla en MariaDB: si P3-16 prefiere leerlo de la
base (como `training_jobs`), se migra sin tocar el contrato HTTP.

## Capturas edge (P4-07)

`GET /ml-api/edge/captures?limit=50` lee todos los objetos bajo
`edge-captures/v1/events/` del bucket indicado por `EDGE_CAPTURES_BUCKET` en
`ml-api`. `limit` es opcional (default `50`), acepta enteros de `1` a `100` y
se aplica después de ordenar por `captured_at` descendente. Un valor inválido,
`?limit=` vacío o parámetros `limit` duplicados responden `400` con
`{"error": "..."}`.

La respuesta contiene `items`, `total` y `bucket`. `total` cuenta **todos los
eventos válidos** encontrados antes de aplicar `limit`; puede ser mayor que
`items.length`. Cada item incluye los campos del [evento edge v1](../contracts/edge-event.v1.schema.json)
más `received_at` e `image_url`. `received_at` se deriva de `LastModified` del
**objeto evento** listado en S3; no forma parte del JSON persistido.
`image_url` es una URL prefirmada temporal de lectura para el frame completo
indicado por `image_key`. No se comprueba cada imagen con `head_object`.

Por ejemplo, con valores ficticios (la URL omite los parámetros de firma):

```json
{
  "items": [
    {
      "schema_version": "1",
      "capture_id": "8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65",
      "captured_at": "2026-10-05T12:34:56-06:00",
      "device_id": "edge-demo-01",
      "model_version": "optimized-v1",
      "model_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "runtime": "local-runtime",
      "predicted_class": "cat",
      "confidence": 0.93,
      "probabilities": {"cat": 0.93, "dog": 0.07},
      "crop": null,
      "preprocess_ms": 12.5,
      "inference_ms": 38.2,
      "image_key": "edge-captures/v1/images/8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65.jpg",
      "received_at": "2026-10-06T12:30:00Z",
      "image_url": "https://example.invalid/temporary-image-url"
    }
  ],
  "total": 1,
  "bucket": "example-edge-bucket"
}
```

Eventos con JSON corrupto o fuera del contrato se omiten y registran en log.
Un evento que desaparece entre el listado y la lectura (`NoSuchKey`) también
se omite. Fallos generales de S3/SSO, incluidos permisos y generación de la
URL prefirmada, responden `503` con un mensaje que no expone credenciales.
