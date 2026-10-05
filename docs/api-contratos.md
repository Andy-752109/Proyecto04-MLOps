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
| `/ml-api/evaluation` | GET | `PendingEndpoint` o `EvaluationLocked` | P3-11: bloquea sin `selection.json`; P3-13/P3-15 llenan los datos |
| `/ml-api/models` | GET | `PendingEndpoint` | Pendiente — P3-14 |
| `/ml-api/inference` | GET | `PendingEndpoint` | Pendiente — P3-16 |

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
