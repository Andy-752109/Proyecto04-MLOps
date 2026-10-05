# Evidencia de recarga limpia (P3-14)

Verificación de que las dos versiones publicadas en S3 se pueden recuperar íntegras y recargar,
y de que cambiar de versión cambia el artefacto cargado (criterio 5.3).

- **Bucket:** `mlops-p3-models-222629887955` (versionado activo)
- **Registro:** `models/registry.json`
- **Fecha:** 2026-10-02

## 1. Versiones registradas

| Versión | Run | Tipo | S3 key | VersionId | SHA-256 del paquete |
|---|---|---|---|---|---|
| 0.1.0 | `b828e032156e423097e90d797b643faa` | smoke, no seleccionada | `models/releases/v0.1.0/model_release_v0.1.0.tar.gz` | `VhbG4LZnDiYc_8hduR4kPUiVfrJzZFBj` | `02d0d454a59dea842c7d89c4f950ee4231a5ae035c5140c32713d4b1c18aaa06` |
| 1.0.0 | `7e7b4a4b35464cfebb6b41714a3ad931` | r02, seleccionada | `models/releases/v1.0.0/model_release_v1.0.0.tar.gz` | `V8H6fouL5HiaUvrxAz1gUI5RbRD008VH` | `bb93a41c8f83a2e42b5f3c337136804f9734687a8ba58360308945347cdfa270` |

El SHA-256 se calcula localmente sobre el archivo. No se usa el ETag de S3.

## 2. Descarga e integridad del paquete

`app/verify_reload.py` lee `models/registry.json`, descarga cada paquete de S3 pidiendo el
`VersionId` registrado, calcula el SHA-256 del archivo descargado y lo compara con el del registro.

| Versión | VersionId descargado | SHA-256 esperado (registry) | SHA-256 calculado | Resultado |
|---|---|---|---|---|
| 0.1.0 | `VhbG4LZnDiYc_8hduR4kPUiVfrJzZFBj` | `02d0d454...18aaa06` | `02d0d454...18aaa06` | ✅ coincide |
| 1.0.0 | `V8H6fouL5HiaUvrxAz1gUI5RbRD008VH` | `bb93a41c...cdfa270` | `bb93a41c...cdfa270` | ✅ coincide |

Los paquetes se extrajeron en carpetas nuevas (`eval_v0.1.0/`, `eval_v1.0.0/`), que se borraron antes
de la descarga.

## 3. Recarga e inferencia

`app/recarga_limpia.py` se corrió en un proceso nuevo para cada versión. Usa solo lo que trae el
paquete descargado:

- `artifacts/checkpoint/best.pt`, cuyo SHA-256 se verifica contra el esperado;
- `config.json`, con los hiperparámetros del run.

Con eso reconstruye el modelo con `training.model.build_model` y carga los pesos con
`weights_only=True`. Después predice 3 recortes y compara clase y probabilidades contra una
referencia, con tolerancia `|Δ prob| ≤ 1e-6`. El script sale con código 1 si algo no coincide.

### 3.1 Versión 1.0.0 (seleccionada): split test contra `reports/evaluation/predictions.csv`

```powershell
cd app
$env:PYTHONPATH="."; uv run python -X utf8 recarga_limpia.py --version 1.0.0 --checkpoint ../eval_v1.0.0/artifacts/checkpoint/best.pt --expected-sha256 f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9 --config-json ../eval_v1.0.0/config.json --reference-csv ../reports/evaluation/predictions.csv --split test --selection ../reports/selection.json
```

- SHA-256 esperado (`selection.json`): `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9`
- SHA-256 calculado: `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9` ✅

| Recorte | Clase referencia | Clase recarga | máx \|Δ prob\| | ¿Coincide? |
|---|---|---|---|---|
| `000038_000071` | dog | dog | 1.40e-08 | ✅ |
| `000578_000580` | dog | dog | 1.38e-09 | ✅ |
| `000173_000199` | dog | dog | 8.38e-08 | ✅ |

**Resultado: COINCIDE.**

`predictions.csv` se generó en el entorno de entrenamiento (Linux aarch64), y la recarga se hizo en
Windows. Las diferencias de ~1e-8 corresponden a la aritmética de punto flotante entre plataformas,
y quedan dos órdenes de magnitud debajo de la tolerancia.

### 3.2 Versión 0.1.0 (smoke, no seleccionada): split val contra referencia del checkpoint original

El test es exclusivo del modelo seleccionado, y el script rechaza evaluar otro checkpoint sobre test.
Por eso la 0.1.0 se compara sobre **val**. Como no existían predicciones guardadas del smoke, la
referencia (`ref_010.csv`, 131 recortes de val) se generó primero con el `best.pt` **original de
MLflow** (snapshot de DVC) y la opción `--write-reference`. Después se comparó contra el `best.pt`
**descargado de S3**:

```powershell
$env:PYTHONPATH="."; uv run python -X utf8 recarga_limpia.py --version 0.1.0 --checkpoint ../eval_v0.1.0/artifacts/checkpoint/best.pt --expected-sha256 9875970463ed8f5977e8fccb5bb25044c7fd75d99268f8b6b4cb5845f77b2a1e --config-json ../eval_v0.1.0/config.json --reference-csv ../ref_010.csv --split val
```

- SHA-256 esperado (tag `checkpoint_sha256` del run): `9875970463ed8f5977e8fccb5bb25044c7fd75d99268f8b6b4cb5845f77b2a1e`
- SHA-256 calculado: `9875970463ed8f5977e8fccb5bb25044c7fd75d99268f8b6b4cb5845f77b2a1e` ✅

| Recorte | Clase referencia | Clase recarga | máx \|Δ prob\| | ¿Coincide? |
|---|---|---|---|---|
| `000252_000286` | cat | cat | 6.94e-17 | ✅ |
| `000194_000221` | cat | cat | 0.00e+00 | ✅ |
| `000102_000118` | dog | dog | 7.46e-17 | ✅ |

**Resultado: COINCIDE.**

## 4. Cambiar de versión cambia el artefacto cargado (5.3)

| | 0.1.0 | 1.0.0 |
|---|---|---|
| Run | `b828e032...` (smoke) | `7e7b4a4b...` (r02) |
| SHA-256 de `best.pt` | `98759704...77b2a1e` | `f759cde2...20ab7d9` |
| `learning_rate` | 0.001 | 0.0003 |
| `hidden_layers` | 0 | 1 |
| `dropout` | 0.0 | 0.3 |

Cada versión trae pesos y configuración distintos, y cada una se recarga con su propio
`config.json`. La 0.1.0 sigue recuperable por su `VersionId` aunque exista la 1.0.0.

## 5. Notas

- **Entorno.** La recarga se hizo en un proceso y carpetas nuevas en la máquina local (Windows),
  no en un contenedor nuevo. Para correrla en un contenedor limpio hacen falta acceso a S3 y red,
  porque `build_model` descarga los pesos ImageNet de torchvision al construir el modelo.
- **Publicaciones anteriores.** Las primeras subidas de 0.1.0 y 1.0.0 estaban incompletas: no
  traían `best.pt` y la tarjeta estaba vacía. Se borraron y se republicaron una sola vez con
  `publish.py` corregido. Como el bucket tiene versionado, esas versiones previas siguen en el
  historial del objeto. Los `VersionId` válidos son los de `models/registry.json` y los de esta
  evidencia.
- **Archivos de salida.** `recarga_1.0.0.txt` y `recarga_0.1.0.txt` tienen la salida completa de
  cada corrida.

