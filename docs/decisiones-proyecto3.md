# Decisiones del Proyecto 3

> **Congeladas antes de cualquier corrida de entrenamiento.** La fecha del commit
> que mergea este archivo es la evidencia de que las clases, la métrica de
> selección y la rejilla se eligieron antes de mirar el test (rúbrica 1.2, 2.1,
> 2.4, 3.1, 3.3 y 5.1). Cambiar algo después exige un commit nuevo que lo
> justifique y, si ya hubo corridas, repetirlas.

## 1. Release de origen

| Campo | Valor |
|---|---|
| Release | `v0.1.1`, la única versión que aprobó la compuerta de calidad (`v0.1.0` reprobó) |
| Imágenes (`data/raw/images.dvc`) | `md5 951150dd4fb053f4665089fcb37a1c87.dir` · 600 archivos |
| Anotaciones (`data/raw/annotations.dvc`) | `md5 c7cb86ae7ece94ef7b853620e464a4d7.dir` · 10 archivos |
| Remote DVC | `s3://mlops-p2-dvc-cache-222629887955` |
| Baseline de P2 | commit `9cfea4c`, tag `p2-final-baseline` = [`karenelizabg/proyecto-fase2-MLOPS@849c812`](https://github.com/karenelizabg/proyecto-fase2-MLOPS/commit/849c8120bd1b8fbc5c2fa40bdf390d16d62ec57f) (entrega final de P2, calificada con 99) |

### Política de compuerta

La compuerta evalúa los siete checks de `app/policies/quality.yaml`. Un check
con `action: fail` que no pasa deja el reporte en `status: failed` y detiene las
etapas de DVC que dependen del marcador de la compuerta (hoy, solo `split`); un
check con `action: warn` que no pasa deja el release en `status: warning`, que
**sí cuenta como aprobado** para este proyecto.

- `app/policies/quality.yaml`: bloquean (`action: fail`) `min_images_per_class`
  (umbral 300), `degenerate_boxes` (umbral 0) y `cross_split_leakage` (umbral 0);
  solo advierten (`action: warn`) `max_imbalance_ratio`, `max_small_object_ratio`,
  `duplicate_similarity_threshold` y `min_spatial_dispersion`.
- `app/presentation/gate.py`, `evaluate_dataset` (líneas 144-149): `status` es
  `failed` si algún check `fail` no pasó, `warning` si solo fallan checks
  `warn`, y `passed` si todo pasa. `main()` (líneas 199-208) devuelve código 1
  cuando el reporte es `failed`.
- `app/dvc_gate_stage.py`: escribe el marcador `reports/.quality_gate.passed`
  solo si el reporte no es `failed`. En `dvc.yaml`, ese marcador se declara como
  salida de la etapa `quality_gate` (línea 36) y **solo** `split` lo lista como
  dependencia (línea 45); `projections` no depende de la compuerta.

**Release de origen verificado.** En `v0.1.1` ningún check `fail` reprobó
(`min_images_per_class = 300.0`, `degenerate_boxes = 0`,
`cross_split_leakage = 0`); el `status: warning` proviene de checks `warn`, por
lo que la compuerta se considera **aprobada** y existe
`reports/.quality_gate.passed`. En `v0.1.0` la compuerta quedó `failed`: su
política aún declaraba las clases `person` (id 1) y `car` (id 2) sin
anotaciones, así que `min_images_per_class` (action `fail`) vio 0 imágenes y
reprobó el release.

**Recuperabilidad de v0.1.0.** v0.1.0 comparte las mismas 600 imágenes de
v0.1.1 (`data/raw/images.dvc` = `951150dd4fb053f4665089fcb37a1c87.dir`, con 600
archivos) y se diferencia solo en las anotaciones
(`8bd7d8e8f4dcb65edf95d8f38822cc53.dir`, 10 archivos). Ambos objetos siguen
presentes en el cache de S3 (`files/md5/.../.dir`), por lo que los datos de
v0.1.0 **sí son recuperables** desde el repo de P2; su reprobación se debe a la
lista de clases, no a la pérdida del dataset.

**Conteos útiles por clase (P3-02).** Contando imágenes originales después de
descartar cajas inválidas con el mismo analizador de la compuerta
(`app/crops/counts.py`, que reutiliza `analyzers.invalid_boxes`), sobre
`data/raw` (600 imágenes, 668 anotaciones, 0 cajas inválidas en este release).
El conteo parte del COCO crudo (`load_raw_dataset`), así que el filtro también
opera cuando el dataset trae cajas degeneradas que la validación estricta
rechazaría:

| Clase | Imágenes originales | Mínimo | ¿Cumple? |
|---|---|---|---|
| `dog` (id 3) | 300 | 300 | Sí, en el mínimo exacto |
| `cat` (id 4) | 301 | 300 | Sí |

`dog` está justo en el mínimo de 300: una sola caja inválida adicional, si es la
única de perro en su imagen, lo dejaría en 299 imágenes útiles (y la compuerta
reprobaría por `degenerate_boxes`). Es la clase a vigilar en P3-04 al construir
los recortes.

Comando de verificación:

```bash
cd app && uv run python -m crops.counts --annotations-dir ../data/raw/annotations
```

## 2. Clases

- **Clases:** `dog` (id COCO 3) y `cat` (id COCO 4), las dos categorías del
  release.
- **Exclusiones:** ninguna otra categoría. Se excluyen las cajas inválidas
  (degeneradas o fuera de la imagen; hoy hay 0) y las imágenes faltantes; cada
  una queda en `exclusions.csv` con su motivo (P3-04).
- **Advertencia de sesgo espacial** de la compuerta: se acepta y queda
  documentada como limitación en la tarjeta del modelo.
- **Unidad de clasificación:** un recorte por caja válida, nunca la imagen
  completa.
- **Conteo en bruto** (antes de filtrar cajas inválidas): `dog` 300 imágenes
  originales y `cat` 301. P3-02 confirma que, tras descartar las cajas
  inválidas, siguen siendo 300 y 301; `dog` está exactamente en el mínimo.

## 3. CNN

- **Arquitectura:** ResNet18 de `torchvision`.
- **Pesos:** `ResNet18_Weights.IMAGENET1K_V1` (preentrenados en ImageNet-1k,
  descargados por `torchvision`).
- **Capas entrenables:** `layer4` y la cabeza. `conv1`, `bn1`, `layer1`–`layer3`
  quedan congeladas (`requires_grad = False`).
- **Cabeza:** `capas_ocultas` capas lineales de 256 unidades con ReLU y
  `dropout`, seguidas de `Linear(→ 2)`.
- **Salida:** logits `[batch, 2]`; `class_map.json` = `{0: "cat", 1: "dog"}`.
- **Por qué preentrenada:** con ~600 imágenes, entrenar desde cero difícilmente
  llega al 85%.

## 4. Early stopping

| Parámetro | Valor |
|---|---|
| Métrica vigilada | `val_loss` (se minimiza) |
| `patience` | 3 épocas |
| `min_delta` | 0.001 |
| Tope | `epocas_maximas` de cada corrida |
| Al terminar | **siempre** se restauran los pesos de la mejor época |

## 5. Métrica de selección

- **Principal:** `val_accuracy` (mayor es mejor).
- **Desempate:** `val_macro_f1`; si persiste, menor `val_loss`.
- **Prohibido:** cualquier métrica `test_*`. `select_candidate.py` falla si la recibe.

## 6. Semillas

| Uso | Semilla |
|---|---|
| Split del manifiesto | 42 |
| DataLoader (`generator`) | 43 |
| Aumentación | 44 |
| Inicialización de la cabeza | 45 |

## 7. Rejilla de experimentos

Cada uno de los 7 parámetros toma al menos 2 valores y ninguna fila se repite.
Las corridas son sobre el mismo manifiesto (P3-06).

| Corrida | Optimizador | Batch | Épocas máx. | LR | Imagen | Capas ocultas | Dropout |
|---|---|---|---|---|---|---|---|
| r01 | adam | 32 | 15 | 1e-3 | 128 | 0 | 0.0 |
| r02 | adam | 32 | 15 | 3e-4 | 128 | 1 | 0.3 |
| r03 | adam | 16 | 15 | 1e-3 | 160 | 1 | 0.3 |
| r04 | adam | 16 | 30 | 3e-4 | 160 | 0 | 0.3 |
| r05 | sgd | 32 | 30 | 1e-2 | 128 | 1 | 0.0 |
| r06 | sgd | 16 | 15 | 1e-2 | 160 | 0 | 0.3 |
| r07 | sgd | 32 | 30 | 3e-3 | 160 | 1 | 0.3 |
| r08 | adam | 32 | 30 | 1e-4 | 160 | 1 | 0.0 |
| r09 | sgd | 16 | 30 | 1e-2 | 128 | 1 | 0.3 |
| r10 | adam | 16 | 15 | 1e-3 | 128 | 1 | 0.5 |

Valores por parámetro: optimizador {adam, sgd} · batch {16, 32} · épocas
{15, 30} · LR {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} · imagen {128, 160} · capas
ocultas {0, 1} · dropout {0.0, 0.3, 0.5}. SGD usa `momentum = 0.9`.

**Corridas extra r11–r12:** solo si la mejor `val_accuracy` de r01–r10 es menor
que 0.90, y siempre antes de cerrar la selección.

**Presupuesto de cómputo:** la campaña la corre **Uriel en su Mac M3 (4 CPU y
8 GB de RAM)**, en segundo plano y conectada a la corriente (`caffeinate -i`).
El entrenamiento corre en el contenedor `trainer-worker`, **solo con CPU**
(Docker en macOS no expone la GPU `mps`), con `torch.set_num_threads(4)` y
`num_workers = 2`. Docker Desktop necesita al menos 6 GB de RAM asignados; los
demás servicios que no se usan durante la campaña (frontend, copilot) se
detienen. Estimado: ~15 min por corrida como máximo, 30 épocas como
tope; 10 corridas más la evaluación final suman unas 3–4 h. Si el smoke test
(P3-10) mide más de 15 min por corrida, antes de lanzar la campaña se reduce
el tamaño de imagen a {112, 128} en un commit que cite esta sección. No se
permite ningún otro cambio.

## 8. MLflow

- Servicio `mlflow` en `docker-compose.yml`, en `localhost:5050` (el puerto
  5000 choca con AirPlay en macOS).
- Registros en MariaDB y artefactos en el bucket `mlflow-artifacts` de MinIO,
  ambos con volumen persistente.
- Después de la campaña y de la selección se toma un snapshot (volcado de la
  base y artefactos) versionado con DVC, para que se pueda restaurar en un clon
  limpio.

### Contrato de cada corrida

Lo que toda corrida de entrenamiento deja en MLflow. Lo escribe
`app/training/tracking.py` (P3-08) y lo leen la campaña (P3-10), la validación
y la selección (P3-11), la página Experiments (P3-12) y el paquete del modelo
(P3-14). Una corrida a la que le falte algo de esta lista no cuenta como válida.

- **Experimento:** `clasificador-perro-gato`, uno solo para el smoke test y la
  campaña.
- **Parámetros:** todos los campos de `TrainingConfig` (los 7 de la rejilla, las
  4 semillas, `patience` y `min_delta`), con `config.model_dump()`. Con SGD,
  también `momentum`.
- **Tags:**

  | Tag | Valor |
  |---|---|
  | `git_commit` | SHA completo de `HEAD` |
  | `git_dirty` | `true` si hay cambios sin commitear; P3-11 descarta esas corridas |
  | `release` | `release.name` de `manifest_meta.json` (`v0.1.1`) |
  | `manifest_id` y `manifest_sha256` | los de `manifest_meta.json` |
  | `classes` | `classes` de `manifest_meta.json` en JSON: `{"0": "cat", "1": "dog"}` |
  | `run_kind` | `smoke` o `campaign`; solo `campaign` cuenta para P3-11 |
  | `grid_row` | la fila de la sección 7 (`r01`…`r12`); vacío en `smoke` |
  | `checkpoint_sha256` | SHA-256 de `checkpoint/best.pt` |
  | `python_version` | `platform.python_version()` |
  | `torch_version` y `torchvision_version` | `torch.__version__` y `torchvision.__version__` |
  | `platform` | `platform.platform()` (sistema y arquitectura donde corrió) |
  | `device` | `cpu`, `cuda` o `mps` |

- **Métricas por época**, con `step` = época (desde 1): `train_loss`,
  `train_accuracy`, `val_loss`, `val_accuracy` y `val_macro_f1`.
  `train_accuracy` se calcula durante la misma pasada de entrenamiento, sin
  recorrer `train` otra vez (duplicaría el tiempo por corrida de la sección 7).
- **Métricas finales**, sin `step`: `best_epoch`, `stopped_epoch`,
  `best_val_loss`, `best_val_accuracy`, `best_val_macro_f1` y `duration_s`.
  Las `best_*` son las de la época restaurada; P3-11 selecciona con ellas y no
  con el último valor por época, que es el de la época de parada.
- **Artefactos:**
  - `checkpoint/best.pt`: `state_dict` de la mejor época (sección 4), el que se
    carga desde un proceso nuevo en el smoke test;
  - `curves.csv`: una fila por época con las cinco métricas por época.
- **Estado:** `FINISHED` si termina; `FAILED` si hay una excepción (se registra
  y se vuelve a lanzar); `KILLED` si se cancela desde Training (P3-09).
- **Prohibido:** registrar cualquier métrica `test_*` o cargar el split `test`
  durante el entrenamiento. El test solo se toca después de MODEL SELECTION
  CLOSED (sección 11).

La prueba de contrato de P3-08 corre una corrida corta contra un MLflow
temporal y verifica cada punto de esta lista, incluido el estado `FAILED` ante
un fallo controlado.

## 9. S3 para modelos

- Todos los buckets del proyecto viven en la cuenta de Karen (`222629887955`):
  el remote DVC, los releases y el bucket de modelos. El equipo entra con IAM
  Identity Center (rol `MLOpsP3`).
- Bucket de modelos: `mlops-p3-models-222629887955` (`us-east-1`), con
  versionado activo, acceso público bloqueado, cifrado SSE-S3 y una política
  que rechaza todo acceso sin HTTPS.
- Ruta de cada versión:
  `s3://mlops-p3-models-222629887955/models/clasificador-perro-gato/<semver>/`.
- Nunca se sobrescribe una versión publicada. El rol `MLOpsP3` puede leer y
  subir, pero no puede borrar versiones de objetos, suspender el versionado ni
  cambiar la política del bucket.

## 10. Versiones del modelo

| Versión | Contenido |
|---|---|
| `0.1.0` | Ensayo: la corrida del smoke test, marcada "ensayo, no seleccionada" |
| `1.0.0` | Final: el candidato elegido en P3-11 con las métricas de P3-13 |

## 11. Custodio del test

**Karen.** Guarda el 10% de prueba "bajo llave" y es dueña de la cuenta donde
viven los buckets. Solo ella ejecuta `final.py --split test`, una única vez,
después de declarar **MODEL SELECTION CLOSED**. Uriel, que corre la campaña y
prepara la selección, no toca el test.
