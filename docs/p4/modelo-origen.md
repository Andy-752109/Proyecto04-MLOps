# Modelo de origen (P3) — ficha del ancla de P4

Ticket P4-01. Este es el modelo del que debe derivar la variante de P4 (M1) y la línea base de la
comparación de calidad. Todo lo de abajo se recuperó y verificó contra S3 y DVC el 2026-10-05; no
se reentrenó ni se volvió a correr `final.py --split test`.

## 1. Referencia P3

| Dato | Valor |
|---|---|
| Repositorio P3 | `proyecto-fase3-MLOPS` (base de este repo; `main` sin historial) |
| Versión del modelo | `1.0.0` (`model_release` v1.0.0, `run_kind=campaign`, fila de rejilla `r02`) |
| `run_id` (MLflow) | `7e7b4a4b35464cfebb6b41714a3ad931` |
| Release de datos | `v0.1.1` |
| Manifiesto | `v0.1.1-53fc84fdaa07` |
| SHA-256 de `manifest.csv` | `53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f` |
| Recortes del manifiesto | 668 = 464 train / 131 val / 73 test |
| Selección | `reports/selection.json` (`selected_at` 2026-10-02T05:43:20Z, métrica `best_val_accuracy`) |

## 2. Paquete y checkpoint

| Dato | Valor |
|---|---|
| Paquete S3 | `s3://mlops-p3-models-222629887955/models/releases/v1.0.0/model_release_v1.0.0.tar.gz` |
| `VersionId` | `V8H6fouL5HiaUvrxAz1gUI5RbRD008VH` |
| SHA-256 del paquete | `bb93a41c8f83a2e42b5f3c337136804f9734687a8ba58360308945347cdfa270` |
| Checkpoint | `artifacts/checkpoint/best.pt` (`state_dict` de PyTorch, se carga con `weights_only=True`) |
| SHA-256 de `best.pt` | `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9` |
| Tamaño de `best.pt` | **45 304 641 bytes** (≈ 43.2 MiB) |

## 3. Arquitectura e hiperparámetros

ResNet-18 (ImageNet `IMAGENET1K_V1`) con la cabeza reemplazada:
`Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 2)`. Solo `conv1`, `bn1`, `layer1–3`
estuvieron congeladas durante el entrenamiento; el checkpoint contiene todos los pesos.

`config.json`: `optimizer=adam`, `learning_rate=0.0003`, `batch_size=32`, `max_epochs=15`,
`patience=3`, `min_delta=0.001`, `image_size=128`, `hidden_layers=1`, `dropout=0.3`, semillas
`seed_split=42`, `seed_train=43`, `seed_aug=44`, `seed_model=45`.

## 4. Clases y orden

`class_map.json`: `{"0": "cat", "1": "dog"}`. El logit 0 es `cat` y el logit 1 es `dog`
(coincide con `classes` de `app/manifest/manifest.yaml`). En los CSV, `prob_cat` es la
probabilidad del índice 0 y `prob_dog` la del índice 1.

## 5. Preprocesamiento exacto (inferencia)

1. Abrir el recorte y convertir a **RGB**.
2. `Resize((128, 128))` **bilineal con antialias** (PIL; `antialias=True` en torchvision). Estira
   al cuadrado: **no** conserva la relación de aspecto y **no** hay *crop*.
3. `ToTensor()` (escala a [0, 1], formato CHW).
4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet).
5. Tensor de entrada `[1, 3, 128, 128]` (float32) → modelo → **2 logits** → `softmax` sobre la
   dimensión de clases → `[prob_cat, prob_dog]`; la clase predicha es el `argmax`.

Fuente: `preprocess.json` del paquete y `app/training/preprocess.py` (rama no-`train`).
`app/edge_model/baseline.py` replica estos pasos.

## 6. Línea base de validación

Producida por `app/edge_model/baseline.py` (sin MLflow, desde el paquete 1.0.0, 1 hilo de CPU,
torch 2.14.0):

| Métrica | Valor |
|---|---|
| Recortes de `val` | 131 (IDs únicos, ambas clases) |
| Aciertos | 126 / 131 |
| Accuracy recalculada | **0.9618320610687023** |
| `best_val_accuracy` (`selection.json`) | 0.9618320610687023 — coincide exactamente |
| Errores | 5 (3 gato→perro, 2 perro→gato) |

Archivos:

- `reports/p4/quality/predictions_val_original.csv` — `crop_id, source_image_id, true_class,
  predicted_class, prob_cat, prob_dog` (131 filas).
- `reports/p4/reference/parity_reference.csv` — 20 recortes de val (10 por clase verdadera,
  semilla 42) con `crop_id, path, true_class, predicted_class, prob_cat, prob_dog`; `path` es
  relativo a `data/derived/crops/`. Para las pruebas de paridad de edge y de conversión.
- `reports/p4/evidence/` — salida de `verify_reload`, `sha256sum` del manifiesto y resumen JSON.

## 7. Cómo reproducirlo

```bash
cd app
uv sync --group ml
uv run python verify_reload.py --version 1.0.0      # descarga por VersionId y extrae a eval_v1.0.0/
cd .. && dvc pull -r prod data/raw/annotations.dvc data/raw/images.dvc
dvc repro crops manifest && sha256sum data/derived/manifests/v0.1.1/manifest.csv
cd app && uv run python -m edge_model.baseline
```

Requiere sesión SSO vigente (`aws sso login --profile mlops-p3`, y `AWS_PROFILE=mlops-p3` para DVC).
