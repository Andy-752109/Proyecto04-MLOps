# P3-06 — Manifiesto 70/20/10 sin fuga

Asigna cada recorte de P3-04 a `train`, `val` o `test`. Es la **única** fuente
de particiones para el entrenamiento de P3; el 70/15/15 de P2
(`app/splits/splits.yaml`, `reports/splits.json`) no se toca.

```bash
dvc repro manifest            # corre `crops` si hace falta y luego el manifiesto
cd app && uv run python -m manifest.check_leakage \
  --release v0.1.1            # código ≠ 0 si hay fuga
dvc push -r prod              # sube manifest.csv (y los recortes) al remote
```

## Cómo reparte

1. Parte del catálogo `data/derived/crops/crops.csv`: las cajas inválidas ya
   se excluyeron en P3-04. Si el catálogo trae una caja que el analizador
   marca como inválida, la etapa falla en lugar de repartirla.
2. Agrupa las imágenes de origen con `splits.stratified.split_dataset` de P2:
   copias exactas (SHA-256), mismo nombre de archivo y pares pHash del
   analizador de duplicados (umbral `duplicate_similarity_threshold` de
   `policies/quality.yaml`). Reparte **grupos completos**, estratificado por
   clase y con la semilla 42.
3. Cada recorte hereda la partición y el `duplicate_group_id` de su imagen: dos
   recortes de la misma foto nunca quedan separados.
4. Falla si una partición se desvía más de **±5 pp** del objetivo (en total o
   dentro de cada clase), si una clase falta en `val` o `test`, o si
   `check_leakage` encuentra intersecciones.

## Contrato de `manifest.csv`

Una fila por recorte, ordenadas por `crop_id`, UTF-8 con `\n`. Sin fechas ni
rutas absolutas: la misma entrada da el mismo SHA-256.

| Columna | Qué es |
|---|---|
| `crop_id` | `<source_image_id>_<annotation_id>` con 6 dígitos cada uno (P3-04) |
| `split` | `train`, `val` o `test` |
| `label` | Índice del modelo: `0` = cat, `1` = dog (orden de `classes`) |
| `category_name` | `cat` o `dog` |
| `category_id` | Id COCO (`4` = cat, `3` = dog); **no** es la etiqueta del modelo |
| `source_image_id`, `source_image_file` | Imagen de origen |
| `annotation_id` | Caja COCO del recorte |
| `duplicate_group_id` | `g` + menor `image_id` del grupo, con 6 dígitos |
| `path` | Ruta del recorte **relativa a `crops_root`** (`data/derived/crops`) |

## Archivos que genera

| Archivo | Dónde vive |
|---|---|
| `data/derived/manifests/<release>/manifest.csv` | DVC (`dvc push`) |
| `reports/manifests/<release>/manifest_meta.json` | Git: `manifest_id`, SHA-256, release y sus md5, referencia de calidad, semilla, proporciones y clases |
| `reports/manifests/<release>/counts.json` | Git: recortes y originales por clase y partición, con su desviación en pp |

Los parámetros viven en `manifest/manifest.yaml`. **Congelado:** si cambian
después de la primera corrida de entrenamiento, se repiten las corridas.
