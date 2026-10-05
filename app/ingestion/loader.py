"""Tier 1 — combina los lotes reales de anotaciones en un solo CocoDataset.

Cada `annotations-lote-*.json` en `data/raw/annotations/` es autocontenido
(ver app/ingestion/README.md). `load_raw_dataset` es el único punto donde se
juntan, sin validarlos; `load_dataset` es la puerta estricta del pipeline. Si dos
lotes reintroducen el bug de [[project_dataset_id_collision_bug]] (ids repetidos
entre archivos), `CocoDataset.model_validate` lo rechaza en `load_dataset`, no
río abajo.
"""

import json
from pathlib import Path

from ingestion.models import CocoDataset


def load_raw_dataset(annotations_dir: Path) -> dict:
    """Combina los lotes en un dict COCO crudo, sin validar el formato.

    Deduplica categorías por id y rechaza el mismo id con nombres distintos
    (colisión real). Devuelve el dict tal cual llega (`images`, `annotations` y
    `categories`), para que quien necesite inspeccionar cajas inválidas antes de
    la validación estricta pueda hacerlo. No repara ni descarta nada.
    """
    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No se encontraron lotes de anotaciones en {annotations_dir}")

    images = []
    annotations = []
    categories_by_id: dict[int, dict] = {}
    categories_source: dict[int, Path] = {}
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        images.extend(raw["images"])
        annotations.extend(raw["annotations"])
        for category in raw["categories"]:
            existing = categories_by_id.get(category["id"])
            if existing is not None and existing["name"] != category["name"]:
                raise ValueError(
                    f"category id={category['id']} tiene nombres distintos: "
                    f"'{existing['name']}' en {categories_source[category['id']]} vs "
                    f"'{category['name']}' en {path}"
                )
            categories_by_id.setdefault(category["id"], category)
            categories_source.setdefault(category["id"], path)

    return {
        "images": images,
        "annotations": annotations,
        "categories": list(categories_by_id.values()),
    }


def load_dataset(annotations_dir: Path) -> CocoDataset:
    """Valida el merge crudo de todos los lotes como un solo `CocoDataset`."""
    return CocoDataset.model_validate(load_raw_dataset(annotations_dir))


def load_image_bytes(coco: CocoDataset, images_dir: Path) -> dict[int, bytes]:
    """Lee del disco los binarios de cada imagen declarada en `coco`.

    Compartido por `presentation.gate` y `presentation.release`: ambos
    necesitan los mismos bytes por `image_id` para correr pHash.
    """
    return {image.id: (images_dir / image.file_name).read_bytes() for image in coco.images}
