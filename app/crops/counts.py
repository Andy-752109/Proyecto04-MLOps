"""P3-02: conteo de imágenes originales por clase tras descartar cajas inválidas.

La compuerta (`analyzers.imbalance`) cuenta imágenes por categoría sin mirar la
validez de cada caja. Para los "conteos útiles" de P3-02 hay que descartar
primero toda anotación con caja inválida: una imagen cuyas únicas cajas de una
clase son inválidas no cuenta para esa clase (rúbrica 1.1 y 1.2). La validez la
define el mismo analizador que alimenta la compuerta (`analyzers.invalid_boxes`),
para no mantener dos criterios distintos.
"""

import argparse
import json
import sys
from pathlib import Path

from analyzers.invalid_boxes import InvalidBoxConfig, analyze_invalid_boxes
from ingestion.loader import load_raw_dataset
from policies.invalid_boxes import load_invalid_box_config


def count_original_images_per_class(coco: dict, config: InvalidBoxConfig) -> dict:
    """Cuenta, por categoría, las imágenes originales con al menos una caja válida.

    Devuelve `categories` (una entrada por categoría, ordenada por
    `category_id`, igual que la compuerta), más `total_original_images`
    (imágenes distintas con alguna caja válida) e `invalid_annotations`
    (anotaciones descartadas). No muta `coco`.
    """
    invalid = {
        sample["annotation_id"]
        for sample in analyze_invalid_boxes(coco, config).details["offending_samples"]
    }

    images_by_category = {category["id"]: set() for category in coco["categories"]}
    valid_images: set = set()
    for annotation in coco["annotations"]:
        category_id = annotation["category_id"]
        if category_id not in images_by_category:
            raise ValueError(f"Unknown category_id: {category_id}")
        if annotation["id"] in invalid:
            continue
        images_by_category[category_id].add(annotation["image_id"])
        valid_images.add(annotation["image_id"])

    ordered = sorted(coco["categories"], key=lambda category: category["id"])
    categories = [
        {
            "category_id": category["id"],
            "category_name": category["name"],
            "image_count": len(images_by_category[category["id"]]),
            "image_ids": sorted(images_by_category[category["id"]]),
        }
        for category in ordered
    ]
    return {
        "categories": categories,
        "total_original_images": len(valid_images),
        "invalid_annotations": len(invalid),
    }


def main(argv: list[str] | None = None) -> int:
    """Imprime los conteos como JSON; pensado para el comando de verificación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directorio con los lotes de anotaciones COCO (`*.json`).",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="`quality.yaml` alterno; por defecto el del paquete de políticas.",
    )
    args = parser.parse_args(argv)

    coco = load_raw_dataset(args.annotations_dir)
    counts = count_original_images_per_class(coco, load_invalid_box_config(args.quality))
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
