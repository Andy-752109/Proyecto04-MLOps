"""P3-04: un recorte por caja válida, con procedencia completa.

Reutiliza el criterio de validez de la compuerta y del conteo de P3-02
(`analyzers.invalid_boxes`): una caja degenerada, fuera de la imagen, con
`area_mismatch` o que referencia una imagen inexistente se descarta y queda
registrada en `exclusions.csv`. Un archivo de imagen ausente en disco también
se excluye (`missing_image_file`) sin reventar.

`plan_crops` es puro (fácil de probar); `build_crops` materializa los recortes
y escribe el catálogo y las exclusiones.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image

from analyzers.invalid_boxes import InvalidBoxConfig, analyze_invalid_boxes
from ingestion.loader import load_raw_dataset
from policies.invalid_boxes import load_invalid_box_config

CATALOG_FILENAME = "crops.csv"
EXCLUSIONS_FILENAME = "exclusions.csv"
CROPS_SUBDIR = "images"
JPEG_QUALITY = 95

CATALOG_FIELDS = [
    "crop_id",
    "source_image_id",
    "source_image_file",
    "annotation_id",
    "category_id",
    "category_name",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "crop_width",
    "crop_height",
    "path",
]
EXCLUSION_FIELDS = [
    "annotation_id",
    "source_image_id",
    "category_id",
    "image_file",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "reason",
]


def make_crop_id(source_image_id: int, annotation_id: int) -> str:
    """Identificador determinístico y único a partir de los ids COCO."""
    return f"{source_image_id:06d}_{annotation_id:06d}"


def _pixel_bounds(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convierte una bbox COCO `[x, y, w, h]` en píxeles, recortada a la imagen."""
    x, y, w, h = bbox
    x0 = max(0, math.floor(x))
    y0 = max(0, math.floor(y))
    x1 = min(width, math.ceil(x + w))
    y1 = min(height, math.ceil(y + h))
    return x0, y0, x1, y1


def _exclusion_row(annotation: dict, image: dict | None, reason: str) -> dict:
    x, y, w, h = annotation["bbox"]
    return {
        "annotation_id": annotation["id"],
        "source_image_id": annotation["image_id"],
        "category_id": annotation["category_id"],
        "image_file": image["file_name"] if image is not None else "",
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "reason": reason,
    }


def plan_crops(
    coco: dict,
    config: InvalidBoxConfig,
    *,
    available_image_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Decide qué cajas producen recorte y cuáles se excluyen, sin tocar disco.

    `available_image_ids`, si se da, marca qué imágenes existen en disco: las
    anotaciones cuya imagen falta se excluyen con `missing_image_file`. Devuelve
    `(crops, exclusions)` ordenados de forma determinística.
    """
    invalid = {
        sample["annotation_id"]: sample
        for sample in analyze_invalid_boxes(coco, config).details["offending_samples"]
    }
    images = {image["id"]: image for image in coco["images"]}
    categories = {category["id"]: category["name"] for category in coco["categories"]}

    crops: list[dict] = []
    exclusions: list[dict] = []
    for annotation in coco["annotations"]:
        image = images.get(annotation["image_id"])
        if available_image_ids is not None and annotation["image_id"] not in available_image_ids:
            exclusions.append(_exclusion_row(annotation, image, "missing_image_file"))
            continue
        if annotation["id"] in invalid:
            reason = ";".join(invalid[annotation["id"]]["reasons"])
            exclusions.append(_exclusion_row(annotation, image, reason))
            continue

        x0, y0, x1, y1 = _pixel_bounds(annotation["bbox"], image["width"], image["height"])
        if x1 <= x0 or y1 <= y0:
            exclusions.append(_exclusion_row(annotation, image, "empty_crop"))
            continue

        bbox_x, bbox_y, bbox_w, bbox_h = annotation["bbox"]
        crops.append(
            {
                "crop_id": make_crop_id(annotation["image_id"], annotation["id"]),
                "source_image_id": annotation["image_id"],
                "source_image_file": image["file_name"],
                "annotation_id": annotation["id"],
                "category_id": annotation["category_id"],
                "category_name": categories.get(annotation["category_id"], ""),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "crop_width": x1 - x0,
                "crop_height": y1 - y0,
            }
        )

    crops.sort(key=lambda crop: (crop["source_image_id"], crop["annotation_id"]))
    exclusions.sort(key=lambda row: row["annotation_id"])
    return crops, exclusions


def _write_catalog(path: Path, crops: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        for crop in crops:
            writer.writerow({field: crop[field] for field in CATALOG_FIELDS})


def _write_exclusions(path: Path, exclusions: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(exclusions)


def build_crops(
    coco: dict,
    images_dir: Path,
    output_dir: Path,
    config: InvalidBoxConfig,
) -> dict:
    """Materializa un recorte por caja válida y escribe el catálogo y las exclusiones."""
    available = {
        image["id"] for image in coco["images"] if (images_dir / image["file_name"]).is_file()
    }
    crops, exclusions = plan_crops(coco, config, available_image_ids=available)
    images = {image["id"]: image for image in coco["images"]}

    (output_dir / CROPS_SUBDIR).mkdir(parents=True, exist_ok=True)
    for crop in crops:
        source = images[crop["source_image_id"]]
        with Image.open(images_dir / source["file_name"]) as image:
            cropped = image.crop((crop["x0"], crop["y0"], crop["x1"], crop["y1"]))
            relative = f"{CROPS_SUBDIR}/{crop['crop_id']}.jpg"
            cropped.save(output_dir / relative, format="JPEG", quality=JPEG_QUALITY)
        crop["path"] = relative

    _write_catalog(output_dir / CATALOG_FILENAME, crops)
    _write_exclusions(output_dir / EXCLUSIONS_FILENAME, exclusions)
    return {
        "crops": len(crops),
        "exclusions": len(exclusions),
        "categories": _category_counts(crops),
        "catalog": str(output_dir / CATALOG_FILENAME),
        "exclusions_csv": str(output_dir / EXCLUSIONS_FILENAME),
    }


def _category_counts(crops: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for crop in crops:
        counts[crop["category_name"]] = counts.get(crop["category_name"], 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    """Materializa los recortes desde la línea de comandos (evidencia de P3-04)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path, required=True, help="Lotes COCO (`*.json`)."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Imágenes originales.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destino de los recortes.")
    parser.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="`quality.yaml` alterno; por defecto el del paquete de políticas.",
    )
    args = parser.parse_args(argv)

    coco = load_raw_dataset(args.annotations_dir)
    summary = build_crops(
        coco, args.images_dir, args.output_dir, load_invalid_box_config(args.quality)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
