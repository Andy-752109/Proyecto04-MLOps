"""P3-06: asigna cada recorte a train/val/test sin separar grupos de duplicados.

Reutiliza el split de P2 (`splits.stratified.split_dataset`): agrupa imágenes
por SHA-256, nombre de archivo y pares pHash del analizador de duplicados, y
reparte grupos completos de forma estratificada y con semilla. Aquí la unidad es
la imagen de origen con al menos un recorte válido; cada recorte hereda la
partición y el `duplicate_group_id` de su imagen, así que dos recortes de la
misma foto (o de dos fotos casi iguales) nunca quedan en particiones distintas.

Todo es puro: sin disco ni red. `dvc_manifest_stage.py` hace la I/O.
"""

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from analyzers.invalid_boxes import InvalidBoxConfig, analyze_invalid_boxes
from manifest.check_leakage import find_leakage
from manifest.config import ManifestConfig
from splits.stratified import SPLIT_NAMES, split_dataset

MANIFEST_FIELDS = [
    "crop_id",
    "split",
    "label",
    "category_name",
    "category_id",
    "source_image_id",
    "source_image_file",
    "annotation_id",
    "duplicate_group_id",
    "path",
]


class ManifestError(ValueError):
    """El manifiesto no cumple el contrato (fuga, desviación, catálogo inválido)."""


@dataclass(frozen=True)
class ManifestResult:
    rows: list[dict]
    counts: dict


def group_id(group: Sequence[int]) -> str:
    """Identificador estable de un grupo: el menor `image_id` que contiene."""
    return f"g{min(group):06d}"


def _validate_crops(coco: dict, crops: Sequence[Mapping], config: ManifestConfig, invalid_config):
    invalid = {
        sample["annotation_id"]
        for sample in analyze_invalid_boxes(coco, invalid_config).details["offending_samples"]
    }
    annotations = {annotation["id"]: annotation for annotation in coco["annotations"]}
    for crop in crops:
        annotation_id = int(crop["annotation_id"])
        if annotation_id in invalid:
            raise ManifestError(
                f"El recorte {crop['crop_id']} viene de una caja inválida (anotación "
                f"{annotation_id}); vuelve a correr la etapa `crops` (P3-04)."
            )
        annotation = annotations.get(annotation_id)
        if annotation is None or (
            annotation["image_id"],
            annotation["category_id"],
        ) != (int(crop["source_image_id"]), int(crop["category_id"])):
            raise ManifestError(
                f"El recorte {crop['crop_id']} no coincide con la anotación {annotation_id} "
                "del release; el catálogo de recortes está desfasado."
            )
        if crop["category_name"] not in config.labels:
            raise ManifestError(
                f"El recorte {crop['crop_id']} es de la clase '{crop['category_name']}', "
                f"que no está en classes={config.classes}."
            )
    crop_ids = [crop["crop_id"] for crop in crops]
    if len(set(crop_ids)) != len(crop_ids):
        raise ManifestError("El catálogo de recortes repite crop_id.")


def _subset(coco: dict, crops: Sequence[Mapping]) -> dict:
    """COCO reducido a las imágenes con recorte y a las anotaciones recortadas."""
    image_ids = {int(crop["source_image_id"]) for crop in crops}
    annotation_ids = {int(crop["annotation_id"]) for crop in crops}
    category_ids = {int(crop["category_id"]) for crop in crops}
    return {
        "images": [image for image in coco["images"] if image["id"] in image_ids],
        "annotations": [a for a in coco["annotations"] if a["id"] in annotation_ids],
        "categories": [c for c in coco["categories"] if c["id"] in category_ids],
    }


def build_manifest(
    coco: dict,
    crops: Sequence[Mapping],
    *,
    image_contents: Mapping[int, bytes],
    duplicate_pairs: Sequence[Mapping],
    config: ManifestConfig,
    invalid_config: InvalidBoxConfig,
) -> ManifestResult:
    """Devuelve las filas del manifiesto (ordenadas por `crop_id`) y sus conteos.

    `duplicate_pairs` son los `image_pairs` de `analyze_duplicates` sobre estas
    mismas imágenes. Falla con `ManifestError` si hay fuga, si una clase falta
    en validación o test, o si una partición se desvía más de
    `max_deviation_pp` del objetivo (en total o dentro de cada clase).
    """
    if not crops:
        raise ManifestError("El catálogo de recortes está vacío.")
    _validate_crops(coco, crops, config, invalid_config)

    subset = _subset(coco, crops)
    kept = {image["id"] for image in subset["images"]}
    split = split_dataset(
        subset,
        config.to_splits_config(),
        image_contents={i: content for i, content in image_contents.items() if i in kept},
        duplicate_pairs=[
            pair
            for pair in duplicate_pairs
            if pair["image_id_a"] in kept and pair["image_id_b"] in kept
        ],
    )
    owner = {i: name for name, ids in split.assignments.items() for i in ids}
    groups = {i: group_id(group) for group in split.groups for i in group}

    rows = sorted(
        (
            {
                "crop_id": crop["crop_id"],
                "split": owner[int(crop["source_image_id"])],
                "label": config.labels[crop["category_name"]],
                "category_name": crop["category_name"],
                "category_id": int(crop["category_id"]),
                "source_image_id": int(crop["source_image_id"]),
                "source_image_file": crop["source_image_file"],
                "annotation_id": int(crop["annotation_id"]),
                "duplicate_group_id": groups[int(crop["source_image_id"])],
                "path": crop["path"],
            }
            for crop in crops
        ),
        key=lambda row: row["crop_id"],
    )

    leakage = find_leakage(rows)
    if any(leakage.values()):
        raise ManifestError(f"Fuga entre particiones: {leakage}")
    counts = compute_counts(rows, config)
    _check_counts(counts, config)
    return ManifestResult(rows=rows, counts=counts)


def compute_counts(rows: Sequence[Mapping], config: ManifestConfig) -> dict:
    """Recortes y originales por partición y clase, con su desviación en pp."""
    total = len(rows)
    per_class_total = {
        name: sum(row["category_name"] == name for row in rows) for name in config.classes
    }
    splits = {}
    for name in SPLIT_NAMES:
        split_rows = [row for row in rows if row["split"] == name]
        fraction = len(split_rows) / total
        classes = {}
        for class_name in config.classes:
            class_rows = [row for row in split_rows if row["category_name"] == class_name]
            class_fraction = (
                len(class_rows) / per_class_total[class_name] if per_class_total[class_name] else 0
            )
            classes[class_name] = {
                "crops": len(class_rows),
                "originals": len({row["source_image_id"] for row in class_rows}),
                "crops_fraction_of_class": round(class_fraction, 6),
                "deviation_pp": round((class_fraction - config.ratios[name]) * 100, 4),
            }
        splits[name] = {
            "target_fraction": config.ratios[name],
            "crops": len(split_rows),
            "originals": len({row["source_image_id"] for row in split_rows}),
            "duplicate_groups": len({row["duplicate_group_id"] for row in split_rows}),
            "crops_fraction": round(fraction, 6),
            "deviation_pp": round((fraction - config.ratios[name]) * 100, 4),
            "classes": classes,
        }
    return {
        "release": config.release,
        "seed": config.seed,
        "max_deviation_pp": config.max_deviation_pp,
        "totals": {
            "crops": total,
            "originals": len({row["source_image_id"] for row in rows}),
            "duplicate_groups": len({row["duplicate_group_id"] for row in rows}),
            "classes": {
                class_name: {
                    "crops": per_class_total[class_name],
                    "originals": len(
                        {r["source_image_id"] for r in rows if r["category_name"] == class_name}
                    ),
                }
                for class_name in config.classes
            },
        },
        "splits": splits,
    }


def _check_counts(counts: dict, config: ManifestConfig) -> None:
    problems = []
    for name, summary in counts["splits"].items():
        if abs(summary["deviation_pp"]) > config.max_deviation_pp:
            problems.append(f"{name}: {summary['deviation_pp']:+.2f} pp en total")
        for class_name, per_class in summary["classes"].items():
            if name != "train" and per_class["crops"] == 0:
                problems.append(f"{name}: la clase '{class_name}' no tiene recortes")
            if abs(per_class["deviation_pp"]) > config.max_deviation_pp:
                problems.append(f"{name}/{class_name}: {per_class['deviation_pp']:+.2f} pp")
    if problems:
        raise ManifestError(
            f"El reparto excede ±{config.max_deviation_pp} pp o deja clases vacías: "
            + "; ".join(problems)
        )


def manifest_csv_bytes(rows: Sequence[Mapping]) -> bytes:
    """CSV canónico (UTF-8, `\\n`, columnas fijas) para que el SHA-256 sea estable."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
    return buffer.getvalue().encode("utf-8")
