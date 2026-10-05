"""P3-04: auditoría de recortes contra su caja COCO.

Toma una muestra aleatoria con semilla fija del catálogo (`crops.csv`) y genera
`reports/crop_audit.md` (tabla con procedencia) y `reports/crop_audit.png` (cada
fila compara la imagen de origen con la caja COCO dibujada contra el recorte).
La comparación visual y la firma son humanas.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from crops.build import CATALOG_FILENAME

AUDIT_SEED = 42
AUDIT_SAMPLES = 10
THUMB_SIZE = (160, 140)
ROW_LABEL_HEIGHT = 16
GAP = 8


def read_catalog(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_audit_samples(
    crops: list[dict], count: int = AUDIT_SAMPLES, seed: int = AUDIT_SEED
) -> list[dict]:
    """Muestra determinística: misma lista y misma semilla dan los mismos recortes.

    Ordena por un hash `sha256` que incluye la semilla, en vez de `random`, para
    no depender del generador pseudoaleatorio (evita el hallazgo S2245).
    """
    if len(crops) <= count:
        return list(crops)

    def rank(crop: dict) -> str:
        return hashlib.sha256(f"{seed}:{crop['crop_id']}".encode()).hexdigest()

    return sorted(crops, key=rank)[:count]


def _source_with_box(sample: dict, images_dir: Path) -> Image.Image:
    source = Image.open(images_dir / sample["source_image_file"]).convert("RGB")
    x, y, w, h = (
        float(sample["bbox_x"]),
        float(sample["bbox_y"]),
        float(sample["bbox_w"]),
        float(sample["bbox_h"]),
    )
    draw = ImageDraw.Draw(source)
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
    return source


def build_contact_sheet(
    samples: list[dict], images_dir: Path, crops_dir: Path, output_png: Path
) -> None:
    strips = []
    for sample in samples:
        strip = Image.new("RGB", (2 * 160 + 3 * GAP, 140 + ROW_LABEL_HEIGHT), "white")
        ImageDraw.Draw(strip).text(
            (GAP, 2), f"{sample['crop_id']} · {sample['source_image_file']}", fill="black"
        )
        source = _source_with_box(sample, images_dir)
        source.thumbnail(THUMB_SIZE)
        crop = Image.open(crops_dir / f"{sample['crop_id']}.jpg").convert("RGB")
        crop.thumbnail(THUMB_SIZE)
        strip.paste(source, (GAP, ROW_LABEL_HEIGHT))
        strip.paste(crop, (2 * GAP + 160, ROW_LABEL_HEIGHT))
        strips.append(strip)

    width = strips[0].width if strips else 1
    height = sum(strip.height for strip in strips) or 1
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for strip in strips:
        canvas.paste(strip, (0, offset))
        offset += strip.height
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, format="PNG")


def build_audit_markdown(samples: list[dict], total: int) -> str:
    lines = [
        "# Auditoría de recortes COCO (P3-04)",
        "",
        f"Muestra de **{len(samples)}** recortes al azar (semilla `{AUDIT_SEED}`) "
        f"de **{total}** recortes válidos. Cada fila compara la imagen de origen con "
        "su caja COCO dibujada (arriba) contra el recorte generado (abajo).",
        "",
        "![Auditoría de recortes](crop_audit.png)",
        "",
        "| # | crop_id | origen | annotation_id | categoría | bbox (x, y, w, h) | recorte |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, sample in enumerate(samples, start=1):
        bbox = f"({sample['bbox_x']}, {sample['bbox_y']}, {sample['bbox_w']}, {sample['bbox_h']})"
        lines.append(
            f"| {index} | `{sample['crop_id']}` | `{sample['source_image_file']}` | "
            f"{sample['annotation_id']} | {sample['category_name']} | {bbox} | "
            f"{sample['crop_width']}x{sample['crop_height']} |"
        )
    lines += [
        "",
        "**Firmado por:** ______________________  **Fecha:** ______________",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-dir", type=Path, required=True, help="Salida de build_crops.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Imágenes originales.")
    parser.add_argument("--reports-dir", type=Path, required=True, help="Destino de la auditoría.")
    args = parser.parse_args(argv)

    crops = read_catalog(args.crops_dir / CATALOG_FILENAME)
    samples = select_audit_samples(crops)
    build_contact_sheet(
        samples, args.images_dir, args.crops_dir / "images", args.reports_dir / "crop_audit.png"
    )
    (args.reports_dir / "crop_audit.md").write_text(
        build_audit_markdown(samples, len(crops)), encoding="utf-8"
    )
    print(f"Auditoría con {len(samples)} de {len(crops)} recortes en {args.reports_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
