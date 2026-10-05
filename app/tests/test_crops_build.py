import csv
import json
from pathlib import Path

from PIL import Image

from crops.audit import main as audit_main
from crops.audit import select_audit_samples
from crops.build import (
    CATALOG_FILENAME,
    EXCLUSIONS_FILENAME,
    build_crops,
    main,
    make_crop_id,
    plan_crops,
)
from policies.invalid_boxes import load_invalid_box_config


def _document() -> dict:
    """COCO de fixture: recortes válidos, caja degenerada, fuera de imagen y dos cajas."""
    return {
        "images": [
            {"id": 3, "file_name": "cat.0.jpg", "width": 100, "height": 80},
            {"id": 4, "file_name": "cat.1.jpg", "width": 100, "height": 80},
            {"id": 5, "file_name": "dog.0.jpg", "width": 100, "height": 80},
            {"id": 6, "file_name": "dog.1.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 3,
                "category_id": 4,
                "bbox": [0, 0, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 4,
                "category_id": 4,
                "bbox": [0, 0, 0, 10],
                "area": 0.0,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 5,
                "category_id": 3,
                "bbox": [0, 0, 20, 20],
                "area": 400.0,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 5,
                "category_id": 3,
                "bbox": [20, 20, 20, 20],
                "area": 400.0,
                "iscrowd": 0,
            },
            {
                "id": 5,
                "image_id": 6,
                "category_id": 3,
                "bbox": [0, 0, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
            {
                "id": 6,
                "image_id": 5,
                "category_id": 3,
                "bbox": [95, 70, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
        ],
    }


def _write_jpeg(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    Image.new("RGB", size, color=(90, 90, 90)).save(path, format="JPEG")


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    (annotations_dir / "lote.json").write_text(json.dumps(_document()), encoding="utf-8")
    # La imagen 6 (dog.1.jpg) falta a propósito.
    for file_name in ("cat.0.jpg", "cat.1.jpg", "dog.0.jpg"):
        _write_jpeg(images_dir / file_name)
    return annotations_dir, images_dir


def _reasons(exclusions: list[dict]) -> dict[int, str]:
    return {row["annotation_id"]: row["reason"] for row in exclusions}


def test_degenerate_box_is_excluded_with_reason():
    crops, exclusions = plan_crops(_document(), load_invalid_box_config())

    reasons = _reasons(exclusions)
    assert "non_positive_width" in reasons[2]
    assert 2 not in {crop["annotation_id"] for crop in crops}


def test_out_of_image_box_is_excluded_with_reason():
    _, exclusions = plan_crops(_document(), load_invalid_box_config())

    assert _reasons(exclusions)[6] == "exceeds_image_width"


def test_two_valid_boxes_on_one_image_produce_two_crops():
    crops, _ = plan_crops(_document(), load_invalid_box_config())

    on_image_five = [crop for crop in crops if crop["source_image_id"] == 5]
    assert [crop["annotation_id"] for crop in on_image_five] == [3, 4]


def test_area_mismatch_is_excluded_like_the_gate_does():
    coco = _document()
    coco["annotations"][0]["area"] = 999.0  # única caja de gato de la imagen 3

    crops, exclusions = plan_crops(coco, load_invalid_box_config())

    assert _reasons(exclusions)[1] == "area_mismatch"
    assert 1 not in {crop["annotation_id"] for crop in crops}


def test_crop_ids_are_deterministic_and_sorted():
    coco = _document()
    config = load_invalid_box_config()
    first_crops, first_exclusions = plan_crops(coco, config)
    second_crops, second_exclusions = plan_crops(coco, config)

    assert first_crops == second_crops
    assert first_exclusions == second_exclusions
    assert make_crop_id(5, 3) == "000005_000003"
    assert [crop["crop_id"] for crop in first_crops] == sorted(
        crop["crop_id"] for crop in first_crops
    )


def test_missing_image_file_is_excluded_without_crashing(tmp_path):
    annotations_dir, images_dir = _write_dataset(tmp_path)
    coco = json.loads((annotations_dir / "lote.json").read_text(encoding="utf-8"))
    output_dir = tmp_path / "derived"

    summary = build_crops(coco, images_dir, output_dir, load_invalid_box_config())

    assert summary["crops"] == 3
    exclusions = list(csv.DictReader((output_dir / EXCLUSIONS_FILENAME).open()))
    reasons = {int(row["annotation_id"]): row["reason"] for row in exclusions}
    assert reasons[5] == "missing_image_file"


def test_build_writes_catalog_and_crops_with_pixel_dimensions(tmp_path):
    annotations_dir, images_dir = _write_dataset(tmp_path)
    coco = json.loads((annotations_dir / "lote.json").read_text(encoding="utf-8"))
    output_dir = tmp_path / "derived"

    build_crops(coco, images_dir, output_dir, load_invalid_box_config())

    with (output_dir / CATALOG_FILENAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    row = next(r for r in rows if r["annotation_id"] == "3")
    assert row["path"] == "images/000005_000003.jpg"
    assert row["category_name"] == "dog"
    assert row["source_image_file"] == "dog.0.jpg"

    with Image.open(output_dir / row["path"]) as crop:
        assert crop.size == (20, 20)


def test_cli_builds_crops_and_reports_a_summary(tmp_path, capsys):
    annotations_dir, images_dir = _write_dataset(tmp_path)
    output_dir = tmp_path / "derived"

    exit_code = main(
        [
            "--annotations-dir",
            str(annotations_dir),
            "--images-dir",
            str(images_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["crops"] == 3
    assert summary["exclusions"] == 3


def test_audit_sample_is_deterministic():
    crops, _ = plan_crops(_document(), load_invalid_box_config())
    catalog = crops * 3  # 12 candidatos para muestrear 10

    first = select_audit_samples(catalog, count=10, seed=42)
    second = select_audit_samples(catalog, count=10, seed=42)

    assert first == second


def test_audit_writes_markdown_and_contact_sheet(tmp_path):
    annotations_dir, images_dir = _write_dataset(tmp_path)
    coco = json.loads((annotations_dir / "lote.json").read_text(encoding="utf-8"))
    output_dir = tmp_path / "derived"
    reports_dir = tmp_path / "reports"
    build_crops(coco, images_dir, output_dir, load_invalid_box_config())

    exit_code = audit_main(
        [
            "--crops-dir",
            str(output_dir),
            "--images-dir",
            str(images_dir),
            "--reports-dir",
            str(reports_dir),
        ]
    )

    assert exit_code == 0
    assert (reports_dir / "crop_audit.png").is_file()
    text = (reports_dir / "crop_audit.md").read_text(encoding="utf-8")
    assert "dog.0.jpg" in text
    assert "Firmado por:" in text
