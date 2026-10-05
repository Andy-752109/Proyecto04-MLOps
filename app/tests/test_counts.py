import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from crops.counts import count_original_images_per_class, main
from ingestion.loader import load_dataset, load_raw_dataset
from policies.invalid_boxes import load_invalid_box_config


def _document() -> dict:
    """COCO de fixture: una imagen por caso de conteo.

    - image 1: una caja de perro válida.
    - image 2: su única caja de perro es degenerada (no cuenta para perro);
      una caja de gato válida sí cuenta.
    - image 3: dos cajas de perro válidas -> la imagen cuenta una sola vez.
    - image 4: su única caja de gato es inválida -> no cuenta para gato y la
      imagen desaparece por completo.
    """
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80},
            {"id": 3, "file_name": "c.jpg", "width": 100, "height": 80},
            {"id": 4, "file_name": "d.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": [0, 0, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 3,
                "bbox": [0, 0, 0, 10],
                "area": 0.0,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 4,
                "bbox": [0, 0, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 3,
                "bbox": [0, 0, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
            {
                "id": 5,
                "image_id": 3,
                "category_id": 3,
                "bbox": [20, 20, 10, 10],
                "area": 100.0,
                "iscrowd": 0,
            },
            {
                "id": 6,
                "image_id": 4,
                "category_id": 4,
                "bbox": [0, 0, -5, 10],
                "area": 0.0,
                "iscrowd": 0,
            },
        ],
    }


def test_counts_images_per_class_after_invalid_filter():
    result = count_original_images_per_class(_document(), load_invalid_box_config())

    assert [category["category_id"] for category in result["categories"]] == [3, 4]
    assert result["categories"][0]["category_name"] == "dog"
    assert result["categories"][0]["image_count"] == 2
    assert result["categories"][0]["image_ids"] == [1, 3]
    assert result["categories"][1]["category_name"] == "cat"
    assert result["categories"][1]["image_count"] == 1
    assert result["categories"][1]["image_ids"] == [2]
    assert result["total_original_images"] == 3
    assert result["invalid_annotations"] == 2


def test_image_with_only_invalid_boxes_disappears_entirely():
    result = count_original_images_per_class(_document(), load_invalid_box_config())
    counted_images = {
        image_id for category in result["categories"] for image_id in category["image_ids"]
    }
    assert 4 not in counted_images


def test_two_boxes_of_the_same_class_count_the_image_once():
    result = count_original_images_per_class(_document(), load_invalid_box_config())
    dog = result["categories"][0]
    assert dog["image_ids"].count(3) == 1


def test_area_mismatch_discards_the_box_like_the_gate_does():
    coco = _document()
    coco["annotations"][0]["area"] = 999.0  # única caja de perro de la image 1
    result = count_original_images_per_class(coco, load_invalid_box_config())
    assert result["categories"][0]["image_ids"] == [3]


def test_unknown_category_id_is_rejected():
    coco = _document()
    coco["annotations"].append(
        {
            "id": 7,
            "image_id": 1,
            "category_id": 99,
            "bbox": [0, 0, 10, 10],
            "area": 100.0,
            "iscrowd": 0,
        }
    )
    config = load_invalid_box_config()
    with pytest.raises(ValueError, match="Unknown category_id"):
        count_original_images_per_class(coco, config)


def test_invalid_box_with_unknown_category_is_also_rejected():
    coco = _document()
    coco["annotations"][1]["category_id"] = 99  # caja degenerada + categoría desconocida
    config = load_invalid_box_config()
    with pytest.raises(ValueError, match="Unknown category_id"):
        count_original_images_per_class(coco, config)


def test_malformed_bbox_is_rejected():
    coco = _document()
    coco["annotations"][0]["bbox"] = [0, 0, float("nan"), 10]
    config = load_invalid_box_config()
    with pytest.raises(ValueError, match="finite bbox"):
        count_original_images_per_class(coco, config)


def test_no_annotations_gives_zero_counts():
    coco = _document()
    coco["annotations"] = []
    result = count_original_images_per_class(coco, load_invalid_box_config())
    assert all(category["image_count"] == 0 for category in result["categories"])
    assert result["total_original_images"] == 0
    assert result["invalid_annotations"] == 0


def test_result_is_deterministic_and_input_not_mutated():
    coco = _document()
    before = deepcopy(coco)
    config = load_invalid_box_config()
    first = count_original_images_per_class(coco, config)
    second = count_original_images_per_class(coco, config)
    assert first == second
    assert coco == before


def _write_document(tmp_path: Path, document: dict) -> Path:
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "lote.json").write_text(json.dumps(document), encoding="utf-8")
    return annotations_dir


def test_raw_loader_keeps_the_degenerate_box_that_strict_load_rejects(tmp_path):
    annotations_dir = _write_document(tmp_path, _document())

    raw = load_raw_dataset(annotations_dir)
    assert any(annotation["bbox"][2] <= 0 for annotation in raw["annotations"])

    with pytest.raises(ValidationError):
        load_dataset(annotations_dir)


def test_counts_filter_degenerate_box_from_a_raw_dataset(tmp_path):
    raw = load_raw_dataset(_write_document(tmp_path, _document()))

    result = count_original_images_per_class(raw, load_invalid_box_config())
    assert result["categories"][0]["image_ids"] == [1, 3]
    assert result["categories"][1]["image_ids"] == [2]
    assert result["total_original_images"] == 3


def test_cli_counts_a_dataset_with_invalid_boxes_without_crashing(tmp_path, capsys):
    annotations_dir = _write_document(tmp_path, _document())

    exit_code = main(["--annotations-dir", str(annotations_dir)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["categories"][0]["image_count"] == 2
    assert payload["categories"][1]["image_count"] == 1
    assert payload["invalid_annotations"] == 2
