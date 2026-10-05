"""P3-06: manifiesto 70/20/10 por grupos de duplicados, sin fuga y reproducible.

Las pruebas usan un COCO sintético con imágenes de ruido (pHash distintos),
copias exactas y casi duplicados, para que los grupos no sean triviales.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from analyzers.duplicates import DuplicateConfig, analyze_duplicates
from crops.build import plan_crops
from manifest.build import (
    MANIFEST_FIELDS,
    ManifestError,
    build_manifest,
    manifest_csv_bytes,
)
from manifest.check_leakage import find_leakage
from manifest.check_leakage import main as check_leakage_main
from manifest.config import ManifestConfig, load_manifest_config
from policies.invalid_boxes import load_invalid_box_config
from splits.models import load_splits_config

CAT, DOG = 4, 3
IMAGES_PER_CLASS = 60


def _noise_png(seed: int, *, brighten: int = 0) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 200, (64, 64, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(pixels.astype(int) + brighten, 0, 255).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _annotation(annotation_id: int, image_id: int, category_id: int, x: int = 0) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, 0, 20, 20],
        "area": 400.0,
        "iscrowd": 0,
    }


def _dataset() -> tuple[dict, dict[int, bytes]]:
    """120 imágenes; algunas con dos cajas, copias exactas y casi duplicados."""
    images, annotations, contents = [], [], {}
    next_annotation = 1
    for index in range(2 * IMAGES_PER_CLASS):
        image_id = index + 1
        category_id = CAT if index < IMAGES_PER_CLASS else DOG
        name = "cat" if category_id == CAT else "dog"
        images.append(
            {"id": image_id, "file_name": f"{name}.{index}.png", "width": 64, "height": 64}
        )
        contents[image_id] = _noise_png(index)
        annotations.append(_annotation(next_annotation, image_id, category_id))
        next_annotation += 1
        if index % 7 == 0:  # dos cajas en la misma imagen: dos recortes, un origen
            annotations.append(_annotation(next_annotation, image_id, category_id, x=30))
            next_annotation += 1

    # Copias exactas (mismos bytes, otro nombre) y casi duplicados (pHash igual).
    contents[2] = contents[1]
    contents[62] = contents[61]
    contents[11] = _noise_png(9, brighten=3)
    contents[71] = _noise_png(69, brighten=3)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": DOG, "name": "dog"}, {"id": CAT, "name": "cat"}],
    }
    return coco, contents


def _config(**overrides) -> ManifestConfig:
    values = {
        "release": "v0.1.1",
        "train": 0.70,
        "val": 0.20,
        "test": 0.10,
        "seed": 42,
        "classes": ["cat", "dog"],
        "max_deviation_pp": 5.0,
    } | overrides
    return ManifestConfig.model_validate(values)


def _crops(coco: dict) -> list[dict]:
    crops, _ = plan_crops(coco, load_invalid_box_config())
    for crop in crops:
        crop["path"] = f"images/{crop['crop_id']}.jpg"
    return crops


def _pairs(contents: dict[int, bytes]) -> list[dict]:
    result = analyze_duplicates(contents, DuplicateConfig(threshold=0.94))
    return result.details["image_pairs"]


def _build(config: ManifestConfig | None = None, coco=None, contents=None, crops=None):
    if coco is None:
        coco, contents = _dataset()
    return build_manifest(
        coco,
        crops if crops is not None else _crops(coco),
        image_contents=contents,
        duplicate_pairs=_pairs(contents),
        config=config or _config(),
        invalid_config=load_invalid_box_config(),
    )


@pytest.fixture(scope="module")
def result():
    return _build()


def test_no_duplicate_group_is_split_across_partitions(result):
    owners: dict[str, set[str]] = {}
    for row in result.rows:
        owners.setdefault(row["duplicate_group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in owners.values())

    by_image = {row["source_image_id"]: row for row in result.rows}
    # Copia exacta (1, 2) y casi duplicado (10, 11): mismo grupo, misma partición.
    for a, b in ((1, 2), (61, 62), (10, 11), (70, 71)):
        assert by_image[a]["duplicate_group_id"] == by_image[b]["duplicate_group_id"]
        assert by_image[a]["split"] == by_image[b]["split"]


def test_crops_of_the_same_source_image_share_partition(result):
    owners: dict[int, set[str]] = {}
    for row in result.rows:
        owners.setdefault(row["source_image_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in owners.values())


def test_same_seed_same_ids_and_same_sha():
    first, second = _build(), _build()
    assert first.rows == second.rows
    assert manifest_csv_bytes(first.rows) == manifest_csv_bytes(second.rows)


def test_different_seed_changes_assignment(result):
    other = _build(_config(seed=7))
    assert [r["split"] for r in other.rows] != [r["split"] for r in result.rows]


def test_every_partition_within_five_pp_overall_and_per_class(result):
    for split, ratio in (("train", 0.70), ("val", 0.20), ("test", 0.10)):
        summary = result.counts["splits"][split]
        assert abs(summary["crops_fraction"] - ratio) * 100 <= 5.0
        for name in ("cat", "dog"):
            per_class = summary["classes"][name]
            assert abs(per_class["crops_fraction_of_class"] - ratio) * 100 <= 5.0


def test_every_class_present_in_validation_and_test(result):
    for split in ("val", "test"):
        for name in ("cat", "dog"):
            assert result.counts["splits"][split]["classes"][name]["crops"] > 0


def test_manifest_covers_every_crop_exactly_once(result):
    coco, _ = _dataset()
    assert sorted(r["crop_id"] for r in result.rows) == sorted(c["crop_id"] for c in _crops(coco))
    assert len({r["crop_id"] for r in result.rows}) == len(result.rows)


def test_labels_follow_class_map_cat_zero_dog_one(result):
    for row in result.rows:
        assert row["label"] == {"cat": 0, "dog": 1}[row["category_name"]]


def test_injected_degenerate_box_does_not_enter():
    coco, contents = _dataset()
    coco["annotations"].append(
        {
            "id": 999,
            "image_id": 5,
            "category_id": CAT,
            "bbox": [0, 0, 0, 10],
            "area": 0.0,
            "iscrowd": 0,
        }
    )
    result = _build(coco=coco, contents=contents)
    assert 999 not in {row["annotation_id"] for row in result.rows}


def test_catalog_with_an_invalid_box_is_rejected():
    """Defensa: si el catálogo trae una caja inválida (P3-04 desfasado), no se reparte."""
    coco, contents = _dataset()
    crops = _crops(coco)
    coco["annotations"][0]["bbox"] = [0, 0, 0, 10]
    coco["annotations"][0]["area"] = 0.0
    with pytest.raises(ManifestError, match="inválida"):
        _build(coco=coco, contents=contents, crops=crops)


def test_deviation_above_tolerance_fails():
    strict = _config(max_deviation_pp=0.01)
    with pytest.raises(ManifestError, match="pp"):
        _build(strict)


@pytest.mark.parametrize(
    ("override", "message"),
    [({"test": 0.2}, r"sumar 1\.0"), ({"release": "latest"}, "release")],
)
def test_config_rejects_bad_ratios_and_release(override: dict, message: str):
    with pytest.raises(ValueError, match=message):
        _config(**override)


def test_repository_config_is_70_20_10_and_p2_splits_untouched():
    config = load_manifest_config()
    assert (config.train, config.val, config.test, config.seed) == (0.70, 0.20, 0.10, 42)
    assert config.classes == ["cat", "dog"]
    p2 = load_splits_config()
    assert (p2.train, p2.val, p2.test, p2.seed) == (0.70, 0.15, 0.15, 42)


def test_manifest_csv_has_the_contract_columns(result):
    reader = csv.DictReader(io.StringIO(manifest_csv_bytes(result.rows).decode("utf-8")))
    assert reader.fieldnames == MANIFEST_FIELDS
    assert {"crop_id", "split", "label", "path", "duplicate_group_id"} <= set(MANIFEST_FIELDS)


def test_find_leakage_is_empty_for_a_clean_manifest(result):
    assert find_leakage(result.rows) == {
        "crop_id": [],
        "source_image_id": [],
        "duplicate_group_id": [],
    }


@pytest.mark.parametrize("column", ["crop_id", "source_image_id", "duplicate_group_id"])
def test_find_leakage_detects_each_kind(result, column):
    rows = [dict(row) for row in result.rows]
    train = next(r for r in rows if r["split"] == "train")
    test = next(r for r in rows if r["split"] == "test")
    test[column] = train[column]
    assert find_leakage(rows)[column]


def test_check_leakage_cli_exit_codes(result, tmp_path: Path, capsys):
    (tmp_path / "v0.1.1").mkdir()
    (tmp_path / "v0.1.1" / "manifest.csv").write_bytes(manifest_csv_bytes(result.rows))
    assert check_leakage_main(["--release", "v0.1.1"], manifests_dir=tmp_path) == 0
    assert json.loads(capsys.readouterr().out)["leakage"] is False

    rows = [dict(row) for row in result.rows]
    val = next(r for r in rows if r["split"] == "val")
    next(r for r in rows if r["split"] == "train")["source_image_id"] = val["source_image_id"]
    (tmp_path / "v9.9.9").mkdir()
    (tmp_path / "v9.9.9" / "manifest.csv").write_bytes(manifest_csv_bytes(rows))
    assert check_leakage_main(["--release", "v9.9.9"], manifests_dir=tmp_path) == 1


@pytest.mark.parametrize("release", ["../../etc", "v0.1.1/../../x", "/tmp/v0.1.1", "latest"])
def test_check_leakage_cli_rejects_paths_outside_the_manifests_dir(tmp_path: Path, release: str):
    assert check_leakage_main(["--release", release], manifests_dir=tmp_path) == 2


def test_stage_writes_manifest_meta_and_counts(tmp_path: Path, monkeypatch):
    """La etapa DVC completa, sobre el fixture: mismo SHA en dos generaciones."""
    import dvc_manifest_stage as stage

    coco, contents = _dataset()
    raw = tmp_path / "raw"
    (raw / "annotations").mkdir(parents=True)
    (raw / "images").mkdir()
    (raw / "annotations" / "annotations-lote-001.json").write_text(json.dumps(coco))
    for image in coco["images"]:
        (raw / "images" / image["file_name"]).write_bytes(contents[image["id"]])
    for name in ("images", "annotations"):
        (raw / f"{name}.dvc").write_text(
            f"outs:\n- md5: {name}0123456789abcdef.dir\n  nfiles: 1\n  path: {name}\n"
        )

    derived = tmp_path / "derived"
    crops_dir = derived / "crops"
    crops_dir.mkdir(parents=True)
    fields = ["crop_id", "source_image_id", "source_image_file", "annotation_id",
              "category_id", "category_name", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
              "crop_width", "crop_height", "path"]  # fmt: skip
    with (crops_dir / "crops.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_crops(coco))

    reports = tmp_path / "reports"
    (reports / "releases" / "v0.1.1").mkdir(parents=True)
    quality = {"schema_version": "1.0", "dataset_version": "v0.1.1", "status": "warning"}
    (reports / "releases" / "v0.1.1" / "quality.json").write_text(json.dumps(quality))

    monkeypatch.setattr(stage, "DATASET_DIR", raw)
    monkeypatch.setattr(stage, "DERIVED_DIR", derived)
    monkeypatch.setattr(stage, "REPORTS_DIR", reports)

    first = stage.write_manifest()
    manifest_path = derived / "manifests" / "v0.1.1" / "manifest.csv"
    first_bytes = manifest_path.read_bytes()
    second = stage.write_manifest()

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert manifest_path.read_bytes() == first_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == first["manifest_sha256"]

    meta = json.loads((reports / "manifests" / "v0.1.1" / "manifest_meta.json").read_text())
    assert meta["manifest_sha256"] == first["manifest_sha256"]
    assert meta["manifest_id"].startswith("v0.1.1-")
    assert meta["release"]["name"] == "v0.1.1"
    assert meta["release"]["images_md5"] == "images0123456789abcdef.dir"
    assert meta["quality_reference"]["status"] == "warning"
    assert meta["seed"] == 42
    assert meta["classes"] == {"0": "cat", "1": "dog"}
    counts = json.loads((reports / "manifests" / "v0.1.1" / "counts.json").read_text())
    assert set(counts["splits"]) == {"train", "val", "test"}
    assert counts["splits"]["test"]["classes"]["dog"]["originals"] > 0
    assert not (reports / "splits.json").exists()  # nunca toca el split de P2


def test_stage_refuses_a_failed_quality_reference(tmp_path: Path, monkeypatch):
    import dvc_manifest_stage as stage

    reports = tmp_path / "reports"
    (reports / "releases" / "v0.1.1").mkdir(parents=True)
    (reports / "releases" / "v0.1.1" / "quality.json").write_text(
        json.dumps({"dataset_version": "v0.1.1", "status": "failed"})
    )
    monkeypatch.setattr(stage, "REPORTS_DIR", reports)
    config = _config()
    with pytest.raises(ManifestError, match="failed"):
        stage.quality_reference(config)
