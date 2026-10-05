"""DVC stage P3-06: manifiesto de recortes 70/20/10, sin fuga y reproducible.

Lee el release crudo (`data/raw`), el catálogo de P3-04
(`data/derived/crops/crops.csv`) y la referencia de calidad del release
(`reports/releases/<release>/quality.json`). Escribe:

- `data/derived/manifests/<release>/manifest.csv` (salida DVC, se sube con `dvc push`);
- `reports/manifests/<release>/manifest_meta.json` y `counts.json` (en Git).

No escribe fechas ni rutas absolutas: dos generaciones dan el mismo SHA-256.
Solo CPU y disco: sin red ni variables sensibles.
"""

import hashlib
import json
import os
from pathlib import Path

import yaml

from analyzers.duplicates import analyze_duplicates
from ingestion.loader import load_raw_dataset
from manifest.build import ManifestError, build_manifest, manifest_csv_bytes
from manifest.check_leakage import find_leakage, read_manifest
from manifest.config import ManifestConfig, load_manifest_config
from policies.duplicates import load_duplicate_config
from policies.invalid_boxes import load_invalid_box_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("DATASET_DIR", REPO_ROOT / "data" / "raw"))
DERIVED_DIR = Path(os.environ.get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", REPO_ROOT / "reports"))

CROPS_ROOT = "data/derived/crops"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _dvc_md5(path: Path) -> str:
    """md5 del directorio que registra un `.dvc` (identifica el contenido del release)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))["outs"][0]["md5"]


def quality_reference(config: ManifestConfig) -> dict:
    """La compuerta de P2 del release; un reporte `failed` detiene el manifiesto."""
    path = REPORTS_DIR / "releases" / config.release / "quality.json"
    content = path.read_bytes()
    report = json.loads(content)
    if report.get("status") == "failed":
        raise ManifestError(
            f"La compuerta de calidad de {config.release} está failed; no se genera manifiesto."
        )
    if report.get("dataset_version") != config.release:
        raise ManifestError(
            f"{path} describe {report.get('dataset_version')}, no {config.release}."
        )
    return {
        "path": f"reports/releases/{config.release}/quality.json",
        "sha256": _sha256(content),
        "status": report["status"],
        "dataset_version": report["dataset_version"],
    }


def _read_crops(path: Path) -> list[dict]:
    rows = read_manifest(path)  # mismo lector CSV; aquí son filas del catálogo
    for row in rows:
        for key in ("source_image_id", "annotation_id", "category_id"):
            row[key] = int(row[key])
    return rows


def write_manifest() -> dict:
    """Genera el manifiesto y sus reportes; devuelve un resumen para los logs."""
    config = load_manifest_config()
    quality = quality_reference(config)

    catalog_path = DERIVED_DIR / "crops" / "crops.csv"
    catalog_bytes = catalog_path.read_bytes()
    crops = _read_crops(catalog_path)

    coco = load_raw_dataset(DATASET_DIR / "annotations")
    images_dir = DATASET_DIR / "images"
    wanted = {crop["source_image_id"] for crop in crops}
    contents = {
        image["id"]: (images_dir / image["file_name"]).read_bytes()
        for image in coco["images"]
        if image["id"] in wanted
    }
    duplicate_config = load_duplicate_config()
    pairs = analyze_duplicates(contents, duplicate_config).details["image_pairs"]

    result = build_manifest(
        coco,
        crops,
        image_contents=contents,
        duplicate_pairs=pairs,
        config=config,
        invalid_config=load_invalid_box_config(),
    )

    manifest_dir = DERIVED_DIR / "manifests" / config.release
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    manifest_bytes = manifest_csv_bytes(result.rows)
    manifest_path.write_bytes(manifest_bytes)

    # Verificación independiente sobre el archivo escrito, no sobre la memoria.
    leakage = find_leakage(read_manifest(manifest_path))
    if any(leakage.values()):
        raise ManifestError(f"Fuga en {manifest_path}: {leakage}")

    manifest_sha = _sha256(manifest_bytes)
    meta = {
        "manifest_id": f"{config.release}-{manifest_sha[:12]}",
        "manifest_sha256": manifest_sha,
        "manifest_path": f"data/derived/manifests/{config.release}/manifest.csv",
        "crops_root": CROPS_ROOT,
        "release": {
            "name": config.release,
            "images_md5": _dvc_md5(DATASET_DIR / "images.dvc"),
            "annotations_md5": _dvc_md5(DATASET_DIR / "annotations.dvc"),
        },
        "crops_catalog_sha256": _sha256(catalog_bytes),
        "quality_reference": quality,
        "seed": config.seed,
        "ratios": config.ratios,
        "max_deviation_pp": config.max_deviation_pp,
        "classes": {str(index): name for name, index in config.labels.items()},
        "duplicate_similarity_threshold": duplicate_config.threshold,
        "duplicate_pairs": len(pairs),
        "rows": len(result.rows),
    }

    reports_dir = REPORTS_DIR / "manifests" / config.release
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports_dir / "counts.json").write_text(
        json.dumps(result.counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "manifest_id": meta["manifest_id"],
        "manifest_sha256": manifest_sha,
        "rows": len(result.rows),
        "splits": {name: s["crops"] for name, s in result.counts["splits"].items()},
    }


if __name__ == "__main__":
    print(json.dumps(write_manifest(), ensure_ascii=False))
