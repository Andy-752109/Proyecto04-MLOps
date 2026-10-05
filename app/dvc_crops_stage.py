"""DVC stage that materializes one crop per valid COCO box.

Solo CPU y disco: sin red ni variables sensibles. Los recortes viven en
`data/derived/crops/` (ignorado por Git; versionado con DVC).
"""

import json
import os
from pathlib import Path

from crops.build import build_crops
from ingestion.loader import load_raw_dataset
from policies.invalid_boxes import load_invalid_box_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("DATASET_DIR", REPO_ROOT / "data" / "raw"))
DERIVED_DIR = Path(os.environ.get("DERIVED_DIR", REPO_ROOT / "data" / "derived"))


def write_crops() -> dict:
    """Materializa los recortes y devuelve un resumen legible para los logs."""
    coco = load_raw_dataset(DATASET_DIR / "annotations")
    return build_crops(
        coco,
        DATASET_DIR / "images",
        DERIVED_DIR / "crops",
        load_invalid_box_config(),
    )


if __name__ == "__main__":
    print(json.dumps(write_crops(), ensure_ascii=False))
