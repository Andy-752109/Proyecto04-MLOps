"""Valida las corridas de la campaña y escribe `reports/experiments_validity.json` (P3-11, #18).

Uso (desde la raíz del repo, con el venv de `app/`):

    app/.venv/bin/python validate_runs.py --release v0.1.1

Sale con código 1 si la validación no pasa (menos de `--min-required` corridas
válidas o algún parámetro con un solo valor). No toca el split `test`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from selection.mlflow_reader import read_runs
from selection.validate import validate_runs

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="v0.1.1", help="Release del manifiesto.")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"),
    )
    parser.add_argument("--manifest-meta", type=Path, help="Por defecto, el del release.")
    parser.add_argument("--out", type=Path, help="Por defecto, reports/experiments_validity.json.")
    parser.add_argument("--min-required", type=int, default=10)
    args = parser.parse_args(argv)

    manifest_meta = args.manifest_meta or (
        REPO_ROOT / "reports" / "manifests" / args.release / "manifest_meta.json"
    )
    out = args.out or (REPO_ROOT / "reports" / "experiments_validity.json")
    meta = json.loads(manifest_meta.read_text(encoding="utf-8"))

    runs = read_runs(args.tracking_uri, run_kind="campaign")
    result = validate_runs(
        runs,
        manifest_sha256=meta["manifest_sha256"],
        classes=meta["classes"],
        min_required=args.min_required,
    )
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result.passed,
                "valid": len(result.valid),
                "invalid": [entry.run_id for entry in result.invalid],
                "per_param_values": result.per_param_values,
                "out": str(out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
