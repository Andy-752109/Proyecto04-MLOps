"""Paridad de salida: el modelo edge sobre las 20 referencias vs el baseline de P4-01.

Usa el mismo preprocesamiento y backend que la app (`edge.preprocess`, `edge.backend`) y
compara contra `reports/p4/reference/parity_reference.csv`. Necesita los recortes de
`data/derived/crops/` (`dvc pull` o se los pasa Karen) y el ONNX de P4-04. Sin torch.

    python -m edge.tools.parity_model --model edge/models/model.onnx [--json salida.json]

Sale con código 1 si alguna clase difiere del baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from edge.backend import CLASSES, create_backend, sha256_file
from edge.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", default="onnxruntime")
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "reports/p4/reference/parity_reference.csv"
    )
    parser.add_argument("--crops", type=Path, default=ROOT / "data/derived/crops")
    parser.add_argument("--json", type=Path, help="guarda el resumen y las filas en JSON")
    args = parser.parse_args()

    with open(args.reference, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["path"] for r in rows if not (args.crops / r["path"]).is_file()]
    if missing:
        sys.exit(f"faltan {len(missing)} recortes en {args.crops} (p. ej. {missing[0]})")

    backend = create_backend(args.runtime)
    backend.load(args.model)
    results = []
    for row in rows:
        with Image.open(args.crops / row["path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        probs = backend.predict(preprocess(rgb))
        predicted = CLASSES[int(np.argmax(probs))]
        delta = max(
            abs(float(probs[0]) - float(row["prob_cat"])),
            abs(float(probs[1]) - float(row["prob_dog"])),
        )
        results.append(
            {
                "crop_id": row["crop_id"],
                "true_class": row["true_class"],
                "baseline_class": row["predicted_class"],
                "edge_class": predicted,
                "prob_cat": float(probs[0]),
                "prob_dog": float(probs[1]),
                "max_abs_delta_prob": delta,
            }
        )

    mismatches = [r for r in results if r["edge_class"] != r["baseline_class"]]
    summary = {
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "runtime": backend.name,
        "references": len(results),
        "class_mismatches": len(mismatches),
        "max_abs_delta_prob": max(r["max_abs_delta_prob"] for r in results),
    }
    for r in results:
        mark = "OK " if r["edge_class"] == r["baseline_class"] else "DIF"
        print(
            f"{mark} {r['crop_id']} base={r['baseline_class']} edge={r['edge_class']} "
            f"max_dp={r['max_abs_delta_prob']:.2e}"
        )
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": summary, "rows": results}, indent=2), encoding="utf-8"
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
