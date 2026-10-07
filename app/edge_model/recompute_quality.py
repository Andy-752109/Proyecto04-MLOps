"""Recálculo independiente (scikit-learn) de `metrics_val.json` desde `comparison_val.csv`.

Reutiliza `recompute.sklearn_metrics` (P3-13). Compara ambos modelos con tolerancia 1e-9 y
verifica la caída en pp. Desde `app/`:

    uv run --group ml python -m edge_model.recompute_quality [--split val]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from edge_model.baseline import CLASSES
from recompute import TOLERANCE, compare, sklearn_metrics

QUALITY = Path(__file__).resolve().parents[2] / "reports" / "p4" / "quality"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val")
    p.add_argument("--dir", type=Path, default=QUALITY)
    args = p.parse_args(argv)
    with open(args.dir / f"comparison_{args.split}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = json.loads((args.dir / f"metrics_{args.split}.json").read_text(encoding="utf-8"))
    true = [CLASSES.index(r["true_class"]) for r in rows]
    failed = False
    got = {}
    for key, column in (("original", "pred_original"), ("optimized", "pred_optimized")):
        got[key] = sklearn_metrics(true, [CLASSES.index(r[column]) for r in rows])
        diffs = compare(expected[key], got[key])
        print(f"{key}: accuracy={got[key]['accuracy']!r} macro_f1={got[key]['macro_f1']!r}")
        for d in diffs:
            print(f"DIFIERE [{key}]: {d}", file=sys.stderr)
        failed |= bool(diffs)
    drop = (got["original"]["accuracy"] - got["optimized"]["accuracy"]) * 100
    if abs(drop - expected["accuracy_drop_pp"]) > TOLERANCE:
        print(
            f"DIFIERE: accuracy_drop_pp json={expected['accuracy_drop_pp']} sk={drop}",
            file=sys.stderr,
        )
        failed = True
    print(f"accuracy_drop_pp={drop!r}")
    print(
        "recompute: NO coincide"
        if failed
        else "recompute: coincide con metrics_%s.json" % args.split
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
