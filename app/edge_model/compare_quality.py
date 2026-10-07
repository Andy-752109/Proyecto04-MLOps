"""Comparación de calidad P4 (original P3 vs. INT8): predicciones por muestra y métricas.

Corre `build/edge/1.0.0-int8/model_int8.onnx` (ONNX Runtime CPU, 1 hilo) sobre los recortes de
`val` (principal) y `test` (solo referencia) con el mismo preprocesamiento que el original,
y lo junta con las predicciones del original:

- val: `reports/p4/quality/predictions_val_original.csv` (P4-01), sin recalcular;
- test: se calcula con el checkpoint 1.0.0 (PyTorch), porque P4-01 solo dejó val.

Escribe `comparison_val.csv`, `metrics_val.json` y, como sección aparte, `comparison_test.csv` y
`metrics_test.json`. Sale con 1 si falla alguna comprobación de integridad. Desde `app/`:

    uv run --group ml python -m edge_model.compare_quality
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from edge_model import quantize
from edge_model.baseline import (
    CLASSES,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    load_package,
    predict,
    preprocessing,
    read_split,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
QUALITY = ROOT / "reports" / "p4" / "quality"
MAX_DROP_PP = quantize.MAX_ACCURACY_DROP_PP
COLUMNS = [
    "crop_id",
    "true_class",
    "pred_original",
    "pred_optimized",
    "prob_cat_original",
    "prob_dog_original",
    "prob_cat_optimized",
    "prob_dog_optimized",
]


def predict_int8(model: Path, rows: list[dict], crops: Path, image_size: int) -> dict[str, list]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    transform = preprocessing(image_size)
    out = {}
    for row in rows:
        x = transform(Image.open(crops / row["path"]).convert("RGB")).unsqueeze(0).numpy()
        logits = session.run(None, {"input": x})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        out[row["crop_id"]] = (e / e.sum()).tolist()
    return out


def metrics(true: list[str], pred: list[str]) -> dict:
    """Cálculo en Python puro (independiente de sklearn, que usa el recálculo)."""
    n = len(true)
    matrix = [[0, 0], [0, 0]]
    for t, p in zip(true, pred, strict=True):
        matrix[CLASSES.index(t)][CLASSES.index(p)] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(2)) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[i]),
        }
    return {
        "accuracy": (matrix[0][0] + matrix[1][1]) / n,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(CLASSES),
        "confusion_matrix": matrix,
        "per_class": per_class,
        "total": n,
    }


def build_rows(rows: list[dict], original: dict[str, dict], int8: dict[str, list]) -> list[dict]:
    out = []
    for row in rows:
        cid = row["crop_id"]
        o, q = original[cid], int8[cid]
        out.append(
            {
                "crop_id": cid,
                "true_class": CLASSES[int(row["label"])],
                "pred_original": o["predicted_class"],
                "pred_optimized": CLASSES[int(np.argmax(q))],
                "prob_cat_original": float(o["prob_cat"]),
                "prob_dog_original": float(o["prob_dog"]),
                "prob_cat_optimized": q[0],
                "prob_dog_optimized": q[1],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(split: str, rows: list[dict]) -> dict:
    true = [r["true_class"] for r in rows]
    orig = metrics(true, [r["pred_original"] for r in rows])
    opt = metrics(true, [r["pred_optimized"] for r in rows])
    disagreements = [r["crop_id"] for r in rows if r["pred_original"] != r["pred_optimized"]]
    return {
        "split": split,
        "rows": len(rows),
        "unique_ids": len({r["crop_id"] for r in rows}),
        "original": orig,
        "optimized": opt,
        "accuracy_drop_pp": (orig["accuracy"] - opt["accuracy"]) * 100,
        "macro_f1_drop_pp": (orig["macro_f1"] - opt["macro_f1"]) * 100,
        "max_accuracy_drop_pp": MAX_DROP_PP,
        "passes_threshold": (orig["accuracy"] - opt["accuracy"]) * 100 <= MAX_DROP_PP,
        "class_disagreements": len(disagreements),
        "disagreement_ids": disagreements,
        "max_abs_delta_prob": max(
            abs(r["prob_cat_original"] - r["prob_cat_optimized"]) for r in rows
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--package-dir", type=Path, default=ROOT / "eval_v1.0.0")
    p.add_argument(
        "--model", type=Path, default=ROOT / "build" / "edge" / "1.0.0-int8" / "model_int8.onnx"
    )
    p.add_argument("--crops", type=Path, default=ROOT / "data" / "derived" / "crops")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "derived" / "manifests" / "v0.1.1" / "manifest.csv",
    )
    p.add_argument(
        "--calibration-ids",
        type=Path,
        default=ROOT / "reports" / "p4" / "conversion" / "calibration_ids.csv",
    )
    p.add_argument("--out-dir", type=Path, default=QUALITY)
    args = p.parse_args(argv)

    torch.set_num_threads(1)
    errors: list[str] = []
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        errors.append("manifest.csv no coincide con el de P4-01")
    with open(args.calibration_ids, newline="", encoding="utf-8") as f:
        calibration = {r["crop_id"] for r in csv.DictReader(f)}

    torch_model, config, _ = load_package(args.package_dir, EXPECTED_CHECKPOINT_SHA256)
    size = config["image_size"]
    results = {}
    for split in ("val", "test"):
        rows = read_split(args.manifest, split)
        ids = {r["crop_id"] for r in rows}
        if split == "val":
            with open(QUALITY / "predictions_val_original.csv", newline="", encoding="utf-8") as f:
                original = {r["crop_id"]: r for r in csv.DictReader(f)}
            if set(original) != ids:
                errors.append("IDs de predictions_val_original.csv distintos al manifiesto (val)")
        else:
            original = {r["crop_id"]: r for r in predict(torch_model, rows, args.crops, size)}
        merged = build_rows(rows, original, predict_int8(args.model, rows, args.crops, size))
        write_csv(args.out_dir / f"comparison_{split}.csv", merged)
        summary = summarize(split, merged)
        summary["calibration_overlap"] = sorted(calibration & ids)
        if summary["calibration_overlap"]:
            errors.append(f"calibración ∩ {split} no vacía")
        if summary["unique_ids"] != len(merged) or set(summary["original"]["per_class"]) != set(
            CLASSES
        ):
            errors.append(f"{split}: IDs duplicados")
        if any(c["support"] == 0 for c in summary["original"]["per_class"].values()):
            errors.append(f"{split}: falta una clase")
        results[split] = summary

    if results["val"]["rows"] != 131:
        errors.append("val no tiene 131 filas")
    header = {
        "model_original": "1.0.0 (PyTorch, checkpoint sha256 " + EXPECTED_CHECKPOINT_SHA256 + ")",
        "model_optimized": "1.0.0-int8 (ONNX Runtime CPU, 1 hilo) sha256 "
        + sha256_file(args.model),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "class_order": list(CLASSES),
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
    }
    for split, name in (("val", "metrics_val.json"), ("test", "metrics_test.json")):
        text = json.dumps({**header, **results[split]}, indent=2)
        (args.out_dir / name).write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                s: {k: r[k] for k in ("rows", "accuracy_drop_pp", "class_disagreements")}
                for s, r in results.items()
            },
            indent=2,
        )
    )
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
