"""Conversión P4-04: checkpoint de P3 1.0.0 -> ONNX FP32 (paridad verificada) -> INT8 estática.

Exporta `eval_v1.0.0/artifacts/checkpoint/best.pt` a ONNX (opset fijo, batch dinámico),
corre ONNX Runtime (CPU) sobre `reports/p4/reference/parity_reference.csv` y compara las
probabilidades contra las de PyTorch. Sale con código 1 si algún |Δprob| >= 1e-4, si cambia el
orden de clases o si la clase predicha difiere. Se corre desde `app/`:

    uv run --group ml python -m edge_model.convert

Tras la paridad FP32 genera la variante INT8 (`edge_model.quantize`); `--fp32-only` la omite.
Las salidas van a `build/edge/1.0.0-fp32/` y `build/edge/1.0.0-int8/` (ignorado por Git: no se
versionan binarios de modelo); las evidencias van a `reports/p4/conversion/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from edge_model import quantize
from edge_model.baseline import (
    CLASSES,
    EXPECTED_CHECKPOINT_SHA256,
    load_package,
    preprocessing,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
OPSET = 17
MAX_ABS_DELTA = 1e-4


def export_onnx(model: torch.nn.Module, image_size: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out)))


def check_parity(
    model: torch.nn.Module, onnx_path: Path, parity_csv: Path, crops: Path, image_size: int
) -> dict:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    transform = preprocessing(image_size)
    with open(parity_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    max_delta_ref = max_delta_torch = 0.0
    mismatches = []
    for row in rows:
        image = Image.open(crops / row["path"]).convert("RGB")
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            torch_probs = torch.softmax(model(x), dim=1)[0].numpy()
        logits = session.run(None, {"input": x.numpy()})[0]
        e = np.exp(logits[0] - logits[0].max())
        onnx_probs = e / e.sum()
        ref = np.array([float(row["prob_cat"]), float(row["prob_dog"])])
        max_delta_ref = max(max_delta_ref, float(np.abs(onnx_probs - ref).max()))
        max_delta_torch = max(max_delta_torch, float(np.abs(onnx_probs - torch_probs).max()))
        predicted = CLASSES[int(onnx_probs.argmax())]
        if predicted != row["predicted_class"]:
            mismatches.append(row["crop_id"])
    return {
        "rows": len(rows),
        "max_abs_delta_vs_reference_csv": max_delta_ref,
        "max_abs_delta_vs_pytorch": max_delta_torch,
        "tolerance": MAX_ABS_DELTA,
        "class_mismatches": mismatches,
        "class_order": list(CLASSES),
        "passed": max_delta_ref < MAX_ABS_DELTA and not mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--package-dir", type=Path, default=ROOT / "eval_v1.0.0")
    p.add_argument("--crops", type=Path, default=ROOT / "data" / "derived" / "crops")
    p.add_argument(
        "--parity-csv",
        type=Path,
        default=ROOT / "reports" / "p4" / "reference" / "parity_reference.csv",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "build" / "edge" / "1.0.0-fp32" / "model_fp32.onnx"
    )
    p.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "p4" / "conversion" / "parity_fp32.json",
    )
    p.add_argument("--fp32-only", action="store_true", help="No generar la variante INT8.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "derived" / "manifests" / "v0.1.1" / "manifest.csv",
    )
    args = p.parse_args(argv)

    torch.set_num_threads(1)
    model, config, checkpoint_sha = load_package(args.package_dir, EXPECTED_CHECKPOINT_SHA256)
    export_onnx(model, config["image_size"], args.out)
    parity = check_parity(model, args.out, args.parity_csv, args.crops, config["image_size"])

    libraries = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }
    report = {
        "source_version": "1.0.0",
        "source_checkpoint_sha256": checkpoint_sha,
        "output_file": args.out.name,
        "output_sha256": sha256_file(args.out),
        "output_bytes": args.out.stat().st_size,
        "format": "onnx",
        "precision": "fp32",
        "opset": OPSET,
        "input_shape": [None, 3, config["image_size"], config["image_size"]],
        "libraries": libraries,
        "command": "uv run --group ml python -m edge_model.convert",
        "parity": parity,
    }
    text = json.dumps(report, indent=2)
    print(text)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n", encoding="utf-8")
    if not parity["passed"]:
        return 1
    if args.fp32_only:
        return 0
    return quantize.run(
        fp32=args.out,
        manifest=args.manifest,
        crops=args.crops,
        package_src=args.package_dir,
        image_size=config["image_size"],
        checkpoint_sha=checkpoint_sha,
        root=ROOT,
        opset=OPSET,
        libraries=libraries,
        command="uv run --group ml python -m edge_model.convert",
    )


if __name__ == "__main__":
    sys.exit(main())
