"""Conversión P4-04 (parte 2): ONNX FP32 -> INT8 estática (QDQ, per-channel) para ONNX Runtime CPU.

Calibra con 100 recortes SOLO de train (semilla 42), verifica contra el manifiesto que ningún
ID (ni imagen origen) pertenezca a val/test, mide accuracy en val contra el original de P3 y
arma el paquete `build/edge/1.0.0-int8/`. Lo invoca `edge_model.convert`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from edge_model.baseline import CLASSES, preprocessing, read_split, sha256_file

CALIBRATION_SIZE = 100
CALIBRATION_SEED = 42
MAX_ACCURACY_DROP_PP = 2.0
MODEL_VERSION = "1.0.0-int8"
MODEL_FILE = "model_int8.onnx"


class CropReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], image_size: int):
        transform = preprocessing(image_size)
        self._batches = iter(
            [{"input": transform(Image.open(p).convert("RGB")).unsqueeze(0).numpy()} for p in paths]
        )

    def get_next(self):
        return next(self._batches, None)


def select_calibration(manifest: Path) -> list[dict]:
    """100 recortes de train (semilla 42) sin ID ni imagen origen compartidos con val/test."""
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    held_out = {r["source_image_id"] for r in rows if r["split"] in ("val", "test")}
    held_out_ids = {r["crop_id"] for r in rows if r["split"] in ("val", "test")}
    pool = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["crop_id"])
    chosen = sorted(
        random.Random(CALIBRATION_SEED).sample(pool, CALIBRATION_SIZE), key=lambda r: r["crop_id"]
    )
    leaked = [
        r["crop_id"]
        for r in chosen
        if r["crop_id"] in held_out_ids or r["source_image_id"] in held_out
    ]
    if leaked or any(r["split"] != "train" for r in chosen):
        raise SystemExit(f"Calibración contaminada con val/test: {leaked}")
    return chosen


def write_calibration_ids(path: Path, chosen: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["crop_id", "source_image_id", "split"])
        w.writerows([r["crop_id"], r["source_image_id"], r["split"]] for r in chosen)
    return sha256_file(path)


def quantize(fp32: Path, out: Path, chosen: list[dict], crops: Path, image_size: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared = out.with_name("model_fp32_prep.onnx")
    quant_pre_process(str(fp32), str(prepared))
    reader = CropReader([crops / r["path"] for r in chosen], image_size)
    quantize_static(
        str(prepared),
        str(out),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=True,  # AVX2 sin VNNI: evita saturación de VPMADDUBSW en U8S8
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepared.unlink()


def _probs(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    logits = session.run(None, {"input": x})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def evaluate(
    fp32: Path,
    int8: Path,
    manifest: Path,
    crops: Path,
    image_size: int,
    original_predictions: Path,
    parity_csv: Path,
) -> dict:
    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    transform = preprocessing(image_size)
    with open(original_predictions, newline="", encoding="utf-8") as f:
        original = {r["crop_id"]: r for r in csv.DictReader(f)}
    rows = read_split(manifest, "val")
    correct8 = correct_orig = agree = 0
    max_delta = 0.0
    for r in rows:
        x = transform(Image.open(crops / r["path"]).convert("RGB")).unsqueeze(0).numpy()
        p8, p32 = _probs(s8, x), _probs(s32, x)
        true = CLASSES[int(r["label"])]
        pred8 = CLASSES[int(p8.argmax())]
        orig = original[r["crop_id"]]
        correct8 += pred8 == true
        correct_orig += orig["predicted_class"] == true
        agree += pred8 == orig["predicted_class"]
        max_delta = max(max_delta, float(np.abs(p8 - p32).max()))
    n = len(rows)
    with open(parity_csv, newline="", encoding="utf-8") as f:
        parity_rows = list(csv.DictReader(f))
    parity_mismatch = []
    for r in parity_rows:
        x = transform(Image.open(crops / r["path"]).convert("RGB")).unsqueeze(0).numpy()
        if CLASSES[int(_probs(s8, x).argmax())] != r["predicted_class"]:
            parity_mismatch.append(r["crop_id"])
    drop_pp = (correct_orig - correct8) / n * 100
    return {
        "val_rows": n,
        "accuracy_original": correct_orig / n,
        "accuracy_int8": correct8 / n,
        "accuracy_drop_pp": drop_pp,
        "max_accuracy_drop_pp": MAX_ACCURACY_DROP_PP,
        "class_agreement_with_original": agree / n,
        "max_abs_delta_prob_vs_fp32_onnx_val": max_delta,
        "parity_reference_rows": len(parity_rows),
        "parity_reference_class_mismatches": parity_mismatch,
        "class_order": list(CLASSES),
        "passed": drop_pp <= MAX_ACCURACY_DROP_PP,
    }


def describe_io(model: Path) -> dict:
    """Nombres, formas y tipos de entrada/salida leídos del .onnx (`None` = dimensión dinámica)."""
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def describe(nodes) -> list[dict]:
        return [{"name": n.name, "shape": list(n.shape), "type": n.type} for n in nodes]

    return {"inputs": describe(session.get_inputs()), "outputs": describe(session.get_outputs())}


def build_package(directory: Path, files: list[Path], out: Path) -> str:
    """tar.gz determinista (orden, mtime y dueño fijos): mismo contenido -> mismo SHA-256."""
    buf = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for path in sorted(files, key=lambda p: p.name):
            info = tarfile.TarInfo(path.name)
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            with open(path, "rb") as f:
                tar.addfile(info, f)
    out.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def run(
    *,
    fp32: Path,
    manifest: Path,
    crops: Path,
    package_src: Path,
    image_size: int,
    checkpoint_sha: str,
    root: Path,
    opset: int,
    libraries: dict,
    command: str,
) -> int:
    out_dir = root / "build" / "edge" / MODEL_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    int8 = out_dir / MODEL_FILE
    reports = root / "reports" / "p4" / "conversion"

    chosen = select_calibration(manifest)
    ids_sha = write_calibration_ids(reports / "calibration_ids.csv", chosen)
    quantize(fp32, int8, chosen, crops, image_size)
    quality = evaluate(
        fp32,
        int8,
        manifest,
        crops,
        image_size,
        root / "reports" / "p4" / "quality" / "predictions_val_original.csv",
        root / "reports" / "p4" / "reference" / "parity_reference.csv",
    )

    registry = json.loads((root / "models" / "registry.json").read_text(encoding="utf-8"))["1.0.0"]
    log = {
        "model_version": MODEL_VERSION,
        "input": {
            "source_version": "1.0.0",
            "source_package_sha256": registry["sha256"],
            "source_package_version_id": registry["VersionId"],
            "source_checkpoint_sha256": checkpoint_sha,
            "intermediate_fp32_sha256": sha256_file(fp32),
        },
        "output": {
            "file": MODEL_FILE,
            "sha256": sha256_file(int8),
            "bytes": int8.stat().st_size,
            "format": "onnx",
            "precision": "int8",
            "runtime": "onnxruntime-cpu",
            "io": describe_io(int8),
        },
        "technique": {
            "name": "static_quantization",
            "quant_format": "QDQ",
            "per_channel": True,
            "reduce_range": True,
            "weight_type": "QInt8",
            "activation_type": "QUInt8",
            "calibrate_method": "MinMax",
            "calibration_samples": CALIBRATION_SIZE,
            "calibration_seed": CALIBRATION_SEED,
            "calibration_split": "train",
        },
        "calibration_ids_sha256": ids_sha,
        "quality": quality,
        "opset": opset,
        "libraries": libraries,
        "command": command,
        "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log_path = reports / "conversion_log.json"
    log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    for name in ("class_map.json", "preprocess.json"):
        shutil.copy(package_src / name, out_dir / name)
    shutil.copy(log_path, out_dir / "conversion_log.json")
    shutil.copy(reports / "calibration_ids.csv", out_dir / "calibration_ids.csv")
    members = [
        out_dir / n
        for n in (
            MODEL_FILE,
            "conversion_log.json",
            "class_map.json",
            "preprocess.json",
            "calibration_ids.csv",
        )
    ]
    package = out_dir.parent / f"edge_package_{MODEL_VERSION}.tar.gz"
    package_sha = build_package(out_dir, members, package)

    summary = {
        "package": package.name,
        "package_sha256": package_sha,
        "package_bytes": package.stat().st_size,
        **log["output"],
        "quality": quality,
    }
    print(json.dumps(summary, indent=2))
    return 0 if quality["passed"] else 1
