"""Benchmark local P4-11: PyTorch 1.0.0 frente a ONNX INT8, sin cámara ni red.

Ejecutar desde la raíz del repositorio en edge-laptop-01: ``python -m edge.benchmark``.
Los JPEG se decodifican y los modelos se verifican/cargan antes de cronometrar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edge.backend import CLASSES, InferenceBackend, OnnxRuntimeBackend, sha256_file
from edge.preprocess import IMAGE_SIZE, preprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256_CANONICAL = (
    "9c21e2439cf3c546e8b24eec02e436dae3f66d017abb138a51ca41708b9da18c"
)
CSV_COLUMNS = (
    "variant",
    "rep",
    "is_warmup",
    "input_id",
    "preprocess_ms",
    "inference_ms",
    "total_ms",
)
METRICS = ("preprocess_ms", "inference_ms", "total_ms")


class BenchmarkError(RuntimeError):
    """Precondición o ejecución inválida; nunca se convierte en una métrica."""


@dataclass(frozen=True)
class ReferenceInput:
    input_id: str
    path: Path
    rgb: np.ndarray
    sha256: str


class TorchBackend:
    """Adaptador de benchmark; no se registra en la app de captura P4-06."""

    def __init__(self, threads: int) -> None:
        self.threads = threads
        self.name = "pytorch"
        self._torch: Any = None
        self._model: Any = None

    def load(self, package_dir: Path) -> None:
        import torch

        from app.edge_model.baseline import EXPECTED_CHECKPOINT_SHA256, load_package

        torch.set_num_threads(self.threads)
        torch.set_num_interop_threads(1)
        model, config, _ = load_package(package_dir, EXPECTED_CHECKPOINT_SHA256)
        if config["image_size"] != IMAGE_SIZE:
            raise BenchmarkError("el modelo original no usa entrada 128×128")
        self._torch = torch
        self._model = model.cpu().eval()
        self.name = f"pytorch-{torch.__version__}"

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise BenchmarkError("el modelo PyTorch no se ha cargado")
        with self._torch.inference_mode():
            logits = self._model(self._torch.from_numpy(tensor))
            probabilities = self._torch.softmax(logits, dim=1)[0].cpu().numpy()
        if probabilities.shape != (len(CLASSES),):
            raise BenchmarkError("PyTorch no devolvió dos probabilidades")
        return probabilities


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"no se pudo leer {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BenchmarkError(f"{path} debe contener un objeto JSON")
    return data


def artifact_info(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError(f"falta el artefacto {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise BenchmarkError(
            f"SHA-256 incorrecto en {path}: esperado {expected_sha256}, real {actual}"
        )
    return {"path": str(path), "sha256": actual, "artifact_bytes": path.stat().st_size}


def size_reduction_percent(original_bytes: int, optimized_bytes: int) -> float:
    if original_bytes <= 0 or optimized_bytes <= 0:
        raise BenchmarkError("los artefactos deben tener tamaño positivo")
    return 100 * (original_bytes - optimized_bytes) / original_bytes


def verify_artifacts(
    original_package: Path, int8_model: Path
) -> dict[str, dict[str, Any]]:
    original_registry = _json(ROOT / "models/registry.json").get("1.0.0")
    edge_registry = _json(ROOT / "models/edge_registry.json").get("1.0.0-int8")
    selection_sha = _json(ROOT / "reports/selection.json").get("checkpoint_sha256")
    class_map = _json(ROOT / "app/class_map.json")
    if class_map != {"0": "cat", "1": "dog"}:
        raise BenchmarkError("app/class_map.json no coincide con cat,dog")
    if not isinstance(original_registry, dict) or not isinstance(edge_registry, dict):
        raise BenchmarkError("faltan las versiones 1.0.0 / 1.0.0-int8 en los registros")
    original_sha = original_registry.get("checkpoint_sha256")
    optimized_sha = edge_registry.get("model_sha256")
    if not isinstance(original_sha, str) or original_sha != selection_sha:
        raise BenchmarkError(
            "SHA del checkpoint original no coincide con la selección P3"
        )
    if not isinstance(optimized_sha, str):
        raise BenchmarkError("el registro INT8 no contiene model_sha256")
    if original_registry.get("data_release") != "v0.1.1":
        raise BenchmarkError("el modelo original no corresponde al release v0.1.1")
    if edge_registry.get("source_checkpoint_sha256") != original_sha:
        raise BenchmarkError("el INT8 no deriva del checkpoint original seleccionado")
    return {
        "original": artifact_info(
            original_package / "artifacts/checkpoint/best.pt", original_sha
        ),
        "optimized": artifact_info(int8_model, optimized_sha),
    }


def load_inputs(reference: Path, crops: Path) -> tuple[list[ReferenceInput], str]:
    try:
        raw = reference.read_bytes()
    except OSError as exc:
        raise BenchmarkError(f"no se pudo leer el CSV {reference}: {exc}") from exc
    # Git puede materializar CRLF en Windows; se fija el contenido lógico, no sus saltos de línea.
    canonical_sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames != [
            "crop_id",
            "path",
            "true_class",
            "predicted_class",
            "prob_cat",
            "prob_dog",
        ]:
            raise BenchmarkError("columnas inesperadas en parity_reference.csv")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise BenchmarkError(f"CSV inválido: {exc}") from exc
    ids = [row["crop_id"] for row in rows]
    if len(rows) != 20 or len(set(ids)) != 20:
        raise BenchmarkError("se requieren exactamente 20 crop_id únicos")
    if Counter(row["true_class"] for row in rows) != {"cat": 10, "dog": 10}:
        raise BenchmarkError("las referencias deben contener 10 cat y 10 dog")
    if canonical_sha != REFERENCE_SHA256_CANONICAL:
        raise BenchmarkError("el CSV no es el conjunto fijo aprobado de P4-01")
    inputs = []
    for row in rows:
        rel = Path(row["path"])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or rel.as_posix() != f"images/{row['crop_id']}.jpg"
        ):
            raise BenchmarkError(f"ruta inválida para {row['crop_id']}")
        image_path = crops / rel
        if not image_path.is_file():
            raise BenchmarkError(f"falta el recorte {image_path}")
        try:
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except (OSError, ValueError) as exc:
            raise BenchmarkError(f"JPEG inválido {image_path}: {exc}") from exc
        inputs.append(
            ReferenceInput(row["crop_id"], image_path, rgb, sha256_file(image_path))
        )
    return inputs, hashlib.sha256(raw).hexdigest()


def measure_once(input_: ReferenceInput, backend: InferenceBackend) -> dict[str, float]:
    t0 = time.perf_counter_ns()
    tensor = preprocess(input_.rgb)
    t1 = time.perf_counter_ns()
    probabilities = backend.predict(tensor)
    t2 = time.perf_counter_ns()
    if t0 > t1 or t1 > t2 or np.asarray(probabilities).shape != (len(CLASSES),):
        raise BenchmarkError("reloj o salida del backend inválidos")
    return {
        "preprocess_ms": (t1 - t0) / 1_000_000,
        "inference_ms": (t2 - t1) / 1_000_000,
        "total_ms": (t2 - t0) / 1_000_000,
    }


def run_samples(
    inputs: list[ReferenceInput],
    backends: dict[str, InferenceBackend],
    warmups: int,
    measurements: int,
) -> list[dict[str, Any]]:
    if len(inputs) != 20 or warmups != 10 or measurements < 100 or measurements % 20:
        raise BenchmarkError(
            "se requieren 20 entradas, 10 warmups y >=100 mediciones en ciclos de 20"
        )
    if set(backends) != {"original", "optimized"}:
        raise BenchmarkError("se requieren ambos backends")
    rows = []
    for index in range(warmups + measurements):
        input_ = (
            inputs[index % len(inputs)]
            if index < warmups
            else inputs[(index - warmups) % 20]
        )
        # Alternar el primer runtime de cada pareja reduce el sesgo por temperatura/orden.
        order = (
            ("original", "optimized") if index % 2 == 0 else ("optimized", "original")
        )
        for variant in order:
            rows.append(
                {
                    "variant": variant,
                    "rep": index + 1,
                    "is_warmup": index < warmups,
                    "input_id": input_.input_id,
                    **measure_once(input_, backends[variant]),
                }
            )
    return rows


def linear_percentile(values: list[float], q: float) -> float:
    if (
        not values
        or not 0 <= q <= 1
        or not all(math.isfinite(v) and v >= 0 for v in values)
    ):
        raise BenchmarkError("percentil sin datos válidos")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_rows(
    rows: list[dict[str, Any]], warmups: int, measurements: int
) -> dict[str, Any]:
    summary = {}
    for variant in ("original", "optimized"):
        own = [r for r in rows if r["variant"] == variant]
        warm = [r for r in own if r["is_warmup"]]
        measured = [r for r in own if not r["is_warmup"]]
        if len(warm) != warmups or len(measured) != measurements:
            raise BenchmarkError(f"conteo inválido para {variant}")
        summary[variant] = {
            "warmup_count": len(warm),
            "measurement_count": len(measured),
            **{
                metric: {
                    "p50": linear_percentile(
                        [float(r[metric]) for r in measured], 0.50
                    ),
                    "p95": linear_percentile(
                        [float(r[metric]) for r in measured], 0.95
                    ),
                }
                for metric in METRICS
            },
        }
    return summary


def quality_evidence(
    path: Path, original_sha: str, optimized_sha: str
) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "pending", "source": str(path)}
    result: dict[str, Any] = {
        "status": "invalid",
        "source": str(path),
        "sha256": sha256_file(path),
    }
    try:
        data = _json(path)
        if (
            data.get("split") != "val"
            or original_sha not in data.get("model_original", "")
            or optimized_sha not in data.get("model_optimized", "")
            or data.get("class_order") != list(CLASSES)
        ):
            raise BenchmarkError("split, modelos o clases de P4-10 no coinciden")
        fields = {
            "samples": data["rows"],
            "accuracy_original": data["original"]["accuracy"],
            "accuracy_int8": data["optimized"]["accuracy"],
            "macro_f1_original": data["original"]["macro_f1"],
            "macro_f1_int8": data["optimized"]["macro_f1"],
            "accuracy_drop_pp": data["accuracy_drop_pp"],
            "class_disagreements": data["class_disagreements"],
        }
        if (
            not isinstance(fields["samples"], int)
            or fields["samples"] != 131
            or not isinstance(fields["class_disagreements"], int)
            or fields["class_disagreements"] < 0
            or any(
                not isinstance(fields[k], (int, float)) or not math.isfinite(fields[k])
                for k in (
                    "accuracy_original",
                    "accuracy_int8",
                    "macro_f1_original",
                    "macro_f1_int8",
                    "accuracy_drop_pp",
                )
            )
        ):
            raise BenchmarkError("métricas de P4-10 inválidas")
    except (BenchmarkError, KeyError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
        return result
    return {**result, "status": "available", **fields}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "is_warmup": str(row["is_warmup"]).lower()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "reports/p4/reference/parity_reference.csv",
    )
    p.add_argument("--crops", type=Path, default=ROOT / "data/derived/crops")
    p.add_argument("--original-package", type=Path, default=ROOT / "eval_v1.0.0")
    p.add_argument(
        "--int8-model", type=Path, default=ROOT / "edge/models/model_int8.onnx"
    )
    p.add_argument(
        "--quality-json",
        type=Path,
        default=ROOT / "reports/p4/quality/metrics_val.json",
    )
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--measurements", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/p4/benchmark")
    p.add_argument("--device-id", default="edge-laptop-01")
    p.add_argument(
        "--power-state", choices=("connected", "battery", "unknown"), default="unknown"
    )
    p.add_argument("--other-processes", default="not_recorded")
    p.add_argument("--temperature-c", type=float)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.threads < 1
        or args.warmups != 10
        or args.measurements < 100
        or args.measurements % 20
    ):
        print(
            "threads >=1, warmups=10 y measurements >=100 múltiplo de 20",
            file=sys.stderr,
        )
        return 2
    if platform.system() != "Windows":
        print(
            "la evidencia final solo puede ejecutarse en edge-laptop-01 (Windows)",
            file=sys.stderr,
        )
        return 2
    raw_path = args.output_dir / "latency_raw.csv"
    summary_path = args.output_dir / "summary.json"
    if raw_path.exists() or summary_path.exists():
        print(
            "el directorio de salida ya contiene resultados; usa uno vacío",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "incomplete",
        "device": {
            "id": args.device_id,
            "hostname": platform.node(),
            "os": platform.platform(),
            "processor": platform.processor(),
        },
        "run_conditions": {
            "power_state": args.power_state,
            "other_processes": args.other_processes,
            "temperature_c": args.temperature_c,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "threads": {
            "pytorch_intra_op": args.threads,
            "pytorch_inter_op": 1,
            "onnx_intra_op": args.threads,
        },
        "protocol": {
            "warmups": args.warmups,
            "measurements": args.measurements,
            "percentile_method": "linear: position=(n-1)*q, interpolación entre vecinos",
            "included": "preprocess(RGB uint8 a tensor) + predict(probabilidades) sobre CPU",
            "excluded": [
                "cámara",
                "disco",
                "decodificación JPEG",
                "carga de modelos",
                "upload",
                "red",
            ],
            "sequence": "20 referencias en orden CSV, repetidas; orden de variantes alternado por rep",
        },
        "variants": {},
    }
    try:
        inputs, reference_sha = load_inputs(args.reference, args.crops)
        report["inputs"] = {
            "reference_csv": str(args.reference),
            "reference_csv_sha256": reference_sha,
            "input_count": len(inputs),
            "input_ids": [item.input_id for item in inputs],
            "jpeg_sha256": {item.input_id: item.sha256 for item in inputs},
        }
        artifacts = verify_artifacts(args.original_package, args.int8_model)
        report["variants"] = {
            "original": {
                "version": "1.0.0",
                "runtime": "pytorch-cpu (pendiente de cargar)",
                "path": artifacts["original"]["path"],
                "checkpoint_sha256": artifacts["original"]["sha256"],
                "artifact_bytes": artifacts["original"]["artifact_bytes"],
            },
            "optimized": {
                "version": "1.0.0-int8",
                "runtime": "onnxruntime-cpu (pendiente de cargar)",
                "path": artifacts["optimized"]["path"],
                "model_sha256": artifacts["optimized"]["sha256"],
                "artifact_bytes": artifacts["optimized"]["artifact_bytes"],
            },
        }
        original_bytes = artifacts["original"]["artifact_bytes"]
        report["size_reduction_percent"] = size_reduction_percent(
            original_bytes, artifacts["optimized"]["artifact_bytes"]
        )
        report["quality_evidence"] = quality_evidence(
            args.quality_json,
            artifacts["original"]["sha256"],
            artifacts["optimized"]["sha256"],
        )
        original = TorchBackend(args.threads)
        original.load(args.original_package)
        optimized = OnnxRuntimeBackend(args.threads)
        optimized.load(args.int8_model)
        report["runtime_versions"].update(
            {"torch": original.name, "onnxruntime": optimized.name}
        )
        rows = run_samples(
            inputs,
            {"original": original, "optimized": optimized},
            args.warmups,
            args.measurements,
        )
        statistics = summarize_rows(rows, args.warmups, args.measurements)
        for variant, backend in (("original", original), ("optimized", optimized)):
            report["variants"][variant].update(
                {"runtime": backend.name, **statistics[variant]}
            )
        write_csv(raw_path, rows)
        report["latency_raw_csv_sha256"] = sha256_file(raw_path)
        report["status"] = "complete"
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - auditar el fallo real del runtime
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(summary_path, report)
        print(f"benchmark incompleto: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    write_json(summary_path, report)
    print(f"benchmark completo: {raw_path} y {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
