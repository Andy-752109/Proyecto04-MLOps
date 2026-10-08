"""P4-11: protocolo y artefactos con imágenes/backends sintéticos, sin red ni modelos reales."""

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from edge import benchmark
from edge.backend import OnnxRuntimeBackend


class FakeBackend:
    name = "fake-cpu"

    def __init__(self) -> None:
        self.loaded_paths: list[Path] = []

    def load(self, path: Path) -> None:
        self.loaded_paths.append(path)

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        assert tensor.shape == (1, 3, 128, 128)
        return np.array([0.8, 0.2])


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def references(self, *, make_images: bool = True) -> tuple[Path, Path]:
        reference = self.tmp / "parity_reference.csv"
        shutil.copy(
            benchmark.ROOT / "reports/p4/reference/parity_reference.csv", reference
        )
        crops = self.tmp / "crops"
        if make_images:
            with reference.open(newline="", encoding="utf-8") as f:
                for index, row in enumerate(csv.DictReader(f)):
                    path = crops / row["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (9, 7), (index, 20, 30)).save(path)
        return reference, crops

    def artifacts(self) -> tuple[Path, Path, str, str]:
        original_package = self.tmp / "original-package"
        checkpoint = original_package / "artifacts/checkpoint/best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        int8_model = self.tmp / "custom" / "model_int8.onnx"
        int8_model.parent.mkdir(parents=True, exist_ok=True)
        int8_model.write_bytes(b"synthetic onnx")
        original_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        optimized_sha = hashlib.sha256(int8_model.read_bytes()).hexdigest()
        documents = {
            "models/registry.json": {
                "1.0.0": {"checkpoint_sha256": original_sha, "data_release": "v0.1.1"}
            },
            "models/edge_registry.json": {
                "1.0.0-int8": {
                    "source_checkpoint_sha256": original_sha,
                    "model_sha256": optimized_sha,
                    "package_sha256": "f" * 64,
                }
            },
            "reports/selection.json": {"checkpoint_sha256": original_sha},
            "app/class_map.json": {"0": "cat", "1": "dog"},
        }
        for relative, data in documents.items():
            path = self.tmp / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        return original_package, int8_model, original_sha, optimized_sha

    def test_verify_artifacts_uses_exact_files_and_versioned_shas(self) -> None:
        package, int8_model, original_sha, optimized_sha = self.artifacts()
        with patch.object(benchmark, "ROOT", self.tmp):
            found = benchmark.verify_artifacts(package, int8_model)
        checkpoint = package / "artifacts/checkpoint/best.pt"
        self.assertEqual(found["original"]["path"], str(checkpoint))
        self.assertEqual(found["original"]["artifact_bytes"], checkpoint.stat().st_size)
        self.assertEqual(found["original"]["sha256"], original_sha)
        self.assertEqual(found["optimized"]["path"], str(int8_model))
        self.assertEqual(
            found["optimized"]["artifact_bytes"], int8_model.stat().st_size
        )
        self.assertEqual(found["optimized"]["sha256"], optimized_sha)

    def test_verify_artifacts_rejects_wrong_sha_and_missing_registry_version(
        self,
    ) -> None:
        package, int8_model, _, optimized_sha = self.artifacts()
        edge_registry = self.tmp / "models/edge_registry.json"
        with patch.object(benchmark, "ROOT", self.tmp):
            # package_sha256 no sustituye model_sha256, aunque sea el SHA del archivo.
            data = json.loads(edge_registry.read_text(encoding="utf-8"))
            data["1.0.0-int8"]["model_sha256"] = "0" * 64
            data["1.0.0-int8"]["package_sha256"] = optimized_sha
            edge_registry.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(benchmark.BenchmarkError, "SHA-256 incorrecto"):
                benchmark.verify_artifacts(package, int8_model)
            edge_registry.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                benchmark.BenchmarkError, "faltan las versiones"
            ):
                benchmark.verify_artifacts(package, int8_model)

    def test_verify_artifacts_checks_original_selection_and_file_sha(self) -> None:
        package, int8_model, _, _ = self.artifacts()
        selection = self.tmp / "reports/selection.json"
        with patch.object(benchmark, "ROOT", self.tmp):
            selection.write_text(
                json.dumps({"checkpoint_sha256": "0" * 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(benchmark.BenchmarkError, "selección P3"):
                benchmark.verify_artifacts(package, int8_model)
            package, int8_model, _, _ = self.artifacts()
            (package / "artifacts/checkpoint/best.pt").write_bytes(b"altered")
            with self.assertRaisesRegex(benchmark.BenchmarkError, "SHA-256 incorrecto"):
                benchmark.verify_artifacts(package, int8_model)

    def test_onnx_backend_applies_threads_and_cpu_provider(self) -> None:
        options = types.SimpleNamespace(intra_op_num_threads=None)
        session = MagicMock()
        session.get_inputs.return_value = [types.SimpleNamespace(name="input")]
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.__version__ = "1.30.0"
        fake_ort.SessionOptions = MagicMock(return_value=options)
        fake_ort.InferenceSession = MagicMock(return_value=session)
        path = self.tmp / "model_int8.onnx"
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            backend = OnnxRuntimeBackend(threads=1)
            backend.load(path)
        self.assertEqual(options.intra_op_num_threads, 1)
        fake_ort.InferenceSession.assert_called_once_with(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def test_torch_backend_applies_threads_cpu_eval_and_inference_mode(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.14.0"
        fake_torch.set_num_threads = MagicMock()
        fake_torch.set_num_interop_threads = MagicMock()
        fake_torch.inference_mode = MagicMock()
        fake_torch.from_numpy = MagicMock(return_value="cpu-tensor")
        probability = MagicMock()
        probability.cpu.return_value.numpy.return_value = np.array([0.8, 0.2])
        fake_torch.softmax = MagicMock(return_value=[probability])
        model = MagicMock(return_value="logits")
        model.cpu.return_value = model
        model.eval.return_value = model
        fake_baseline = types.ModuleType("app.edge_model.baseline")
        fake_baseline.EXPECTED_CHECKPOINT_SHA256 = "a" * 64
        fake_baseline.load_package = MagicMock(
            return_value=(model, {"image_size": 128}, "a" * 64)
        )
        package = self.tmp / "original-package"
        with patch.dict(
            sys.modules, {"torch": fake_torch, "app.edge_model.baseline": fake_baseline}
        ):
            backend = benchmark.TorchBackend(threads=3)
            backend.load(package)
            result = backend.predict(np.zeros((1, 3, 128, 128), dtype=np.float32))
        fake_torch.set_num_threads.assert_called_once_with(3)
        fake_torch.set_num_interop_threads.assert_called_once_with(1)
        fake_baseline.load_package.assert_called_once_with(package, "a" * 64)
        model.cpu.assert_called_once_with()
        model.eval.assert_called_once_with()
        model.cuda.assert_not_called()
        fake_torch.inference_mode.assert_called_once_with()
        fake_torch.from_numpy.assert_called_once()
        self.assertEqual(result.tolist(), [0.8, 0.2])

    def test_fixed_reference_and_missing_or_duplicate_inputs(self) -> None:
        reference, crops = self.references()
        inputs, digest = benchmark.load_inputs(reference, crops)
        self.assertEqual(len(inputs), 20)
        self.assertEqual(len({item.input_id for item in inputs}), 20)
        self.assertEqual(digest, hashlib.sha256(reference.read_bytes()).hexdigest())
        (crops / "images" / f"{inputs[0].input_id}.jpg").unlink()
        with self.assertRaisesRegex(benchmark.BenchmarkError, "falta el recorte"):
            benchmark.load_inputs(reference, crops)
        with reference.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows[1]["crop_id"] = rows[0]["crop_id"]
        with reference.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "20 crop_id únicos"):
            benchmark.load_inputs(reference, crops)

    def test_crlf_checkout_keeps_the_fixed_logical_reference(self) -> None:
        reference, crops = self.references()
        reference.write_bytes(reference.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(len(benchmark.load_inputs(reference, crops)[0]), 20)

    def test_protocol_and_csv_columns_without_external_services(self) -> None:
        reference, crops = self.references()
        inputs, _ = benchmark.load_inputs(reference, crops)
        rows = benchmark.run_samples(
            inputs, {"original": FakeBackend(), "optimized": FakeBackend()}, 10, 100
        )
        self.assertEqual(len(rows), 220)
        for variant in ("original", "optimized"):
            own = [r for r in rows if r["variant"] == variant]
            self.assertEqual(len([r for r in own if r["is_warmup"]]), 10)
            self.assertEqual(len([r for r in own if not r["is_warmup"]]), 100)
            self.assertEqual(
                [r["input_id"] for r in own[10:]],
                [item.input_id for item in inputs] * 5,
            )
            self.assertTrue(all(r["total_ms"] >= r["preprocess_ms"] for r in own))
        self.assertEqual(
            {r["rep"]: r["input_id"] for r in rows if r["variant"] == "original"},
            {r["rep"]: r["input_id"] for r in rows if r["variant"] == "optimized"},
        )
        out = self.tmp / "latency_raw.csv"
        benchmark.write_csv(out, rows)
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            saved = list(reader)
            self.assertEqual(reader.fieldnames, list(benchmark.CSV_COLUMNS))
        self.assertEqual(len(saved), 220)
        self.assertEqual(sum(r["is_warmup"] == "true" for r in saved), 20)
        result = benchmark.summarize_rows(rows, 10, 100)
        from_csv = [
            float(r["total_ms"])
            for r in saved
            if r["variant"] == "original" and r["is_warmup"] == "false"
        ]
        self.assertAlmostEqual(
            result["original"]["total_ms"]["p95"],
            benchmark.linear_percentile(from_csv, 0.95),
        )

    def test_percentile_is_linear_and_excludes_warmups(self) -> None:
        self.assertEqual(benchmark.linear_percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(benchmark.linear_percentile([1, 2, 3, 4], 0.95), 3.85)
        rows = []
        for variant in ("original", "optimized"):
            for rep in range(1, 111):
                value = 100000 if rep <= 10 else rep - 10
                rows.append(
                    {
                        "variant": variant,
                        "is_warmup": rep <= 10,
                        **{k: value for k in benchmark.METRICS},
                    }
                )
        result = benchmark.summarize_rows(rows, 10, 100)
        self.assertEqual(result["original"]["total_ms"]["p50"], 50.5)
        self.assertAlmostEqual(result["optimized"]["inference_ms"]["p95"], 95.05)

    def test_artifact_size_sha_and_reduction_formula(self) -> None:
        original = self.tmp / "best.pt"
        optimized = self.tmp / "model.onnx"
        original.write_bytes(b"x" * 100)
        optimized.write_bytes(b"y" * 25)
        a = benchmark.artifact_info(
            original, hashlib.sha256(original.read_bytes()).hexdigest()
        )
        b = benchmark.artifact_info(
            optimized, hashlib.sha256(optimized.read_bytes()).hexdigest()
        )
        self.assertEqual((a["artifact_bytes"], b["artifact_bytes"]), (100, 25))
        self.assertEqual(
            benchmark.size_reduction_percent(a["artifact_bytes"], b["artifact_bytes"]),
            75,
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "SHA-256 incorrecto"):
            benchmark.artifact_info(original, "0" * 64)

    def test_quality_is_optional_and_checks_model_provenance(self) -> None:
        path = self.tmp / "metrics_val.json"
        self.assertEqual(
            benchmark.quality_evidence(path, "a" * 64, "b" * 64)["status"], "pending"
        )
        data = {
            "split": "val",
            "rows": 131,
            "class_order": ["cat", "dog"],
            "model_original": "checkpoint sha256 " + "a" * 64,
            "model_optimized": "model sha256 " + "b" * 64,
            "original": {"accuracy": 0.9, "macro_f1": 0.8},
            "optimized": {"accuracy": 0.9, "macro_f1": 0.8},
            "accuracy_drop_pp": 0.0,
            "class_disagreements": 0,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        evidence = benchmark.quality_evidence(path, "a" * 64, "b" * 64)
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["samples"], 131)
        self.assertEqual(
            evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(
            benchmark.quality_evidence(path, "c" * 64, "b" * 64)["status"], "invalid"
        )

    def test_quality_reads_versioned_p4_10_evidence(self) -> None:
        root = benchmark.ROOT
        path = root / "reports/p4/quality/metrics_val.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        original_sha = json.loads(
            (root / "models/registry.json").read_text(encoding="utf-8")
        )["1.0.0"]["checkpoint_sha256"]
        optimized_sha = json.loads(
            (root / "models/edge_registry.json").read_text(encoding="utf-8")
        )["1.0.0-int8"]["model_sha256"]

        evidence = benchmark.quality_evidence(path, original_sha, optimized_sha)

        self.assertEqual(evidence["status"], "available")
        self.assertEqual(
            evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(evidence["samples"], data["rows"])
        self.assertEqual(evidence["accuracy_original"], data["original"]["accuracy"])
        self.assertEqual(evidence["accuracy_int8"], data["optimized"]["accuracy"])
        self.assertEqual(evidence["macro_f1_original"], data["original"]["macro_f1"])
        self.assertEqual(evidence["macro_f1_int8"], data["optimized"]["macro_f1"])
        self.assertEqual(evidence["accuracy_drop_pp"], data["accuracy_drop_pp"])
        self.assertEqual(evidence["class_disagreements"], data["class_disagreements"])

    def test_quality_rejects_invalid_p4_10_json_and_provenance(self) -> None:
        source = benchmark.ROOT / "reports/p4/quality/metrics_val.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        original_sha = json.loads(
            (benchmark.ROOT / "models/registry.json").read_text(encoding="utf-8")
        )["1.0.0"]["checkpoint_sha256"]
        optimized_sha = json.loads(
            (benchmark.ROOT / "models/edge_registry.json").read_text(encoding="utf-8")
        )["1.0.0-int8"]["model_sha256"]
        path = self.tmp / "metrics_val.json"

        path.write_text("{", encoding="utf-8")
        malformed = benchmark.quality_evidence(path, original_sha, optimized_sha)
        self.assertEqual(malformed["status"], "invalid")
        self.assertIn("no se pudo leer", malformed["reason"])

        missing = json.loads(json.dumps(data))
        del missing["optimized"]["macro_f1"]
        path.write_text(json.dumps(missing), encoding="utf-8")
        incomplete = benchmark.quality_evidence(path, original_sha, optimized_sha)
        self.assertEqual(incomplete["status"], "invalid")
        self.assertIn("macro_f1", incomplete["reason"])

        path.write_text(json.dumps(data), encoding="utf-8")
        mismatch = benchmark.quality_evidence(path, "0" * 64, optimized_sha)
        self.assertEqual(mismatch["status"], "invalid")
        self.assertIn("modelos", mismatch["reason"])

    def test_perf_counter_ns_defines_the_three_durations(self) -> None:
        input_ = benchmark.ReferenceInput(
            "synthetic",
            self.tmp / "unused.jpg",
            np.zeros((2, 2, 3), dtype=np.uint8),
            "0" * 64,
        )
        with patch.object(
            benchmark.time, "perf_counter_ns", side_effect=[100, 2_000_100, 5_000_100]
        ):
            durations = benchmark.measure_once(input_, FakeBackend())
        self.assertEqual(
            durations, {"preprocess_ms": 2.0, "inference_ms": 3.0, "total_ms": 5.0}
        )

    def test_complete_report_is_recalculable_without_real_models(self) -> None:
        reference, crops = self.references()
        package = self.tmp / "chosen-package"
        int8_model = self.tmp / "chosen-int8.onnx"
        original_backend = FakeBackend()
        optimized_backend = FakeBackend()
        artifacts = {
            "original": {"path": "best.pt", "sha256": "a" * 64, "artifact_bytes": 100},
            "optimized": {
                "path": "model.onnx",
                "sha256": "b" * 64,
                "artifact_bytes": 25,
            },
        }
        output = self.tmp / "complete"
        with (
            patch.object(benchmark.platform, "system", return_value="Windows"),
            patch.object(
                benchmark, "load_inputs", wraps=benchmark.load_inputs
            ) as load_inputs,
            patch.object(
                benchmark, "verify_artifacts", return_value=artifacts
            ) as verify,
            patch.object(
                benchmark, "TorchBackend", return_value=original_backend
            ) as torch_factory,
            patch.object(
                benchmark, "OnnxRuntimeBackend", return_value=optimized_backend
            ) as onnx_factory,
        ):
            code = benchmark.main(
                [
                    "--reference",
                    str(reference),
                    "--crops",
                    str(crops),
                    "--original-package",
                    str(package),
                    "--int8-model",
                    str(int8_model),
                    "--threads",
                    "3",
                    "--quality-json",
                    str(self.tmp / "missing-metrics.json"),
                    "--output-dir",
                    str(output),
                ]
            )
        self.assertEqual(code, 0)
        load_inputs.assert_called_once_with(reference, crops)
        verify.assert_called_once_with(package, int8_model)
        torch_factory.assert_called_once_with(3)
        onnx_factory.assert_called_once_with(3)
        self.assertEqual(original_backend.loaded_paths, [package])
        self.assertEqual(optimized_backend.loaded_paths, [int8_model])
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["size_reduction_percent"], 75)
        self.assertEqual(report["variants"]["original"]["checkpoint_sha256"], "a" * 64)
        self.assertEqual(report["variants"]["optimized"]["model_sha256"], "b" * 64)
        self.assertEqual(report["quality_evidence"]["status"], "pending")
        self.assertEqual(
            report["quality_evidence"]["source"], str(self.tmp / "missing-metrics.json")
        )
        raw = output / "latency_raw.csv"
        self.assertEqual(
            report["latency_raw_csv_sha256"],
            hashlib.sha256(raw.read_bytes()).hexdigest(),
        )
        with raw.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for variant in ("original", "optimized"):
            measured = [
                float(r["total_ms"])
                for r in rows
                if r["variant"] == variant and r["is_warmup"] == "false"
            ]
            self.assertEqual(len(measured), 100)
            self.assertAlmostEqual(
                report["variants"][variant]["total_ms"]["p50"],
                benchmark.linear_percentile(measured, 0.5),
            )
            self.assertAlmostEqual(
                report["variants"][variant]["total_ms"]["p95"],
                benchmark.linear_percentile(measured, 0.95),
            )

    def test_original_failure_records_error_and_sizes_without_latency(self) -> None:
        reference, crops = self.references()
        inputs, digest = benchmark.load_inputs(reference, crops)
        artifacts = {
            "original": {"path": "best.pt", "sha256": "a" * 64, "artifact_bytes": 100},
            "optimized": {
                "path": "model.onnx",
                "sha256": "b" * 64,
                "artifact_bytes": 25,
            },
        }
        output = self.tmp / "failed"
        with (
            patch.object(benchmark.platform, "system", return_value="Windows"),
            patch.object(benchmark, "load_inputs", return_value=(inputs, digest)),
            patch.object(benchmark, "verify_artifacts", return_value=artifacts),
            patch.object(
                benchmark.TorchBackend,
                "load",
                side_effect=RuntimeError("error real PyTorch"),
            ),
        ):
            code = benchmark.main(
                [
                    "--reference",
                    str(reference),
                    "--crops",
                    str(crops),
                    "--quality-json",
                    str(self.tmp / "missing-metrics.json"),
                    "--output-dir",
                    str(output),
                ]
            )
        self.assertEqual(code, 1)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "incomplete")
        self.assertIn("error real PyTorch", summary["error"]["message"])
        self.assertEqual(summary["size_reduction_percent"], 75)
        self.assertNotIn("total_ms", summary["variants"]["original"])
        self.assertFalse((output / "latency_raw.csv").exists())


if __name__ == "__main__":
    unittest.main()
