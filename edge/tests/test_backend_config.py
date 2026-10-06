"""Capa del modelo (ONNX Runtime) y lectura de config.yaml."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from fixtures import write_config, write_tiny_model

from edge.backend import OnnxRuntimeBackend, create_backend, softmax
from edge.config import ConfigError, load_config

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config.example.yaml"


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.model = self.tmp / "model.onnx"
        write_tiny_model(self.model)

    def test_onnxruntime_returns_two_probabilities(self) -> None:
        backend = create_backend("onnxruntime")
        backend.load(self.model)
        red = np.zeros((1, 3, 128, 128), dtype=np.float32)
        red[0, 0] = 1.0
        probs = backend.predict(red)
        self.assertEqual(probs.shape, (2,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)
        self.assertGreater(probs[0], probs[1])  # rojo -> cat en el modelo diminuto
        self.assertTrue(backend.name.startswith("onnxruntime-"))

    def test_predict_before_load_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            OnnxRuntimeBackend().predict(np.zeros((1, 3, 128, 128), dtype=np.float32))

    def test_unknown_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime desconocido"):
            create_backend("tensorrt")

    def test_generic_resnet18_runs_with_p3_interface(self) -> None:
        import onnx

        from edge.tools.generic_resnet18 import build

        path = self.tmp / "generic.onnx"
        onnx.save(build(), path)
        backend = create_backend("onnxruntime")
        backend.load(path)
        probs = backend.predict(np.zeros((1, 3, 128, 128), dtype=np.float32))
        self.assertEqual(probs.shape, (2,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)

    def test_softmax_is_stable(self) -> None:
        probs = softmax(np.array([1000.0, 0.0]))
        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(float(probs[0]), 1.0)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_example_config_loads(self) -> None:
        config = load_config(EXAMPLE_CONFIG)
        self.assertEqual(config.mode, "manual")
        self.assertIsNone(config.crop)
        self.assertEqual(config.model.path, EXAMPLE_CONFIG.parent / "models" / "model.onnx")
        self.assertEqual(config.data_dir, EXAMPLE_CONFIG.parent / "data")

    def test_relative_paths_resolve_against_config_folder(self) -> None:
        config = load_config(write_config(self.tmp, "a" * 64))
        self.assertEqual(config.model.path, (self.tmp / "models" / "model.onnx").resolve())
        self.assertEqual(config.log_path, (self.tmp / "data" / "captures.jsonl").resolve())

    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, "config.example.yaml"):
            load_config(self.tmp / "nope.yaml")

    def test_invalid_values(self) -> None:
        cases = {
            "mode": {"capture": {"mode": "video"}},
            "sha": {"model": {"path": "m.onnx", "version": "v", "sha256": "xyz"}},
            "crop keys": {"crop": {"x": 0, "y": 0, "width": 10}},
            "crop float": {"crop": {"x": 0.5, "y": 0, "width": 10, "height": 10}},
            "crop zero": {"crop": {"x": 0, "y": 0, "width": 0, "height": 10}},
            "interval": {"capture": {"mode": "interval", "interval_seconds": 0}},
        }
        for name, override in cases.items():
            with self.subTest(case=name), self.assertRaises(ConfigError):
                load_config(write_config(self.tmp, "a" * 64, **override))

    def test_crop_is_parsed(self) -> None:
        crop = {"x": 10, "y": 20, "width": 100, "height": 80}
        config = load_config(write_config(self.tmp, "a" * 64, crop=crop))
        self.assertEqual(config.crop, crop)

    def test_example_has_no_secrets(self) -> None:
        raw = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        text = str(raw).lower()
        for word in ("secret", "token", "password", "access_key"):
            self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
