"""Ruta completa de una captura, sin cámara real y sin red."""

import contextlib
import io
import json
import shutil
import socket
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from fixtures import solid_frame_bgr, write_config, write_tiny_model
from PIL import Image

from edge.__main__ import main
from edge.backend import create_backend
from edge.config import load_config
from edge.event_validator import validate_event
from edge.pipeline import EdgeApp, ModelVerificationError, read_log

CAPTURED_AT = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)
RED = (230, 30, 30)
BLUE = (30, 30, 230)
REAL_REGISTRY = Path(__file__).resolve().parents[2] / "models" / "edge_registry.json"
REGISTRY_VERSION = "1.0.0-int8"


def real_registry_entry() -> dict:
    """Copia de la entrada `1.0.0-int8` de `models/edge_registry.json` (#4)."""
    return dict(json.loads(REAL_REGISTRY.read_text(encoding="utf-8"))[REGISTRY_VERSION])


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def temp_dir(test: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def no_network(*args, **kwargs):
    raise AssertionError("la ruta de captura intentó abrir un socket")


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = temp_dir(self)
        self.sha = write_tiny_model(self.tmp / "models" / "model.onnx")

    def app(self, **overrides: object) -> EdgeApp:
        config = load_config(write_config(self.tmp, self.sha, **overrides))
        app = EdgeApp(config, create_backend(config.model.runtime))
        app.start()
        return app

    def test_capture_writes_valid_event_image_and_log(self) -> None:
        app = self.app()
        with mock.patch.object(socket, "socket", no_network):
            capture = app.process(solid_frame_bgr(RED), CAPTURED_AT)
        event = capture.event
        validate_event(event)
        self.assertEqual(event["predicted_class"], "cat")
        self.assertEqual(event["model_sha256"], self.sha)
        self.assertEqual(event["captured_at"], "2026-10-06T18:00:00+00:00")
        self.assertIsNone(event["crop"])
        self.assertEqual(event["image_key"], f"edge-captures/v1/images/{event['capture_id']}.jpg")
        self.assertEqual(image_size(capture.image_path), (320, 240))
        self.assertIsNone(capture.crop_path)
        (line,) = read_log(app.config.log_path)
        self.assertEqual(line["event"], event)
        self.assertEqual(line["image_path"], f"images/{event['capture_id']}.jpg")

    def test_prediction_changes_with_input(self) -> None:
        app = self.app()
        classes = [
            app.process(solid_frame_bgr(c), CAPTURED_AT).event["predicted_class"]
            for c in (RED, BLUE)
        ]
        self.assertEqual(classes, ["cat", "dog"])

    def test_crop_is_recorded_and_saved(self) -> None:
        crop = {"x": 40, "y": 20, "width": 200, "height": 160}
        capture = self.app(crop=crop).process(solid_frame_bgr(BLUE), CAPTURED_AT)
        self.assertEqual(capture.event["crop"], {**crop, "frame_width": 320, "frame_height": 240})
        self.assertEqual(image_size(capture.crop_path), (200, 160))
        self.assertEqual(image_size(capture.image_path), (320, 240))  # frame completo

    def test_center_square_crop_in_event(self) -> None:
        capture = self.app(crop={"center_square": 0.8}).process(solid_frame_bgr(RED), CAPTURED_AT)
        self.assertEqual(
            capture.event["crop"],
            {
                "x": 64,
                "y": 24,
                "width": 192,
                "height": 192,
                "frame_width": 320,
                "frame_height": 240,
            },
        )
        self.assertEqual(image_size(capture.crop_path), (192, 192))
        validate_event(capture.event)

    def test_crop_outside_frame_is_rejected(self) -> None:
        app = self.app(crop={"x": 300, "y": 0, "width": 100, "height": 100})
        with self.assertRaises(ValueError):
            app.process(solid_frame_bgr(RED), CAPTURED_AT)
        self.assertEqual(read_log(app.config.log_path), [])

    def test_restart_keeps_log_and_ids(self) -> None:
        first = self.app()
        ids = {
            first.process(solid_frame_bgr(RED), CAPTURED_AT).event["capture_id"] for _ in range(3)
        }
        second = self.app()
        ids.add(second.process(solid_frame_bgr(BLUE), CAPTURED_AT).event["capture_id"])
        lines = read_log(second.config.log_path)
        self.assertEqual(len(lines), 4)
        self.assertEqual(len(ids), 4)

    def test_wrong_sha_aborts_before_loading(self) -> None:
        config = load_config(write_config(self.tmp, "0" * 64))
        with self.assertRaisesRegex(ModelVerificationError, "no coincide"):
            EdgeApp(config, create_backend("onnxruntime")).start()

    def test_missing_model_aborts(self) -> None:
        (self.tmp / "models" / "model.onnx").unlink()
        with self.assertRaisesRegex(ModelVerificationError, "no hay modelo"):
            self.app()

    def registry_app(self, entry: object, version: str = REGISTRY_VERSION) -> EdgeApp:
        """App que verifica contra un registro con la forma de `models/edge_registry.json`."""
        registry = self.tmp / "edge_registry.json"
        registry.write_text(json.dumps({version: entry}), encoding="utf-8")
        model = {
            "path": "models/model.onnx",
            "version": REGISTRY_VERSION,
            "sha256": self.sha,
            "registry": "edge_registry.json",
        }
        return self.app(model=model)

    def test_registry_with_real_shape_must_agree_with_config(self) -> None:
        # La entrada real de #4, con el SHA del modelo diminuto en lugar del INT8.
        self.registry_app({**real_registry_entry(), "model_sha256": self.sha})
        with self.assertRaisesRegex(ModelVerificationError, "model_sha256"):
            self.registry_app({**real_registry_entry(), "model_sha256": "1" * 64})

    def test_registry_package_sha_is_not_the_model_sha(self) -> None:
        entry = {**real_registry_entry(), "package_sha256": self.sha}
        with self.assertRaisesRegex(ModelVerificationError, "no coincide con model_sha256"):
            self.registry_app(entry)

    def test_registry_without_model_sha256_is_a_verification_error(self) -> None:
        entry = real_registry_entry()
        del entry["model_sha256"]
        for case in (entry, {"sha256": self.sha}, "no-es-objeto"):
            with self.subTest(case=case):
                with self.assertRaisesRegex(ModelVerificationError, "no tiene model_sha256"):
                    self.registry_app(case)

    def test_registry_missing_version_or_unreadable(self) -> None:
        with self.assertRaisesRegex(ModelVerificationError, "no está en"):
            self.registry_app(real_registry_entry(), version="otra-version")
        (self.tmp / "edge_registry.json").write_text("{no es json", encoding="utf-8")
        model = {
            "path": "models/model.onnx",
            "version": REGISTRY_VERSION,
            "sha256": self.sha,
            "registry": "edge_registry.json",
        }
        with self.assertRaisesRegex(ModelVerificationError, "no se pudo leer el registro"):
            self.app(model=model)


class FakeCamera:
    frames = (solid_frame_bgr(RED), solid_frame_bgr(BLUE))

    def __init__(self, config: object) -> None:
        self.reads = 0

    def read(self):
        frame = self.frames[self.reads % len(self.frames)]
        self.reads += 1
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = temp_dir(self)
        self.config = write_config(self.tmp, write_tiny_model(self.tmp / "models" / "model.onnx"))

    def run_cli(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with (
            mock.patch("edge.__main__.Camera", FakeCamera),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = main(["--config", str(self.config), *args])
        return code, out.getvalue()

    def tearDown(self) -> None:
        import logging

        for handler in list(logging.getLogger("edge").handlers):
            handler.close()
            logging.getLogger("edge").removeHandler(handler)

    def test_interval_mode_captures_count_frames(self) -> None:
        code, out = self.run_cli("run", "--mode", "interval", "--count", "3")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("CAT"), 2)
        self.assertEqual(out.count("DOG"), 1)
        self.assertEqual(len(read_log(self.tmp / "data" / "captures.jsonl")), 3)
        log_text = (self.tmp / "data" / "edge.log").read_text(encoding="utf-8")
        self.assertIn("verificado (sha256=", log_text)

    def test_manual_mode_captures_on_enter_and_quits_on_q(self) -> None:
        with mock.patch("builtins.input", side_effect=["", "", "q"]):
            code, _ = self.run_cli("run", "--mode", "manual")
        self.assertEqual(code, 0)
        self.assertEqual(len(read_log(self.tmp / "data" / "captures.jsonl")), 2)

    def test_status_reports_model_and_log(self) -> None:
        self.run_cli("run", "--count", "1")
        code, out = self.run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("sha256      : OK", out)
        self.assertIn("capturas    : 1", out)

    def test_status_shows_the_camera_name_when_configured(self) -> None:
        self.config = write_config(
            self.tmp,
            write_tiny_model(self.tmp / "models" / "model.onnx"),
            camera={"name": "GENERAL WEBCAM"},
        )
        _, out = self.run_cli("status")
        self.assertIn("cámara      : GENERAL WEBCAM", out)

    def test_cameras_command_lists_devices_with_their_index(self) -> None:
        names = ["USB2.0 UVC HD Webcam", "GENERAL WEBCAM"]
        with mock.patch("edge.__main__.list_camera_names", return_value=names):
            code, out = self.run_cli("cameras")
        self.assertEqual(code, 0)
        self.assertIn("0: USB2.0 UVC HD Webcam", out)
        self.assertIn("1: GENERAL WEBCAM", out)

    def test_cameras_command_without_devices_exits_with_error(self) -> None:
        with mock.patch("edge.__main__.list_camera_names", return_value=[]):
            code, out = self.run_cli("cameras")
        self.assertEqual(code, 1)
        self.assertIn("no se encontraron cámaras", out)

    def test_cameras_command_does_not_need_a_valid_config(self) -> None:
        self.config = self.tmp / "missing.yaml"
        with mock.patch("edge.__main__.list_camera_names", return_value=["GENERAL WEBCAM"]):
            code, _ = self.run_cli("cameras")
        self.assertEqual(code, 0)

    def test_bad_sha_exits_with_error(self) -> None:
        self.config = write_config(self.tmp, "0" * 64)
        code, _ = self.run_cli("run", "--count", "1")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
