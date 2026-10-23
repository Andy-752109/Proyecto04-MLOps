"""La ventana manejada por código: sin cámara, sin red y con el modelo diminuto de las pruebas.

Las pruebas de la ventana necesitan tkinter y una pantalla: sin ellas se saltan (el CI de Linux
no tiene pantalla). El comando `gui` y los avisos de `run_gui` sin pantalla sí corren siempre.
"""

import contextlib
import gc
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
from botocore.exceptions import EndpointConnectionError
from fixtures import FakeS3, solid_frame_bgr, write_config, write_tiny_model
from PIL import Image

from edge.__main__ import main
from edge.backend import create_backend
from edge.camera import CameraError
from edge.config import load_config
from edge.pipeline import EdgeApp, read_log
from edge.preview import Frame, FrameGrabber
from edge.sync import BackgroundSender, UploadLog
from edge.uploader import event_body

try:
    import tkinter as tk

    from edge import gui
except ImportError:  # Python sin tkinter (p. ej. Linux sin python3-tk)
    tk = gui = None

BLUE = (30, 30, 230)
BUCKET = "mlops-p4-edge-captures-test"
NO_DISPLAY = "no display name and no $DISPLAY environment variable"


class StillSource:
    """Cámara de mentira: siempre el mismo cuadro azul (el modelo de prueba lo llama dog)."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def read_live(self) -> np.ndarray:
        return solid_frame_bgr(BLUE)

    def close(self) -> None:
        self.closed.set()


class GatedS3(FakeS3):
    """S3 que no responde hasta que se suelta `gate`: un envío que sigue en cola."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    def put_object(self, **kwargs):
        self.gate.wait(10)
        return super().put_object(**kwargs)


@unittest.skipIf(tk is None, "esta instalación de Python no tiene tkinter")
class WindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.sha = write_tiny_model(self.tmp / "models" / "model.onnx")
        # Tk no tolera que un objeto suyo se libere desde otro hilo (Tcl_AsyncDelete): sin
        # recolección automática, la basura de cada ventana se limpia aquí, en el hilo principal.
        gc.collect()
        gc.disable()
        self.addCleanup(self.collect_garbage)
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"sin pantalla: {error}")
        self.addCleanup(self.destroy_root)

    def collect_garbage(self) -> None:
        gc.collect()
        gc.enable()

    def destroy_root(self) -> None:
        with contextlib.suppress(tk.TclError):  # la ventana ya se cerró sola
            self.root.destroy()

    def pump(self, condition: Callable[[], object], timeout: float = 5.0) -> bool:
        """Deja correr Tk (eventos y `after`) hasta que se cumpla la condición."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.root.update()
            except tk.TclError:  # la ventana se destruyó
                return bool(condition())
            if condition():
                return True
            time.sleep(0.01)
        return False

    def build(
        self,
        source: StillSource | None = None,
        s3: FakeS3 | None = None,
        sha: str | None = None,
        open_source: Callable[[], StillSource] | None = None,
    ) -> "gui.EdgeGui":
        overrides: dict = {"crop": {"center_square": 0.8}}
        if s3 is not None:
            overrides["aws"] = {"bucket": BUCKET}
        config = load_config(write_config(self.tmp, sha or self.sha, **overrides))
        app = EdgeApp(config, create_backend(config.model.runtime))
        sender = None
        if s3 is not None:
            sender = BackgroundSender(
                config.aws, UploadLog(config.upload_log_path), lambda profile, region: s3
            ).start()
        source = source or StillSource()
        grabber = FrameGrabber(open_source or (lambda: source), max_fps=100).start()
        window = gui.EdgeGui(self.root, config, app, grabber, sender, "cámara de prueba", "pista")
        self.addCleanup(self.stop_threads, window)
        return window

    def stop_threads(self, window: "gui.EdgeGui") -> None:
        window.grabber.stop()
        window._pool.shutdown(wait=True)
        if window.sender is not None:
            window.sender.close(5)

    def ready(self, window: "gui.EdgeGui") -> None:
        enabled = lambda: not window.capture_button.instate(["disabled"])  # noqa: E731
        self.assertTrue(self.pump(enabled), window.message.get())

    def rows(self, window: "gui.EdgeGui") -> tuple[str, ...]:
        return window.tree.get_children()

    def test_button_enables_once_model_and_camera_are_ready(self) -> None:
        window = self.build()
        self.assertTrue(window.capture_button.instate(["disabled"]))
        self.ready(window)
        self.assertEqual(window.model_text.get(), "verificado y listo")
        self.assertRegex(window.camera_text.get(), r"^En vivo · 320x240 · ")

    def test_capture_classifies_the_frame_on_screen_and_saves_it(self) -> None:
        window = self.build()
        self.ready(window)
        window.capture()
        self.assertTrue(self.pump(lambda: self.rows(window)), window.message.get())
        (record,) = read_log(window.config.log_path)
        event = record["event"]
        self.assertEqual(event["predicted_class"], "dog")
        self.assertEqual(window.class_label.cget("text"), "DOG")
        self.assertEqual(window.id_var.get(), event["capture_id"])
        (row,) = self.rows(window)
        self.assertEqual(row, event["capture_id"])
        shown = [str(v) for v in window.tree.item(row, "values")]
        self.assertEqual(shown[0], "1")
        self.assertEqual(shown[2:], ["DOG", f"{event['confidence']:.3f}", "solo local"])
        data = window.config.data_dir
        with Image.open(data / record["image_path"]) as saved:
            self.assertEqual(saved.size, (320, 240))  # el cuadro completo, como en `run`
        with Image.open(data / record["crop_path"]) as crop:
            self.assertEqual(crop.size, (192, 192))
        self.assertTrue(window.thumbnail_label.cget("image"))
        self.assertTrue(self.pump(lambda: not window.capture_button.instate(["disabled"])))

    def test_space_captures_but_not_while_typing_in_the_seconds_field(self) -> None:
        window = self.build()
        self.ready(window)
        window._on_key(types.SimpleNamespace(widget=window.interval_box))
        self.assertFalse(self.pump(lambda: self.rows(window), timeout=0.3))
        window._on_key(types.SimpleNamespace(widget=self.root))
        self.assertTrue(self.pump(lambda: self.rows(window)))
        self.assertEqual(len(read_log(window.config.log_path)), 1)

    def test_model_that_does_not_verify_blocks_the_capture_and_says_why(self) -> None:
        with self.assertLogs("edge", level="ERROR"):
            window = self.build(sha="0" * 64)
            self.assertTrue(self.pump(lambda: window.model_text.get() == "NO disponible"))
        self.assertIn("SHA-256 del modelo no coincide", window.message.get())
        self.assertTrue(window.capture_button.instate(["disabled"]))
        window.capture()
        self.assertTrue(window.message.get().startswith("No se puede capturar"))
        self.assertEqual(read_log(window.config.log_path), [])

    def test_camera_that_does_not_open_shows_the_reason_in_the_window(self) -> None:
        def broken() -> StillSource:
            raise CameraError("no se pudo abrir la cámara 1 (api auto)")

        with self.assertLogs("edge", level="ERROR"):
            window = self.build(open_source=broken)
            shown = lambda: "pista" in window.canvas.itemcget(window._text_item, "text")  # noqa: E731
            self.assertTrue(self.pump(shown))
        text = window.canvas.itemcget(window._text_item, "text")
        self.assertIn("no se pudo abrir la cámara 1", text)
        self.assertIn("no se pudo abrir la cámara 1", window.camera_text.get())
        self.assertTrue(window.capture_button.instate(["disabled"]))
        # El modelo termina de cargar después: su aviso de "Listo" no tapa el de la cámara.
        self.assertTrue(self.pump(lambda: window.model_text.get() == "verificado y listo"))
        self.assertIn("no se pudo abrir la cámara 1", window.message.get())
        self.assertFalse(window.message.get().startswith("Listo"))
        window.capture()
        self.assertIn("la cámara no está disponible", window.message.get())

    def test_frame_older_than_a_second_blocks_the_capture(self) -> None:
        window = self.build()
        self.ready(window)
        old = Frame(solid_frame_bgr(BLUE), time.monotonic() - 5, 1)
        self.assertEqual(window._blocked_reason(old), "la cámara dejó de entregar cuadros")

    def test_auto_capture_takes_a_photo_every_interval_until_turned_off(self) -> None:
        window = self.build()
        self.ready(window)
        window.interval_var.set("0.05")
        window.auto.set(True)
        window._on_auto_toggle()
        self.assertTrue(self.pump(lambda: len(self.rows(window)) >= 2))
        window.auto.set(False)
        self.assertTrue(self.pump(lambda: not window._busy))
        self.pump(lambda: False, timeout=0.3)
        self.assertEqual(len(self.rows(window)), len(read_log(window.config.log_path)))
        self.assertEqual(window.countdown.get(), "")

    def test_invalid_interval_falls_back_to_the_config(self) -> None:
        window = self.build()
        for text in ("", "abc", "0", "-3"):
            with self.subTest(text=text):
                window.interval_var.set(text)
                self.assertEqual(window._interval(), window.config.interval_seconds)

    def test_upload_status_follows_the_sender_in_the_table(self) -> None:
        s3 = FakeS3()
        window = self.build(s3=s3)
        self.ready(window)
        window.capture()
        self.assertTrue(self.pump(lambda: self.rows(window)))
        (row,) = self.rows(window)
        self.assertTrue(self.pump(lambda: window.tree.set(row, "envio") == "enviada"))
        self.assertEqual(len(s3.keys("edge-captures/v1/events/")), 1)
        self.assertIn("1 enviadas", window.upload_text.get())

    def test_failed_upload_is_marked_and_points_to_retry(self) -> None:
        s3 = FakeS3()
        s3.fail_next = [EndpointConnectionError(endpoint_url="https://s3")]
        with self.assertLogs("edge", level="ERROR"):
            window = self.build(s3=s3)
            self.ready(window)
            window.capture()
            self.assertTrue(self.pump(lambda: self.rows(window)))
            (row,) = self.rows(window)
            self.assertTrue(self.pump(lambda: window.tree.set(row, "envio") == "FALLÓ"))
        self.assertIn("failed", window.tree.item(row, "tags"))
        self.assertIn("Reenviar fallidas", window.upload_text.get())
        self.assertIn("python -m edge retry", window.upload_text.get())

    def take_photo(self, window: "gui.EdgeGui") -> str:
        """Captura una foto y devuelve su capture_id (la fila nueva de la tabla)."""
        before = set(self.rows(window))
        self.assertTrue(
            self.pump(lambda: not window._busy and window.capture_button.instate(["!disabled"]))
        )
        window.capture()
        self.assertTrue(self.pump(lambda: set(self.rows(window)) - before))
        return (set(self.rows(window)) - before).pop()

    def envio(self, window: "gui.EdgeGui", row: str) -> str:
        return window.tree.set(row, "envio")

    def test_retry_button_resends_only_the_failed_capture(self) -> None:
        s3 = FakeS3()
        s3.fail_next = [EndpointConnectionError(endpoint_url="https://s3")]  # la 1.ª foto falla
        with self.assertLogs("edge", level="ERROR"):
            window = self.build(s3=s3)
            self.ready(window)
            failed_id = self.take_photo(window)
            self.assertTrue(self.pump(lambda: self.envio(window, failed_id) == "FALLÓ"))
        sent_id = self.take_photo(window)
        self.assertTrue(self.pump(lambda: self.envio(window, sent_id) == "enviada"))
        self.assertEqual(window.retry_button.cget("text"), "Reenviar fallidas (1)")
        self.assertFalse(window.retry_button.instate(["disabled"]))
        calls_before = list(s3.calls)

        window.retry_failed()
        self.assertTrue(self.pump(lambda: self.envio(window, failed_id) == "enviada"))

        sent_event = f"edge-captures/v1/events/{sent_id}.json"
        sent_calls = [k for k in s3.calls if sent_id in k]
        self.assertEqual(sent_calls, [k for k in calls_before if sent_id in k])  # no se tocó
        self.assertIn(
            f"edge-captures/v1/events/{failed_id}.json", s3.keys("edge-captures/v1/events/")
        )
        self.assertIn(sent_event, s3.keys("edge-captures/v1/events/"))
        self.assertTrue(self.pump(lambda: window.retry_button.instate(["disabled"])))
        self.assertEqual(window.retry_button.cget("text"), "Reenviar fallidas")
        self.assertNotIn("FALLARON", window.upload_text.get())
        self.assertEqual(window.tree.item(failed_id, "tags"), "")
        latest = UploadLog(window.config.upload_log_path).latest()
        self.assertEqual(latest[failed_id]["upload_status"], "sent")

    def test_retry_sends_the_same_event_it_saved(self) -> None:
        s3 = FakeS3()
        s3.fail_next = [EndpointConnectionError(endpoint_url="https://s3")]
        with self.assertLogs("edge", level="ERROR"):
            window = self.build(s3=s3)
            self.ready(window)
            capture_id = self.take_photo(window)
            self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "FALLÓ"))
        window.retry_failed()
        self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "enviada"))
        (record,) = read_log(window.config.log_path)
        key = f"edge-captures/v1/events/{capture_id}.json"
        self.assertEqual(s3.objects[key], event_body(record["event"]))
        self.assertEqual(len(read_log(window.config.log_path)), 1)  # no se volvió a inferir

    def test_row_does_not_flip_back_to_failed_while_the_retry_is_in_flight(self) -> None:
        s3 = GatedS3()
        s3.gate.set()  # la 1.ª foto llega a S3...
        s3.fail_next = [EndpointConnectionError(endpoint_url="https://s3")]  # ...y falla
        with self.assertLogs("edge", level="ERROR"):
            window = self.build(s3=s3)
            # S3 se libera antes de detener los hilos: ninguno sobrevive a la prueba.
            self.addCleanup(s3.gate.set)
            self.ready(window)
            capture_id = self.take_photo(window)
            self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "FALLÓ"))
        s3.gate.clear()  # el reenvío se queda esperando a S3
        window.retry_failed()
        self.assertEqual(self.envio(window, capture_id), "enviando…")
        self.assertTrue(window.retry_button.instate(["disabled"]))
        # Más de un ciclo de revisión de envíos: el resultado viejo (failed) no debe volver.
        self.pump(lambda: False, timeout=1.2)
        self.assertEqual(self.envio(window, capture_id), "enviando…")
        s3.gate.set()
        self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "enviada"))

    def test_retry_does_not_touch_captures_still_in_the_queue(self) -> None:
        s3 = GatedS3()  # S3 "colgado": la foto sigue en cola, no fallida
        window = self.build(s3=s3)
        # S3 se libera antes de detener los hilos: ninguno sobrevive a la prueba.
        self.addCleanup(s3.gate.set)
        self.ready(window)
        capture_id = self.take_photo(window)
        self.assertTrue(window.retry_button.instate(["disabled"]))
        window.retry_failed()
        self.assertEqual(window.sender.submitted, 1)
        self.assertEqual(self.envio(window, capture_id), "enviando…")

    def test_retry_button_stays_disabled_in_local_mode_and_without_failures(self) -> None:
        window = self.build()  # sin bucket: solo local
        self.ready(window)
        capture_id = self.take_photo(window)
        self.assertEqual(self.envio(window, capture_id), "solo local")
        window.retry_failed()  # no hace nada y no falla
        self.assertTrue(window.retry_button.instate(["disabled"]))

    def test_failed_captures_from_a_failing_retry_stay_failed(self) -> None:
        s3 = FakeS3()
        s3.fail_next = [EndpointConnectionError(endpoint_url="https://s3")] * 2
        with self.assertLogs("edge", level="ERROR"):
            window = self.build(s3=s3)
            self.ready(window)
            capture_id = self.take_photo(window)
            self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "FALLÓ"))
            window.retry_failed()  # también falla (sin red)
            self.assertTrue(self.pump(lambda: len(s3.calls) >= 2))
            self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "FALLÓ"))
        self.assertFalse(window.retry_button.instate(["disabled"]))
        window.retry_failed()  # y con red vuelve a salir
        self.assertTrue(self.pump(lambda: self.envio(window, capture_id) == "enviada"))

    def test_close_without_pending_uploads_stops_the_camera_and_closes(self) -> None:
        source = StillSource()
        window = self.build(source=source)
        self.ready(window)
        window._close()
        self.assertTrue(source.closed.wait(3))
        with self.assertRaises(tk.TclError):
            self.root.winfo_exists()

    def test_close_waits_for_the_queued_uploads(self) -> None:
        s3 = GatedS3()
        window = self.build(s3=s3)
        # S3 se libera antes de detener los hilos: ninguno sobrevive a la prueba.
        self.addCleanup(s3.gate.set)
        self.ready(window)
        window.capture()
        self.assertTrue(self.pump(lambda: self.rows(window)))
        window._close()
        self.assertFalse(self.pump(lambda: window._finished, timeout=0.3))
        self.assertIn("Esperando 1 envíos", window.message.get())
        s3.gate.set()
        self.assertTrue(self.pump(lambda: window._finished))
        self.assertEqual(len(s3.keys("edge-captures/v1/events/")), 1)

    def test_closing_twice_does_not_wait_for_the_uploads(self) -> None:
        s3 = GatedS3()
        window = self.build(s3=s3)
        # S3 se libera antes de detener los hilos: ninguno sobrevive a la prueba.
        self.addCleanup(s3.gate.set)
        self.ready(window)
        window.capture()
        self.assertTrue(self.pump(lambda: self.rows(window)))
        window._close()
        window._close()
        self.assertTrue(window._finished)
        with self.assertRaises(tk.TclError):
            self.root.winfo_exists()


@unittest.skipIf(tk is None, "esta instalación de Python no tiene tkinter")
class RunGuiWithoutDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.config = load_config(write_config(tmp, write_tiny_model(tmp / "models" / "m.onnx")))

    def run_without_display(self, **env: str) -> list[str]:
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(gui.tk, "Tk", side_effect=tk.TclError(NO_DISPLAY)),
            self.assertLogs("edge", level="ERROR") as logs,
        ):
            self.assertEqual(gui.run_gui(self.config), 1)
        return logs.output

    def test_without_a_screen_it_says_so_and_points_to_run(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("SSH_CONNECTION", None)
            os.environ.pop("SSH_CLIENT", None)
            output = "\n".join(self.run_without_display())
        self.assertIn(NO_DISPLAY, output)
        self.assertIn("necesita una pantalla", output)
        self.assertIn("python -m edge run", output)

    def test_over_ssh_it_explains_the_window_opens_on_the_laptop(self) -> None:
        output = "\n".join(self.run_without_display(SSH_CONNECTION="10.0.0.2 5000 10.0.0.9 22"))
        self.assertIn("sesión es por SSH", output)
        self.assertIn("pantalla de la laptop", output)
        self.assertIn("python -m edge run", output)


class GuiCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.config_path = write_config(tmp, write_tiny_model(tmp / "models" / "m.onnx"))
        patcher = mock.patch("edge.__main__.setup_logging")  # sin archivos de log abiertos
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_gui_opens_the_window_with_the_config(self) -> None:
        with mock.patch.dict(
            sys.modules, {"edge.gui": types.SimpleNamespace(run_gui=mock.Mock(return_value=0))}
        ):
            run_gui = sys.modules["edge.gui"].run_gui
            self.assertEqual(main(["--config", str(self.config_path), "gui"]), 0)
        (config, image), _ = run_gui.call_args
        self.assertEqual(config.device_id, "edge-test-01")
        self.assertIsNone(image)

    def test_gui_accepts_an_image_and_a_bucket_override(self) -> None:
        fake = types.SimpleNamespace(run_gui=mock.Mock(return_value=0))
        with mock.patch.dict(sys.modules, {"edge.gui": fake}):
            code = main(
                ["--config", str(self.config_path), "gui", "--image", "foto.jpg", "--bucket", ""]
            )
        self.assertEqual(code, 0)
        (config, image), _ = fake.run_gui.call_args
        self.assertEqual(image, Path("foto.jpg"))
        self.assertEqual(config.aws.bucket, "")

    def test_gui_returns_the_exit_code_of_the_window(self) -> None:
        fake = types.SimpleNamespace(run_gui=mock.Mock(return_value=1))
        with mock.patch.dict(sys.modules, {"edge.gui": fake}):
            self.assertEqual(main(["--config", str(self.config_path), "gui"]), 1)

    def test_without_tkinter_it_fails_with_a_message_instead_of_a_traceback(self) -> None:
        # `None` en sys.modules hace que `from edge.gui import ...` lance ImportError.
        with (
            mock.patch.dict(sys.modules, {"edge.gui": None}),
            self.assertLogs("edge", level="ERROR") as logs,
        ):
            self.assertEqual(main(["--config", str(self.config_path), "gui"]), 1)
        self.assertIn("no se pudo cargar la interfaz gráfica", logs.output[0])
        self.assertIn("tkinter", logs.output[0])

    def test_gui_needs_a_valid_config(self) -> None:
        self.assertEqual(main(["--config", "no-existe.yaml", "gui"]), 2)


if __name__ == "__main__":
    unittest.main()
