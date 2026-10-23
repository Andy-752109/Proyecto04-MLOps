"""Apertura de la cámara sin hardware: backend de OpenCV y descarte de cuadros."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from edge.camera import Camera, CameraError, ImageSource, list_camera_names
from edge.config import CameraConfig


class FakeCapture:
    def __init__(self, index: int, api: int, opened: bool = True) -> None:
        self.index, self.api, self.opened = index, api, opened
        self.grabs = 0
        self.reads = 0
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # API de OpenCV
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def grab(self) -> bool:
        self.grabs += 1
        return True

    def read(self):
        self.reads += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def use_fast_clock(test: unittest.TestCase) -> None:
    """El vaciado espera `settle_seconds` de reloj: las pruebas no deben dormir de verdad."""
    ticks = iter(range(1_000_000))
    patcher = mock.patch("edge.camera.time.monotonic", side_effect=lambda: float(next(ticks)))
    patcher.start()
    test.addCleanup(patcher.stop)


class CameraTests(unittest.TestCase):
    def setUp(self) -> None:
        use_fast_clock(self)

    def open(
        self, config: CameraConfig, opened: bool = True, names: list[str] | None = None
    ) -> FakeCapture:
        created = []

        def factory(index, api):
            created.append(FakeCapture(index, api, opened))
            return created[-1]

        list_names = None if names is None else (lambda: names)
        with mock.patch.object(cv2, "VideoCapture", side_effect=factory):
            camera = Camera(config, list_names=list_names)
            camera.read()
        return created[0]

    def test_auto_uses_directshow_on_windows(self) -> None:
        with mock.patch.object(sys, "platform", "win32"):
            capture = self.open(CameraConfig(index=1))
        self.assertEqual((capture.index, capture.api), (1, cv2.CAP_DSHOW))

    def test_auto_uses_default_backend_elsewhere(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            capture = self.open(CameraConfig(index=0))
        self.assertEqual(capture.api, cv2.CAP_ANY)

    def test_flush_reads_frames_instead_of_grabbing_them(self) -> None:
        # `grab()` de DirectShow no espera un cuadro nuevo: devuelve el cuadro guardado al instante.
        # Hay que leer (read) para que el controlador entregue cuadros recientes.
        capture = self.open(CameraConfig(warmup_frames=10, settle_seconds=0))
        self.assertEqual(capture.grabs, 0)
        self.assertEqual(capture.reads, 10 + 10 + 1)  # al abrir + antes de leer + la lectura

    def test_flush_lasts_at_least_settle_seconds(self) -> None:
        clock = {"now": 0.0}

        def monotonic() -> float:
            clock["now"] += 0.1  # cada consulta del reloj avanza 100 ms
            return clock["now"]

        with mock.patch("edge.camera.time.monotonic", side_effect=monotonic):
            capture = self.open(CameraConfig(warmup_frames=2, settle_seconds=0.5))
        # 0.5 s a 0.1 s por vuelta: más de las 2 lecturas mínimas, en cada vaciado
        self.assertGreater(capture.reads, 2 + 2 + 1)

    def test_warmup_frames_are_a_minimum_even_if_settle_is_zero(self) -> None:
        capture = self.open(CameraConfig(warmup_frames=3, settle_seconds=0))
        self.assertEqual(capture.reads, 3 + 3 + 1)

    def open_camera(self, config: CameraConfig) -> tuple[Camera, FakeCapture]:
        capture = FakeCapture(0, cv2.CAP_ANY)
        with mock.patch.object(cv2, "VideoCapture", return_value=capture):
            return Camera(config), capture

    def test_read_live_reads_one_frame_without_flushing(self) -> None:
        # La vista en vivo ya lee sin parar: vaciar en cada cuadro la dejaría a ~2 imágenes/s.
        camera, capture = self.open_camera(CameraConfig(warmup_frames=10, settle_seconds=0))
        before = capture.reads
        frame = camera.read_live()
        self.assertEqual(capture.reads, before + 1)
        self.assertEqual((frame.dtype, frame.shape), (np.uint8, (480, 640, 3)))

    def test_read_live_raises_when_the_camera_stops_delivering(self) -> None:
        camera, capture = self.open_camera(CameraConfig(warmup_frames=0, settle_seconds=0))
        capture.read = lambda: (False, None)
        with self.assertRaisesRegex(CameraError, "no devolvió un frame"):
            camera.read_live()

    def test_native_resolution_does_not_set_size(self) -> None:
        self.assertEqual(self.open(CameraConfig()).props, {})
        props = self.open(CameraConfig(width=640, height=480)).props
        self.assertEqual(props[cv2.CAP_PROP_FRAME_WIDTH], 640)

    def test_unopened_camera_raises(self) -> None:
        with self.assertRaisesRegex(CameraError, "camera.index"):
            self.open(CameraConfig(index=3), opened=False)


class CameraByNameTests(unittest.TestCase):
    """`camera.name` elige el dispositivo por nombre: el orden de los índices no es estable."""

    USB = "GENERAL WEBCAM"
    BUILT_IN = "USB2.0 UVC HD Webcam"

    open = CameraTests.open

    def setUp(self) -> None:
        use_fast_clock(self)

    def test_name_selects_the_matching_device_index(self) -> None:
        config = CameraConfig(index=1, name=self.USB)
        capture = self.open(config, names=[self.USB, self.BUILT_IN])
        self.assertEqual(capture.index, 0)

    def test_name_follows_the_device_when_the_order_changes(self) -> None:
        config = CameraConfig(index=1, name=self.USB)
        capture = self.open(config, names=[self.BUILT_IN, self.USB])
        self.assertEqual(capture.index, 1)

    def test_name_match_is_case_insensitive_substring(self) -> None:
        capture = self.open(CameraConfig(name="general web"), names=[self.BUILT_IN, self.USB])
        self.assertEqual(capture.index, 1)

    def test_without_name_the_index_is_used_and_devices_are_not_listed(self) -> None:
        def fail() -> list[str]:
            raise AssertionError("no debe listar dispositivos sin camera.name")

        created = []

        def factory(index, api):
            created.append(FakeCapture(index, api))
            return created[-1]

        with mock.patch.object(cv2, "VideoCapture", side_effect=factory):
            Camera(CameraConfig(index=2), list_names=fail)
        self.assertEqual(created[0].index, 2)

    def test_unknown_name_lists_the_available_cameras(self) -> None:
        with self.assertRaises(CameraError) as ctx:
            self.open(CameraConfig(name="Logitech"), names=[self.USB, self.BUILT_IN])
        message = str(ctx.exception)
        self.assertIn("Logitech", message)
        self.assertIn(self.USB, message)
        self.assertIn(self.BUILT_IN, message)

    def test_name_without_device_list_is_an_error(self) -> None:
        with self.assertRaisesRegex(CameraError, "camera.index"):
            self.open(CameraConfig(name=self.USB, index=1), names=[])

    def test_open_failure_reports_the_resolved_index(self) -> None:
        with self.assertRaisesRegex(CameraError, "cámara 1"):
            self.open(CameraConfig(name=self.USB), opened=False, names=[self.BUILT_IN, self.USB])

    def test_list_camera_names_is_empty_outside_windows(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(list_camera_names(), [])


class ImageSourceTests(unittest.TestCase):
    """`ImageSource` reemplaza a la cámara en `edge run --image`: mismo contrato de `read()`."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_png(self, path: Path, frame: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", frame)  # sin pérdida: el frame vuelve idéntico
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())
        return path

    def test_read_returns_a_bgr_uint8_frame(self) -> None:
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:] = (10, 20, 200)  # B, G, R
        frame[0, 0] = (1, 2, 3)
        with ImageSource(self.write_png(self.tmp / "frame.png", frame)) as source:
            read = source.read()
        self.assertEqual((read.dtype, read.shape), (np.uint8, (36, 64, 3)))
        np.testing.assert_array_equal(read, frame)  # orden de canales BGR, igual que Camera

    def test_grayscale_image_still_has_three_channels(self) -> None:
        gray = np.full((20, 30), 128, dtype=np.uint8)
        with ImageSource(self.write_png(self.tmp / "gray.png", gray)) as source:
            self.assertEqual(source.read().shape, (20, 30, 3))

    def test_every_read_returns_the_same_image_as_an_independent_copy(self) -> None:
        frame = np.full((8, 8, 3), 77, dtype=np.uint8)
        source = ImageSource(self.write_png(self.tmp / "frame.png", frame))
        first = source.read()
        first[:] = 0
        np.testing.assert_array_equal(source.read(), frame)

    def test_read_live_is_a_copy_of_the_same_frame(self) -> None:
        frame = np.full((8, 8, 3), 55, dtype=np.uint8)
        source = ImageSource(self.write_png(self.tmp / "frame.png", frame))
        live = source.read_live()
        live[:] = 0
        np.testing.assert_array_equal(source.read_live(), frame)

    def test_path_with_accents_opens(self) -> None:
        # cv2.imread falla con estas rutas en Windows; el clon puede estar en una carpeta así.
        path = self.write_png(
            self.tmp / "Integración" / "foto ñandú.png", np.zeros((8, 8, 3), "u1")
        )
        self.assertEqual(ImageSource(path).read().shape, (8, 8, 3))

    def test_missing_file_raises_camera_error(self) -> None:
        with self.assertRaisesRegex(CameraError, "no existe la imagen .*nada.jpg"):
            ImageSource(self.tmp / "nada.jpg")

    def test_directory_raises_camera_error(self) -> None:
        with self.assertRaisesRegex(CameraError, "es una carpeta"):
            ImageSource(self.tmp)

    def test_file_that_is_not_an_image_raises_camera_error(self) -> None:
        path = self.tmp / "notas.jpg"
        path.write_text("esto no es un JPG", encoding="utf-8")
        with self.assertRaisesRegex(CameraError, "no es una imagen legible"):
            ImageSource(path)

    def test_empty_file_raises_camera_error(self) -> None:
        path = self.tmp / "vacio.jpg"
        path.write_bytes(b"")
        with self.assertRaisesRegex(CameraError, "no es una imagen legible"):
            ImageSource(path)


if __name__ == "__main__":
    unittest.main()
