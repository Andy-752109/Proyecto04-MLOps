"""Apertura de la cámara sin hardware: backend de OpenCV y descarte de cuadros."""

import sys
import unittest
from unittest import mock

import cv2
import numpy as np

from edge.camera import Camera, CameraError, list_camera_names
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


class CameraTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
