"""Apertura de la cámara sin hardware: backend de OpenCV y descarte de cuadros."""

import sys
import unittest
from unittest import mock

import cv2
import numpy as np

from edge.camera import Camera, CameraError
from edge.config import CameraConfig


class FakeCapture:
    def __init__(self, index: int, api: int, opened: bool = True) -> None:
        self.index, self.api, self.opened = index, api, opened
        self.grabs = 0
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
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class CameraTests(unittest.TestCase):
    def open(self, config: CameraConfig, opened: bool = True) -> FakeCapture:
        created = []

        def factory(index, api):
            created.append(FakeCapture(index, api, opened))
            return created[-1]

        with mock.patch.object(cv2, "VideoCapture", side_effect=factory):
            camera = Camera(config)
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

    def test_discards_warmup_frames_on_open_and_before_each_read(self) -> None:
        capture = self.open(CameraConfig(warmup_frames=10))
        self.assertEqual(capture.grabs, 20)

    def test_native_resolution_does_not_set_size(self) -> None:
        self.assertEqual(self.open(CameraConfig()).props, {})
        props = self.open(CameraConfig(width=640, height=480)).props
        self.assertEqual(props[cv2.CAP_PROP_FRAME_WIDTH], 640)

    def test_unopened_camera_raises(self) -> None:
        with self.assertRaisesRegex(CameraError, "camera.index"):
            self.open(CameraConfig(index=3), opened=False)


if __name__ == "__main__":
    unittest.main()
