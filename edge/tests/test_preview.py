"""Vista en vivo sin pantalla: el hilo que lee la cámara y el dibujo de la zona recortada."""

import threading
import time
import unittest
from collections.abc import Callable

import numpy as np

from edge.camera import CameraError
from edge.preview import DIM_OUTSIDE_CROP, FrameGrabber, fit_size, render_preview


def wait_for(condition: Callable[[], object], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


class FakeSource:
    """Cámara de mentira: cada lectura es un cuadro con su número en el canal azul."""

    def __init__(self, failures: list[Exception | None] | None = None) -> None:
        self.failures = failures or []
        self.reads = 0
        self.closed = threading.Event()
        self.threads: set[int] = set()

    def read_live(self) -> np.ndarray:
        self.threads.add(threading.get_ident())
        if self.failures:
            error = self.failures.pop(0)
            if error is not None:
                raise error
        self.reads += 1
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = self.reads % 256
        return frame

    def close(self) -> None:
        self.threads.add(threading.get_ident())
        self.closed.set()


class FitSizeTests(unittest.TestCase):
    def test_wide_frame_is_limited_by_width(self) -> None:
        self.assertEqual(fit_size(640, 480, 320, 320), (320, 240))

    def test_tall_frame_is_limited_by_height(self) -> None:
        self.assertEqual(fit_size(480, 640, 320, 320), (240, 320))

    def test_small_frame_grows_to_fill_the_window(self) -> None:
        self.assertEqual(fit_size(100, 50, 400, 400), (400, 200))

    def test_never_returns_an_empty_image(self) -> None:
        self.assertEqual(fit_size(1000, 10, 10, 10), (10, 1))


class RenderPreviewTests(unittest.TestCase):
    def frame(self, width: int = 200, height: int = 100) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (0, 0, 200)  # B, G, R: rojo
        return frame

    def test_colors_are_shown_as_rgb(self) -> None:
        image = render_preview(self.frame(), (200, 100), None)
        self.assertEqual(image.size, (200, 100))
        self.assertEqual(image.getpixel((10, 10)), (200, 0, 0))

    def test_fits_the_window_without_deforming(self) -> None:
        self.assertEqual(render_preview(self.frame(), (100, 100), None).size, (100, 50))

    def test_center_square_dims_outside_and_keeps_inside(self) -> None:
        # Cuadrado de 50 px (0.5 x el lado menor) centrado: x 75..125, y 25..75.
        image = render_preview(self.frame(), (200, 100), {"center_square": 0.5})
        self.assertEqual(image.getpixel((100, 50)), (200, 0, 0))
        self.assertEqual(image.getpixel((10, 10)), (int(200 * DIM_OUTSIDE_CROP), 0, 0))
        self.assertEqual(image.getpixel((75, 25)), (255, 255, 255))  # borde de la zona

    def test_zone_follows_the_scale_of_the_window(self) -> None:
        image = render_preview(self.frame(), (100, 100), {"center_square": 0.5})  # 100x50
        self.assertEqual(image.getpixel((50, 25)), (200, 0, 0))
        self.assertEqual(image.getpixel((5, 5)), (int(200 * DIM_OUTSIDE_CROP), 0, 0))

    def test_pixel_crop_marks_that_rectangle(self) -> None:
        crop = {"x": 100, "y": 0, "width": 100, "height": 100}
        image = render_preview(self.frame(), (200, 100), crop)
        self.assertEqual(image.getpixel((150, 50)), (200, 0, 0))
        self.assertEqual(image.getpixel((20, 50)), (int(200 * DIM_OUTSIDE_CROP), 0, 0))

    def test_without_the_option_or_without_crop_nothing_is_dimmed(self) -> None:
        for crop, show in (({"center_square": 0.5}, False), (None, True)):
            with self.subTest(crop=crop, show=show):
                image = render_preview(self.frame(), (200, 100), crop, show_crop=show)
                self.assertEqual(image.getpixel((10, 10)), (200, 0, 0))

    def test_crop_larger_than_the_frame_is_clamped_instead_of_failing(self) -> None:
        crop = {"x": 150, "y": 0, "width": 500, "height": 500}
        image = render_preview(self.frame(), (200, 100), crop)
        self.assertEqual(image.getpixel((170, 50)), (200, 0, 0))

    def test_crop_completely_outside_the_frame_draws_nothing(self) -> None:
        crop = {"x": 900, "y": 900, "width": 50, "height": 50}
        image = render_preview(self.frame(), (200, 100), crop)
        self.assertEqual(image.getpixel((10, 10)), (200, 0, 0))


class FrameGrabberTests(unittest.TestCase):
    def grabber(self, source: FakeSource, **options: float) -> FrameGrabber:
        grabber = FrameGrabber(lambda: source, **options)
        self.addCleanup(grabber.stop)
        return grabber

    def test_keeps_the_latest_frame_and_numbers_them(self) -> None:
        source = FakeSource()
        grabber = self.grabber(source, max_fps=200).start()
        self.assertTrue(wait_for(lambda: (grabber.latest() or 0) and grabber.latest().number >= 3))
        first = grabber.latest()
        self.assertTrue(wait_for(lambda: grabber.latest().number > first.number))
        self.assertIsNone(grabber.error)
        self.assertEqual(grabber.latest().image.shape, (4, 6, 3))

    def test_there_is_no_frame_before_the_source_opens(self) -> None:
        opened = threading.Event()
        release = threading.Event()

        def slow_open() -> FakeSource:
            opened.set()
            release.wait(3)
            return FakeSource()

        grabber = FrameGrabber(slow_open).start()
        self.addCleanup(grabber.stop)
        self.assertTrue(opened.wait(3))
        self.assertIsNone(grabber.latest())
        self.assertIsNone(grabber.error)
        release.set()
        self.assertTrue(wait_for(lambda: grabber.latest() is not None))

    def test_source_is_opened_read_and_closed_in_the_same_thread(self) -> None:
        source = FakeSource()
        opening: list[int] = []

        def open_source() -> FakeSource:
            opening.append(threading.get_ident())
            return source

        grabber = FrameGrabber(open_source, max_fps=200).start()
        self.assertTrue(wait_for(lambda: grabber.latest() is not None))
        grabber.stop()
        self.assertTrue(source.closed.is_set())
        self.assertEqual(source.threads | set(opening), {opening[0]})
        self.assertNotEqual(opening[0], threading.get_ident())

    def test_stop_ends_the_thread(self) -> None:
        source = FakeSource()
        grabber = self.grabber(source, max_fps=200).start()
        self.assertTrue(wait_for(lambda: grabber.latest() is not None))
        grabber.stop()
        reads = source.reads
        time.sleep(0.05)
        self.assertEqual(source.reads, reads)

    def test_camera_that_does_not_open_leaves_the_message(self) -> None:
        def broken() -> FakeSource:
            raise CameraError("no se pudo abrir la cámara 1")

        with self.assertLogs("edge", level="ERROR"):
            grabber = FrameGrabber(broken).start()
            self.addCleanup(grabber.stop)
            self.assertTrue(wait_for(lambda: grabber.error))
        self.assertEqual(grabber.error, "no se pudo abrir la cámara 1")
        self.assertIsNone(grabber.latest())

    def test_unexpected_error_when_opening_shows_the_type(self) -> None:
        def broken() -> FakeSource:
            raise ImportError("sin cv2")

        with self.assertLogs("edge", level="ERROR"):
            grabber = FrameGrabber(broken).start()
            self.addCleanup(grabber.stop)
            self.assertTrue(wait_for(lambda: grabber.error))
        self.assertEqual(grabber.error, "ImportError: sin cv2")

    def test_a_few_failed_reads_do_not_stop_the_preview(self) -> None:
        source = FakeSource(failures=[CameraError("x")] * 3)
        grabber = self.grabber(source, max_fps=200, patience_s=5).start()
        self.assertTrue(wait_for(lambda: grabber.latest() is not None))
        self.assertIsNone(grabber.error)

    def test_lost_camera_sets_the_error_and_closes_the_source(self) -> None:
        source = FakeSource(failures=[CameraError("la cámara 0 no devolvió un frame")] * 10_000)
        with self.assertLogs("edge", level="ERROR"):
            grabber = self.grabber(source, patience_s=0.2).start()
            self.assertTrue(wait_for(lambda: grabber.error))
        self.assertEqual(grabber.error, "la cámara 0 no devolvió un frame")
        self.assertTrue(source.closed.wait(3))

    def test_frame_rate_is_capped(self) -> None:
        source = FakeSource()  # una imagen fija se lee al instante
        grabber = self.grabber(source, max_fps=50).start()
        time.sleep(0.3)
        grabber.stop()
        self.assertGreater(source.reads, 0)
        self.assertLess(source.reads, 40)  # sin tope serían decenas de miles


if __name__ == "__main__":
    unittest.main()
