"""Preprocesamiento edge: forma, valores y paridad con torchvision (si está instalado)."""

import csv
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edge.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    bgr_to_rgb,
    crop_frame,
    preprocess,
    resolve_crop,
)

ROOT = Path(__file__).resolve().parents[2]
PARITY_CSV = ROOT / "reports" / "p4" / "reference" / "parity_reference.csv"
CROPS_DIR = ROOT / "data" / "derived" / "crops"
HAS_TORCHVISION = all(importlib.util.find_spec(m) for m in ("torch", "torchvision"))
PARITY_TOLERANCE = 1e-5


def torchvision_tensor(image: Image.Image, size: int = 128) -> np.ndarray:
    """El preprocesamiento de P3 (`app/edge_model/baseline.py`), como referencia."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0).numpy()


class PreprocessTests(unittest.TestCase):
    def test_shape_dtype_and_layout(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        tensor = preprocess(rgb)
        self.assertEqual(tensor.shape, (1, 3, 128, 128))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags["C_CONTIGUOUS"])

    def test_solid_color_normalization(self) -> None:
        rgb = np.full((50, 70, 3), (200, 100, 30), dtype=np.uint8)
        tensor = preprocess(rgb)
        expected = (np.array([200, 100, 30], dtype=np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        for channel in range(3):
            np.testing.assert_allclose(tensor[0, channel], expected[channel], atol=1e-6)

    def test_bgr_to_rgb_swaps_channels(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[..., 0] = 255  # azul en BGR
        self.assertTrue((bgr_to_rgb(frame)[..., 2] == 255).all())
        self.assertTrue((bgr_to_rgb(frame)[..., 0] == 0).all())

    def test_rejects_non_uint8_or_non_rgb(self) -> None:
        with self.assertRaises(ValueError):
            preprocess(np.zeros((10, 10, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            preprocess(np.zeros((10, 10), dtype=np.uint8))

    def test_crop_frame(self) -> None:
        frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        region = crop_frame(frame, {"x": 2, "y": 1, "width": 4, "height": 3})
        np.testing.assert_array_equal(region, frame[1:4, 2:6])
        self.assertIs(crop_frame(frame, None), frame)
        np.testing.assert_array_equal(
            crop_frame(frame, {"x": 0, "y": 0, "width": 8, "height": 6}), frame
        )
        for bad in (
            {"x": 5, "y": 0, "width": 4, "height": 3},
            {"x": 0, "y": 4, "width": 1, "height": 3},
        ):
            with self.subTest(crop=bad), self.assertRaises(ValueError):
                crop_frame(frame, bad)

    def test_center_square_80_percent(self) -> None:
        self.assertEqual(
            resolve_crop({"center_square": 0.8}, 640, 480),
            {"x": 128, "y": 48, "width": 384, "height": 384},
        )
        self.assertEqual(
            resolve_crop({"center_square": 0.8}, 480, 640),  # vertical
            {"x": 48, "y": 128, "width": 384, "height": 384},
        )
        self.assertEqual(
            resolve_crop({"center_square": 1.0}, 640, 480),
            {"x": 80, "y": 0, "width": 480, "height": 480},
        )

    def test_center_square_always_fits(self) -> None:
        for width, height in ((640, 480), (1280, 720), (1920, 1080), (321, 241), (7, 5)):
            for fraction in (0.1, 0.5, 0.8, 1.0):
                with self.subTest(size=(width, height), fraction=fraction):
                    crop = resolve_crop({"center_square": fraction}, width, height)
                    self.assertEqual(crop["width"], crop["height"])
                    self.assertLessEqual(crop["x"] + crop["width"], width)
                    self.assertLessEqual(crop["y"] + crop["height"], height)
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    self.assertEqual(crop_frame(frame, crop).shape[:2], (crop["height"],) * 2)

    def test_resolve_crop_passes_pixels_and_none(self) -> None:
        pixels = {"x": 1, "y": 2, "width": 3, "height": 4}
        self.assertEqual(resolve_crop(pixels, 640, 480), pixels)
        self.assertIsNone(resolve_crop(None, 640, 480))


@unittest.skipUnless(
    HAS_TORCHVISION, "torch/torchvision no instalados (corre en el entorno ml de app/)"
)
class TorchvisionParityTests(unittest.TestCase):
    def test_random_images_match_torchvision(self) -> None:
        rng = np.random.default_rng(42)
        for shape in ((128, 128), (240, 320), (97, 211), (600, 450)):
            with self.subTest(shape=shape):
                rgb = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
                delta = np.abs(preprocess(rgb) - torchvision_tensor(Image.fromarray(rgb))).max()
                self.assertLess(delta, PARITY_TOLERANCE)

    @unittest.skipUnless(CROPS_DIR.is_dir(), "faltan los recortes en data/derived/crops (dvc pull)")
    def test_parity_reference_crops_match_torchvision(self) -> None:
        with open(PARITY_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 20)
        for row in rows:
            with self.subTest(crop_id=row["crop_id"]):
                image = Image.open(CROPS_DIR / row["path"]).convert("RGB")
                delta = np.abs(preprocess(np.asarray(image)) - torchvision_tensor(image)).max()
                self.assertLess(delta, PARITY_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
