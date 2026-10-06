"""Pruebas offline del contrato y su validador reutilizable."""

import json
import math
import unittest
from pathlib import Path

from jsonschema import FormatChecker

from edge.event_validator import EventValidationError, validate_event

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"


def load_example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


class EventValidatorTests(unittest.TestCase):
    def test_all_valid_examples_pass(self) -> None:
        names = ("valid-cat-no-crop.json", "valid-dog-with-crop.json")
        for name in names:
            with self.subTest(name=name):
                validate_event(load_example(name))

    def test_all_invalid_examples_fail(self) -> None:
        names = (
            "invalid-confidence-high.json",
            "invalid-captured-at-no-timezone.json",
            "invalid-unknown-class.json",
            "invalid-image-key-mismatch.json",
            "invalid-uuid-not-v4.json",
            "invalid-additional-property.json",
            "invalid-crop-out-of-frame.json",
        )
        for name in names:
            with self.subTest(name=name), self.assertRaises(EventValidationError):
                validate_event(load_example(name))

    def test_uuid_v4_and_both_timezone_forms_pass(self) -> None:
        for name in ("valid-cat-no-crop.json", "valid-dog-with-crop.json"):
            with self.subTest(name=name):
                validate_event(load_example(name))

    def test_crop_forms_pass(self) -> None:
        cat = load_example("valid-cat-no-crop.json")
        dog = load_example("valid-dog-with-crop.json")
        self.assertIsNone(cat["crop"])
        self.assertIsInstance(dog["crop"], dict)
        validate_event(cat)
        validate_event(dog)

    def test_crop_inside_and_exact_frame_edges_pass(self) -> None:
        event = load_example("valid-dog-with-crop.json")
        validate_event(event)

        right_edge = load_example("valid-dog-with-crop.json")
        right_edge["crop"]["x"] = right_edge["crop"]["frame_width"] - right_edge["crop"]["width"]
        validate_event(right_edge)

        bottom_edge = load_example("valid-dog-with-crop.json")
        bottom_edge["crop"]["y"] = (
            bottom_edge["crop"]["frame_height"] - bottom_edge["crop"]["height"]
        )
        validate_event(bottom_edge)

    def test_crop_outside_frame_fails(self) -> None:
        with self.assertRaisesRegex(EventValidationError, r"crop: x \+ width excede frame_width"):
            validate_event(load_example("invalid-crop-out-of-frame.json"))

        bottom_overflow = load_example("valid-dog-with-crop.json")
        bottom_overflow["crop"]["y"] = (
            bottom_overflow["crop"]["frame_height"] - bottom_overflow["crop"]["height"] + 1
        )
        with self.assertRaisesRegex(EventValidationError, r"crop: y \+ height excede frame_height"):
            validate_event(bottom_overflow)

    def test_confidence_bounds_and_class(self) -> None:
        for value in (-0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(EventValidationError):
                event = load_example("valid-cat-no-crop.json")
                event["confidence"] = value
                validate_event(event)
        with self.assertRaisesRegex(EventValidationError, "predicted_class"):
            validate_event(load_example("invalid-unknown-class.json"))

    def test_captured_at_requires_timezone(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "captured_at"):
            validate_event(load_example("invalid-captured-at-no-timezone.json"))

    def test_invalid_dates_raise_event_validation_error(self) -> None:
        self.assertFalse(FormatChecker().conforms("no-es-fecha", "date-time"))
        for value in ("no-es-fecha", "2026-13-45T10:00:00Z", "2026-10-05T12:34:56"):
            with self.subTest(value=value), self.assertRaises(EventValidationError):
                event = load_example("valid-cat-no-crop.json")
                event["captured_at"] = value
                validate_event(event)

    def test_valid_dates_with_z_and_offset(self) -> None:
        for value in ("2026-10-05T18:35:10Z", "2026-10-05T12:34:56-06:00"):
            with self.subTest(value=value):
                event = load_example("valid-cat-no-crop.json")
                event["captured_at"] = value
                validate_event(event)

    def test_image_key_must_match_capture_id(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "image_key"):
            validate_event(load_example("invalid-image-key-mismatch.json"))

    def test_uuid_must_be_v4(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "capture_id"):
            validate_event(load_example("invalid-uuid-not-v4.json"))

    def test_uuid_must_be_canonical_lowercase(self) -> None:
        event = load_example("valid-cat-no-crop.json")
        validate_event(event)

        both_upper = json.loads(json.dumps(event))
        both_upper["capture_id"] = event["capture_id"].upper()
        both_upper["image_key"] = event["image_key"].replace(
            event["capture_id"], both_upper["capture_id"]
        )
        with self.assertRaises(EventValidationError):
            validate_event(both_upper)

        mixed = json.loads(json.dumps(event))
        mixed["image_key"] = event["image_key"].replace(
            event["capture_id"], event["capture_id"].upper()
        )
        with self.assertRaises(EventValidationError):
            validate_event(mixed)

    def test_additional_properties_fail_at_every_object_level(self) -> None:
        base = load_example("valid-dog-with-crop.json")
        for location in ("event", "probabilities", "crop"):
            with self.subTest(location=location), self.assertRaises(EventValidationError):
                event = json.loads(json.dumps(base))
                target = event if location == "event" else event[location]
                target["unexpected"] = True
                validate_event(event)

    def test_probabilities_and_crop_ranges(self) -> None:
        for field in ("cat", "dog"):
            with self.subTest(field=field), self.assertRaises(EventValidationError):
                event = load_example("valid-cat-no-crop.json")
                event["probabilities"][field] = 1.01
                validate_event(event)
        event = load_example("valid-dog-with-crop.json")
        event["crop"]["width"] = 0
        with self.assertRaises(EventValidationError):
            validate_event(event)

    def test_nonfinite_numbers_fail(self) -> None:
        for field in ("confidence", "preprocess_ms", "inference_ms"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    event = load_example("valid-cat-no-crop.json")
                    event[field] = value
                    with self.assertRaises(EventValidationError):
                        validate_event(event)
        for class_name in ("cat", "dog"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(class_name=class_name, value=value):
                    event = load_example("valid-cat-no-crop.json")
                    event["probabilities"][class_name] = value
                    with self.assertRaises(EventValidationError):
                        validate_event(event)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(crop_width=value):
                event = load_example("valid-dog-with-crop.json")
                event["crop"]["width"] = value
                with self.assertRaises(EventValidationError):
                    validate_event(event)

    def test_nonstandard_json_constants_fail_validation(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaises(EventValidationError):
                event = json.loads(
                    (EXAMPLES / "valid-cat-no-crop.json")
                    .read_text(encoding="utf-8")
                    .replace('"preprocess_ms": 12.5', f'"preprocess_ms": {constant}')
                )
                validate_event(event)


if __name__ == "__main__":
    unittest.main()
