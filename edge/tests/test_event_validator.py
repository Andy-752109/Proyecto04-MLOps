"""Pruebas offline del contrato y su validador reutilizable."""

import json
import unittest
from pathlib import Path

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

    def test_image_key_must_match_capture_id(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "image_key"):
            validate_event(load_example("invalid-image-key-mismatch.json"))

    def test_uuid_must_be_v4(self) -> None:
        with self.assertRaisesRegex(EventValidationError, "capture_id"):
            validate_event(load_example("invalid-uuid-not-v4.json"))

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


if __name__ == "__main__":
    unittest.main()
