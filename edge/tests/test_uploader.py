"""Uploader idempotente: parámetros exactos (Stubber) y reintentos (S3 en memoria)."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
from botocore.stub import ANY, Stubber
from fixtures import FakeS3

from edge import uploader as uploader_module
from edge.uploader import Uploader, event_body, event_key

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"
IMAGE = EXAMPLES / "test-capture.jpg"
BUCKET = "mlops-p4-edge-captures-test"


def load_event(name: str = "valid-cat-no-crop.json") -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


class StubberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = boto3.client(
            "s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x"
        )
        self.stub = Stubber(self.client)
        self.event = load_event()
        self.image_key = self.event["image_key"]
        self.event_key = event_key(self.event["capture_id"])

    def expected(self, key: str, content_type: str) -> dict:
        return {
            "Bucket": BUCKET,
            "Key": key,
            "Body": ANY,
            "ContentType": content_type,
            "IfNoneMatch": "*",
        }

    def test_image_then_event_with_if_none_match(self) -> None:
        self.stub.add_response("put_object", {}, self.expected(self.image_key, "image/jpeg"))
        self.stub.add_response("put_object", {}, self.expected(self.event_key, "application/json"))
        with self.stub:
            result = Uploader(BUCKET, self.client).upload(self.event, IMAGE)
        self.stub.assert_no_pending_responses()
        self.assertEqual(result.status, "sent")
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.upload_ms, 0)

    def test_412_on_event_means_already_sent(self) -> None:
        for _ in range(2):
            self.stub.add_client_error(
                "put_object", service_error_code="PreconditionFailed", http_status_code=412
            )
        with self.stub:
            result = Uploader(BUCKET, self.client).upload(self.event, IMAGE)
        self.assertEqual(result.status, "already_sent")

    def test_access_denied_is_failed_with_aws_message(self) -> None:
        self.stub.add_client_error(
            "put_object",
            service_error_code="AccessDenied",
            service_message="no",
            http_status_code=403,
        )
        with self.stub:
            result = Uploader(BUCKET, self.client).upload(self.event, IMAGE)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "AccessDenied: no")


class RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = FakeS3()
        self.uploader = Uploader(BUCKET, self.s3)
        self.event = load_event("valid-dog-with-crop.json")

    def test_double_send_creates_one_object_of_each_type(self) -> None:
        first = self.uploader.upload(self.event, IMAGE)
        second = self.uploader.upload(self.event, IMAGE)
        self.assertEqual((first.status, second.status), ("sent", "already_sent"))
        self.assertEqual(len(self.s3.keys("edge-captures/v1/images/")), 1)
        self.assertEqual(len(self.s3.keys("edge-captures/v1/events/")), 1)

    def test_image_is_uploaded_before_event(self) -> None:
        self.uploader.upload(self.event, IMAGE)
        self.assertEqual(
            self.s3.calls, [self.event["image_key"], event_key(self.event["capture_id"])]
        )

    def test_uploaded_bytes(self) -> None:
        self.uploader.upload(self.event, IMAGE)
        self.assertEqual(self.s3.objects[self.event["image_key"]], IMAGE.read_bytes())
        stored = self.s3.objects[event_key(self.event["capture_id"])]
        self.assertEqual(json.loads(stored), self.event)
        self.assertEqual(stored, event_body(self.event))

    def test_network_failure_before_image_then_retry(self) -> None:
        self.s3.fail_next = [EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")]
        failed = self.uploader.upload(self.event, IMAGE)
        self.assertEqual(failed.status, "failed")
        self.assertIn("EndpointConnectionError", failed.error)
        self.assertEqual(self.s3.objects, {})
        self.assertEqual(self.uploader.upload(self.event, IMAGE).status, "sent")

    def test_failure_between_image_and_event_then_retry(self) -> None:
        # La imagen llegó pero el evento no: la captura no cuenta como completa.
        self.s3.fail_next = [None, EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")]
        self.assertEqual(self.uploader.upload(self.event, IMAGE).status, "failed")
        self.assertEqual(self.s3.keys("edge-captures/v1/events/"), [])
        retry = self.uploader.upload(self.event, IMAGE)
        self.assertEqual(retry.status, "sent")  # imagen 412 (ya estaba) + evento nuevo
        self.assertEqual(len(self.s3.objects), 2)

    def test_missing_credentials_is_failed(self) -> None:
        self.s3.fail_next = [NoCredentialsError()]
        result = self.uploader.upload(self.event, IMAGE)
        self.assertEqual(result.status, "failed")
        self.assertIn("NoCredentialsError", result.error)

    def test_invalid_event_is_not_uploaded(self) -> None:
        result = self.uploader.upload(load_event("invalid-confidence-high.json"), IMAGE)
        self.assertEqual(result.status, "failed")
        self.assertIn("EventValidationError", result.error)
        self.assertEqual(self.s3.calls, [])

    def test_missing_image_is_not_uploaded(self) -> None:
        result = self.uploader.upload(self.event, Path("no-existe.jpg"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.s3.calls, [])

    def test_event_body_is_deterministic(self) -> None:
        reordered = dict(reversed(list(self.event.items())))
        self.assertEqual(event_body(reordered), event_body(self.event))

    def test_empty_bucket_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Uploader("", self.s3)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_cli(self, s3: FakeS3, event: Path = EXAMPLES / "valid-cat-no-crop.json") -> tuple:
        out = io.StringIO()
        args = ["--bucket", BUCKET, "--event", str(event), "--image", str(IMAGE)]
        with (
            mock.patch.object(uploader_module, "make_s3_client", return_value=s3),
            redirect_stdout(out),
        ):
            code = uploader_module.main(args)
        return code, json.loads(out.getvalue())

    def test_cli_twice(self) -> None:
        s3 = FakeS3()
        self.assertEqual(self.run_cli(s3)[1]["status"], "sent")
        code, second = self.run_cli(s3)
        self.assertEqual((code, second["status"]), (0, "already_sent"))
        self.assertEqual(len(s3.objects), 2)

    def test_cli_bad_event_file_is_failed_not_traceback(self) -> None:
        # Pendiente de la revisión de #20: el CLI no debe tronar con un traceback.
        invalid_json = self.tmp / "roto.json"
        invalid_json.write_text("{no es json", encoding="utf-8")
        not_object = self.tmp / "lista.json"
        not_object.write_text("[]", encoding="utf-8")
        for event in (self.tmp / "no-existe.json", invalid_json, not_object):
            with self.subTest(event=event.name):
                s3 = FakeS3()
                code, result = self.run_cli(s3, event)
                self.assertEqual((code, result["status"]), (1, "failed"))
                self.assertTrue(result["error"])
                self.assertEqual(s3.calls, [])

    def test_cli_client_error_is_failed(self) -> None:
        out = io.StringIO()
        args = ["--bucket", BUCKET, "--event", str(EXAMPLES / "valid-cat-no-crop.json")]
        with (
            mock.patch.object(
                uploader_module, "make_s3_client", side_effect=RuntimeError("perfil no existe")
            ),
            redirect_stdout(out),
        ):
            code = uploader_module.main([*args, "--image", str(IMAGE)])
        result = json.loads(out.getvalue())
        self.assertEqual((code, result["status"]), (1, "failed"))
        self.assertIn("perfil no existe", result["error"])


if __name__ == "__main__":
    unittest.main()
