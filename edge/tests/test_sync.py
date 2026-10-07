"""P4-09: envío desde la app, falla visible, reintento sin duplicados y exportación.

Sin red ni AWS: un S3 en memoria con `If-None-Match: *` que además conoce sus buckets, para
reproducir la falla del procedimiento ("apuntar a un bucket inexistente").
"""

import contextlib
import csv
import io
import json
import logging
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from botocore.exceptions import ClientError, EndpointConnectionError
from fixtures import FakeS3, solid_frame_bgr, write_config, write_tiny_model
from test_pipeline import CAPTURED_AT, FakeCamera

from edge.__main__ import main
from edge.backend import create_backend
from edge.config import AwsConfig, ConfigError, load_config
from edge.export import COLUMNS, export
from edge.pipeline import EdgeApp, read_log
from edge.sync import (
    BackgroundSender,
    UnknownCaptureError,
    UploadLog,
    image_path,
    retry,
    retry_ids,
    status_of,
)
from edge.uploader import event_body, event_key

BUCKET = "mlops-p4-edge-captures-test"
MISSING_BUCKET = "mlops-p4-edge-captures-no-existe"
RED = (230, 30, 30)


def no_such_bucket() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "PutObject",
    )


class BucketS3(FakeS3):
    """FakeS3 que solo acepta `BUCKET`; los demás responden NoSuchBucket."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: threading.Event | None = None

    def put_object(self, *, Bucket, **kwargs):
        if self.gate is not None:
            self.gate.wait(5)
        if Bucket != BUCKET:
            self.calls.append(kwargs["Key"])
            raise no_such_bucket()
        return super().put_object(Bucket=Bucket, **kwargs)

    def objects_for(self, capture_id: str) -> list[str]:
        return sorted(k for k in self.objects if capture_id in k)


def temp_dir(test: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = temp_dir(self)
        self.sha = write_tiny_model(self.tmp / "models" / "model.onnx")
        self.config = load_config(write_config(self.tmp, self.sha, aws={"bucket": BUCKET}))
        self.app = EdgeApp(self.config, create_backend("onnxruntime"))
        self.app.start()
        self.s3 = BucketS3()
        self.factory = mock.Mock(return_value=self.s3)
        self.log = UploadLog(self.config.upload_log_path)

    def capture(self):
        return self.app.process(solid_frame_bgr(RED), CAPTURED_AT)

    def sender(self, aws: AwsConfig | None = None) -> BackgroundSender:
        return BackgroundSender(aws or self.config.aws, self.log, self.factory).start()

    def statuses(self, capture_id: str) -> list[str]:
        return [r["upload_status"] for r in self.log.read() if r["capture_id"] == capture_id]

    def test_capture_is_sent_with_the_same_event_as_the_local_log(self) -> None:
        capture = self.capture()
        sender = self.sender()
        sender.submit(capture.event, capture.image_path)
        self.assertEqual(sender.close(5), 0)
        cid = capture.event["capture_id"]
        self.assertEqual(self.statuses(cid), ["pending", "sent"])
        (local,) = read_log(self.config.log_path)
        self.assertEqual(self.s3.objects[event_key(cid)], event_body(local["event"]))
        self.assertEqual(
            self.s3.objects[capture.event["image_key"]], capture.image_path.read_bytes()
        )
        last = self.log.latest()[cid]
        self.assertIsNone(last["error"])
        self.assertGreaterEqual(last["upload_ms"], 0)
        self.assertEqual(last["bucket"], BUCKET)

    def test_submit_does_not_wait_for_the_upload(self) -> None:
        self.s3.gate = threading.Event()  # S3 "colgado" hasta que lo soltemos
        sender = self.sender()
        first = self.capture()
        sender.submit(first.event, first.image_path)
        second = self.capture()  # la inferencia sigue mientras el envío está bloqueado
        sender.submit(second.event, second.image_path)
        self.assertEqual(len(read_log(self.config.log_path)), 2)
        self.assertEqual(sender.results, [])
        self.s3.gate.set()
        self.assertEqual(sender.close(5), 0)
        self.assertEqual([r.status for r in sender.results], ["sent", "sent"])

    def test_failure_is_visible_and_retry_sends_once(self) -> None:
        capture = self.capture()
        cid = capture.event["capture_id"]
        with self.assertLogs("edge", level="ERROR") as logs:
            sender = self.sender(AwsConfig(bucket=MISSING_BUCKET))
            sender.submit(capture.event, capture.image_path)
            sender.close(5)
        self.assertIn(cid, logs.output[0])
        self.assertIn("NoSuchBucket", logs.output[0])
        failed = self.log.latest()[cid]
        self.assertEqual((failed["upload_status"], failed["bucket"]), ("failed", MISSING_BUCKET))
        self.assertIn("NoSuchBucket", failed["error"])
        self.assertEqual(self.s3.objects_for(cid), [])

        (result,) = retry(self.config, [cid], self.factory)
        self.assertEqual(result.status, "sent")
        (again,) = retry(self.config, [cid], self.factory)
        self.assertEqual(again.status, "already_sent")
        self.assertEqual(
            self.s3.objects_for(cid),
            [f"edge-captures/v1/events/{cid}.json", f"edge-captures/v1/images/{cid}.jpg"],
        )
        self.assertEqual(self.statuses(cid), ["pending", "failed", "sent", "already_sent"])

    def test_retry_after_failure_between_image_and_event(self) -> None:
        capture = self.capture()
        cid = capture.event["capture_id"]
        self.s3.fail_next = [None, EndpointConnectionError(endpoint_url="https://s3")]
        with self.assertLogs("edge", level="ERROR"):
            sender = self.sender()
            sender.submit(capture.event, capture.image_path)
            sender.close(5)
        self.assertEqual(self.s3.objects_for(cid), [f"edge-captures/v1/images/{cid}.jpg"])
        (result,) = retry(self.config, [cid], self.factory)
        self.assertEqual(result.status, "sent")  # 412 en la imagen, crea el evento
        self.assertEqual(len(self.s3.objects_for(cid)), 2)

    def test_retry_sends_byte_identical_event_read_back_from_the_log(self) -> None:
        capture = self.capture()
        live = event_body(capture.event)
        retry(self.config, [capture.event["capture_id"]], self.factory)
        self.assertEqual(self.s3.objects[event_key(capture.event["capture_id"])], live)

    def test_client_error_fails_every_upload_without_stopping(self) -> None:
        factory = mock.Mock(side_effect=RuntimeError("The config profile (x) could not be found"))
        captures = [self.capture() for _ in range(2)]
        with self.assertLogs("edge", level="ERROR"):
            sender = BackgroundSender(self.config.aws, self.log, factory).start()
            for capture in captures:
                sender.submit(capture.event, capture.image_path)
            sender.close(5)
        self.assertEqual([r.status for r in sender.results], ["failed", "failed"])
        self.assertIn("could not be found", sender.results[0].error)
        self.assertEqual(factory.call_count, 2)  # vuelve a intentar crear el cliente

        with self.assertLogs("edge", level="ERROR"):
            results = retry(self.config, [c.event["capture_id"] for c in captures], factory)
        self.assertEqual([r.status for r in results], ["failed", "failed"])

    def test_retry_ids_pending_and_unknown(self) -> None:
        sent, failed, never = (self.capture() for _ in range(3))
        self.log.append(sent.event["capture_id"], "sent", BUCKET, upload_ms=1.0)
        self.log.append(failed.event["capture_id"], "failed", BUCKET, error="x", upload_ms=1.0)
        pending = retry_ids(self.config, True, [])
        self.assertEqual(pending, [failed.event["capture_id"], never.event["capture_id"]])
        with self.assertRaises(UnknownCaptureError):
            retry_ids(self.config, False, ["00000000-0000-4000-8000-000000000000"])
        latest = self.log.latest()
        self.assertEqual(status_of(latest, never.event["capture_id"]), "pending")

    def test_image_path_is_resolved_against_data_dir(self) -> None:
        capture = self.capture()
        (record,) = read_log(self.config.log_path)
        self.assertEqual(image_path(self.config, record), capture.image_path)

    def test_upload_log_rejects_unknown_status(self) -> None:
        with self.assertRaises(ValueError):
            self.log.append("x", "enviado", BUCKET)

    def test_export_csv_and_json_with_upload_status(self) -> None:
        sent, failed = self.capture(), self.capture()
        self.log.append(sent.event["capture_id"], "pending", BUCKET)
        self.log.append(sent.event["capture_id"], "failed", BUCKET, error="sin red", upload_ms=5.0)
        self.log.append(sent.event["capture_id"], "sent", BUCKET, upload_ms=80.0)
        self.log.append(
            failed.event["capture_id"], "failed", BUCKET, error="sin red", upload_ms=5.0
        )
        out = self.tmp / "export"
        json_path, csv_path, rows = export(self.config, out)
        self.assertEqual(rows, 2)
        items = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(items[0]["event"], read_log(self.config.log_path)[0]["event"])
        self.assertEqual(items[0]["upload"]["status"], "sent")
        self.assertEqual(items[0]["upload"]["attempts"], 2)  # pending no cuenta
        with open(csv_path, newline="", encoding="utf-8") as f:
            table = list(csv.DictReader(f))
        self.assertEqual(list(table[0]), COLUMNS)
        self.assertEqual(table[0]["capture_id"], sent.event["capture_id"])
        self.assertEqual(table[1]["upload_status"], "failed")
        self.assertEqual(table[1]["upload_error"], "sin red")
        self.assertEqual(float(table[0]["prob_cat"]), sent.event["probabilities"]["cat"])


class AwsConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = temp_dir(self)

    def load(self, aws: object):
        return load_config(write_config(self.tmp, "a" * 64, aws=aws)).aws

    def test_defaults_and_values(self) -> None:
        self.assertEqual(self.load(None), AwsConfig())
        self.assertFalse(self.load({"bucket": ""}).enabled)
        aws = self.load({"bucket": BUCKET, "profile": "mlops-p3", "region": "us-east-1"})
        self.assertEqual(aws, AwsConfig(BUCKET, "mlops-p3", "us-east-1"))
        self.assertTrue(aws.enabled)

    def test_invalid(self) -> None:
        for aws in ({"bucket": BUCKET, "key": "x"}, {"profile": " "}, "bucket"):
            with self.subTest(aws=aws), self.assertRaises(ConfigError):
                self.load(aws)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = temp_dir(self)
        sha = write_tiny_model(self.tmp / "models" / "model.onnx")
        self.config_path = write_config(self.tmp, sha, aws={"bucket": BUCKET})
        self.s3 = BucketS3()

    def tearDown(self) -> None:
        for handler in list(logging.getLogger("edge").handlers):
            handler.close()
            logging.getLogger("edge").removeHandler(handler)

    def cli(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch("edge.__main__.Camera", FakeCamera),
            mock.patch("edge.sync.make_s3_client", return_value=self.s3),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = main(["--config", str(self.config_path), *args])
        for handler in list(logging.getLogger("edge").handlers):
            handler.close()
            logging.getLogger("edge").removeHandler(handler)
        return code, out.getvalue(), err.getvalue()

    def ids(self) -> list[str]:
        return [r["event"]["capture_id"] for r in read_log(self.tmp / "data" / "captures.jsonl")]

    def test_run_sends_every_capture(self) -> None:
        code, _, _ = self.cli("run", "--count", "2")
        self.assertEqual(code, 0)
        for cid in self.ids():
            self.assertEqual(len(self.s3.objects_for(cid)), 2)
        code, out, _ = self.cli("status")
        self.assertIn(f"bucket      : {BUCKET}", out)
        self.assertIn("envíos      : pending 0, sent 2, already_sent 0, failed 0", out)

    def test_failure_procedure_with_missing_bucket_then_retry(self) -> None:
        code, _, err = self.cli("run", "--count", "1", "--bucket", MISSING_BUCKET)
        self.assertEqual(code, 0)  # la captura se guarda aunque el envío falle
        (cid,) = self.ids()
        self.assertIn(f"envío {cid} FALLÓ", err)
        self.assertIn("NoSuchBucket", err)
        self.assertIn(f"python -m edge retry {cid}", err)
        log_text = (self.tmp / "data" / "edge.log").read_text(encoding="utf-8")
        self.assertIn("NoSuchBucket", log_text)
        _, out, _ = self.cli("status")
        self.assertIn("failed 1", out)

        code, out, _ = self.cli("retry", cid)
        self.assertEqual(code, 0)
        self.assertIn(f"{cid} sent", out)
        code, out, _ = self.cli("retry", cid)
        self.assertIn(f"{cid} already_sent", out)
        self.assertEqual(len(self.s3.objects_for(cid)), 2)

    def test_retry_pending_only_resends_unsent(self) -> None:
        self.cli("run", "--count", "1")
        self.cli("run", "--count", "2", "--bucket", MISSING_BUCKET)
        sent_first, *failed = self.ids()
        code, out, _ = self.cli("retry", "--pending")
        self.assertEqual(code, 0)
        self.assertNotIn(sent_first, out)
        for cid in failed:
            self.assertIn(f"{cid} sent", out)
        self.assertIn("resumen: sent 2", out)
        _, out, _ = self.cli("retry", "--pending")
        self.assertIn("no hay capturas pendientes", out)

    def test_retry_failure_exit_code_and_arguments(self) -> None:
        self.cli("run", "--count", "1", "--bucket", MISSING_BUCKET)
        code, out, _ = self.cli("retry", "--pending", "--bucket", MISSING_BUCKET)
        self.assertEqual(code, 1)
        self.assertIn("failed", out)
        code, _, err = self.cli("retry", "00000000-0000-4000-8000-000000000000")
        self.assertEqual(code, 2)
        self.assertIn("no están en", err)
        for args in (("retry",), ("retry", "--pending", self.ids()[0])):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.cli(*args)

    def test_without_bucket_nothing_is_sent(self) -> None:
        self.config_path = write_config(self.tmp, self.config_sha(), aws={"bucket": ""})
        code, _, err = self.cli("run", "--count", "1")
        self.assertEqual(code, 0)
        self.assertIn("solo se guardan en local", err)
        self.assertEqual(self.s3.calls, [])
        code, _, err = self.cli("retry", "--pending")
        self.assertEqual(code, 2)

    def config_sha(self) -> str:
        return load_config(self.config_path).model.sha256

    def test_export_command(self) -> None:
        self.cli("run", "--count", "2")
        code, out, _ = self.cli("export", "--out", str(self.tmp / "out"))
        self.assertEqual(code, 0)
        self.assertIn("2 capturas", out)
        items = json.loads((self.tmp / "out" / "events_local.json").read_text(encoding="utf-8"))
        self.assertEqual([i["upload"]["status"] for i in items], ["sent", "sent"])
        self.assertTrue((self.tmp / "out" / "events_local.csv").is_file())
        self.cli("export")
        self.assertTrue((self.tmp / "data" / "export" / "events_local.json").is_file())


if __name__ == "__main__":
    unittest.main()
