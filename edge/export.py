"""`edge export`: los eventos locales a CSV y JSON, con su estado de envío (P4-09).

- `events_local.json`: lista de `{"event", "image_path", "crop_path", "upload"}`, con el evento
  tal como está en `captures.jsonl` (el mismo que se sube a S3).
- `events_local.csv`: una fila por captura, con el evento aplanado y el estado de envío.

`upload` resume `uploads.jsonl`: último estado, error y `upload_ms`, número de intentos
(sin contar los `pending` de la cola) y hora del último registro.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from edge.config import EdgeConfig
from edge.pipeline import read_log
from edge.sync import UploadLog

EVENT_COLUMNS = [
    "capture_id",
    "captured_at",
    "device_id",
    "model_version",
    "model_sha256",
    "runtime",
    "predicted_class",
    "confidence",
    "prob_cat",
    "prob_dog",
    "preprocess_ms",
    "inference_ms",
    "image_key",
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
    "crop_frame_width",
    "crop_frame_height",
]
LOCAL_COLUMNS = ["image_path", "crop_path"]
UPLOAD_COLUMNS = [
    "upload_status",
    "upload_error",
    "upload_ms",
    "upload_attempts",
    "upload_bucket",
    "upload_last_at",
]
COLUMNS = EVENT_COLUMNS + LOCAL_COLUMNS + UPLOAD_COLUMNS


def upload_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        return {
            "status": "pending",
            "error": None,
            "upload_ms": None,
            "attempts": 0,
            "bucket": None,
            "last_at": None,
        }
    last = attempts[-1]
    return {
        "status": last["upload_status"],
        "error": last["error"],
        "upload_ms": last["upload_ms"],
        "attempts": sum(a["upload_status"] != "pending" for a in attempts),
        "bucket": last["bucket"],
        "last_at": last["at"],
    }


def collect(config: EdgeConfig) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for attempt in UploadLog(config.upload_log_path).read():
        by_id.setdefault(attempt["capture_id"], []).append(attempt)
    return [
        {
            "event": record["event"],
            "image_path": record["image_path"],
            "crop_path": record["crop_path"],
            "upload": upload_summary(by_id.get(record["event"]["capture_id"], [])),
        }
        for record in read_log(config.log_path)
    ]


def flatten(item: dict[str, Any]) -> dict[str, Any]:
    event, upload = item["event"], item["upload"]
    crop = event["crop"] or {}
    row = {key: event[key] for key in EVENT_COLUMNS if key in event}
    row["prob_cat"] = event["probabilities"]["cat"]
    row["prob_dog"] = event["probabilities"]["dog"]
    for key in ("x", "y", "width", "height", "frame_width", "frame_height"):
        row[f"crop_{key}"] = crop.get(key)
    row["image_path"] = item["image_path"]
    row["crop_path"] = item["crop_path"]
    for key in ("status", "error", "upload_ms", "attempts", "bucket", "last_at"):
        row[f"upload_{key}" if key != "upload_ms" else key] = upload[key]
    return row


def export(config: EdgeConfig, out_dir: Path) -> tuple[Path, Path, int]:
    items = collect(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "events_local.json"
    csv_path = out_dir / "events_local.csv"
    json_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flatten(item) for item in items)
    return json_path, csv_path, len(items)
