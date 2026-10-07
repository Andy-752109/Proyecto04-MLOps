"""Fixtures compartidas: un modelo ONNX diminuto, una config temporal y un S3 en memoria.

El modelo imita la interfaz del ResNet18 de P3 (entrada [1, 3, S, S], salida [1, 2] logits)
con GlobalAveragePool + Gemm, para probar la app sin descargar nada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from botocore.exceptions import ClientError

from edge.backend import sha256_file


def write_tiny_model(path: Path, image_size: int = 128) -> str:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # logit_cat sube con el rojo y logit_dog con el azul: el color decide la clase.
    weights = np.array([[4.0, 0.0, -4.0], [-4.0, 0.0, 4.0]], dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"]),
            helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, image_size, image_size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return sha256_file(path)


def write_config(directory: Path, sha256: str, **overrides: object) -> Path:
    config = {
        "device_id": "edge-test-01",
        "camera": {"index": 0, "warmup_frames": 0},
        "model": {"path": "models/model.onnx", "version": "tiny-test", "sha256": sha256},
        "crop": None,
        "capture": {"mode": "interval", "interval_seconds": 0.01},
        "data_dir": "data",
    }
    config.update(overrides)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def solid_frame_bgr(rgb: tuple[int, int, int], height: int = 240, width: int = 320) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = rgb[::-1]
    return frame


def precondition_failed() -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "PreconditionFailed",
                "Message": "At least one of the pre-conditions you specified did not hold",
            },
            "ResponseMetadata": {"HTTPStatusCode": 412},
        },
        "PutObject",
    )


class FakeS3:
    """S3 mínimo en memoria con la semántica de `If-None-Match: *`."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.fail_next: list[Exception | None] = []

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        self.calls.append(Key)
        if self.fail_next:
            error = self.fail_next.pop(0)
            if error is not None:
                raise error
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise precondition_failed()
        self.objects[Key] = Body
        return {"ETag": '"etag"'}

    def keys(self, prefix: str) -> list[str]:
        return [k for k in self.objects if k.startswith(prefix)]
