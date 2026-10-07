"""`describe_io` lee nombres, formas y tipos de entrada/salida del .onnx."""

from pathlib import Path

import onnx
from edge_model.quantize import describe_io
from onnx import TensorProto, helper


def _identity_model(path: Path) -> Path:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 128, 128])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3, 128, 128])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["logits"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def test_describe_io_reports_names_shapes_and_types(tmp_path: Path) -> None:
    io = describe_io(_identity_model(tmp_path / "m.onnx"))
    assert io == {
        "inputs": [{"name": "input", "shape": ["batch", 3, 128, 128], "type": "tensor(float)"}],
        "outputs": [{"name": "logits", "shape": ["batch", 3, 128, 128], "type": "tensor(float)"}],
    }
