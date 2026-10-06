"""ResNet18 genérico en ONNX, sin torch: misma arquitectura y cómputo que el modelo de P3.

Pesos aleatorios (semilla fija): sirve para probar el runtime y medir latencia en el
dispositivo mientras llega el ONNX real de P4-04, **no** para clasificar. BatchNorm va
plegada en las convoluciones, como queda tras exportar un modelo en `eval()`.

Cabeza de P3: Linear(512, 256) -> ReLU -> Linear(256, 2) (el Dropout no existe en
inferencia). Entrada [1, 3, 128, 128], salida [1, 2] logits.

    python -m edge.tools.generic_resnet18 --setup    # modelo + edge/config.yaml
    python -m edge.tools.generic_resnet18 ruta/salida.onnx
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np

STAGES = ((64, 1), (128, 2), (256, 2), (512, 2))  # (canales, stride del primer bloque)


class _Graph:
    def __init__(self, seed: int) -> None:
        from onnx import helper, numpy_helper

        self.helper = helper
        self.numpy_helper = numpy_helper
        self.rng = np.random.default_rng(seed)
        self.nodes: list = []
        self.initializers: list = []
        self.count = 0

    def name(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}_{self.count}"

    def weight(self, shape: tuple[int, ...], fan_in: int) -> str:
        name = self.name("w")
        values = self.rng.normal(0, np.sqrt(2.0 / fan_in), shape).astype(np.float32)
        self.initializers.append(self.numpy_helper.from_array(values, name))
        return name

    def bias(self, size: int) -> str:
        name = self.name("b")
        self.initializers.append(self.numpy_helper.from_array(np.zeros(size, np.float32), name))
        return name

    def op(self, op_type: str, inputs: list[str], **attrs: object) -> str:
        out = self.name(op_type.lower())
        self.nodes.append(self.helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def conv(self, x: str, cin: int, cout: int, k: int, stride: int, relu: bool) -> str:
        w = self.weight((cout, cin, k, k), cin * k * k)
        pad = k // 2
        y = self.op("Conv", [x, w, self.bias(cout)], strides=[stride, stride], pads=[pad] * 4)
        return self.op("Relu", [y]) if relu else y

    def linear(self, x: str, fin: int, fout: int) -> str:
        return self.op("Gemm", [x, self.weight((fout, fin), fin), self.bias(fout)], transB=1)


def build(image_size: int = 128, seed: int = 0):
    from onnx import TensorProto, checker, helper

    g = _Graph(seed)
    x = g.conv("input", 3, 64, 7, 2, relu=True)
    x = g.op("MaxPool", [x], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    cin = 64
    for cout, stride in STAGES:
        for block in range(2):
            s = stride if block == 0 else 1
            y = g.conv(x, cin, cout, 3, s, relu=True)
            y = g.conv(y, cout, cout, 3, 1, relu=False)
            shortcut = g.conv(x, cin, cout, 1, s, relu=False) if (s != 1 or cin != cout) else x
            x = g.op("Relu", [g.op("Add", [y, shortcut])])
            cin = cout
    x = g.op("Flatten", [g.op("GlobalAveragePool", [x])])
    x = g.op("Relu", [g.linear(x, 512, 256)])
    logits = g.linear(x, 256, 2)

    graph = helper.make_graph(
        g.nodes,
        "generic_resnet18",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, image_size, image_size])],
        [helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, 2])],
        initializer=g.initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    checker.check_model(model)
    return model


EDGE_DIR = Path(__file__).resolve().parents[1]
SETUP_MODEL = EDGE_DIR / "models" / "generic-resnet18.onnx"
SETUP_CONFIG = EDGE_DIR / "config.yaml"


def write_config(config: Path, model: Path, sha256: str) -> None:
    """Crea `config.yaml` desde el ejemplo (si no existe) y fija model.path/version/sha256."""
    if not config.exists():
        shutil.copyfile(EDGE_DIR / "config.example.yaml", config)
    text = config.read_text(encoding="utf-8")
    model_path = os.path.relpath(model, config.parent).replace(os.sep, "/")
    replacements = {
        r"^(  path: )\S+": rf"\g<1>{model_path}",
        r"^(  version: )\S+": r"\g<1>generic-resnet18",
        r'^(  sha256: )"?[0-9a-fA-F]*"?': rf'\g<1>"{sha256}"',
    }
    for pattern, value in replacements.items():
        text, count = re.subn(pattern, value, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"no encontré {pattern!r} en {config}")
    config.write_text(text, encoding="utf-8")


def main() -> None:
    import onnx

    from edge.backend import sha256_file

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, nargs="?", help="ruta del .onnx a generar")
    parser.add_argument(
        "--setup",
        action="store_true",
        help=f"genera {SETUP_MODEL.relative_to(EDGE_DIR.parent)} y lo pone en edge/config.yaml",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = SETUP_MODEL if args.setup else args.output
    if output is None:
        parser.error("indica la ruta de salida o usa --setup")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(args.image_size, args.seed), output)
    sha256 = sha256_file(output)
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB) sha256={sha256}")
    if args.setup:
        write_config(SETUP_CONFIG, output, sha256)
        print(f"{SETUP_CONFIG} listo con ese modelo y su sha256")


if __name__ == "__main__":
    main()
