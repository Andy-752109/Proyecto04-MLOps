"""Capa del modelo: la app solo conoce `InferenceBackend`.

El runtime definitivo lo decide P4-03. Para cambiarlo se agrega otra implementación y se
registra en `BACKENDS`; la captura, el preprocesamiento y el evento no cambian.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

CLASSES = ("cat", "dog")  # logit 0 = cat, logit 1 = dog (class_map.json de P3)


class InferenceBackend(Protocol):
    """Ejecuta el modelo sobre un tensor [1, 3, S, S] y devuelve probabilidades."""

    name: str

    def load(self, model_path: Path) -> None: ...

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        """Devuelve un vector float con una probabilidad por clase, en el orden de CLASSES."""
        ...


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxRuntimeBackend:
    """ONNX Runtime en CPU. El modelo devuelve 2 logits; el softmax se aplica aquí."""

    def __init__(self, threads: int | None = None) -> None:
        import onnxruntime as ort

        self._ort = ort
        self._threads = threads
        self._session = None
        self._input_name = ""
        self.name = f"onnxruntime-{ort.__version__}"

    def load(self, model_path: Path) -> None:
        options = self._ort.SessionOptions()
        if self._threads:
            options.intra_op_num_threads = self._threads
        self._session = self._ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("load() no se ha llamado")
        (logits,) = self._session.run(None, {self._input_name: tensor})
        logits = np.asarray(logits).reshape(-1)
        if logits.shape != (len(CLASSES),):
            raise ValueError(
                f"el modelo devolvió {logits.shape}, se esperaban {len(CLASSES)} logits"
            )
        return softmax(logits)


BACKENDS = {"onnxruntime": OnnxRuntimeBackend}


def create_backend(name: str, threads: int | None = None) -> InferenceBackend:
    try:
        factory = BACKENDS[name]
    except KeyError:
        raise ValueError(f"runtime desconocido {name!r}; opciones: {sorted(BACKENDS)}") from None
    return factory(threads=threads)
