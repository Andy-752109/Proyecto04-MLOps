"""Valida rutas y argumentos que llegan por la línea de comandos (P3-14).

`safe_path` normaliza una ruta con `os.path.realpath` y comprueba con `startswith` que quede
dentro de `base` (por defecto la raíz del repo): la forma que SonarCloud (python:S2083) reconoce
como sanitizador de Path Traversal. Las rutas relativas se resuelven contra el directorio actual,
así que `../eval_v1.0.0/...` desde `app/` sigue funcionando. Los scripts lo aplican solo en
`main()`; las funciones que reciben rutas ya validadas (`publish()`, `build_card()`) no lo usan.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mlflow_dump import ROOT

VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
RUN_ID_RE = re.compile(r"[0-9a-f]{32}")


def safe_path(p: str | os.PathLike, base: str | os.PathLike = ROOT) -> Path:
    """Devuelve la ruta resuelta; sale con un mensaje claro si queda fuera de `base`."""
    base_real = os.path.realpath(base)
    real = os.path.realpath(p)
    # normcase solo para comparar: en Windows ignora mayúsculas/minúsculas (C:\ vs c:\).
    base_cmp, real_cmp = os.path.normcase(base_real), os.path.normcase(real)
    if real_cmp != base_cmp and not real_cmp.startswith(base_cmp.rstrip(os.sep) + os.sep):
        raise SystemExit(f"Ruta fuera de {base_real}: {p}")
    return Path(real)


def safe_version(value: str) -> str:
    if not VERSION_RE.fullmatch(value):
        raise SystemExit(f"Versión inválida '{value}': se espera X.Y.Z (p. ej. 1.0.0).")
    return value


def safe_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise SystemExit(f"run_id inválido '{value}': se esperan 32 caracteres hexadecimales.")
    return value
