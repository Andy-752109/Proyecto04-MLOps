"""Lee un run de MLflow desde el respaldo SQL del snapshot (mlflow-store/mlflow.sql).

No usa la librería de MLflow ni mlflow.db. Para revisar qué trae un run:

    uv run python mlflow_dump.py <run_id>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def snapshot_dir() -> Path:
    """Carpeta del snapshot de MLflow: MLFLOW_SNAPSHOT o, por defecto, mlflow-store/."""
    return Path(os.environ.get("MLFLOW_SNAPSHOT") or ROOT / "mlflow-store")


# Orden de columnas de MLflow cuando el INSERT no las nombra.
SCHEMAS = {
    "params": ["key", "value", "run_uuid"],
    "tags": ["key", "value", "run_uuid"],
    "latest_metrics": ["key", "value", "timestamp", "step", "is_nan", "run_uuid"],
    "metrics": ["key", "value", "timestamp", "run_uuid", "step", "is_nan"],
}
_INSERT = re.compile(r"INSERT\s+INTO\s+[`\"]?(\w+)[`\"]?\s*(?:\(([^)]*)\))?", re.IGNORECASE)
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


def _scalar(token: str):
    if token.upper() == "NULL":
        return None
    for cast in (int, float):
        try:
            return cast(token)
        except ValueError:
            pass
    return token


def _quoted(text: str, i: int) -> tuple[str, int]:
    """Lee una cadena entre comillas simples; `text[i]` es la comilla de apertura."""
    n, buf = len(text), []
    i += 1
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            buf.append(_ESCAPES.get(text[i + 1], text[i + 1]))
            i += 2
        elif c == "'" and i + 1 < n and text[i + 1] == "'":
            buf.append("'")
            i += 2
        elif c == "'":
            return "".join(buf), i + 1
        else:
            buf.append(c)
            i += 1
    return "".join(buf), i


def _bare(text: str, i: int) -> tuple[object, int]:
    """Lee un valor sin comillas (número, NULL, texto suelto) hasta la coma o el paréntesis."""
    j = i
    while j < len(text) and text[j] not in ",)":
        j += 1
    return _scalar(text[i:j].strip()), j


def _tuple_values(text: str, i: int) -> tuple[list, int]:
    """Lee los valores de una tupla; `i` va justo después del "(". Devuelve el índice tras ")"."""
    values = []
    while i < len(text) and text[i] != ")":
        if text[i] == "'":
            value, i = _quoted(text, i)
            values.append(value)
        elif text[i] in ", \t\r\n":
            i += 1
        else:
            value, i = _bare(text, i)
            values.append(value)
    return values, i + 1


def _tuples(text: str):
    """Extrae las tuplas (...) de un fragmento de INSERT, respetando comillas y escapes."""
    i = 0
    while i < len(text):
        if text[i] != "(":
            i += 1
            continue
        values, i = _tuple_values(text, i + 1)
        yield values


def _columns_for(match: re.Match, table: str) -> list[str] | None:
    """Columnas del INSERT: las nombradas o, si no hay, el orden de MLflow para la tabla."""
    if match.group(2):
        return [c.strip(' `"') for c in match.group(2).split(",")]
    return SCHEMAS.get(table)


def _rows(dump: Path):
    table, columns = None, None
    for line in dump.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _INSERT.search(line)
        if match:
            table = match.group(1).lower()
            columns = _columns_for(match, table)
            rest = line[match.end() :]
        elif table and line.lstrip().startswith(("(", "VALUES", "values")):
            rest = line
        else:
            table = None
            continue
        if not columns:
            continue
        for values in _tuples(rest):
            if len(values) == len(columns):
                yield table, dict(zip(columns, values, strict=True))


def load_run(run_id: str, dump: Path | None = None) -> dict:
    """Devuelve params, tags y métricas (último valor) de un run."""
    dump = dump or snapshot_dir() / "mlflow.sql"
    if not dump.exists():
        sys.exit(f"No existe el respaldo de MLflow: {dump} (falta dvc pull?)")
    params: dict = {}
    tags: dict = {}
    latest: dict = {}
    history: dict = defaultdict(list)
    for table, row in _rows(dump):
        if row.get("run_uuid") != run_id:
            continue
        if table == "params":
            params[row["key"]] = row["value"]
        elif table == "tags":
            tags[row["key"]] = row["value"]
        elif table == "latest_metrics":
            latest[row["key"]] = row["value"]
        elif table == "metrics":
            history[row["key"]].append(
                (row.get("step") or 0, row.get("timestamp") or 0, row["value"])
            )
    metrics = {key: max(values)[2] for key, values in history.items()}
    metrics.update(latest)
    if not params or not tags:
        sys.exit(f"El run {run_id} no tiene params/tags en {dump}.")
    return {"run_id": run_id, "params": params, "tags": tags, "metrics": metrics}


def artifacts_dir(run_id: str) -> Path:
    """Carpeta de artefactos del run dentro del snapshot de DVC."""
    root = snapshot_dir() / "artifacts"
    found = [p for p in root.glob(f"*/{run_id}/artifacts") if p.is_dir()]
    if len(found) != 1:
        sys.exit(f"Se esperaba 1 carpeta de artefactos para {run_id} en {root}; hay {len(found)}.")
    return found[0]


def typed_params(params: dict) -> dict:
    """MLflow guarda los params como texto: '32' -> 32, '0.0' -> 0.0, 'adam' -> 'adam'."""
    return {
        key: _scalar(value) if isinstance(value, str) else value for key, value in params.items()
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python mlflow_dump.py <run_id>")
    print(json.dumps(load_run(sys.argv[1]), indent=2, ensure_ascii=False))
