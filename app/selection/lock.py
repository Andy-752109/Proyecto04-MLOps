"""El candado (chequeo 5 de #18): sin `selection.json`, o si el hash del test no
coincide con el manifiesto, la evaluación se niega a correr.

`test_ids_sha256` es el sello del test: SHA-256 de los `crop_id` del split
`test` del manifiesto, ordenados y unidos por `\\n`. Cambiar el test después de
sellar la selección rompe el candado.
"""

import hashlib
from pathlib import Path

from manifest.check_leakage import read_manifest
from selection.contracts import Selection, SelectionLocked

TEST_SPLIT = "test"


def test_ids_sha256(manifest_csv: Path) -> str:
    rows = read_manifest(manifest_csv)
    ids = sorted(str(row["crop_id"]) for row in rows if row["split"] == TEST_SPLIT)
    if not ids:
        raise SelectionLocked(f"el manifiesto {manifest_csv} no tiene recortes de test")
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 de un archivo, por bloques (para verificar el `manifest.csv`)."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_selection(path: Path) -> Selection:
    if not path.is_file():
        raise SelectionLocked(f"no existe {path} (MODEL SELECTION no cerrada)")
    return Selection.model_validate_json(path.read_text(encoding="utf-8"))


def require_closed(*, split: str, selection_path: Path, manifest_csv: Path) -> Selection | None:
    """Para `test` exige selección cerrada y hash coincidente.

    `validation` está exento: es el ensayo previo al cierre (el #20 se corre
    así antes de que Karen declare MODEL SELECTION CLOSED).
    """
    if split != TEST_SPLIT:
        return None
    selection = load_selection(selection_path)
    expected = test_ids_sha256(manifest_csv)
    if selection.test_ids_sha256 != expected:
        raise SelectionLocked(
            "el hash del test no coincide con el manifiesto; se niega la evaluación"
        )
    return selection


def lock_reason(*, selection_path: Path, derived_dir: Path) -> str | None:
    """Motivo por el que la API de Evaluation debe negarse; `None` si está abierta."""
    if not selection_path.is_file():
        return "selección no cerrada (no existe reports/selection.json)"
    try:
        selection = Selection.model_validate_json(selection_path.read_text(encoding="utf-8"))
    except Exception as error:  # contrato roto: mejor negarse que servir algo falso
        return f"selection.json inválido: {error}"
    manifest_csv = derived_dir / "manifests" / selection.release / "manifest.csv"
    if not manifest_csv.is_file():
        return f"no existe el manifiesto {manifest_csv}"
    try:
        expected = test_ids_sha256(manifest_csv)
    except SelectionLocked as error:
        return str(error)
    if selection.test_ids_sha256 != expected:
        return "el hash del test no coincide con el manifiesto"
    return None
