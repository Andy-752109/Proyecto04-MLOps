"""P3-06: verifica que ningún recorte, imagen de origen ni grupo cruce particiones.

    uv run python -m manifest.check_leakage --release v0.1.1

Lee `data/derived/manifests/<release>/manifest.csv`, imprime un JSON con las
intersecciones y termina con código 1 si hay fuga. No acepta rutas libres: el
release se valida y el archivo tiene que quedar dentro del directorio de
manifiestos.
"""

import argparse
import csv
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

LEAKAGE_KEYS = ("crop_id", "source_image_id", "duplicate_group_id")
RELEASE_PATTERN = re.compile(r"v\d+\.\d+\.\d+")
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "data" / "derived" / "manifests"


def find_leakage(rows: Sequence[Mapping]) -> dict[str, list[dict]]:
    """Por cada llave, los valores que aparecen en más de una partición."""
    leakage = {}
    for key in LEAKAGE_KEYS:
        owners: dict[str, set[str]] = {}
        for row in rows:
            owners.setdefault(str(row[key]), set()).add(row["split"])
        leakage[key] = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(owners.items())
            if len(splits) > 1
        ]
    return leakage


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest_path(release: str, manifests_dir: Path = MANIFESTS_DIR) -> Path:
    """Ruta de `manifest.csv` de un release, sin salir de `manifests_dir`."""
    if not RELEASE_PATTERN.fullmatch(release):
        raise ValueError(f"release inválido: {release!r} (se espera vMAYOR.MENOR.PARCHE)")
    base = manifests_dir.resolve()
    path = (base / release / "manifest.csv").resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"{path} queda fuera de {base}")
    return path


def main(argv: list[str] | None = None, *, manifests_dir: Path = MANIFESTS_DIR) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="Release del manifiesto, p. ej. v0.1.1.")
    args = parser.parse_args(argv)

    try:
        path = manifest_path(args.release, manifests_dir)
    except ValueError as error:
        print(f"check_leakage: {error}", file=sys.stderr)
        return 2
    rows = read_manifest(path)
    leakage = find_leakage(rows)
    leaked = any(leakage.values())
    print(
        json.dumps(
            {
                "manifest": str(path),
                "rows": len(rows),
                "leakage": leaked,
                "intersections": {key: len(values) for key, values in leakage.items()},
                "details": leakage,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
