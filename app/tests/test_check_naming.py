"""`check-naming.sh` acepta ramas/títulos p2, p3 y p4 a la vez.

Corre el script bash real como subproceso -- lo mismo que invoca
`pr-hygiene.yml` en cada PR -- en vez de reimplementar la regex en Python.
Así la prueba detecta un regreso real si alguien rompe el script, no solo
si rompe una copia paralela de la lógica.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check-naming.sh"


def _run(branch: str, title: str) -> subprocess.CompletedProcess:
    if sys.platform == "win32":
        pytest.skip("check-naming.sh es un script bash; no corre en el runner de Windows")
    return subprocess.run(
        ["bash", str(SCRIPT), branch, title],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("branch", "title"),
    [
        # p4 -- convención del Proyecto 4.
        ("p4-02-x", "P4-02: x"),
        # p3 -- sigue funcionando.
        ("p3-03-infra-contratos", "P3-03: Infraestructura y contratos"),
        ("p3-03-p3-05-algo", "P3-03 P3-05: algo"),
        ("p3-14-15-16-modelo", "P3-14/15/16: modelo"),
        # p2 -- sigue funcionando.
        ("p2-52-copilot-llm-client", "P2-52: Copilot con cliente LLM"),
        ("p2-22-23-24-quality-gate", "P2-22/23/24: compuerta de calidad"),
        ("p2-26-p2-27-lotes", "P2-26 P2-27: lotes 7 y 8"),
        # tipo/descripcion, sin ticket -- no depende del prefijo p2/p3/p4.
        ("fix/annotation-id-collisions", "fix: main tiene un test roto"),
        ("feat/ui-experiments-page", "feat(ui): página de experimentos"),
    ],
)
def test_valid_branch_and_title_pass(branch: str, title: str) -> None:
    result = _run(branch, title)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("branch", "title", "reason"),
    [
        ("P4-02-mayusculas", "P4-02: x", "la rama P4 debe ir en minúsculas"),
        ("p4-02-x", "p4-02: x", "el título P4 debe ir en mayúsculas"),
        ("P3-03-mayusculas", "P3-03: algo", "la rama debe ir en minúsculas"),
        ("p3-03-algo", "P3 03 algo", "falta el guion y los dos puntos en el título"),
        ("p3-03-algo", "p3-03: algo", "el título va con el prefijo en mayúsculas"),
        ("random-branch-name", "random: algo", "tipo 'random' no está en la lista permitida"),
    ],
)
def test_invalid_branch_or_title_fails(branch: str, title: str, reason: str) -> None:
    result = _run(branch, title)
    assert result.returncode != 0, f"debía fallar ({reason}) pero pasó: {result.stdout}"
