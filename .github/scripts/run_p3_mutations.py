"""P3-17 (#27): 5 mutaciones específicas de la campaña, cada una en un
`git worktree` temporal -- nunca se toca el checkout real, ni siquiera a
medias (un worktree es un checkout aparte, no el mismo árbol de trabajo).
Cada mutación debe hacer fallar la prueba que la cubre; si alguna sobrevive,
el script termina en error. Escribe la evidencia en
`app/tests/evidence/p3-mutations.md`.

Uso (desde la raíz del repo, con `app/.venv` ya resuelto -- `uv sync
--group ml` en `app/`):

    python .github/scripts/run_p3_mutations.py
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
EVIDENCE_PATH = APP_ROOT / "tests" / "evidence" / "p3-mutations.md"


@dataclass(frozen=True)
class Mutation:
    name: str
    description: str
    target: str  # ruta relativa a app/
    original: bytes
    mutant: bytes
    test_path: str  # ruta relativa a app/, qué correr con pytest


MUTATIONS = (
    Mutation(
        name="grupo duplicado en train y test",
        description=(
            "`find_leakage` ya no detecta un `duplicate_group_id` (ni `crop_id`/"
            "`source_image_id`) que aparece en dos particiones -- sección 8: una "
            "fuga real pasaría sin que nadie se entere."
        ),
        target="manifest/check_leakage.py",
        original=b"if len(splits) > 1",
        mutant=b"if len(splits) > 2",
        test_path="tests/test_manifest.py",
    ),
    Mutation(
        name="predicción cambiada en la matriz de confusión",
        description=(
            "`confusion_matrix` acumula en `[predicted][true]` en vez de "
            "`[true][predicted]` -- la matriz queda transpuesta; precisión y "
            "recall se intercambian sin que ninguna métrica agregada avise."
        ),
        target="evaluation/metrics.py",
        original=b"matrix[prediction.true_label][prediction.predicted_label] += 1",
        mutant=b"matrix[prediction.predicted_label][prediction.true_label] += 1",
        test_path="tests/test_p3_13.py",
    ),
    Mutation(
        name="aumentación en validación",
        description=(
            "`get_preprocessing_transforms` aplica el pipeline de 'train' "
            "(recorte aleatorio, flip, color jitter) a cualquier split que no "
            "sea 'test' -- validation deja de ser determinista."
        ),
        target="training/preprocess.py",
        original=b'if split == "train":',
        mutant=b'if split != "test":',
        test_path="tests/test_p3_05.py",
    ),
    Mutation(
        name="restaurar la última época en vez de la mejor",
        description=(
            "`EarlyStopping` guarda los pesos en cada llamada, no solo cuando "
            "`val_loss` mejora -- al terminar, `best_weights` es en realidad la "
            "última época, no la de mejor validación."
        ),
        target="training/trainer.py",
        original=b"if val_loss < self.best_loss - self.min_delta:",
        mutant=b"if True:  # val_loss < self.best_loss - self.min_delta",
        test_path="tests/test_p3_08.py",
    ),
    Mutation(
        name="caja degenerada",
        description=(
            "`analyze_invalid_boxes` ya no marca una bbox de ancho exactamente "
            "cero como inválida (solo ancho negativo) -- una caja degenerada "
            "pasaría la compuerta de calidad."
        ),
        target="analyzers/invalid_boxes.py",
        original=b"if width <= 0:",
        mutant=b"if width < 0:",
        test_path="tests/test_invalid_boxes.py",
    ),
)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _apply_and_test(
    mutation: Mutation, worktree_app: Path, python_bin: Path
) -> tuple[bool, str]:
    target_path = worktree_app / mutation.target
    source = target_path.read_bytes()
    count = source.count(mutation.original)
    if count != 1:
        return False, f"el fragmento original apareció {count} veces (se esperaba 1)"

    target_path.write_bytes(source.replace(mutation.original, mutation.mutant))
    result = _run(
        [str(python_bin), "-m", "pytest", "-q", mutation.test_path], cwd=worktree_app
    )
    killed = result.returncode != 0
    return killed, (result.stdout + result.stderr).strip()


def _write_evidence(rows: list[tuple[Mutation, bool, str]]) -> None:
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evidencia de mutaciones (P3-17, #27)",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        (
            "Cada mutación corre en un `git worktree` temporal (nunca se toca el "
            "checkout real) y debe hacer fallar la prueba que la cubre."
        ),
        "",
        "| Mutación | Archivo | Prueba | Resultado |",
        "|---|---|---|---|",
    ]
    for mutation, killed, _output in rows:
        result = "matada" if killed else "**SOBREVIVIÓ**"
        lines.append(
            f"| {mutation.name} | `{mutation.target}` | `{mutation.test_path}` | {result} |"
        )
    lines.append("")
    for mutation, killed, output in rows:
        lines += [
            f"## {mutation.name}",
            "",
            mutation.description,
            "",
            f"- Archivo: `{mutation.target}`",
            f"- Prueba: `{mutation.test_path}`",
            f"- Resultado: {'matada' if killed else 'SOBREVIVIÓ'}",
            "",
            "```",
            output[-2000:],
            "```",
            "",
        ]
    EVIDENCE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    python_bin = APP_ROOT / ".venv" / "bin" / "python"
    if not python_bin.is_file():
        print(
            f"No existe {python_bin}; corre 'uv sync --group ml' en app/ primero.",
            file=sys.stderr,
        )
        return 2

    # Antes de tocar nada: el checkout real puede traer trabajo en curso sin
    # commitear (normal durante desarrollo) -- lo que importa es que el
    # script no le agregue NADA nuevo, no que esté vacío.
    status_before = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT).stdout

    suffix = uuid.uuid4().hex[:8]
    worktree_dir = REPO_ROOT.parent / f"p3-mutations-{suffix}"
    created = _run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), "HEAD"], cwd=REPO_ROOT
    )
    if created.returncode != 0:
        print(created.stderr, file=sys.stderr)
        return 2

    rows: list[tuple[Mutation, bool, str]] = []
    try:
        for mutation in MUTATIONS:
            killed, output = _apply_and_test(mutation, worktree_dir / "app", python_bin)
            rows.append((mutation, killed, output))
            print(f"[{'matada' if killed else 'SOBREVIVIÓ'}] {mutation.name}")
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=REPO_ROOT)

    # El worktree vive fuera del repo (REPO_ROOT.parent) y nunca se tocó el
    # checkout real -- esto confirma que de verdad quedó exactamente como
    # estaba antes (ADVERTENCIA si no, no si ya había cambios previos). Se
    # revisa ANTES de escribir la evidencia: ese archivo es la salida
    # esperada del script, no "suciedad" que haya que detectar.
    status_after = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT).stdout
    status_changed = status_after != status_before

    _write_evidence(rows)

    if status_changed:
        print("ADVERTENCIA: git status cambió durante las mutaciones:", file=sys.stderr)
        print(status_after, file=sys.stderr)
        return 1

    survived = [mutation.name for mutation, killed, _ in rows if not killed]
    if survived:
        print(
            f"\n{len(survived)} mutación(es) sobrevivieron: {', '.join(survived)}",
            file=sys.stderr,
        )
        return 1

    print(f"\nLas {len(MUTATIONS)} mutaciones fueron detectadas. git status limpio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
