"""Pruebas de `safe_path` (P3-14): las rutas de la línea de comandos deben quedar dentro del repo.

Cubre la validación en sí y que `main()` de card.py y publish.py la apliquen a sus argumentos.
Las rutas de `tmp_path` están fuera del repo a propósito: así se prueba el rechazo sin tocar
nada del proyecto.
"""

import os
from pathlib import Path

import card
import publish
import pytest
from safe_path import safe_path, safe_run_id, safe_version

RUN_ID = "a" * 32


def test_relative_paths_used_in_practice_stay_inside_the_repo(tmp_path, monkeypatch):
    # Se corre desde app/ con ../eval_v1.0.0/..., ../ref_010.csv y ../reports/...
    (tmp_path / "app").mkdir()
    monkeypatch.chdir(tmp_path / "app")

    for relative in ("../eval_v1.0.0/config.json", "../ref_010.csv", "../reports/selection.json"):
        assert safe_path(relative, base=tmp_path) == (tmp_path / relative[3:]).resolve()


def test_a_path_inside_the_base_is_accepted_and_returned_resolved(tmp_path):
    inside = safe_path(tmp_path / "a" / ".." / "b.txt", base=tmp_path)

    assert inside == (tmp_path / "b.txt").resolve()


def test_the_base_itself_is_accepted(tmp_path):
    assert safe_path(tmp_path, base=tmp_path) == tmp_path.resolve()


@pytest.mark.parametrize("escape", ["..", "../otra", "a/../../fuera.txt"])
def test_a_path_that_escapes_the_base_is_rejected_with_a_clear_message(tmp_path, escape):
    base = tmp_path / "repo"
    base.mkdir()

    with pytest.raises(SystemExit, match="Ruta fuera de"):
        safe_path(base / escape, base=base)


def test_a_sibling_folder_with_the_same_prefix_is_rejected(tmp_path):
    base, sibling = tmp_path / "repo", tmp_path / "repo-copia"
    base.mkdir()
    sibling.mkdir()

    with pytest.raises(SystemExit, match="Ruta fuera de"):
        safe_path(sibling / "x.txt", base=base)


@pytest.mark.skipif(os.name != "nt", reason="solo Windows ignora mayúsculas en las rutas")
def test_windows_paths_differing_only_in_case_are_accepted(tmp_path):
    shouting = str(tmp_path / "SUB" / "Archivo.TXT").swapcase()

    assert safe_path(shouting, base=tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="solo Windows tiene unidades")
def test_a_path_on_another_drive_is_rejected(tmp_path):
    other = "Z:\\" if Path(tmp_path).drive.upper() != "Z:" else "Y:\\"

    with pytest.raises(SystemExit, match="Ruta fuera de"):
        safe_path(other + "x.txt", base=tmp_path)


@pytest.mark.parametrize("version", ["1.0.0", "0.1.0", "10.20.30"])
def test_valid_versions_pass(version):
    assert safe_version(version) == version


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0/../x", "..\\1.0.0", "1.0.0\n", ""])
def test_invalid_versions_are_rejected(version):
    with pytest.raises(SystemExit, match="Versión inválida"):
        safe_version(version)


@pytest.mark.parametrize("run_id", [RUN_ID, "0123456789abcdef" * 2])
def test_valid_run_ids_pass(run_id):
    assert safe_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["zz", "A" * 32, "a" * 31, "../" + "a" * 29, ""])
def test_invalid_run_ids_are_rejected(run_id):
    with pytest.raises(SystemExit, match="run_id inválido"):
        safe_run_id(run_id)


def test_card_main_rejects_an_output_outside_the_repo(tmp_path):
    outside = tmp_path / "tarjeta.md"

    with pytest.raises(SystemExit, match="Ruta fuera de"):
        card.main(["--run-id", RUN_ID, "--version", "1.0.0", "--output", str(outside)])
    assert not outside.exists()


def test_card_main_rejects_a_reports_dir_outside_the_repo(tmp_path):
    with pytest.raises(SystemExit, match="Ruta fuera de"):
        card.main(["--run-id", RUN_ID, "--version", "1.0.0", "--reports-dir", str(tmp_path)])


@pytest.mark.parametrize("flag", ["--card", "--registry"])
def test_publish_main_rejects_card_and_registry_outside_the_repo(tmp_path, flag):
    argv = ["--run-id", RUN_ID, "--version", "1.0.0", "--dry-run", flag, str(tmp_path / "x")]

    with pytest.raises(SystemExit, match="Ruta fuera de"):
        publish.main(argv)


def test_publish_main_rejects_a_malformed_version_before_touching_paths(tmp_path):
    argv = ["--run-id", RUN_ID, "--version", "../../1", "--dry-run"]

    with pytest.raises(SystemExit, match="Versión inválida"):
        publish.main(argv)
