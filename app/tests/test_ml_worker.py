"""`ml_worker.__main__._wait_for_dependencies` (P3-03).

`ml_worker` importa `torch` a nivel de módulo a propósito (confirma en el
arranque real que el grupo `ml` cargó, ver el docstring del módulo) -- así
que esta prueba se salta si `torch` no está instalado, igual que CI no
instala el grupo `ml` para el resto de `app/` (ver CONTRIBUTING.md). El
resto del bucle real de entrenamiento lo prueba P3-09, no este ticket.
"""

import pytest

pytest.importorskip("torch")

from ml_worker.__main__ import _wait_for_dependencies


class _FailTwiceThenSucceedConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, _query):
        return None


class _FailTwiceThenSucceedEngine:
    def __init__(self):
        self.attempts = 0

    def connect(self):
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("MariaDB todavía no acepta conexiones")
        return _FailTwiceThenSucceedConnection()


def test_retries_until_mariadb_is_reachable(monkeypatch):
    engine = _FailTwiceThenSucceedEngine()
    monkeypatch.setattr("ml_worker.__main__.get_engine", lambda: engine)
    monkeypatch.setattr("ml_worker.__main__.time.sleep", lambda _seconds: None)

    _wait_for_dependencies(retries=5, delay_seconds=0)

    assert engine.attempts == 3


def test_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr("ml_worker.__main__.get_engine", lambda: _FailTwiceThenSucceedEngine())
    monkeypatch.setattr("ml_worker.__main__.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="No se pudo conectar"):
        _wait_for_dependencies(retries=2, delay_seconds=0)
