"""`mlflow_ops.artifacts_sync`/`ensure_bucket` (P3-03), contra un doble de MinIO.

No toca un MinIO real -- CI no levanta uno para `app/` (ver CONTRIBUTING.md)
-- pero el comportamiento real (backup/restore de verdad contra el MinIO del
docker-compose local) se verificó a mano como parte de este ticket; ver
`docs/api-contratos.md`/el PR. Este doble solo implementa los cuatro
métodos que `mlflow_ops` de verdad usa.
"""

from pathlib import Path

from mlflow_ops import artifacts_sync, ensure_bucket


class _Object:
    def __init__(self, name: str):
        self.object_name = name


class _FakeMinioClient:
    def __init__(self, existing_buckets: set[str] | None = None):
        self.buckets = existing_buckets or set()
        self.objects: dict[str, bytes] = {}
        self.made_buckets: list[str] = []

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self.buckets

    def make_bucket(self, bucket: str) -> None:
        self.buckets.add(bucket)
        self.made_buckets.append(bucket)

    def list_objects(self, bucket: str, recursive: bool = False):
        assert recursive, "artifacts_sync siempre debe listar recursivo"
        prefix = f"{bucket}/"
        return [_Object(name[len(prefix) :]) for name in self.objects if name.startswith(prefix)]

    def fget_object(self, bucket: str, object_name: str, file_path: str) -> None:
        Path(file_path).write_bytes(self.objects[f"{bucket}/{object_name}"])

    def fput_object(self, bucket: str, object_name: str, file_path: str) -> None:
        self.objects[f"{bucket}/{object_name}"] = Path(file_path).read_bytes()


def test_ensure_bucket_creates_it_only_if_missing(monkeypatch):
    client = _FakeMinioClient()
    monkeypatch.setattr(ensure_bucket, "get_minio_client", lambda: client)

    ensure_bucket.ensure_bucket("mlflow-artifacts")
    ensure_bucket.ensure_bucket("mlflow-artifacts")

    assert client.made_buckets == ["mlflow-artifacts"]


def test_backup_then_restore_round_trips_every_file(monkeypatch, tmp_path):
    client = _FakeMinioClient(existing_buckets={"mlflow-artifacts"})
    client.objects["mlflow-artifacts/1/run.json"] = b'{"run_id": "abc"}'
    client.objects["mlflow-artifacts/1/model.pt"] = b"\x00\x01fake-weights"
    monkeypatch.setattr(artifacts_sync, "get_minio_client", lambda: client)

    backup_dir = tmp_path / "backup"
    count = artifacts_sync.backup(backup_dir)

    assert count == 2
    assert (backup_dir / "1" / "run.json").read_bytes() == b'{"run_id": "abc"}'
    assert (backup_dir / "1" / "model.pt").read_bytes() == b"\x00\x01fake-weights"

    # Restaurar sobre un bucket vacío en otro cliente -- simula un clon limpio.
    empty_client = _FakeMinioClient()
    monkeypatch.setattr(artifacts_sync, "get_minio_client", lambda: empty_client)
    monkeypatch.setattr(ensure_bucket, "get_minio_client", lambda: empty_client)

    restored = artifacts_sync.restore(backup_dir)

    assert restored == 2
    assert empty_client.objects == client.objects
