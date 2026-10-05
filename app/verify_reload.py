"""Verifica que cada versión de models/registry.json se descarga de S3 por VersionId
y que su SHA-256 coincide con el registro. Sale con código 1 si algo falla.

    uv run python verify_reload.py [--profile mlops-p3] [--registry ../models/registry.json]
        [--version 1.0.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import boto3
from mlflow_dump import ROOT
from safe_path import safe_path


def calculate_sha256(filepath: Path) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_s3_path(s3_path: str) -> tuple[str, str]:
    bucket, _, key = s3_path.removeprefix("s3://").partition("/")
    return bucket, key


def verify_version(s3, version: str, data: dict, dist: Path) -> bool:
    bucket, key = split_s3_path(data["s3_path"])
    version_id = data.get("VersionId")
    local_tar = dist / f"downloaded_v{version}.tar.gz"
    extra = {"VersionId": version_id} if version_id and version_id.lower() != "null" else {}
    print(f"Verificando {version} (VersionId {version_id})...")
    try:
        s3.download_file(Bucket=bucket, Key=key, Filename=str(local_tar), ExtraArgs=extra)
    except Exception as e:
        print(f"  ERROR al descargar {version}: {e}")
        return False

    local_sha = calculate_sha256(local_tar)
    match = local_sha == data["sha256"]
    print(f"  SHA-256 esperado:  {data['sha256']}")
    print(f"  SHA-256 calculado: {local_sha}  {'COINCIDE' if match else 'NO COINCIDE'}")
    if match:
        extract_dir = ROOT / f"eval_v{version}"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")
        print(f"  Extraído en {extract_dir}/")
    local_tar.unlink()
    return match


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--profile", default="mlops-p3")
    p.add_argument("--registry", type=Path, default=ROOT / "models" / "registry.json")
    p.add_argument("--version", help="Verifica solo esta versión del registro (p. ej. 1.0.0).")
    args = p.parse_args(argv)

    registry_path = safe_path(args.registry)
    if not registry_path.exists():
        print(f"No se encontró {registry_path}")
        return 1
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    if args.version:
        if args.version not in registry:
            print(f"La versión {args.version} no está en {registry_path}")
            return 1
        registry = {args.version: registry[args.version]}

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    s3 = boto3.Session(profile_name=args.profile).client("s3")
    results = [verify_version(s3, v, d, dist) for v, d in registry.items()]
    print("\nResultado:", "TODO COINCIDE" if all(results) else "HAY FALLOS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
