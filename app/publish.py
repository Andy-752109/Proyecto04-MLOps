"""Empaqueta y publica una versión del modelo en S3 (P3-14). Nunca sobrescribe.

El paquete lleva: artefactos del run (incluye artifacts/checkpoint/best.pt), config.json,
architecture.json, class_map.json, preprocess.json, dependencias, la tarjeta y package.json
(versión, run_id, releases y SHA-256 de cada archivo). Falla con código 1 si falta algo,
si la versión ya existe en S3 o si el bucket no tiene versionado.

Se corre desde app/ (lee training/, así que necesita torch y torchvision):

    uv run python publish.py --run-id <run_id> --version 1.0.0 --dry-run      # solo arma y revisa
    uv run python publish.py --run-id <run_id> --version 1.0.0 --bucket <bucket>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from mlflow_dump import ROOT, artifacts_dir, load_run, typed_params
from safe_path import safe_path, safe_run_id, safe_version

sys.path.insert(0, str(ROOT / "app"))

CONFIG_JSON = "config.json"
ARCHITECTURE_JSON = "architecture.json"
CLASS_MAP_JSON = "class_map.json"
PREPROCESS_JSON = "preprocess.json"
PACKAGE_JSON = "package.json"

REQUIRED = (
    "artifacts/checkpoint/best.pt",
    CONFIG_JSON,
    ARCHITECTURE_JSON,
    CLASS_MAP_JSON,
    PREPROCESS_JSON,
    "model_card.md",
    "dependencies/pyproject.toml",
    PACKAGE_JSON,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def stage_files(run: dict, version: str, staging: Path, card: Path) -> list[tuple[Path, str]]:
    """Reúne (archivo, nombre dentro del paquete). Falla si falta algo."""
    run_id, tags = run["run_id"], run["tags"]
    files: list[tuple[Path, str]] = []

    art = artifacts_dir(run_id)
    best = art / "checkpoint" / "best.pt"
    if not best.exists():
        sys.exit(f"Falta {best}")
    if sha256(best) != tags.get("checkpoint_sha256"):
        sys.exit("El SHA-256 de best.pt no coincide con el tag checkpoint_sha256 del run.")
    files += [
        (f, "artifacts/" + f.relative_to(art).as_posix())
        for f in sorted(art.rglob("*"))
        if f.is_file()
    ]

    if not card.exists():
        sys.exit(f"Falta {card}: genera la tarjeta con card.py primero.")
    if run_id not in card.read_text(encoding="utf-8"):
        sys.exit(f"{card.name} no es de este run.")
    files.append((card, "model_card.md"))

    config = typed_params(run["params"])
    for key in ("image_size", "hidden_layers", "dropout"):
        if key not in config:
            sys.exit(f"Falta el param '{key}' en el run.")
    write_json(staging / CONFIG_JSON, config)

    from training.model import export_architecture_and_class_map
    from training.preprocess import get_preprocessing_transforms

    arch_path, cmap_path = staging / ARCHITECTURE_JSON, staging / CLASS_MAP_JSON
    export_architecture_and_class_map(str(arch_path), str(cmap_path))
    class_map = json.loads(cmap_path.read_text())
    if class_map != json.loads(tags["classes"]):
        sys.exit("El class_map de training.model no coincide con el tag 'classes' del run.")
    arch = json.loads(arch_path.read_text())
    arch.update(
        {
            "builder": "training.model.build_model",
            "config_file": CONFIG_JSON,
            "hidden_layers": config["hidden_layers"],
            "dropout": config["dropout"],
            "num_classes": len(class_map),
        }
    )
    write_json(arch_path, arch)

    pipeline = get_preprocessing_transforms("test", config["image_size"])
    write_json(
        staging / PREPROCESS_JSON,
        {
            "entry_point": "training.preprocess.get_preprocessing_transforms",
            "split": "test",
            "image_size": config["image_size"],
            "transforms": [repr(t) for t in pipeline.transforms],
        },
    )
    files += [
        (staging / n, n) for n in (CONFIG_JSON, ARCHITECTURE_JSON, CLASS_MAP_JSON, PREPROCESS_JSON)
    ]

    for name in ("pyproject.toml", "uv.lock"):
        dep = ROOT / "app" / name
        if dep.exists():
            files.append((dep, f"dependencies/{name}"))
        elif name == "pyproject.toml":
            sys.exit(f"Falta {dep}")

    package = {
        "version": version,
        "model_release": f"v{version}",
        "run_id": run_id,
        "run_kind": tags.get("run_kind"),
        "data_release": tags.get("release"),
        "manifest_id": tags.get("manifest_id"),
        "checkpoint_sha256": tags["checkpoint_sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {arc: sha256(src) for src, arc in files},
    }
    write_json(staging / PACKAGE_JSON, package)
    files.append((staging / PACKAGE_JSON, PACKAGE_JSON))
    return files


def build_package(run: dict, version: str, card_path: Path, workdir: Path) -> Path:
    staging = workdir / "build" / f"package_v{version}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    files = stage_files(run, version, staging, card_path)

    dist = workdir / "dist"
    dist.mkdir(exist_ok=True)
    tar_path = dist / f"model_release_v{version}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for src, arc in files:
            tar.add(src, arcname=arc)
    with tarfile.open(tar_path, "r:gz") as tar:
        missing = set(REQUIRED) - set(tar.getnames())
    if missing:
        sys.exit(f"Paquete incompleto, faltan: {sorted(missing)}")
    print(f"Paquete completo: {tar_path} ({len(files)} archivos)")
    return tar_path


def s3_exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def publish(
    run_id: str,
    version: str,
    s3,
    bucket: str,
    registry_path: Path,
    card_path: Path,
    workdir: Path,
) -> int:
    """Arma el paquete, lo sube con el cliente `s3` dado y lo registra. Nunca sobrescribe."""
    from botocore.exceptions import ClientError

    run = load_run(run_id)
    key = f"models/releases/v{version}/model_release_v{version}.tar.gz"

    if s3_exists(s3, bucket, key):
        print(f"ERROR: la versión {version} ya existe en s3://{bucket}/{key}. No se sobrescribe.")
        return 1
    try:
        status = s3.get_bucket_versioning(Bucket=bucket).get("Status")
        if status != "Enabled":
            print(f"ERROR: el bucket no tiene versionado activo (Status={status}).")
            return 1
    except ClientError as e:
        print(f"Aviso: no se pudo consultar el versionado del bucket ({e}).")

    tar_path = build_package(run, version, card_path, workdir)
    package_sha = sha256(tar_path)
    print(f"SHA-256 del paquete: {package_sha}")

    print(f"Subiendo a s3://{bucket}/{key} ...")
    s3.upload_file(str(tar_path), bucket, key)
    version_id = s3.head_object(Bucket=bucket, Key=key).get("VersionId")
    if not version_id or version_id == "null":
        print("ERROR: S3 no devolvió VersionId; no se registra la versión.")
        return 1

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    registry[version] = {
        "s3_path": f"s3://{bucket}/{key}",
        "sha256": package_sha,
        "VersionId": version_id,
        "run_id": run_id,
        "checkpoint_sha256": run["tags"]["checkpoint_sha256"],
        "data_release": run["tags"].get("release"),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    registry_path.write_text(  # NOSONAR: ruta validada con safe_path en main()
        json.dumps(registry, indent=4) + "\n", encoding="utf-8"
    )
    print(f"Publicado. VersionId {version_id}. registry.json actualizado.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--bucket")
    p.add_argument("--profile", default="mlops-p3")
    p.add_argument(
        "--endpoint-url", help="S3 compatible (p. ej. MinIO); sin perfil, credenciales por env"
    )
    p.add_argument("--registry", type=Path, default=ROOT / "models" / "registry.json")
    p.add_argument("--card", type=Path, help="por defecto build/model_card_<versión>.md")
    p.add_argument("--dry-run", action="store_true", help="arma y revisa el paquete sin subirlo")
    args = p.parse_args(argv)
    if not args.dry_run and not args.bucket:
        p.error("--bucket es obligatorio salvo con --dry-run")
    version, run_id = safe_version(args.version), safe_run_id(args.run_id)
    card_path = safe_path(args.card or ROOT / "build" / f"model_card_{version}.md")
    registry = safe_path(args.registry)

    if args.dry_run:
        tar_path = build_package(load_run(run_id), version, card_path, ROOT)
        print(f"SHA-256 del paquete: {sha256(tar_path)}")
        return 0

    import boto3
    from botocore.config import Config

    if args.endpoint_url:
        s3 = boto3.client(
            "s3",
            endpoint_url=args.endpoint_url,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(s3={"addressing_style": "path"}),
        )
    else:
        s3 = boto3.Session(profile_name=args.profile).client("s3")
    return publish(run_id, version, s3, args.bucket, registry, card_path, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
