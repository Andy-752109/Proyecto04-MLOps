"""Cliente y operaciones mínimas contra el bucket de modelos en AWS S3 (P3-15).

El bucket de modelos (`mlops-p3-models-222629887955`, ver docs/decisiones-proyecto3.md
sección 9) vive en AWS y se accede con el perfil SSO del equipo. Igual que
`object_store.py` para MinIO, este es el único lugar autorizado a crear clientes
`boto3`; `ml-api` lo consume a través de callables inyectados, nunca importa
`boto3` directo.

Todas las operaciones aceptan `version_id`: el registry de P3-14 guarda el
`VersionId` con el que se publicó cada paquete, y si la misma key se republica
hay que leer/descargar exactamente esa versión, no la última.
"""

import json
import tarfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from storage.settings import Settings

# Códigos con los que S3 dice "no existe" en un head-object.
_MISSING_CODES = {"404", "NoSuchKey", "NoSuchObject", "NotFound"}


def get_model_s3_client():
    settings = Settings()
    session = (
        boto3.Session(profile_name=settings.aws_profile)
        if settings.aws_profile
        else boto3.Session()
    )
    return session.client(
        "s3",
        region_name=settings.aws_region,
        config=Config(signature_version="s3v4"),
    )


def head_object(client, *, bucket: str, key: str, version_id: str | None = None) -> dict | None:
    """Estado en vivo (`head-object`); `None` si el objeto no existe."""
    kwargs = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        kwargs["VersionId"] = version_id
    try:
        stat = client.head_object(**kwargs)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in _MISSING_CODES:
            return None
        raise
    return {
        "version_id": stat.get("VersionId"),
        "size_bytes": stat.get("ContentLength"),
        "last_modified": stat.get("LastModified"),
    }


def presigned_get_url(
    client,
    *,
    bucket: str,
    key: str,
    version_id: str | None = None,
    expires_seconds: int = 3600,
) -> str:
    """URL prefirmada de descarga. El bucket es privado: sin firma no se baja."""
    params = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        params["VersionId"] = version_id
    return client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_seconds)


def read_package_info(client, *, bucket: str, key: str, version_id: str | None = None) -> dict:
    """Lee `model_card.md` y `package.json` de dentro del `.tar.gz` publicado.

    P3-14 empaqueta la tarjeta y el `package.json` (con `manifest_id` y
    `run_kind`) *dentro* del paquete; el registry no los repite. `card` es la
    tarjeta en texto; si al paquete le falta algo, el campo queda en `None` en
    vez de tumbar la página.

    Se lee en streaming (`r|gz`) sobre el `Body` de `get_object`: nunca se
    materializa el `.tar.gz` completo en memoria.
    """
    kwargs = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        kwargs["VersionId"] = version_id
    stream = client.get_object(**kwargs)["Body"]
    info: dict = {"card": None, "run_kind": None, "manifest_id": None}
    found: set[str] = set()
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            if member.name == "model_card.md":
                info["card"] = tar.extractfile(member).read().decode("utf-8")
                found.add("card")
            elif member.name == "package.json":
                package = json.loads(tar.extractfile(member).read().decode("utf-8"))
                info["run_kind"] = package.get("run_kind")
                info["manifest_id"] = package.get("manifest_id")
                found.add("package")
            if found == {"card", "package"}:
                # Ya se leyeron los dos archivos: no se sigue recorriendo el tar.
                break
    return info
