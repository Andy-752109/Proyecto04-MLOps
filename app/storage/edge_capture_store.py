"""Operaciones S3 de lectura para eventos edge; el cliente se inyecta en tests."""

import boto3
from botocore.config import Config

from storage.settings import Settings

EVENT_PREFIX = "edge-captures/v1/events/"


def get_edge_s3_client(settings: Settings):
    session = (
        boto3.Session(profile_name=settings.aws_profile)
        if settings.aws_profile
        else boto3.Session()
    )
    return session.client(
        "s3", region_name=settings.aws_region, config=Config(signature_version="s3v4")
    )


def list_event_objects(client, *, bucket: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=EVENT_PREFIX):
        yield from page.get("Contents", [])


def read_event_json(client, *, bucket: str, key: str) -> bytes:
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        return body.read()
    finally:
        body.close()
