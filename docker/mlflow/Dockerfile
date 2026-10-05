# La imagen oficial de MLflow no trae driver de MariaDB (pymysql) ni cliente
# S3 (boto3) -- los necesita para el backend store (MariaDB) y el artifact
# store (MinIO) de P3-03. Se instalan aquí, en el build, no con un `pip
# install` en el entrypoint del contenedor: así no depende de que PyPI esté
# alcanzable cada vez que el servicio arranca o se reinicia.
#
# Versiones fijadas al mismo pin que app/uv.lock (pymysql, boto3), para no
# resolver una versión distinta a la que ya se prueba ahí. --only-binary
# :all:: solo wheels ya publicadas, sin ejecutar scripts de compilación de
# terceros (mismo criterio que app/Dockerfile).
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN useradd --system --no-create-home mlflow \
    && pip install --no-cache-dir --only-binary :all: pymysql==1.2.3 boto3==1.43.105

USER mlflow
