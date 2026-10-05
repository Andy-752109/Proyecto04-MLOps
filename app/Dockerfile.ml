# Variante de Dockerfile con el grupo de dependencias `ml` (torch, torchvision,
# mlflow, scikit-learn, pillow fijados -- P3-03). La usan `trainer-worker` y
# `ml-api`, que sí necesitan entrenar/cargar el modelo; `app` y `copilot`
# siguen construyéndose con el Dockerfile base, sin torch (el Copilot no debe
# cargarlo: es un proceso aparte, más pesado, y sin ninguna razón para
# entrenar ni cargar un checkpoint).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

# P3-08: `training.tracking.get_git_info()` registra el commit real de cada
# corrida (contrato de MLflow, sección 8). Se hornea en el build, no se lee
# del host en vivo -- revisión de Uriel sobre este PR: el código que entrena
# es el que se copió a la imagen (los COPY de abajo), así que el commit que
# cuenta es el de ESE momento, no el que esté en el checkout del host cuando
# arranca el contenedor (si alguien cambia de rama sin reconstruir, MLflow
# registraría un commit que no es el que de verdad corrió). `docker-compose.yml`
# pasa estos ARG calculados por el host al invocar `docker compose build`;
# `get_git_info()` (P3-08) ya prioriza estas variables sobre `git` en vivo.
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIRTY=${GIT_DIRTY}

WORKDIR /app

# A diferencia del Dockerfile base: SÍ necesita un home real. trainer-worker
# monta ~/.aws de solo lectura ahí (docker-compose.yml) para el perfil SSO
# del equipo, y boto3 resuelve las credenciales vía $HOME/.aws/credentials.
# `HOME` se define recién antes de `USER appuser` (no aquí): si se define
# antes, `uv sync` (que todavía corre como root) escribe su propio caché
# bajo $HOME/.cache/uv como root, y appuser ya no puede crear nada dentro de
# ese ~/.cache más tarde (p. ej. los pesos de torchvision) -- probado real,
# "Permission denied: '/home/appuser/.cache/torch'" al entrenar.
RUN useradd --system --create-home --home-dir /home/appuser appuser

COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=180
ENV UV_CONCURRENT_DOWNLOADS=2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --no-build --group ml

COPY ingestion/ ./ingestion/
COPY analyzers/ ./analyzers/
COPY policies/ ./policies/
COPY splits/ ./splits/
COPY storage/ ./storage/
COPY presentation/ ./presentation/
COPY projections/ ./projections/
COPY crops/ ./crops/
# P3-06/P3-07: ml-api los necesita para reverificar el manifiesto antes de
# crear un training_job (P3-09); trainer-worker los necesita para entrenar.
COPY manifest/ ./manifest/
COPY selection/ ./selection/
# P3-16: ml-api necesita CLASSES (el orden cat/dog) para /predict.
COPY evaluation/ ./evaluation/
COPY training/ ./training/
COPY mlflow_ops/ ./mlflow_ops/
COPY ml_worker/ ./ml_worker/
COPY ml_api/ ./ml_api/

ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/home/appuser
# Por si algún paso anterior ya dejó algo ahí como root (defensivo).
RUN chown -R appuser:appuser /home/appuser
USER appuser

CMD ["python", "-m", "ml_worker"]
