# P4-17 · Imagen MinIO descargable en Compose local

Prueba realizada el 12 de octubre de 2026 en Docker Desktop 4.95.0, Engine 29.8.2 (Linux arm64) y Compose v5.6.0. El proyecto aislado fue `p4-17-minio-test`; no utilizó volúmenes ni contenedores anteriores. El [override de prueba](override.yml) se combinó con `docker-compose.yml` y un archivo de entorno temporal fuera del repositorio que contenía únicamente valores ficticios.

## Reproducción

Desde la raíz del repositorio, prepara fuera de él un archivo de entorno temporal con valores ficticios para `MARIADB_ROOT_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` y `AWS_PROFILE`. Asigna su ruta a `P4_TEST_ENV` antes de ejecutar:

```bash
docker compose -p p4-17-minio-test --env-file "$P4_TEST_ENV" -f docker-compose.yml -f reports/p4/operation/minio-image/override.yml pull --policy always minio
docker compose -p p4-17-minio-test --env-file "$P4_TEST_ENV" -f docker-compose.yml -f reports/p4/operation/minio-image/override.yml up -d --build frontend
docker compose -p p4-17-minio-test --env-file "$P4_TEST_ENV" -f docker-compose.yml -f reports/p4/operation/minio-image/override.yml ps
curl -f http://127.0.0.1:19000/minio/health/ready
curl -f http://127.0.0.1:18080/api/health
```

Para repetir la lectura y escritura, sube un PNG sintético con `curl -F 'image=@/ruta/a/imagen-de-prueba.png;type=image/png' http://127.0.0.1:18080/api/images`. Usa el `id` devuelto para descargar `/api/images/<id>/file` y compara los SHA-256 de ambos archivos. El override evita publicar MariaDB y backend en el host, monta las carpetas de configuración del backend en solo lectura y no inicia Copilot ni `ml-api`.

## Descarga y arranque

- [Salida de `docker compose pull --policy always minio`](pull.txt): `cgr.dev/chainguard/minio:latest` se descargó del registro, con transferencia de capas, y terminó correctamente. No se borraron imágenes para realizar la prueba.
- [Estado de `docker compose ps`](ps.txt): `minio`, `mariadb`, `backend` y `frontend` quedaron `Up`; MariaDB quedó `healthy`.
- [Arranque de MinIO](minio-startup.txt): versión `RELEASE.2026-09-22T19-25-18Z`; los endpoints `/minio/health/live` y `/minio/health/ready` respondieron HTTP 200. El contenedor se ejecutó como UID 65532 y pudo escribir en el volumen nuevo `p4-17-minio-test_minio_data`.

## Escritura, lectura y portal

- [Prueba de lectura y escritura](read-write.txt): el backend cargó una imagen PNG sintética de un píxel mediante `POST /api/images` (201) y recuperó los mismos 69 bytes mediante `GET /api/images/1/file` (200); los SHA-256 coincidieron. La creación del bucket y los metadatos ocurrió solo en los volúmenes de prueba.
- [Comprobación del portal local](portal.txt): `/`, `/edge/captures` y `/api/health` respondieron HTTP 200. La respuesta de `/edge/captures` confirma la ruta de la SPA; el contenido de capturas no pudo validarse porque no se inició `ml-api` ni se utilizaron credenciales AWS. `/ml-api/edge/captures` respondió 502 en este entorno mínimo.

El proyecto Docker de prueba se dejó en ejecución para revisión, con puertos `127.0.0.1:18080`, `19000` y `19001`. No se incluyeron credenciales ni respuestas de AWS en estas evidencias.
