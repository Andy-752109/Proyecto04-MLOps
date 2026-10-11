#!/bin/sh
# P4-15: escribe el htpasswd del portal con PORTAL_USER y PORTAL_PASSWORD (del
# .env de la EC2) y arranca nginx con el entrypoint normal de la imagen.
# La contraseña entra por stdin, no como argumento visible en `ps`.
set -eu
: "${PORTAL_USER:?Falta PORTAL_USER}"
: "${PORTAL_PASSWORD:?Falta PORTAL_PASSWORD}"

# SHA-512 con `mkpasswd` de busybox (la imagen nginx:alpine no trae openssl).
HASH="$(printf '%s' "$PORTAL_PASSWORD" | mkpasswd -m sha512 -P 0)"
umask 027
printf '%s:%s\n' "$PORTAL_USER" "$HASH" > /etc/nginx/portal.htpasswd
chgrp nginx /etc/nginx/portal.htpasswd

exec /docker-entrypoint.sh "$@"
