#!/usr/bin/env bash
# Comprueba la convención de nombre de rama y de título de PR (ver CONTRIBUTING.md).
#
# Uso: bash .github/scripts/check-naming.sh "<rama>" "<título del PR>"
#
# Lo llama .github/workflows/pr-hygiene.yml, y sirve igual en local antes de abrir el PR:
#   bash .github/scripts/check-naming.sh "$(git branch --show-current)" "P2-47: calidad y CI"
set -u

# Sin esto, en algunas configuraciones regionales [a-z] también acepta mayúsculas.
export LC_ALL=C

BRANCH="${1:-}"
TITLE="${2:-}"

TYPES='feat|fix|test|chore|docs|refactor|ci|build|perf|style'
# Prefijo de ticket en minúsculas (ramas) y en mayúsculas (títulos de PR): p2
# para Fase 2, p3 para Fase 3. Ambos siguen siendo válidos a la vez porque
# main puede tener trabajo pendiente de las dos fases al mismo tiempo.
PREFIX='p2|p3'
TITLE_PREFIX='P2|P3'
# p2-52-copilot-llm-client · p3-03-infra-contratos · p2-22-23-24-quality-gate
# p2-26-p2-27-lotes · p3-03-p3-05-algo · fix/nombre-corto
BRANCH_RE="^(($PREFIX)-[0-9]+(-[0-9]+)*(-($PREFIX)-[0-9]+(-[0-9]+)*)*-[a-z0-9]+(-[a-z0-9]+)*|($TYPES)/[a-z0-9]+(-[a-z0-9]+)*)\$"
# P2-52: texto · P3-03: texto · P2-22/23/24: texto · P2-26 P2-27: texto
# P3-03 P3-05: texto · fix: texto · feat(ui): texto
TITLE_RE="^(($TITLE_PREFIX)-[0-9]+(/[0-9]+| ($TITLE_PREFIX)-[0-9]+)*|($TYPES)(\\([a-z0-9-]+\\))?): .+"

status=0

if ! [[ "$BRANCH" =~ $BRANCH_RE ]]; then
  echo "::error title=Nombre de rama::'$BRANCH' no sigue la convención. Usa p2-NN-descripcion o" \
    "p3-NN-descripcion (varios tickets: p2-22-23-24-descripcion) o tipo/descripcion con tipo en" \
    "($TYPES), todo en minúsculas y con guiones. Ver CONTRIBUTING.md." >&2
  status=1
fi

if ! [[ "$TITLE" =~ $TITLE_RE ]]; then
  echo "::error title=Título del PR::'$TITLE' no sigue la convención. Usa 'P2-NN: descripción' o" \
    "'P3-NN: descripción' (varios tickets: 'P2-NN P2-MM: ...' o 'P2-22/23/24: ...') o" \
    "'tipo: descripción' con tipo en ($TYPES). Ver CONTRIBUTING.md." >&2
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "La rama y el título cumplen la convención."
fi
exit "$status"
