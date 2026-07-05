#!/usr/bin/env bash
# Ripristina i venv dal backup NAS (rollback della migrazione #103). Ogni tar torna alla SUA cartella
# originale (i venv sono relocatable solo sullo stesso path → shebang/paths assoluti combaciano).
#
# Uso:
#   bash scripts/restore_venvs_from_nas.sh                 # ripristina TUTTI i venv del manifest
#   bash scripts/restore_venvs_from_nas.sh inference_server_venv   # solo quelli che matchano il filtro
#   DRYRUN=1 bash scripts/restore_venvs_from_nas.sh        # mostra solo cosa farebbe
set -uo pipefail

NAS_HOST="${NAS_HOST:-nuspy@100.84.254.116}"
NAS_PORT="${NAS_PORT:-9222}"
NAS_DIR="${NAS_DIR:-/home/nuspy/ltx_venv_backups}"
STAGING="${STAGING:-/c/TEMP/venv_backup_staging}"
DRYRUN="${DRYRUN:-0}"
FILTER="${1:-}"

run() { if [ "$DRYRUN" = "1" ]; then echo "  [dry] $*"; else eval "$*"; fi; }

mkdir -p "$STAGING"
echo "=== Scarico il manifest dal NAS ==="
scp -P "$NAS_PORT" "$NAS_HOST:$NAS_DIR/venv_manifest.txt" "$STAGING/venv_manifest.txt" || {
  echo "ERRORE: manifest non trovato sul NAS ($NAS_DIR)"; exit 1; }

while IFS= read -r V; do
  [ -z "$V" ] && continue
  name="$(echo "$V" | sed 's#^/##; s#/#_#g')"
  if [ -n "$FILTER" ] && ! echo "$V" | grep -q "$FILTER"; then continue; fi
  tarf="$STAGING/${name}.tar"
  echo "→ ripristino $V"
  run "scp -P $NAS_PORT '$NAS_HOST:$NAS_DIR/${name}.tar' '$tarf'"
  if [ -d "$V" ]; then
    echo "  ⚠ esiste già $V → lo rinomino in ${V}.OLD prima di estrarre"
    run "mv '$V' '${V}.OLD.$$'"
  fi
  run "mkdir -p '$(dirname "$V")'"
  run "tar -C '$(dirname "$V")' -xf '$tarf'"
  run "rm -f '$tarf'"
  echo "  ✓ $V ripristinato (vecchio in ${V}.OLD.* se c'era)"
done < "$STAGING/venv_manifest.txt"
echo "=== Fatto. Verifica: <venv>/Scripts/python.exe -c \"import torch; print(torch.__version__)\" ==="
