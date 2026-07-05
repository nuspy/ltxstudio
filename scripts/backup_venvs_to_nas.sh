#!/usr/bin/env bash
# Backup dei venv Python sul NAS (rete di sicurezza per la migrazione CUDA13/Torch2.11/Py3.11 — task #103).
# Ogni venv viene impacchettato in UN tar (trasferimento veloce vs migliaia di file su SMB) e copiato sul NAS
# via SSH/Tailscale (auth con chiave: NESSUNA password nel file). 14 TB liberi sul NAS bastano ampiamente.
#
# Uso:
#   bash scripts/backup_venvs_to_nas.sh              # esegue il backup
#   DRYRUN=1 bash scripts/backup_venvs_to_nas.sh     # mostra solo cosa farebbe
#   NAS_DIR=/Volume1/backups/ltx_venvs bash scripts/backup_venvs_to_nas.sh   # override destinazione
#
# Ripristino: scripts/restore_venvs_from_nas.sh
set -uo pipefail

# --- Config (override via env) ---
NAS_HOST="${NAS_HOST:-nuspy@100.84.254.116}"     # Tailscale (key-based login attivo)
NAS_PORT="${NAS_PORT:-9222}"
NAS_DIR="${NAS_DIR:-/home/nuspy/ltx_venv_backups}"   # cartella di destinazione sul NAS (crea se assente)
STAGING="${STAGING:-/c/TEMP/venv_backup_staging}"    # dove creo i tar prima di spedirli
DRYRUN="${DRYRUN:-0}"

# --- Venv da salvare (aggiungi/rimuovi secondo la tua macchina) ---
VENVS=(
  "/e/AI/Comfy_Nexus.bak/ComfyUI/venv"          # runtime LTX-2 (torch 2.8)
  "/e/AI/Tools/inference_server_venv"            # backend nativo (torch 2.11)
  "/e/AI/Tools/ltx2_venv"                        # worker video LTX-2
  "/e/AI/Tools/ACE-Step/venv15"                  # ACE-Step 1.5
  "/e/AI/Tools/ai-toolkit/venv"                  # LoRA training
)

run() { if [ "$DRYRUN" = "1" ]; then echo "  [dry] $*"; else eval "$*"; fi; }

echo "=== Backup venv → NAS ($NAS_HOST:$NAS_PORT:$NAS_DIR) ==="
mkdir -p "$STAGING"
run "ssh -p $NAS_PORT $NAS_HOST 'mkdir -p $NAS_DIR'"

for V in "${VENVS[@]}"; do
  if [ ! -d "$V" ]; then echo "  ⏭  assente, salto: $V"; continue; fi
  name="$(echo "$V" | sed 's#^/##; s#/#_#g')"        # /e/AI/... → e_AI_...
  tarf="$STAGING/${name}.tar"
  size=$(du -sh "$V" 2>/dev/null | cut -f1)
  echo "→ $V  ($size)"
  run "tar -C '$(dirname "$V")' -cf '$tarf' '$(basename "$V")'"
  run "scp -P $NAS_PORT '$tarf' '$NAS_HOST:$NAS_DIR/'"
  run "rm -f '$tarf'"                                 # libera lo staging locale
  echo "  ✓ $name.tar sul NAS"
done

# manifest (path originale di ogni venv, per il ripristino esatto)
manifest="$STAGING/venv_manifest.txt"
printf "%s\n" "${VENVS[@]}" > "$manifest"
run "scp -P $NAS_PORT '$manifest' '$NAS_HOST:$NAS_DIR/'"
echo "=== Fatto. Manifest + tar su $NAS_DIR. Ripristino: scripts/restore_venvs_from_nas.sh ==="
