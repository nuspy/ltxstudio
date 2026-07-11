#!/usr/bin/env bash
# Driver esaustivo resiliente: esegue run_real.py a blocchi ISOLATI (ogni blocco = processo separato →
# un crash/OOM ferma solo il suo blocco, il driver prosegue al successivo). I risultati per-caso restano in
# output/tests/REAL/<name>.txt. Progresso in output/tests/_exhaustive_progress.txt.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY="E:/AI/Comfy_Nexus.bak/ComfyUI/venv/Scripts/python.exe"
TOTAL="${TOTAL:-622}"; STEP="${STEP:-20}"
PROG="output/tests/_exhaustive_progress.txt"
echo "START $(date '+%H:%M:%S') total=$TOTAL step=$STEP" > "$PROG"
for ((s=0; s<TOTAL; s+=STEP)); do
  echo "chunk start=$s $(date '+%H:%M:%S')" >> "$PROG"
  QT_QPA_PLATFORM=offscreen "$PY" -u tests/gui_regression/run_real.py --start "$s" --limit "$STEP" \
      >> "output/tests/_exhaustive.log" 2>&1
  echo "chunk start=$s DONE rc=$? $(date '+%H:%M:%S')" >> "$PROG"
done
echo "ALL DONE $(date '+%H:%M:%S')" >> "$PROG"
