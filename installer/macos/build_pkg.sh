#!/usr/bin/env bash
# LTX Studio — build .app → .dmg per macOS. L'app installata mostra il wizard «Setup modelli» al primo avvio.
# Firma+notarizzazione via build/sign.py (codesign Developer ID + notarytool) → evita Gatekeeper.
#
# Uso:  APP_SRC=dist/LTXStudio.app VER=1.2.3 bash installer/macos/build_pkg.sh
# Prereq: create-dmg (brew install create-dmg), Xcode CLT. Firma: APPLE_SIGN_IDENTITY, APPLE_NOTARY_PROFILE.
set -euo pipefail

APP_SRC="${APP_SRC:-dist/LTXStudio.app}"
VER="${VER:-0.0.0}"
OUT="dist/LTXStudio-${VER}.dmg"
VOLNAME="LTX Studio"

[ -d "$APP_SRC" ] || { echo "App bundle non trovato: $APP_SRC (compila prima l'app in .app)"; exit 1; }

echo "→ creo il DMG $OUT"
create-dmg \
  --volname "$VOLNAME" \
  --window-size 540 380 \
  --icon-size 100 \
  --icon "$(basename "$APP_SRC")" 140 190 \
  --app-drop-link 400 190 \
  "$OUT" "$APP_SRC" || hdiutil create -volname "$VOLNAME" -srcfolder "$APP_SRC" -ov -format UDZO "$OUT"

echo "→ firma + notarizzazione (build/sign.py, se le credenziali Apple sono presenti)"
python3 build/sign.py --os macos --out dist || python3 - <<'PY'
import sys; sys.path.insert(0,".")
from build import sign
sign.sign_artifacts("dist","macos",{})
PY

echo "OK → $OUT (firmato+notarizzato se le credenziali Apple erano configurate)."
