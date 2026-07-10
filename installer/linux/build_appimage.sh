#!/usr/bin/env bash
# LTX Studio — build AppImage per Linux. L'app mostra il wizard «Setup modelli» al primo avvio.
# Firma: SHA256SUMS + GPG via build/sign.py (GPG_SIGN_KEY) → integrità verificabile.
#
# Uso:  APP_SRC=dist/LTXStudio VER=1.2.3 bash installer/linux/build_appimage.sh
# Prereq: appimagetool nel PATH (https://github.com/AppImage/AppImageKit).
set -euo pipefail

APP_SRC="${APP_SRC:-dist/LTXStudio}"        # cartella con l'eseguibile + runtime (Nuitka onedir)
VER="${VER:-0.0.0}"
APPDIR="dist/LTXStudio.AppDir"
OUT="dist/LTXStudio-${VER}-x86_64.AppImage"

[ -d "$APP_SRC" ] || { echo "Build non trovato: $APP_SRC"; exit 1; }

rm -rf "$APPDIR"; mkdir -p "$APPDIR/usr/bin"
cp -r "$APP_SRC"/* "$APPDIR/usr/bin/"

cat > "$APPDIR/AppRun" <<'SH'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/LTXStudio" "$@"
SH
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/ltxstudio.desktop" <<DESK
[Desktop Entry]
Name=LTX Studio
Exec=LTXStudio
Icon=ltxstudio
Type=Application
Categories=Graphics;AudioVideo;
DESK
# icona (usa un placeholder se assente)
cp "assets/icon.png" "$APPDIR/ltxstudio.png" 2>/dev/null || touch "$APPDIR/ltxstudio.png"

echo "→ appimagetool → $OUT"
ARCH=x86_64 appimagetool "$APPDIR" "$OUT"

echo "→ SHA256SUMS + firma GPG (build/sign.py)"
python3 - <<PY
import sys; sys.path.insert(0,".")
from build import sign
sign.sign_artifacts("dist","linux",{})
PY

echo "OK → $OUT"
