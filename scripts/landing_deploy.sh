#!/usr/bin/env bash
# Deploy della landing TaDaMoo su tadamoo.com come HOMEPAGE.
#
# Il sito è servito dal container docker "website" (nginx:alpine) che monta /opt/website/html (ro).
# La home ATTUALE è la piattaforma trading ("TaDaMoo - To Da Moon! AI & ML Trading Platform"):
# la RINOMINIAMO in index.html.bak prima di pubblicare (scelta utente, 12/07) → rollback = un mv.
# Gli altri file (ac.html, dashboard.html, panels.html, hai-ai/, AI-Forge/) NON vengono toccati.
#
# Uso:  bash scripts/landing_deploy.sh [--dry-run]
set -euo pipefail

SRC="E:/ai/ltxstudio/marketing/tadamoo_landing"
HOST="root@tadamoo.com"
WEBROOT="/opt/website/html"
DRY="${1:-}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1) Preflight locale"
[ -f "$SRC/index.html" ] || { echo "index.html mancante"; exit 1; }
NASS=$(find "$SRC/assets" -maxdepth 1 -type f ! -name '_*' ! -name 'placeholder*' | wc -l)
SIZE=$(du -sh --exclude=_work "$SRC/assets" | cut -f1)
echo "   index.html: $(stat -c%s "$SRC/index.html" 2>/dev/null || echo '?') byte"
echo "   assets:     $NASS file, $SIZE (esclusi _work/)"

say "2) Rinomino la homepage attuale in index.html.bak"
if [ "$DRY" = "--dry-run" ]; then
  echo "   (dry-run) mv $WEBROOT/index.html → $WEBROOT/index.html.bak"
else
  ssh "$HOST" "set -e
    cd $WEBROOT
    if [ -f index.html ]; then
      if [ -f index.html.bak ]; then                 # già rinominata da un deploy precedente:
        mv -f index.html index.html.prev             # non calpestare il .bak originale (trading)
        echo '   index.html.bak esisteva già → la vecchia va in index.html.prev'
      else
        mv index.html index.html.bak
        echo '   homepage trading rinominata: index.html.bak'
      fi
    fi
  "
fi

say "3) Upload landing (index.html + buy.html + assets/)"
if [ "$DRY" = "--dry-run" ]; then
  echo "   (dry-run) tar assets/ + scp index.html buy.html → $HOST:$WEBROOT/"
else
  # niente rsync in Git-Bash → tar-pipe: UN solo trasferimento invece di 39 connessioni scp.
  # Escludo i file di lavoro (_work/, _gen_state.json, _optimized.json) e i chunk intermedi.
  ssh "$HOST" "mkdir -p $WEBROOT/assets"
  tar -C "$SRC" -czf - \
      --exclude='assets/_work' --exclude='assets/_*.json' --exclude='assets/*_chunk*.mp4' \
      assets \
    | ssh "$HOST" "tar -C $WEBROOT -xzf -"
  scp "$SRC/index.html" "$HOST:$WEBROOT/index.html"
  scp "$SRC/buy.html"   "$HOST:$WEBROOT/buy.html"
  ssh "$HOST" "chmod -R a+r $WEBROOT/assets $WEBROOT/index.html $WEBROOT/buy.html"
fi

say "4) Verifica live"
if [ "$DRY" != "--dry-run" ]; then
  echo -n "   <title> servito: "
  curl -sk --max-time 15 https://tadamoo.com/ | grep -oiE '<title>[^<]*</title>' | head -1
  echo -n "   hero video:      "
  curl -skI --max-time 15 https://tadamoo.com/assets/hero-bg-01.mp4 | head -1
  echo -n "   immagine t2i:    "
  curl -skI --max-time 15 https://tadamoo.com/assets/img-t2i-01.webp | head -1
fi

say "FATTO — rollback: ssh $HOST 'mv $WEBROOT/index.html.bak $WEBROOT/index.html'"
