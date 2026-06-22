"""Downloader pesi da HuggingFace per il wizard GUI (subprocess).

Locale-first: scarica i pesi UNA VOLTA nelle cartelle locali; poi l'inferenza è offline.
L'HF token (env HF_TOKEN) serve solo per i repo gated (es. FLUX.1-dev).

Uso:
  python scripts/download_weights.py --ids flux1-dev,heartcodec
  python scripts/download_weights.py --all
Stampa righe "[weights] <id> ..." parsabili dalla GUI.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.weights_catalog import dest_for               # logica destinazione type/family

CATALOG = Path(__file__).resolve().parents[1] / "data" / "weights_catalog.json"


def log(m: str) -> None:
    print(m, flush=True)


def _ensure_hf() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001
        log("[weights] installo huggingface_hub…")
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"]).returncode
        return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="id separati da virgola")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base-dir", default=None, help="cartella base (override); i tool-dest la ignorano")
    args = ap.parse_args()

    catalog = json.loads(CATALOG.read_text("utf-8")).get("weights", [])
    wanted = [e for e in catalog if args.all or e["id"] in args.ids.split(",")]
    if not wanted:
        log("[weights] nessun peso selezionato."); return 2
    if not _ensure_hf():
        log("[weights] ERRORE: huggingface_hub non installabile."); return 3

    from huggingface_hub import snapshot_download, hf_hub_download
    token = os.environ.get("HF_TOKEN") or None
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    fail = 0
    for e in wanted:
        dest = str(dest_for(e, args.base_dir)); Path(dest).mkdir(parents=True, exist_ok=True)
        gated = " (gated: serve token)" if e.get("gated") else ""
        log(f"[weights] ↓ {e['id']} — {e.get('label', e['repo_id'])} ~{e.get('approx_gb', '?')}GB{gated}")
        try:
            if e.get("kind") == "file":
                hf_hub_download(repo_id=e["repo_id"], filename=e["filename"], local_dir=dest, token=token)
            else:
                snapshot_download(repo_id=e["repo_id"], local_dir=dest, token=token)
            log(f"[weights] ✓ {e['id']} completato → {dest}")
        except Exception as ex:                          # noqa: BLE001
            fail += 1
            msg = str(ex)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                log(f"[weights] ✗ {e['id']}: accesso negato — accetta la licenza su HF e imposta l'HF token.")
            else:
                log(f"[weights] ✗ {e['id']}: {msg[:300]}")
    log(f"[weights] FINE: {len(wanted) - fail}/{len(wanted)} ok.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
