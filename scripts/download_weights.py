"""Downloader pesi per il wizard GUI (subprocess) — CLI sopra il MOTORE UNICO core/downloader.

Locale-first: scarica i pesi UNA VOLTA nelle cartelle locali; poi l'inferenza è offline.
L'HF token (env HF_TOKEN) serve solo per i repo gated (es. FLUX.1-dev, LTX-2.5).

Novità FASE B (weights-unification):
  - fast_downloader multi-connessione con resume (prima hf_hub single-stream che si piantava su xet);
  - dipendenze scaricate PRIMA e in automatico (es. krea_rt → base Wan) — opt-out --no-deps;
  - preflight dei repo GATED (l'errore arriva SUBITO e con l'url della licenza);
  - verifica d'integrità post-download (sha256/struttura: un GGUF troncato fallisce qui);
  - voci `no_remote` rifiutate con messaggio chiaro (prima: KeyError silenziato);
  - progresso a BYTE: righe "[weights] … {id}: {MB}MB (v={MB/s})".

Uso:
  python scripts/download_weights.py --ids flux1-dev,heartcodec [--base-dir DIR] [--no-deps]
  python scripts/download_weights.py --all
Output parsabile dalla GUI: "[weights] ✓ <id> completato → <dest>" / "[weights] ✗ <id>: …".
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.net_tls import use_system_ca  # noqa: E402
use_system_ca(inject=False)            # bundle certifi+Windows: dietro antivirus/proxy TLS (core/net_tls)
from core import downloader as dl  # noqa: E402
from core import weights_catalog as wc  # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="id separati da virgola")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base-dir", default=None, help="cartella base (override); i tool-dest la ignorano")
    ap.add_argument("--no-deps", action="store_true", help="NON espandere le dipendenze")
    ap.add_argument("--accept-license", default="", help="id (virgola) o «all»: conferma di licenza per le voci con license_ack (MagicQuill…); senza, quelle voci NON si scaricano")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    ids = [i for i in args.ids.split(",") if i.strip()]
    if args.all:
        ids = [e["id"] for e in wc.load()]
    entries = ([e for e in wc.load() if e["id"] in set(ids)] if args.no_deps
               else dl.plan_ids(ids))
    if not entries:
        log("[weights] nessun peso selezionato.")
        return 2

    token = dl.resolve_token()
    fail = 0
    accepted = {a.strip() for a in args.accept_license.split(",") if a.strip()}
    for e in entries:
        if e.get("license_ack") and "all" not in accepted and e["id"] not in accepted and not wc.is_present(e, args.base_dir):
            log(f"[weights] ✗ {e['id']}: licenza {e.get('license', '?')} non confermata — il download parte solo dal dialog "
                f"con avviso di licenza (o con --accept-license {e['id']}).")
            fail += 1
            continue
        if not wc.downloadable(e):
            log(f"[weights] ✗ {e['id']}: nessuna fonte remota (no_remote) — "
                f"{e.get('note') or 'usa scan/link o import manuale.'}")
            fail += 1
            continue
        if wc.is_present(e, args.base_dir):
            log(f"[weights] ✓ {e['id']} completato → già presente")
            continue
        gated = " (gated: serve token)" if e.get("gated") else ""
        log(f"[weights] ↓ {e['id']} — {e.get('label', e.get('repo_id', ''))} "
            f"~{e.get('approx_gb', '?')}GB{gated}")
        state = {"bytes": 0, "t0": time.monotonic(), "last": 0.0}

        def _prog(n, fname, _s=state, _id=e["id"]):
            _s["bytes"] += n
            now = time.monotonic()
            if now - _s["last"] >= 2.0:
                _s["last"] = now
                mb = _s["bytes"] / 2**20
                spd = mb / max(now - _s["t0"], 0.001)
                log(f"[weights] … {_id}: {mb:,.0f}MB (v={spd:,.1f}MB/s) {fname[-60:]}")

        try:
            dest = dl.download_entry(e, base_dir=args.base_dir, token=token,
                                     progress=_prog, log=log)
            log(f"[weights] ✓ {e['id']} completato → {dest}")
        except dl.GatedRepoError as ex:
            fail += 1
            log(f"[weights] ✗ {e['id']}: accesso negato — accetta la licenza su "
                f"{ex.license_url} e imposta l'HF token.")
        except dl.IntegrityError as ex:
            fail += 1
            log(f"[weights] ✗ {e['id']}: file corrotto/incompleto ({ex}) — riprova (resume attivo).")
        except Exception as ex:                          # noqa: BLE001
            fail += 1
            log(f"[weights] ✗ {e['id']}: {str(ex)[:300]}")
    log(f"[weights] FINE: {len(entries) - fail}/{len(entries)} ok.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
