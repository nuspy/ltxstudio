"""Installer comune di LTX Studio (uguale per tutte le edizioni; parametrizzato dal
manifest dell'edizione). Verifica i prerequisiti runtime, mostra un RIEPILOGO dei
download con le DIMENSIONI e scarica i modelli BASE dell'edizione dal registro unico
(gli altri al primo uso, via wizard «Setup modelli» / tab «Pesi modelli»).

Il backend è NATIVO: niente ComfyUI, niente custom nodes (rimossi 17/08 — erano il
relitto del design pre-migrazione; l'unica integrazione ComfyUI rimasta è lo shim
opzionale `core/comfy_compat`, installabile dal tab «Importa modelli» per chi VUOLE
usare LTX-2.3 dentro un proprio ComfyUI).

Uso:
  python installer/install.py --dry-run            # mostra il piano + dimensioni
  python installer/install.py --edition premium    # forza un'edizione (override del manifest)
  python installer/install.py                       # installa (chiede conferma)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from installer import ui                    # noqa: E402
from installer.steps import prereqs         # noqa: E402


def _edition_info(forced: str | None) -> dict:
    """Manifest dell'edizione: dal binario bakeato (core.features) o forzato."""
    if forced:
        # override esplicito (test/dev): deriva i permessi dall'id edizione.
        return {"edition": forced, "allows_adult": forced in ("premium", "studio"),
                "allows_studio": forced == "studio"}
    try:
        from core import features
        return features.manifest()
    except Exception:
        return {"edition": "free", "allows_adult": False}


def build_plan(allows_adult: bool) -> dict:
    """MODELLI dal REGISTRO UNICO (weights_catalog v2): `install_role: base` + capability
    dell'edizione (manifest bakeato via core.features) decidono cosa scarica l'installer;
    tutto il resto è on-demand (wizard primo avvio / tab Pesi)."""
    from core import weights_catalog as wc
    base_models = [
        {"id": e["id"], "label": e.get("label", e["id"]),
         "size_mb": int(float(e.get("approx_gb", 0)) * 1024), "_entry": e}
        for e in wc.base_weights_for_edition(allows_adult=allows_adult)]
    on_demand = [
        {"id": e["id"], "label": e.get("label", e["id"]),
         "size_mb": int(float(e.get("approx_gb", 0)) * 1024)}
        for e in wc.load()
        if wc.downloadable(e) and wc.install_role(e) != "base"
        and wc.wanted(e, check_present=False, allows_adult=allows_adult)]
    total = sum(m.get("size_mb", 0) for m in base_models)
    return {"base_models": base_models, "on_demand": on_demand, "total_mb": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    man = _edition_info(args.edition)
    edition = man.get("edition", "free")
    allows_adult = bool(man.get("allows_adult", False))
    print(f"== Installer LTX Studio — edizione: {edition} (adult={allows_adult}) ==\n")

    # 1) prerequisiti runtime (python / driver NVIDIA / torch / ffmpeg)
    pr = prereqs.check_all()
    ui.print_prereqs(pr)

    # 2) piano + dimensioni
    plan = build_plan(allows_adult)
    ui.print_plan(plan)

    if args.dry_run:
        print("\n[dry-run] nessun download eseguito.")
        return 0
    if not ui.confirm(plan):
        print("Annullato."); return 1

    # 3) modelli base dell'edizione (gli on-demand li scaricano wizard/tab al primo uso)
    _download_base_models(plan["base_models"])
    print("\nInstallazione completata.")
    return 0


def _download_base_models(models: list) -> None:
    """Download REALE dei pesi base via il motore unico. Resume attivo; un peso alla volta
    (connessione metered); gated → messaggio con l'url della licenza; il fallito non blocca
    gli altri (si può rilanciare: idempotente)."""
    import time
    from core import downloader as dl
    from core import weights_catalog as wc
    from core import weights_tooltips
    # CATENA: i pesi base tirano le loro dipendenze (deps-first) — es. il checkpoint LTX tira
    # Gemma TE + upscaler; l'utente non deve conoscere la composizione interna.
    entries = dl.plan_ids([m["id"] for m in models])
    fail = 0
    for e in entries:
        if not wc.downloadable(e):
            continue
        m = {"id": e["id"], "label": e.get("label", e["id"]),
             "size_mb": int(float(e.get("approx_gb", 0)) * 1024)}
        if wc.is_present(e):
            print(f"  [modello base] ✓ {m['label']} — già presente")
            continue
        print(f"  [modello base] ↓ {weights_tooltips.short_line(e)}")
        state = {"b": 0, "t0": time.monotonic(), "last": 0.0}

        def _prog(n, fname, _s=state):
            _s["b"] += n
            now = time.monotonic()
            if now - _s["last"] >= 3.0:
                _s["last"] = now
                mb = _s["b"] / 2**20
                print(f"    … {mb:,.0f} MB ({mb / max(now - _s['t0'], 0.001):,.1f} MB/s)", flush=True)

        try:
            dest = dl.download_entry(e, progress=_prog)
            print(f"  [modello base] ✓ {m['label']} → {dest}")
        except dl.GatedRepoError as ex:
            fail += 1
            print(f"  [modello base] ✗ {m['label']}: repo GATED — accetta la licenza su "
                  f"{ex.license_url}, imposta HF_TOKEN e rilancia l'installer.")
        except Exception as ex:                      # noqa: BLE001
            fail += 1
            print(f"  [modello base] ✗ {m['label']}: {str(ex)[:200]} (rilancia: resume attivo)")
    if fail:
        print(f"  [modello base] {fail} download NON completati — rilancia l'installer o usa il tab Pesi.")


if __name__ == "__main__":
    sys.exit(main())
