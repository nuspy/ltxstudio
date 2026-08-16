"""Installer comune di LTX Studio (uguale per tutte le edizioni; parametrizzato dal
manifest dell'edizione). Verifica i prerequisiti, individua ComfyUI (lo CHIEDE se
non standard), mostra un RIEPILOGO dei download con le DIMENSIONI, scarica i modelli
BASE dell'edizione (gli altri al primo uso) e installa nodi/patch. Fallback mirror
tadamoo.com solo per nodi + comfy_compat (mai i modelli).

Uso:
  python installer/install.py --dry-run            # mostra il piano + dimensioni
  python installer/install.py --edition premium    # forza un'edizione (override del manifest)
  python installer/install.py                       # installa (chiede conferma)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from installer import mirror, ui            # noqa: E402
from installer.steps import prereqs         # noqa: E402

import json                                 # noqa: E402

SPEC = json.loads((Path(__file__).parent / "install_spec.json").read_text("utf-8"))


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


def _wants(item: dict, edition: str, allows_adult: bool) -> bool:
    eds = item.get("editions")
    if eds is not None and edition not in eds:
        return False
    if item.get("sensitivity") == "adult" and not allows_adult:
        return False
    return True


def build_plan(edition: str, allows_adult: bool) -> dict:
    """Nodi/deps da install_spec; MODELLI dal REGISTRO UNICO (weights_catalog v2).
    Prima i modelli stavano in una lista PARALLELA di install_spec (id diversi dal catalogo,
    gating a `editions[]` invece che a capability) e il download era uno stub: due manifesti
    non sincronizzati e zero pesi scaricati. Ora: `install_role: base` + capability
    dell'edizione (manifest bakeato via core.features) decidono cosa scarica l'installer;
    tutto il resto è on-demand (primo uso / tab Pesi)."""
    nodes = [n for n in SPEC["custom_nodes"] if _wants(n, edition, allows_adult)]
    deps = list(SPEC.get("python_deps", []))
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
    total = sum(n.get("size_mb", 0) for n in nodes) + sum(d.get("size_mb", 0) for d in deps) \
        + sum(m.get("size_mb", 0) for m in base_models)
    return {"nodes": nodes, "deps": deps, "base_models": base_models,
            "on_demand": on_demand, "total_mb": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--comfy-root", default=None)
    args = ap.parse_args(argv)

    man = _edition_info(args.edition)
    edition = man.get("edition", "free")
    allows_adult = bool(man.get("allows_adult", False))
    print(f"== Installer LTX Studio — edizione: {edition} (adult={allows_adult}) ==\n")

    # 1) prerequisiti (+ ComfyUI: chiede la cartella se non standard)
    pr = prereqs.check_all(args.comfy_root)
    ui.print_prereqs(pr)

    # 2) piano + dimensioni
    plan = build_plan(edition, allows_adult)
    ui.print_plan(plan)

    if args.dry_run:
        print("\n[dry-run] nessun download eseguito.")
        return 0
    if not ui.confirm(plan):
        print("Annullato."); return 1

    # 3) install (idempotente; riusa gli script esistenti)
    _install(plan, pr)
    print("\nInstallazione completata.")
    return 0


def _install(plan: dict, pr: dict) -> None:
    # patch compatibilità ComfyUI (shim) — riusa core.comfy_compat
    try:
        from core import comfy_compat
        st = comfy_compat.install()
        print(f"  comfy_compat: installato={st.get('installed')} v{st.get('installed_version')}")
    except Exception as e:                       # noqa: BLE001
        print(f"  [warn] comfy_compat: {e} → provo il mirror")
        mirror.fetch_node(SPEC, "comfy_compat", pr.get("comfy_root"))

    # custom nodes (origine → fallback mirror)
    for n in plan["nodes"]:
        if n.get("bundled"):
            continue
        ok = _git_clone(n, pr.get("comfy_root"))
        if not ok and n.get("mirror"):
            mirror.fetch_node(SPEC, n["id"], pr.get("comfy_root"))

    # deps python (SageAttention…)
    for d in plan["deps"]:
        subprocess.run([sys.executable, "-m", "pip", "install", d["pip"]])

    # modelli base dell'edizione (gli on-demand li scaricano i downloader al primo uso)
    _download_base_models(plan["base_models"])


def _git_clone(node: dict, comfy_root: str | None) -> bool:
    if not comfy_root:
        return False
    dest = Path(comfy_root) / "custom_nodes" / node["id"]
    if dest.exists():
        return True
    if not shutil.which("git"):
        return False
    r = subprocess.run(["git", "clone", "--depth", "1", node["repo"], str(dest)])
    return r.returncode == 0


def _download_base_models(models: list) -> None:
    """Download REALE dei pesi base via il motore unico (prima era uno stub che stampava).
    Resume attivo; un peso alla volta (connessione metered); gated → messaggio con l'url
    della licenza; il fallito non blocca gli altri (si può rilanciare: idempotente)."""
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
