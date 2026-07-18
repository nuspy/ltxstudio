"""Cast automatico per Lucky Director — replica in AUTOMATICO lo step "asset" di Cinema Studio.

Lucky Director = Cinema Studio con tutti i passaggi auto-accettati. In Cinema Studio l'utente, nell'
asset_manager, crea i personaggi e genera/carica le loro foto di riferimento; poi ogni shot che cita
`@Nome` usa quelle foto per la coerenza (PuLID/Kontext o Krea 2 Edit). In Lucky Director quello step
mancava: il copione mette i `@Nome`, ma nessuno creava gli asset né le foto → `resolve_refs(@Nome)` non
trovava nulla → volti diversi ad ogni scena.

`build_cast(pid)` colma il buco: estrae i `@Nome` unici dal breakdown (scene+shot), crea un asset character
per ciascuno e genera UNA foto ritratto di riferimento (t2i, una volta), salvata come `image` dell'asset.
Idempotente (non rigenera se l'asset ha già una foto). Da qui `resolve_refs` trova i riferimenti e la
coerenza volto (già propagata in `gen_service.gen_image`) si aggancia.
"""
from __future__ import annotations

import json

from core.film_studio import db, gen_service
from core.film_studio.gen_service import _REF_RE


def collect_names(pid: int) -> list[str]:
    """`@Nome` unici (prima grafia vista) da tutte le descrizioni di scene e shot del progetto."""
    seen: dict[str, str] = {}
    for sc in db.query("scene", "project_id=?", (pid,)):
        texts = [sc.get("description") or ""]
        for sh in db.query("shot", "scene_id=?", (sc["id"],)):
            texts.append(sh.get("description") or "")
        for t in texts:
            for m in _REF_RE.finditer(t):
                seen.setdefault(m.group(1).lower(), m.group(1))
    return list(seen.values())


def extract_appearances(script: str, names: list[str]) -> dict[str, str]:
    """1 call LLM → {Nome: descrizione fisica in una riga} per ritratti fedeli. Vuoto in caso di errore."""
    if not names or not script:
        return {}
    try:
        from core import llm_service as L
        sys = ("You are a casting director. For each character NAME, give ONE short line describing a "
               "photorealistic portrait: apparent age, hair, build, ethnicity, distinctive features. "
               "Output ONLY a JSON object {\"Name\": \"description\"}. No prose.")
        usr = "NAMES: " + ", ".join(names) + "\n\nSCRIPT (excerpt):\n" + script[:4000]
        out = L.complete(sys, usr, max_tokens=700) or ""
        s = out[out.find("{"): out.rfind("}") + 1]
        d = json.loads(s) if s else {}
        # normalizza le chiavi (case-insensitive) sui nomi richiesti
        low = {k.lower(): v for k, v in d.items() if isinstance(v, str)}
        return {n: low.get(n.lower(), "") for n in names if low.get(n.lower())}
    except Exception as ex:                                       # noqa: BLE001
        print(f"[cast_auto] extract_appearances fallito: {ex}", flush=True)
        return {}


def _asset_has_image(aid: int) -> bool:
    return bool(db.query("image", "owner_kind='asset' AND owner_id=?", (aid,)))


def build_cast(pid: int, *, img_arch: str = "flux", model: str = "", width: int = 768, height: int = 768,
               progress=None) -> dict:
    """Per ogni `@Nome` del progetto: asset character (se manca) + UNA foto ritratto di riferimento (t2i).
    Idempotente. Ritorna {Nome: path|'exists'}. Il ritratto è pura t2i (è la SORGENTE dei riferimenti);
    la coerenza vera avviene poi negli shot via gen_service.gen_image(coherence_engine=…)."""
    def log(m):
        print(f"[cast_auto] {m}", flush=True)
        if progress:
            try:
                progress("cast", 0, 0, m, 0.02)
            except Exception:                                     # noqa: BLE001
                pass

    names = collect_names(pid)
    if not names:
        log("nessun @Nome nel copione → niente cast (film senza personaggi ricorrenti)")
        return {}
    log(f"personaggi trovati: {', '.join(names)}")
    rows = db.query("script", "project_id=?", (pid,), order="version")
    script = rows[-1]["text"] if rows else ""
    appear = extract_appearances(script, names)
    # CRITICO: libera Gemma dalla VRAM prima dei ritratti Krea2 — altrimenti l'LLM (~24GB con
    # device_map=auto nel processo GUI) va in contesa col subprocess immagine → crash "non generata".
    try:
        from core import llm_service as _L
        _L.unload()
    except Exception:                                             # noqa: BLE001
        pass

    outdir = db.project_dir(pid) / "cast"
    outdir.mkdir(parents=True, exist_ok=True)
    made: dict[str, str] = {}
    for i, name in enumerate(names):
        assets = db.find_assets(name_like=name)
        asset = next((a for a in assets if (a.get("name") or "").lower() == name.lower()), None)
        if asset and _asset_has_image(asset["id"]):
            made[name] = "exists"; continue                       # già ha una foto → idempotente
        desc = appear.get(name) or f"a person named {name}"
        if not asset:
            aid = db.insert("asset", type="character", name=name, description=desc,
                            fields="{}", tags="", scope="project")
        else:
            aid = asset["id"]
        out = str(outdir / f"{name}.png")
        prompt = (f"cinematic photorealistic headshot portrait of {desc}, neutral expression, "
                  f"soft studio lighting, plain background, sharp focus")
        try:
            gen_service.gen_image(prompt, out, width=width, height=height, arch=img_arch, model=model,
                                  structure=False, seed=1000 + i)
            db.insert("image", owner_kind="asset", owner_id=aid, angle="front", repr="front",
                      file=out, seed=1000 + i, prompt=prompt, version=1)
            made[name] = out
            log(f"ritratto {name} → {out}")
        except Exception as ex:                                   # noqa: BLE001
            log(f"ritratto {name} FALLITO: {ex}")
    return made
