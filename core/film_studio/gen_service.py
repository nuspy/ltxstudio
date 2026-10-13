"""Generation Service dell'AI Film Studio — facciata sul backend NATIVO (zero ComfyUI).

Espone le primitive che servono allo storyboard/scenografia, risolvendo i due percorsi di coerenza:
- Rapido: immagini di riferimento dell'asset + PuLID (volto) + multi-ref.
- LoRA: carica le LoRA degli @Nome (trigger) con peso; keyframe come conditioning aggiuntivo.

Immagini → core/native_driver "image"; video → worker ltx2 (riusa _NativeLtx2Worker tramite subprocess
del worker, qui via native? no: il video nativo gira col worker ltx2_infer come negli altri moduli).
@Reference: (project, "@Nome#Variante") → AssetVariation → {ref images | Lora+trigger}.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

from core import native_driver
from core.film_studio import db

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
# NB: nessuna costante GGUF di default. Il GGUF si usa SOLO se scelto dall'utente (passa da
# core/ltx_video_job): forzarlo costava ~500 s/step (dequant per-forward) e OOM col modello residente.
NEG = "blurry, low quality, deformed, distorted face, extra limbs, watermark, text"

_REF_RE = re.compile(r"@([A-Za-z0-9_]+)(?:#([A-Za-z0-9_]+))?")


# ─────────────── prompt → JSON strutturato (schema PromptBuilder), condiviso ───────────────
def _apply_prompt_json(prompt: str, negative: str, loras: list | None, out_path: str,
                       *, kind: str, structure: bool) -> tuple[str, str]:
    """Converte il prompt nello schema JSON e lo ricompone (trigger LoRA preservati). Salva il JSON accanto
    all'output per riapertura/modifica. No-op se disattivato (LTX_PROMPT_JSON=0) o structure=False.
    Restituisce (prompt_ricomposto, negative_unito)."""
    from core import prompt_struct
    if not structure or not prompt_struct.enabled():
        return prompt, negative
    triggers = [l.get("trigger", "") for l in (loras or []) if l.get("trigger")]
    new_prompt, extra_neg, pj = prompt_struct.structure_and_compose(prompt, kind=kind, triggers=triggers)
    if extra_neg:
        negative = (negative + ", " + extra_neg).strip(", ") if negative else extra_neg
    try:                                                     # sidecar JSON (non fatale)
        with open(out_path + ".prompt.json", "w", encoding="utf-8") as _f:
            json.dump(pj, _f, ensure_ascii=False, indent=2)
    except Exception:                                        # noqa: BLE001
        pass
    return new_prompt, negative


# ─────────────── @Reference resolver ───────────────
def resolve_refs(text: str, path_mode: str = "fast", flavor: str = "image") -> dict:
    """Estrae gli @Nome[#Variante] dal testo e li risolve in {ref_images, loras, trigger_map, clean_text}.
    path_mode: 'fast' (immagini+PuLID) | 'lora' (LoRA+trigger) | 'mixed' (per-asset).
    flavor: 'image' (keyframe) | 'video' (shot) → sceglie la LoRA del flavor giusto (B2: prima si caricava
    sempre flavor='image' anche per gli shot video → LoRA video mai usate).
    #Variante ora è ONORATA (B1): immagini+LoRA della VARIAZIONE (es. @Marco#insanguinato), con fallback
    all'asset se la variante non ha propri asset."""
    refs, ref_images, loras, trigger_map = [], [], [], {}
    # Separati per TIPO di asset: sul video i soggetti (actor/prop) sono gli slot MSR, mentre
    # luoghi e architetture forniscono l'AMBIENTE — che sulla 2.5 è lo slot senza il quale il
    # modello prende lo sfondo dai ritratti dei personaggi e la scena descritta sparisce.
    subject_images: list[str] = []
    place_images: list[str] = []
    for m in _REF_RE.finditer(text or ""):
        name, variant = m.group(1), m.group(2)
        # più recente per primo: se due film hanno un personaggio omonimo (es. Lucky Director rilanciato),
        # usa quello del film CORRENTE (l'asset creato più di recente da cast_auto), non uno vecchio.
        assets = sorted(db.find_assets(name_like=name), key=lambda a: a.get("created", 0) or 0, reverse=True)
        asset = next((a for a in assets if a["name"].lower() == name.lower()), assets[0] if assets else None)
        if not asset:
            continue
        var = None                                           # risolvi la variante citata (#Variante)
        if variant:
            vrows = db.query("asset_variation", "asset_id=? AND name=?", (asset["id"], variant))
            var = vrows[0] if vrows else None
        refs.append((asset, variant))
        mode = path_mode
        if path_mode == "mixed":
            mode = (json.loads(asset.get("fields") or "{}").get("path_mode")) or "fast"
        if mode == "lora":
            lr = []
            if var:                                          # LoRA della VARIAZIONE (flavor giusto), poi fallback asset
                lr = db.query("lora", "variation_id=? AND flavor=?", (var["id"], flavor), order="version")
            if not lr:
                lr = db.query("lora", "asset_id=? AND flavor=?", (asset["id"], flavor), order="version")
            if lr:
                _w = float(lr[-1].get("weight") or 0.9)      # peso dalla LoRA (col opz.), default 0.9
                loras.append({"file": lr[-1]["file"], "trigger": lr[-1]["trigger"], "weight": _w})
                trigger_map[name] = lr[-1]["trigger"]
                continue
        # Rapido (o LoRA assente): immagini di riferimento della VARIAZIONE se citata, altrimenti dell'asset
        imgs = []
        if var:
            imgs = db.query("image", "owner_kind='variation' AND owner_id=?", (var["id"],))
        if not imgs:
            imgs = db.query("image", "owner_kind='asset' AND owner_id=?", (asset["id"],))
        _files = [i["file"] for i in imgs[:4] if i.get("file")]
        ref_images += _files
        if (asset.get("type") or "").lower() in ("location", "architecture"):
            place_images += _files
        else:
            subject_images += _files
    clean = _REF_RE.sub(lambda m: trigger_map.get(m.group(1), m.group(1)), text or "")
    return {"refs": refs, "ref_images": ref_images, "loras": loras,
            "trigger_map": trigger_map, "clean_text": clean,
            "subject_images": subject_images, "place_images": place_images}


def style_suffix(cfg: dict) -> str:
    """Style Bible del progetto → coda iniettata in OGNI prompt (logline/palette/look/registi). Condivisa
    tra storyboard (UI) e auto_run (Lucky Director). Vuoto se non impostata."""
    sb = (cfg or {}).get("style_bible") or {}
    bits = []
    if sb.get("look"):
        bits.append(sb["look"])
    if sb.get("palette"):
        bits.append("color palette: " + sb["palette"])
    if sb.get("directors"):
        bits.append("in the visual style of " + sb["directors"])
    if sb.get("logline"):
        bits.append("overall tone: " + sb["logline"])
    return (", " + ", ".join(bits)) if bits else ""


def compose_shot_prompt(sh: dict, state: dict, guard: dict, *, path_mode: str = "fast",
                        flavor: str = "image") -> tuple[str, dict]:
    """Costruisce il prompt di generazione di uno shot (identico a storyboard._gen): @ref risolti +
    camera/luce + tag di continuity + 'required' + Style Bible. Ritorna (prompt, refs). CONDIVISO
    tra Cinema Studio e Lucky Director. La conversione in JSON avviene poi dentro gen_image/gen_shot_video."""
    from core.film_studio import continuity as ct
    required = (guard or {}).get("required", "")
    refs = resolve_refs(sh.get("description", ""), path_mode, flavor=flavor)
    cam = (sh.get("camera") or "").strip()
    prompt = (refs["clean_text"] + ". " + (cam + ". " if cam else "")
              + ct.to_prompt(state) + ((", " + required) if required else "")
              + style_suffix(guard))
    return prompt, refs


# ─────────────── immagini (keyframe / asset sheet) ───────────────
def _gen_image_qwen21(prompt, out_png, negative, width, height, seed, dit, loras, refs) -> str:
    """Keyframe con Qwen-Image 2.1 (worker dedicato). Riferimenti dei personaggi della scena, in ordine e senza
    doppioni, come «image 1…10»: la coerenza viene dal modello stesso (niente PuLID, che è solo FLUX)."""
    from core.qwen21_spec import DITS, MAX_REFS, TURBO_STEPS, DEFAULT_STEPS, DEFAULT_GUIDANCE
    if dit not in DITS:
        raise RuntimeError("Film Studio: scegli il modello Qwen-Image 2.1 nella barra dei modelli del progetto")
    meta = DITS[dit]
    uniq = list(dict.fromkeys(refs))[:MAX_REFS]
    params = {"arch": "qwen_image21", "qwen21_dit": dit, "prompt": prompt, "negative": negative,
              "width": width, "height": height, "seed": seed, "output": out_png, "ref_images": uniq,
              "steps": TURBO_STEPS if meta.get("turbo") else int(meta.get("steps", DEFAULT_STEPS)),
              "guidance": 1.0 if meta.get("turbo") else float(meta.get("cfg", DEFAULT_GUIDANCE))}
    if loras:
        params["loras"] = loras
    res = native_driver.run_and_wait("image", params, timeout=1800)
    p = (res.get("output") or [out_png]); p = p[0] if isinstance(p, list) else p
    if p and os.path.exists(p) and p != out_png:
        import shutil; shutil.copyfile(p, out_png)
    if res.get("status") != "success" and not os.path.exists(out_png):
        raise RuntimeError(res.get("error", "gen_image qwen_image21 fallita"))
    return out_png



def gen_image(prompt: str, out_png: str, *, width: int = 1024, height: int = 1024, seed: int = 7,
              face_ref: str = "", ref_image: str = "", ref_images: list | None = None,
              loras: list | None = None, arch: str = "flux", model: str = "", negative: str = "",
              structure: bool = True, coherence_engine: str = "flux",
              grounding_px: int = 1024, krea2_variant: str = "turbo",
              face_strength: float = 0.85, qwen21_dit: str = "") -> str:
    """Genera UNA immagine col motore nativo. face_ref→PuLID; ref_image→Kontext; loras opzionali.
    arch/model: architettura+checkpoint scelti nel progetto (B5: prima hardcoded flux). negative: estende il
    negative di default (continuity elementi vietati). structure: converte il prompt in JSON strutturato.

    coherence_engine: motore per la COERENZA VOLTO/personaggio quando ci sono riferimenti (switch utente,
    NON sostituisce FLUX):
      • "flux" (default) → PuLID (face_ref) / Kontext (ref_images), come prima;
      • "krea2_edit" → Krea 2 Edit in-context: usa i riferimenti dei personaggi DI QUESTA scena
        (face_ref+ref_images, già risolti per-scena da resolve_refs → ogni scena porta l'attore giusto)
        come immagini di riferimento; il prompt dello shot è l'istruzione. Fino a 2 personaggi (multi-ref);
        con 3+ riferimenti ricade su FLUX per quella scena (log);
      • "qwen_image21" → Qwen-Image 2.1: i riferimenti della scena (fino a 10) entrano come «image N».
    arch "qwen_image21": keyframe con Qwen-Image 2.1; il DiT è `qwen21_dit` (scelto nel progetto, mai d'ufficio)."""
    prompt, negative = _apply_prompt_json(prompt, negative, loras, out_png, kind="image", structure=structure)
    neg = NEG + (", " + negative if negative else "")

    # ── Qwen-Image 2.1: keyframe (arch) e/o coerenza personaggi (riferimenti della scena ≤10) ────────
    if str(arch).lower() == "qwen_image21" or str(coherence_engine).lower() == "qwen_image21":
        return _gen_image_qwen21(prompt, out_png, neg, width, height, seed, qwen21_dit, loras,
                                 [r for r in ([face_ref] + list(ref_images or []) + [ref_image]) if r])

    # ── Coerenza personaggio via Krea 2 Edit (in-context) ─────────────────────────────────────────
    _krefs = []
    for r in ([face_ref] + list(ref_images or []) + [ref_image]):
        if r and r not in _krefs:
            _krefs.append(r)
    if str(coherence_engine).lower() == "krea2_edit" and _krefs:
        if len(_krefs) > 2:
            print(f"[film] krea2_edit: {len(_krefs)} riferimenti in scena > 2 → FLUX per questo shot", flush=True)
        else:
            kp = {"arch": "krea2_edit", "krea2_variant": krea2_variant, "prompt": prompt, "negative": neg,
                  "grounding_px": int(grounding_px), "seed": seed, "output": out_png,
                  "steps": 8 if krea2_variant != "raw" else 20,
                  "guidance": 0.0 if krea2_variant != "raw" else 2.0}
            if len(_krefs) == 1:
                kp["image_path"] = _krefs[0]
            else:
                kp["images"] = _krefs[:2]
            res = native_driver.run_and_wait("image", kp, timeout=900)
            p = (res.get("output") or [out_png]); p = p[0] if isinstance(p, list) else p
            if p and os.path.exists(p) and p != out_png:
                import shutil; shutil.copyfile(p, out_png)
            if res.get("status") != "success" and not os.path.exists(out_png):
                raise RuntimeError(res.get("error", "gen_image krea2_edit fallita"))
            return out_png

    params = {"arch": arch, "prompt": prompt, "negative": neg, "width": width, "height": height,
              "steps": 22, "guidance": 3.5, "seed": seed, "output": out_png}
    if str(arch).lower() == "krea2":
        # Krea 2 Turbo è DISTILLATO (guidance-distilled): vuole ~8 step e guidance 0.0 (niente CFG). Con i
        # valori FLUX (22 step / guidance 3.5) la CFG sul negative genera un reticolo colorato sul volto
        # (bug KAIRA — ritratti cast rovinati). Il ramo krea2_edit già usava 8/0.0; qui allineo il t2i puro.
        params["steps"] = 8 if krea2_variant != "raw" else 52
        params["guidance"] = 0.0 if krea2_variant != "raw" else 3.5
    if model:                                                # checkpoint esplicito (altrimenti default del server)
        params["model"] = model
    if face_ref:
        # forza PuLID regolabile: a 0.85 l'identità è fedelissima ma si porta dietro TUTTO il volto
        # di partenza, età compresa, e il prompt non riesce più a ringiovanire o invecchiare il
        # personaggio. Abbassandola (0.5-0.7) restano i tratti e il prompt torna a contare.
        params["pulid_face"] = face_ref
        params["pulid_strength"] = max(0.0, min(1.0, float(face_strength)))
    # Kontext/multi-ref SOLO se non c'è un volto PuLID (attori→PuLID; props/luoghi/architetture→Kontext ref).
    _refs = [r for r in ([ref_image] + list(ref_images or [])) if r]
    if _refs and not face_ref:
        params["arch"] = "kontext"; params["image_path"] = _refs[0]
        if len(_refs) > 1:
            params["ref_images"] = _refs                     # multi-reference (composizione senza incollaggio)
    if loras:
        params["loras"] = loras
    # STICKY: un film genera N keyframe di fila (uno per shot). Senza, il server rimette il
    # transformer in GPU a OGNI keyframe (~5s di warmup a testa: su 18 shot sono ~1,5 minuti buttati).
    # Residente per tutto il batch: ~24s → ~5s a immagine. La VRAM viene comunque sfrattata prima dei
    # video (free_for("video") in ltx_video_job), quindi non ruba spazio al transformer 22B. (13/07)
    params["placement_mode"] = "sticky"
    res = native_driver.run_and_wait("image", params, timeout=900)
    p = (res.get("output") or [out_png])
    p = p[0] if isinstance(p, list) else p
    if p and os.path.exists(p) and p != out_png:
        import shutil
        shutil.copyfile(p, out_png)
    if res.get("status") != "success" and not os.path.exists(out_png):
        raise RuntimeError(res.get("error", "gen_image fallita"))
    return out_png


LTX2_SERVER = os.environ.get("LTX2_SERVER", "http://127.0.0.1:8190")   # worker video persistente (opzionale)


def _wait_free_vram(min_mb: int = 24000, max_s: int = 30) -> int:
    """Attende il rilascio EFFETTIVO della VRAM (asincrono rispetto a /free e unload): se un worker
    22B parte con la VRAM ancora occupata, le sue allocazioni finiscono in SHARED memory (WDDM) e
    ci restano anche a VRAM svuotata → GPU all'1%, clip eterne. Poll via nvidia-smi (MAI torch.cuda
    nel processo app). La soglia è clampata alla VRAM fisica (dinamica per PC): su una scheda
    piccola una soglia fissa non si raggiunge mai e l'attesa degenererebbe in uno sleep pieno."""
    free_mb = 0
    try:
        from core import gpu_mem as _gm
        total_mb = _gm.total_vram_mb()
        if total_mb:
            min_mb = min(min_mb, max(total_mb - 2048, 1024))
        for _ in range(max_s):
            free_mb = _gm.free_vram_mb()
            if free_mb >= min_mb:
                break
            time.sleep(1)
        print(f"[film] VRAM libera pre-worker: {free_mb}MB (soglia {min_mb})", flush=True)
    except Exception:                                        # noqa: BLE001
        pass
    return free_mb


def _ltx2_server_up() -> bool:
    """True se il server video persistente risponde (modello già caricato in VRAM)."""
    import json as _j
    import urllib.request as _u
    try:
        with _u.urlopen(LTX2_SERVER + "/system_stats", timeout=1.5) as r:
            return bool(_j.loads(r.read()).get("up", False))
    except Exception:                                       # noqa: BLE001
        return False


def _submit_persistent(params: dict, out_mp4: str, timeout: int = 1800) -> str:
    """Invia il job al server persistente e attende l'output (poll /history). Il modello resta caricato."""
    import json as _j
    import time as _t
    import urllib.request as _u
    data = _j.dumps({"params": params}).encode()
    req = _u.Request(LTX2_SERVER + "/submit", data=data, headers={"Content-Type": "application/json"})
    jid = _j.loads(_u.urlopen(req, timeout=30).read())["prompt_id"]
    t0 = _t.time()
    while _t.time() - t0 < timeout:
        with _u.urlopen(LTX2_SERVER + f"/history/{jid}", timeout=10) as r:
            h = _j.loads(r.read())
        st = h.get("status")
        if st == "success":
            out = (h.get("output") or [out_mp4])[0]
            if out and os.path.exists(out) and out != out_mp4:
                import shutil
                shutil.copyfile(out, out_mp4)
            return out_mp4
        if st == "error":
            raise RuntimeError(str(h.get("error", "ltx2 server error"))[:500])
        _t.sleep(1.0)
    raise TimeoutError("ltx2 server: timeout")


# ─────────────── video (shot) ───────────────
def msr_background_for(keyframes: list | None, out_mp4: str, width: int, height: int,
                       *, log=print) -> str:
    """Sfondo per la variante pseudo-video (LTX 2.3), che senza non parte.

    Serve il KEYFRAME dello shot: è il contesto dell'inquadratura, ha già il formato del video, e
    nel ciclo Cinema Studio esiste quasi sempre (i keyframe si generano prima dei video).

    Senza keyframe si torna a stringa vuota, cioè MSR NON si attiva. Qui c'era un ripiego con una
    tela neutra: PROVATO E SCARTATO. La guida viene seguita con `control_strength=1.0` e uno
    sfondo piatto non porta informazione, così il modello riproduce la tela — clip annacquate,
    sfocate, coi soggetti che si sgranano in punti (stessa "poltiglia" che il worker documenta
    per le guide di formato sbagliato, ltx2_infer.py:1915). Il confronto A/B è netto: con il
    keyframe il risultato è ottimo, con la tela neutra è peggiore che senza MSR del tutto.
    """
    for kf in (keyframes or []):
        p = kf.get("path") if isinstance(kf, dict) else str(kf or "")
        if p and os.path.exists(p):
            log(f"[msr] sfondo = keyframe dello shot ({os.path.basename(p)})")
            return p
    log("[msr] nessun keyframe da usare come sfondo → riferimenti multipli NON attivati "
        "(su LTX 2.3 una guida senza sfondo reale peggiora la clip). Genera prima il keyframe, "
        "indica msr_background, oppure usa la 2.5 che non richiede sfondo.")
    return ""


def _aspect_warning(ref_images: list, width: int, height: int, log=print) -> None:
    """Avvisa se i riferimenti hanno un formato molto diverso dallo shot.

    `fit_on_white` scala il soggetto DENTRO il frame e lo centra su bianco: un ritratto verticale
    dentro un frame orizzontale lascia grosse bande bianche, che entrano nella guida e finiscono
    nella clip. I riferimenti vanno generati nello stesso formato del video.
    """
    if not width or not height:
        return
    target = width / height
    for p in ref_images or []:
        try:
            from PIL import Image
            with Image.open(p) as im:
                r = im.width / im.height
        except Exception:                                    # noqa: BLE001
            continue
        if r and (r / target > 1.5 or target / r > 1.5):
            log(f"[msr] ⚠ {os.path.basename(p)} è {im.width}x{im.height} ma lo shot è "
                f"{width}x{height}: il soggetto verrà centrato fra bande bianche che entrano "
                f"nella guida. Genera i riferimenti nel formato dello shot.")


def msr_references(ref_images: list | None, release: str, *, background: str = "",
                   strength: float = 1.0) -> list[dict]:
    """Ritratti/luoghi della scena → riferimenti MSR, solo se la release li regge davvero.

    I due meccanismi hanno prerequisiti diversi (vedi workers/ltx_msr.py):
      · 2.5 → slot embedding: bastano le immagini, nessuno sfondo richiesto;
      · 2.3 → pseudo-video IC-LoRA: lo SFONDO è obbligatorio, senza il worker esce con errore.
    Perciò sulla 2.3 l'attivazione automatica avviene solo se uno sfondo c'è. La strategia vera la
    determina il worker aprendo il file della LoRA: qui si decide solo se PROVARCI.
    """
    from core import ltx_release as lrel
    from workers import ltx_msr

    paths, seen = [], set()
    for p in (ref_images or []):
        p = str(p or "")
        if p and p not in seen and os.path.exists(p):
            seen.add(p)
            paths.append(p)
    if not paths:
        return []
    rel = lrel.get(release)
    if rel.caps.get("multiref") != lrel.CAP_YES or not os.path.exists(rel.msr_lora or ""):
        return []
    if not background:
        # Sfondo obbligatorio su LTX, come nel pannello della GUI: senza, la 2.3 non parte proprio
        # e la 2.5 produce clip in cui l'ambiente è quello dei ritratti invece della scena chiesta.
        # Meglio non attivare i riferimenti che consegnare una clip sbagliata in silenzio.
        return []
    # Quanti soggetti REGGE la release: la 2.3 ne prende 4 (il 5º slot è lo sfondo) e
    # `build_pseudo_video` tronca in silenzio; la 2.5 ha 5 slot pieni. Il taglio va fatto qui,
    # dove si può dirlo, invece di far sparire un personaggio senza un avviso.
    cap = ltx_msr.MAX_SUBJECTS if rel.id == lrel.DEFAULT_RELEASE else ltx_msr.MAX_REFERENCES
    if len(paths) > cap:
        print(f"[msr] {len(paths)} riferimenti ma LTX {rel.id} ne regge {cap}: "
              f"scartati {[os.path.basename(p) for p in paths[cap:]]}", flush=True)
    # `strength` per-riferimento: sullo slot embedding (2.5) è QUESTA a decidere quanto il
    # riferimento pesa contro il prompt — `references_to_inputs` la legge da ogni voce e il
    # default globale non la sovrascrive. Fissarla a 1.0 rendeva `msr_strength` inefficace, e i
    # riferimenti finivano per riprodursi tali e quali ignorando la scena descritta.
    out = [{"path": p, "kind": "image", "strength": float(strength)} for p in paths[:cap]]

    # SFONDO come riferimento, esattamente come fa il pannello della GUI (get_references: lo
    # accoda con role='background'). Serve a ENTRAMBE le release, per motivi diversi:
    #   · 2.3 → riempie i fotogrammi liberi della pseudo-sequenza;
    #   · 2.5 → `references_to_inputs` filtra solo per kind, quindi lo sfondo diventa uno slot a
    #     pieno titolo: è il riferimento che porta l'AMBIENTE. Senza, il modello l'ambiente se lo
    #     prende dai ritratti dei soggetti e la scena descritta nel prompt viene ignorata.
    if background and os.path.exists(background):
        out.append({"path": background, "kind": "image", "role": "background",
                    "strength": float(strength)})
    return out


def gen_shot_video(prompt: str, out_mp4: str, *, width: int = 768, height: int = 512, fps: int = 25,
                   duration_s: float = 4.0, seed: int = 7, keyframes: list | None = None,
                   loras: list | None = None, audio_path: str = "", model_gguf: str = "",
                   negative: str = "", structure: bool = True, video_arch: str = "ltx",
                   lingbot_variant: str = "dense", run_refiner: bool = False,
                   ltx_release: str | None = None, model: str = "", references=None,
                   ref_images: list | None = None, native_audio: bool = False,
                   msr_strength: float = 1.0, msr_background: str = "",
                   msr_spacing: int = 1, steps: int = 8, guidance: float = 3.0,
                   audio_lowpass: int | None = None) -> str:
    """Genera il video di uno shot col worker nativo. keyframes=[{path,frame,strength}].
    model_gguf: transformer scelto nel progetto (B5), altrimenti il distilled Q4 di default.
    video_arch: 'ltx' (default) o 'lingbot' (2ª architettura, worker/venv dedicati).
    structure: converte il prompt in JSON strutturato e lo ricompone (vedi gen_image).

    ltx_release/model: release LTX (2.3 default | 2.5) e transformer esplicito.
    references: riferimenti multipli (MSR) per la coerenza di cast e luoghi fra shot diversi —
    sulla 2.5 entrano come slot embedding a frame negativi, sulla 2.3 come pseudo-video IC-LoRA
    (che pretende anche `msr_background`). La LoRA MSR la risolve `apply_ltx_release`."""
    prompt, negative = _apply_prompt_json(prompt, negative, loras, out_mp4, kind="video", structure=structure)
    nf = max(9, int(round(duration_s * fps)) // 8 * 8 + 1)
    neg = NEG + (", " + negative if negative else "")
    video_arch = (video_arch or "ltx").lower()
    params = {"prompt": prompt, "negative": neg, "width": width,
              # steps/guidance NON più cablati: la GUI li prende dallo slider, qui erano fissi e il
              # ciclo Cinema Studio non poteva alzarli. (Sull'AUDIO però non incidono: misurato a
              # 8/20/30 passi, rumore di fondo -88,8 / -80,3 / -89,2 dB, nessun andamento. Il timbro
              # metallico veniva dal BWE, non dal denoise — vedi `audio_lowpass`.)
              "height": height, "num_frames": nf, "fps": fps,
              "steps": int(steps), "guidance": float(guidance),
              "seed": seed, "output": out_mp4, "arch": video_arch}
    if video_arch == "lingbot":
        params["lingbot_variant"] = lingbot_variant
        params["steps"] = 40                                  # default ufficiale LingBot
        if run_refiner:
            params["run_refiner"] = True
    # #158: GGUF/offload solo se SCELTI nel progetto (B5). In auto decide il worker: fp8 residente
    # se entra in VRAM (4s/step), GGUF+cpu come fallback per GPU piccole. Prima si forzava SEMPRE
    # il Q4 col dequant per-forward → ~470s/step su ogni shot del film.
    if model_gguf and video_arch != "lingbot":
        params["gguf_transformer"] = model_gguf
        params["offload"] = "cpu"
    # libera i server delle ALTRE categorie (es. FLUX 22GB dei keyframe) prima del video 22B:
    # senza, il transformer LTX spillava in shared-VRAM (clip da 21+ min nel film di test).
    try:
        from core import mem_coordinator as _mc
        _mc.free_for("video", timeout=20)
    except Exception:                                        # noqa: BLE001
        pass
    # e sfratta Gemma (device_map='auto' l'ha messa in VRAM al copione e nessuno la scaricava):
    # il prompt JSON di questo shot è già stato calcolato sopra; al prossimo shot si ricarica
    # (VRAM di nuovo libera a worker uscito) o va in CPU da sola se stretta.
    try:
        from core import lyrics_llm as _lllm
        _lllm.unload()
    except Exception:                                        # noqa: BLE001
        pass
    if keyframes:
        params["keyframes"] = keyframes
    if loras:
        params["loras"] = loras
    if audio_path:
        params["audio_path"] = audio_path
    if native_audio:
        # Senza questo flag il worker BUTTA la traccia che LTX-2 produce comunque, perché non
        # condizionata sarebbe rumore. Con il flag restano suoni e parlato ricavati dal prompt.
        params["native_audio"] = True
    if audio_lowpass is not None:
        # Taglio della banda che il vocoder BWE della 2.5 sintetizza sopra i 16 kHz interni: None
        # lascia decidere il worker (8000 Hz sulla 2.5, niente sulla 2.3), 0 disattiva il filtro.
        params["audio_lowpass"] = int(audio_lowpass)
    # ── Release LTX ────────────────────────────────────────────────────────────────────────────
    # Non dichiarata → si deduce dal transformer scelto nel progetto, come fa la finestra principale
    # (main_window: detect_release sul checkpoint). Così basta scegliere il modello 2.5 nel progetto
    # perché Cinema Studio usi la 2.5, senza una seconda impostazione da tenere allineata a mano.
    if video_arch == "ltx" and not ltx_release:
        from core import ltx_release as _lr
        ltx_release = _lr.detect_release(model or model_gguf or "") or None

    # ── Riferimenti multipli (MSR) ─────────────────────────────────────────────────────────────
    # Sulla 2.5 i ritratti del cast presenti nella scena entrano come slot embedding a frame
    # negativi: è ciò che tiene lo stesso volto/luogo coerente da uno shot all'altro. `references`
    # esplicite hanno la precedenza; altrimenti si derivano dai ref_images della scena.
    # ── Percorso A: LoRA del personaggio → INFERENZA DIRETTA ───────────────────────────────────
    # Con la LoRA video il personaggio sta nei PESI del transformer: non serve né un keyframe che
    # lo mostri né lo slot MSR che lo descriva. Anzi, entrambi sono dannosi — il keyframe a piena
    # forza congela la clip e i riferimenti MSR competono con la scena. Quindi t2v diretto.
    _has_video_lora = bool(loras)
    if _has_video_lora and (references or ref_images):
        print(f"[film] LoRA del personaggio presenti ({len(loras)}): inferenza diretta, "
              f"i riferimenti MSR non servono", flush=True)
        references, ref_images = None, None

    if (references or ref_images) and video_arch == "ltx":
        from core import ltx_release as _lr
        _rel_id = ltx_release or _lr.DEFAULT_RELEASE
        # La 2.3 (pseudo-video IC-LoRA) pretende uno sfondo: lo si ricava dallo shot stesso invece
        # di far fallire il job. Vale sia per i riferimenti derivati sia per quelli passati a mano
        # (a meno che tra questi ce ne sia già uno con role='background'). La 2.5 non ne ha bisogno.
        _has_bg = any(isinstance(r, dict) and r.get("role") == "background"
                      for r in (references or []))
        # Lo sfondo serve a entrambe le release (vedi msr_references): sulla 2.3 riempie la
        # pseudo-sequenza, sulla 2.5 è lo slot che porta l'ambiente. Il pannello della GUI lo
        # rende obbligatorio su LTX senza distinguere la release, e ha ragione.
        if not msr_background and not _has_bg:
            msr_background = msr_background_for(keyframes, out_mp4, width, height)
        if not references:
            references = msr_references(ref_images, _rel_id, background=msr_background,
                                        strength=msr_strength)
            if references:
                _aspect_warning([r["path"] for r in references], width, height)
                print(f"[film] MSR: {len(references)} riferimenti di scena → coerenza cast/luoghi "
                      f"(LTX {_rel_id})", flush=True)

    # Valida le componenti su disco e risolve la LoRA Multiple-Subject-Reference della release:
    # senza, il worker morirebbe a metà job con un errore poco leggibile.
    if ltx_release or model or references:
        from api.handlers._common import apply_ltx_release
        apply_ltx_release(params, ltx_release, model, references, msr_strength, msr_background,
                          msr_spacing)

    # ── Ancoraggio del primo fotogramma (2.5, slot embedding) ──────────────────────────────────
    # I riferimenti stanno a frame NEGATIVI e vengono concatenati senza clamp
    # (ltx_msr.install_slot_embedding): se nessun conditioning occupa il frame 0, affiorano nei
    # primi ~10 fotogrammi come sagome in dissolvenza sopra la scena. Sulla 2.3 il problema non
    # si vede perché la guida viene ritagliata (LTXVCropGuides); qui quel taglio non c'è.
    # Con un keyframe a frame 0 la clip parte pulita — verificato confrontando le due varianti —
    # quindi in sua assenza si usa lo SFONDO come ancora: è già l'immagine della scena.
    # NIENTE ancoraggio automatico dello sfondo sui keyframe. Provato e RIMOSSO: la stessa
    # immagine a frame 0 e all'ultimo frame con strength 1.0 fa interpolare il modello fra due
    # fotogrammi identici e la clip esce FERMA (misurato: YDIF medio 0.34, cioè una foto animata).
    # Stabilita' e immobilita' non sono la stessa cosa. La contaminazione dei riferimenti va
    # risolta a monte, nel worker, non tenendo ferma la scena a forza.
    # chunking SOLO oltre i 20s (indicazione utente 12/07: sotto, LTX-2 genera in passata unica —
    # niente giunzioni e niente reload per-chunk). Prima la soglia era 150 frame (~6s): ogni shot
    # cinematografico da 15-18s finiva spezzato in 3-4 chunk senza necessità.
    _max_single = int(20 * fps)
    if nf > _max_single:
        params["chunk_frames"] = _max_single - (_max_single - 1) % 8   # multiplo 8+1 compatibile LTX
    # Worker VIDEO PERSISTENTE (server :8190, modello già caricato) se attivo → niente reload di LTX ~22B ad
    # ogni shot (era il collo di bottiglia di «Genera Tutti»). Il server persistente serve SOLO il modello
    # LTX-2 → ogni altra architettura (LingBot, krea_rt→Self-Forcing, self_forcing) DEVE andare al subprocess
    # dedicato via run_clip/_ARCH_RUNNER: instradarla qui produrrebbe clip col modello SBAGLIATO (LTX-2).
    # ⚠ Il server persistente è avviato SEMPRE col venv 2.3 (core/ltx2_service.py:279-288): un job
    # con release diversa finirebbe sul PACCHETTO 2.3, che non sa caricare un text encoder da file
    # singolo → «No files matching pattern 'tokenizer.model'». Release ≠ default → subprocess.
    from core import ltx_release as _lrel
    _rel_default = params.get("ltx_release", _lrel.DEFAULT_RELEASE) == _lrel.DEFAULT_RELEASE
    if video_arch == "ltx" and _rel_default and _ltx2_server_up():
        # PRIMO job (server senza pipeline in cache): il SUO caricamento sceglie fp8-vs-GGUF in base
        # alla VRAM libera in quell'istante → aspetta il rilascio effettivo come per il subprocess
        # (visto live: shot_4 servito col subprocess morente = GGUF + 40GB alloc su 32). Dai job
        # successivi il server tiene la VRAM di proposito e non si aspetta nulla.
        try:
            import urllib.request as _ur
            _st = json.loads(_ur.urlopen(f"{LTX2_SERVER}/system_stats", timeout=5).read())
            if int(_st.get("cached_pipelines", 0)) == 0:
                _wait_free_vram()
            else:
                # pipe calda: la VRAM è legittimamente del server → si aspetta solo il rilascio
                # EFFETTIVO dei server immagini appena liberati (free_for sopra risponde prima
                # che il driver restituisca la VRAM — WDDM), non l'intero fabbisogno.
                _wait_free_vram(min_mb=8000, max_s=15)
        except Exception:                                    # noqa: BLE001
            pass
        return _submit_persistent(params, out_mp4)
    # ramo subprocess: stessa attesa (le allocazioni WDDM finite in shared NON migrano più).
    _wait_free_vram()
    # Lancio CENTRALIZZATO (core/ltx_video_job): identico a videoclip/LoRA-video, così la policy del
    # video (#158) sta in un posto solo e non può più restare indietro in un percorso soltanto.
    # free_vram=False: la VRAM è già stata liberata qui sopra (free_for + _wait_free_vram).
    from core import ltx_video_job
    ltx_video_job.run_clip(params, free_vram=False)
    return out_mp4
