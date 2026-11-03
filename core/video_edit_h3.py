"""«Modifica Video AI» con il motore MiniMax H3 (edit con maschera): costruzione dei params del worker — modulo PURO (senza Qt/torch).

Condiviso da GUI (ui/main_window.py::_VideoEditAIWorker) e, in futuro, REST API. Il worker H3 e `workers/minimax_h3_infer.py`
(ramo `h3_edit` -> `workers/h3_edit_run.py`), nel venv di H3 (core.ltx_video_job.MINIMAX_H3_PY).
"""
from __future__ import annotations

import os

from core import minimax_h3_spec as S
from core import video_edit_engines as E

# profili H3 per l'edit: l'edit usa SEMPRE i DiT Ref2VA (foto + video di riferimento + audio); l'utente sceglie il profilo, mai in automatico
H3_PROFILES: tuple[str, ...] = ("int8", "fp8", "nvfp4", "int4")
H3_PROFILE_I18N = {"int8": "INT8 ConvRot (21 GB)", "fp8": "FP8 (21 GB)", "nvfp4": "NVFP4 (12,5 GB, RTX 50xx)", "int4": "INT4 W4A8 (12 GB)"}
DEFAULT_TURBO = "v4"                          # LoRA turbo larryvrh v4 (ha la variante Ref2VA, 8 passi, shift video 12)
DEFAULT_SHORT_EDGE = 480                      # lato corto della tela (Openjutsu: 480p in 6-13 min per 6 s su 16 GB; default 736p)
SHORT_EDGES = (480, 576, 736)


class H3EditError(RuntimeError):
    pass


# ── modello H3 (DiT) per l'edit: i profili ufficiali + i DiT personalizzati che sanno fare Ref2VA (S.CUSTOM_DITS); catalogo pesi per ognuno ──────────────
CUSTOM_MODELS: tuple[str, ...] = tuple(k for k, v in S.CUSTOM_DITS.items() if "ref2va" in v["modes"])
CUSTOM_TE_PROFILE = "nvfp4"                   # i DiT personalizzati non hanno un profilo: l'encoder e quello del profilo nvfp4 (unico presente di default), dichiarato nel log
H3_MODELS: tuple[str, ...] = H3_PROFILES + CUSTOM_MODELS
H3_MODEL_LABELS = {**H3_PROFILE_I18N, **{k: (v["label"] if v.get("official") else f"{v['label']} (finetune, 18+ possibile)") for k, v in S.CUSTOM_DITS.items()}}


def model_dit_id(model: str) -> str:
    """ID di catalogo del DiT per l'edit (Ref2VA) di una voce del menu «Modello H3»."""
    if model in S.CUSTOM_DITS:
        return model
    if model in S.PROFILES:
        return S.PROFILES[model][1]
    raise H3EditError(f"Modello H3 sconosciuto: {model!r} ({' | '.join(H3_MODELS)})")


def model_choices(present=None) -> list[dict]:
    """Voci del menu «Modello H3» per l'edit (TUTTE, anche quelle non sul disco: la GUI le segna e offre il download alla prima selezione).
    [{id, label, kind: profile|custom, dit, installed}]"""
    present = present or E._present
    return [{"id": m, "label": H3_MODEL_LABELS.get(m, m), "kind": "custom" if m in S.CUSTOM_DITS else "profile",
             "dit": model_dit_id(m), "installed": bool(present(model_dit_id(m)))} for m in H3_MODELS]


# ── velocita: LoRA turbo per Ref2VA (scegliere una turbo imposta i SUOI passi e il SUO shift: lo dice la tabella S.TURBO) ──────────────────────────────────
_TURBO_DEFECT_CACHE: dict = {}


def turbo_lora_defect(key: str) -> str:
    """"adaln" se la LoRA turbo Ref2VA di `key` e sul disco, e stata convertita da una versione ComfyUI (metadato `source` con «comfyui») e NON ha gli adattatori AdaLN
    (`transformer_blocks.N.adaln_proj.*`): difetto noto della conversione ComfyUI sui pruned (issue ModelTC/Minimax-H3-Turbo#7: la sorgente tocca l'AdaLN a 2688 canali,
    il pruned ne ha 8, le 51 coppie si perdono), la turbo v4 ne ha 100 e la 4 passi v0.1 zero → distillazione monca. Le turbo lightx2v in formato DIFFUSERS (r4v10, r8v11)
    non hanno coppie AdaLN per costruzione (1,38 GB contro 1,96 della versione Comfy) e NON sono segnate: la resa si giudica dal giro dal vivo. "" se a posto o non installata."""
    import json as _j
    import re as _re
    import struct as _st
    lid = (S.TURBO.get(key, ({}, 0, 0))[0] or {}).get("ref2va")
    if not lid:
        return ""
    try:
        from core import model_paths as mp
        path = mp.p(lid, "")
        stt = os.stat(path)
    except Exception:                                        # noqa: BLE001
        return ""
    sig = (path, stt.st_mtime_ns, stt.st_size)
    if sig not in _TURBO_DEFECT_CACHE:
        try:
            with open(path, "rb") as f:
                n = _st.unpack("<Q", f.read(8))[0]
                hdr = _j.loads(f.read(n)) if 0 < n < 100_000_000 else {}
            keys = [k for k in hdr if k != "__metadata__"]
            src = str((hdr.get("__metadata__") or {}).get("source", "")).lower()
            has = any(_re.search(r"transformer_blocks\.\d+\.adaln_proj", k) for k in keys)
            _TURBO_DEFECT_CACHE[sig] = "" if (has or not keys or "comfyui" not in src) else "adaln"
        except Exception:                                    # noqa: BLE001
            _TURBO_DEFECT_CACHE[sig] = ""
    return _TURBO_DEFECT_CACHE[sig]


def turbo_choices() -> list[dict]:
    """[{key, nfe, shift, ref2va (esiste per Ref2VA), lora}] nell'ordine della tabella. Le turbo senza versione Ref2VA (8 passi fl2v, DMAD) restano in lista
    DISATTIVATE (l'edit lavora in Ref2VA): la GUI mostra il motivo."""
    return [{"key": k, "nfe": nfe, "shift": shift, "ref2va": "ref2va" in modes, "lora": modes.get("ref2va", ""),
             "defect": turbo_lora_defect(k) if "ref2va" in modes else ""}
            for k, (modes, nfe, shift) in S.TURBO.items()]


def turbo_info(key: str, model: str = "") -> tuple[int, float]:
    """(passi, shift video) che il worker usera: col DiT con turbo nei pesi (10Eros-Max) i suoi, altrimenti quelli della LoRA turbo scelta."""
    meta = S.CUSTOM_DITS.get(model or "", {})
    if meta.get("turbo_fused"):
        return int(meta["nfe"]), float(meta["video_shift"])
    if key not in S.TURBO:
        raise H3EditError(f"LoRA turbo sconosciuta: {key!r}")
    return S.TURBO[key][1], S.TURBO[key][2]


# ── LoRA dell'utente per l'edit: architettura dalle CHIAVI dell'header (mai dai metadati); Ref2VA/FL2VA non si distinguono dalle chiavi (stesse chiavi e forme) ──
_H3_LORA_KEY = None
# LoRA di Alissonerdx (VFX Edit, Style Transfer, Motion Transfer «guide latents», Head Swap, LMS): il video sorgente entra come LATENTI ALLINEATI (MiniMaxH3AddGuide,
# frame_idx 0), non come riferimento nativo <Video 1>: il nostro edit (Ref2VA + riferimento video) non ha questa guida → fuori dal menu finche non c'e (DECISIONE RICHIESTA).
_GUIDE_TOKENS = ("vfx_edit", "style_transfer", "motion_transfer", "head_swap", "_lms_")
_FL2VA_ONLY_TOKENS = ("fl2v", "i2v")          # LoRA addestrate per prima/ultima immagine: l'edit lavora in Ref2VA (DiT diverso, adaln diverso)
_LORA_CACHE: dict = {}


LORA_REASONS = {"": "", "missing": "file non trovato", "twin": "esiste gia la versione nativa (-native): usa quella",
                "convert": "non e in formato nativo: convertila dal tab «Pesi» (l'edit usa solo LoRA native)",
                "guide": "richiede la guida di latenti allineata (AddGuide, frame 0) che l'edit H3 di LTX Studio non ha ancora",
                "arch": "non e una LoRA di MiniMax H3 (le chiavi dell'header non hanno i moduli del DiT H3)",
                "fl2va": "solo FL2VA (prima/ultima immagine): l'edit lavora in Ref2VA"}


def lora_edit_compat(path: str, name: str = "") -> tuple[bool, str]:
    """(utilizzabile nell'edit H3, motivo in italiano se no): vedi lora_edit_check (codice del motivo, per la GUI)."""
    ok, code = lora_edit_check(path, name)
    return ok, LORA_REASONS[code]


def lora_edit_check(path: str, name: str = "") -> tuple[bool, str]:
    """(utilizzabile nell'edit H3, codice del motivo: "" | missing | arch | fl2va). 1) architettura: dalle chiavi dell'header deve avere i moduli del DiT H3 (`transformer_blocks.N.attn.to_qkv`
    con pesi LoRA); 2) modalita: dalle chiavi non si distingue (le turbo ref2va e fl2va hanno le stesse chiavi e forme), quindi vale la modalita DICHIARATA nel
    nome del file (fl2v/i2v = solo FL2VA → disattivata col motivo; ref2v/ref2va = verificata; altrimenti ammessa, modalita non dichiarata)."""
    import json as _j
    import re as _re
    import struct as _st
    global _H3_LORA_KEY
    if _H3_LORA_KEY is None:
        _H3_LORA_KEY = _re.compile(r"^transformer_blocks\.\d+\.attn\.to_qkv\.(lora_(A|B|up|down)\.weight|lokr_|diff)")
    try:
        stt = os.stat(path)
    except OSError:
        return False, "missing"
    sig = (path, stt.st_mtime_ns, stt.st_size)
    arch_ok = _LORA_CACHE.get(sig)
    if arch_ok is None:
        try:
            with open(path, "rb") as f:
                n = _st.unpack("<Q", f.read(8))[0]
                keys = [k for k in _j.loads(f.read(n)) if k != "__metadata__"] if 0 < n < 100_000_000 else []
            arch_ok = any(_H3_LORA_KEY.search(k) for k in keys)
        except Exception:                                    # noqa: BLE001 — non e un safetensors leggibile
            arch_ok = False
        _LORA_CACHE[sig] = arch_ok
    if not arch_ok:
        return False, "arch"
    low = (name or os.path.basename(path)).lower()
    try:
        from core import native_convert as _nc
        if not _nc.is_native(path):
            stem, ext = os.path.splitext(path)
            return False, ("twin" if os.path.isfile(stem + "-native" + ext) else "convert")
    except Exception:                                        # noqa: BLE001
        pass
    if any(t in low for t in _GUIDE_TOKENS):
        return False, "guide"
    if _re.search(r"ref2v", low):
        return True, ""
    if any(t in low for t in _FL2VA_ONLY_TOKENS):
        return False, "fl2va"
    return True, ""


def installed_profiles(present=None) -> list[str]:
    """Profili con il DiT Ref2VA su disco (il resto dei pesi lo controlla `video_edit_engines.missing_message`)."""
    present = present or E._present
    ids = {"int8": "mmh3-ref2va-int8", "fp8": "mmh3-ref2va-fp8", "nvfp4": "mmh3-ref2va-nvfp4", "int4": "mmh3-ref2va-int4"}
    return [p for p in H3_PROFILES if present(ids[p])]


def track_dir_of(video: str) -> str:
    return os.path.splitext(video)[0] + "_track"


def build_params(cfg: dict, out: str, *, action: str | None = None, present=None) -> dict:
    """`params` per workers/minimax_h3_infer.py (senza il tracking SAM2, che lo fa il chiamante e riempie `mask_dir`).

    cfg: video, mode (change_background|replace_char), object (cosa tenere / chi sostituire), prompt (il nuovo luogo / note), ref_image
    (foto del luogo, opzionale, o del nuovo personaggio, obbligatoria), h3_profile, short_edge, seed, engine."""
    mode = cfg.get("mode")
    act = action or E.action_of_mode(mode or "")
    if act not in (E.CHANGE_BACKGROUND, E.REPLACE_CHARACTER):
        raise H3EditError(f"Il motore H3 non fa l'azione «{act}».")
    eng = cfg.get("engine") or "h3_edit"
    msg = E.missing_message(eng, present)
    if msg:
        raise H3EditError(msg)
    prof = str(cfg.get("h3_profile") or "")                    # voce del menu «Modello H3»: un profilo ufficiale o un DiT personalizzato
    custom = prof if prof in CUSTOM_MODELS else str(cfg.get("h3_dit") or "")
    if custom:
        if custom not in CUSTOM_MODELS:
            raise H3EditError(f"DiT H3 «{custom}» non adatto all'edit (serve Ref2VA): {' | '.join(CUSTOM_MODELS)}")
        pres = present or E._present
        if not pres(custom):
            raise H3EditError(f"Il DiT «{S.CUSTOM_DITS[custom]['label']}» non e installato: scaricalo dal tab «Pesi» (nessuna scelta automatica).")
        prof = CUSTOM_TE_PROFILE
    elif prof not in installed_profiles(present):
        raise H3EditError("Scegli il profilo H3 (int8/fp8/nvfp4/int4) o un DiT personalizzato di cui hai il DiT Ref2VA installato: nessuna scelta automatica.")
    turbo = str(cfg.get("h3_turbo") or DEFAULT_TURBO)
    if turbo not in S.TURBO or "ref2va" not in S.TURBO[turbo][0]:
        raise H3EditError(f"La LoRA turbo «{turbo}» non esiste per Ref2VA (l'edit lavora in Ref2VA): scegli {' | '.join(t['key'] for t in turbo_choices() if t['ref2va'])}.")
    if turbo_lora_defect(turbo):
        raise H3EditError(f"La LoRA turbo «{turbo}» e sul disco ma senza gli adattatori AdaLN (difetto della conversione ComfyUI sui pruned, ModelTC/Minimax-H3-Turbo#7): "
                          "la distillazione e monca. Usa la turbo v4 o attendi la versione diffusers.")
    te = str(cfg.get("h3_te") or "")
    if te and te not in S.TE_IDS:
        raise H3EditError(f"Encoder testo H3 sconosciuto: {te!r} ({' | '.join(S.TE_IDS)})")
    pres_te = present or E._present
    te_id = te or (S.PROFILES[prof][2] if prof in S.PROFILES else "")
    if te_id and not pres_te(te_id):
        raise H3EditError(f"L'encoder testo «{te_id}» non e installato: scegli un altro nel menu «Encoder testo» (es. nvfp4, uguale per tutti i confronti) o scaricalo dal tab «Pesi».")
    loras = []
    for lo in cfg.get("loras") or []:
        nm = str((lo or {}).get("path") or (lo or {}).get("name") or "")
        if not nm:
            continue
        full = nm if os.path.isabs(nm) else os.path.join(_lora_root(), nm)
        ok, why = lora_edit_compat(full, nm)
        if not ok:
            raise H3EditError(f"LoRA «{os.path.basename(nm)}» non usabile nell'edit H3: {why}")
        loras.append({"name": nm, "strength": float((lo or {}).get("strength", 1.0))})
    if cfg.get("h3_veda"):
        pres = present or E._present
        if not pres(S.VEDA_R2VA):
            raise H3EditError("Attenzione sparsa Veda: il predittore (mmh3-veda-r2va) non e installato: scaricalo dal tab «Pesi».")
    lv = str(cfg.get("h3_lightvae") or "")
    if lv and lv not in S.LIGHTVAE:
        raise H3EditError(f"LightVAE sconosciuto: {lv!r} ({' | '.join(S.LIGHTVAE)})")
    if lv:
        pres = present or E._present
        miss = [i for i in S.LIGHTVAE[lv] if not pres(i)]
        if miss:
            raise H3EditError("LightVAE: pesi mancanti " + ", ".join(miss) + ": scaricali dal tab «Pesi».")
    video = str(cfg.get("video") or "")
    if not os.path.isfile(video):
        raise H3EditError(f"Video sorgente non trovato: {video}")
    keep_text = str(cfg.get("object") or "").strip()
    if not keep_text:
        raise H3EditError("Indica la persona (a parole, es. «person»): serve a SAM2 per la maschera." if act == E.CHANGE_BACKGROUND
                          else "Indica il personaggio da sostituire (a parole, es. «person»).")
    pic = str(cfg.get("ref_image") or "").strip()
    if act == E.REPLACE_CHARACTER and (not pic or not os.path.isfile(pic)):
        raise H3EditError("Sostituisci personaggio: scegli la foto del nuovo personaggio.")
    if act == E.CHANGE_BACKGROUND and not pic and not str(cfg.get("prompt") or "").strip():
        raise H3EditError("Cambia sfondo: descrivi il nuovo luogo a parole o scegli la foto del luogo.")
    ep = E.ENGINES[eng].params
    e = {"action": act, "video": video, "mask_dir": os.path.join(track_dir_of(video), "masks"),
         "short_edge": int(cfg.get("short_edge") or DEFAULT_SHORT_EDGE), "keep_text": keep_text,
         "prompt_text": str(cfg.get("prompt") or "").strip(), "picture": pic if pic and os.path.isfile(pic) else "",
         "picture_desc": str(cfg.get("picture_desc") or ""), "keep_audio": bool(cfg.get("keep_audio", True)),
         "clip_s": float(ep["clip_s"]), "overlap": int(ep["overlap_frames"]), "grow_px": int(ep["grow_px"]),
         "feather_px": int(ep["feather_px"]), "headroom": float(ep["headroom"]), "hold_frames": int(ep["pose_hold_frames"]),
         "person_grow_px": int(ep["person_grow_px"]), "replace_expand": float(ep.get("replace_expand", 0.0))}
    out_p = {"arch": "minimax_h3", "h3_edit": e, "h3_profile": prof, "h3_turbo": turbo, "h3_video_shift": float(ep["video_shift"]),
             "seed": int(cfg.get("seed") or 42), "output": out, **E.engine_params(act, eng)}
    # opzioni di velocita / modello: presenti SOLO se l'utente le ha scelte (default invariati: senza, i params sono quelli di sempre)
    if custom:
        out_p["h3_dit"] = custom
    if te:
        out_p["h3_te"] = te
    if cfg.get("h3_veda"):
        out_p["h3_veda"] = True
    if lv:
        out_p["h3_lightvae"] = lv
    if loras:
        out_p["loras"] = loras
    return out_p


def _lora_root() -> str:
    from core import model_paths as mp
    return os.path.join(str(mp.models_root()), "LoRA")
