"""Filtro dei modelli PER ARCHITETTURA, guidato dall'header del file (non dal nome).

Prima i menu del tab Immagini/Video decidevano l'architettura con sottostringhe del nome
(«flux_2» dentro «PornFlux_20», «krea» dentro i Krea 2, il catalogo esterno che etichetta come LTX un
file di Qwen21): un FLUX.1 compariva fra i FLUX.2, i Krea 2 fra gli UNET FLUX, un Qwen fra i video LTX,
mentre un FLUX.2 Klein con un nome qualsiasi spariva dal menu FLUX.2.

Ordine di affidabilità di `classify()`:
 1. detector dell'header (`core.model_detect`: modelspec/chiavi/config, dunque il CONTENUTO del file);
 2. cartella (Krea2, Qwen21, MiniMax…) per i file convertiti «-native» che l'header non rende riconoscibili;
 3. nome del file (ultima spiaggia, solo se 1 e 2 non dicono nulla).
Non sceglie e non sostituisce nessun modello: dice solo a quale famiglia appartiene un file.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

UNKNOWN = "unknown"
_ROOT_SUBDIRS = ("diffusion_models", "Checkpoints", "checkpoints")

# cartella (segmento esatto, minuscolo) → famiglia. Solo per ciò che l'header non dice.
_FOLDER_FAMILY = {
    "flux": "flux1", "z-image": "zimage", "zimage": "zimage", "ernie": "ernie",
    "krea2": "krea2", "qwen21": "qwen_image", "qwen": "qwen_image",
    "ltx": "ltxv", "ltxvideo": "ltxv", "ltx-2": "ltxv",
    "minimax": "minimax_h3", "minimax-h3": "minimax_h3",
    "wan": "wan", "wan13b": "wan",
}

# parole del nome → famiglia (ultima spiaggia)
_NAME_FAMILY = (
    (re.compile(r"flux_2(?!\d)|flux2|flux 2(?!\d)|klein"), "flux2"),
    (re.compile(r"flux|getphat|fuxcapacity"), "flux1"),
    (re.compile(r"z-image|z_image|zimage"), "zimage"),
    (re.compile(r"ernie"), "ernie"),
    (re.compile(r"qwen-image|qwen_image|qwen21"), "qwen_image"),
    (re.compile(r"krea-?2"), "krea2"),
    (re.compile(r"minimax|_h3_|_h3\.|-h3-"), "minimax_h3"),
    (re.compile(r"ltx"), "ltxv"),
)

# segnali di RUOLO nel nome (l'header FLUX.1 non distingue Kontext/Redux dal dev: stesse chiavi)
_NOT_T2I = re.compile(r"fill|kontext|redux|klein")


def is_flux2_name(name: str) -> bool:
    """FLUX.2 dal solo nome, senza il falso positivo «PornFlux_20» (flux_2 seguito da una cifra)."""
    return bool(re.search(r"flux_2(?!\d)|flux2|flux 2(?!\d)", (name or "").lower()))


def is_flux2(name: str) -> bool:
    """UNICA definizione di «questo file è FLUX.2 (Dev o Klein)»: lo dice l'header se il file si trova su disco;
    altrimenti (nome non risolvibile, es. grafo ComfyUI) il nome, senza il falso positivo «flux_2» in «Flux_20».
    Sostituisce le copie sparse (gen_pipeline, inpaint_pipeline, frame1_tab, restore_tab, model_selection, img_router)."""
    c = classify(name)
    return c["family"] == "flux2" if c["source"] != "none" else is_flux2_name(name)


def models_root() -> Path:
    from core import offline_inventory
    return Path(offline_inventory.MODELS_ROOT)


def resolve(rel: str) -> str | None:
    """Path assoluto di un modello dal nome relativo all'inventario (None se non è un file noto)."""
    if not rel:
        return None
    if os.path.isfile(rel):
        return rel
    rel_os = rel.replace("/", os.sep).replace("\\", os.sep)
    for sub in _ROOT_SUBDIRS:
        p = os.path.join(str(models_root()), sub, rel_os)
        if os.path.isfile(p):
            return p
    return None


def _from_folder(rel: str) -> str:
    parts = [s.lower() for s in re.split(r"[\\/]", rel or "")[:-1]]
    for seg in reversed(parts):
        if seg in _FOLDER_FAMILY:
            return _FOLDER_FAMILY[seg]
    return UNKNOWN


def _from_name(rel: str) -> str:
    n = os.path.basename((rel or "").replace("\\", "/")).lower()
    for rx, fam in _NAME_FAMILY:
        if rx.search(n):
            return fam
    return UNKNOWN


_MEMO: dict = {}


def classify(rel: str) -> dict:
    """{family, variant, role, source} del modello `rel` (nome relativo all'inventario o path assoluto).
    source ∈ header | folder | name | none. Il risultato è memoizzato per (path, mtime, size)."""
    p = resolve(rel)
    sig = None
    if p:
        try:
            st = os.stat(p)
            sig = (p, st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
    key = (rel, sig)
    if key in _MEMO:
        return _MEMO[key]
    out = {"family": UNKNOWN, "variant": "", "role": "", "source": "none"}
    if p and not ("svdq" in os.path.basename(p).lower() or "nunchaku" in p.lower()):
        try:
            from core import model_detect
            det = model_detect.get(p) or {}
        except Exception:                                     # noqa: BLE001
            det = {}
        fam = det.get("family") or UNKNOWN
        if fam != UNKNOWN:
            out = {"family": fam, "variant": det.get("variant") or "", "role": det.get("role") or "",
                   "source": "header" if det.get("source") in ("metadata", "keys", "config") else "folder"}
    if p and ("svdq" in os.path.basename(p).lower() or "nunchaku" in p.lower()):
        # Nunchaku/SVDQuant (pesi quantizzati .qweight): l'header NON ha le chiavi FLUX standard, quindi il detector
        # ripiega sulla cartella («Flux» → flux1) anche per un Klein. La famiglia la dice il nome; il formato è
        # marcato `packed` perché nessuna pipeline FLUX di questo backend lo carica come transformer.
        n = os.path.basename(p).lower()
        out = {"family": "flux2" if (is_flux2_name(n) or "klein" in n or "flux.2" in n) else "flux1",
               "variant": "klein" if "klein" in n else "", "role": "", "source": "name", "packed": "nunchaku"}
    if out["family"] == UNKNOWN:
        fam = _from_folder(rel)
        if fam != UNKNOWN:
            out = {"family": fam, "variant": "", "role": "", "source": "folder"}
    if out["family"] == UNKNOWN:
        fam = _from_name(rel)
        if fam != UNKNOWN:
            out = {"family": fam, "variant": "", "role": "", "source": "name"}
    if len(_MEMO) > 4096:
        _MEMO.clear()
    _MEMO[key] = out
    return out


def clear_memo() -> None:
    _MEMO.clear()


def family(rel: str) -> str:
    return classify(rel)["family"]


def _flags(rel: str) -> set:
    n = os.path.basename((rel or "").replace("\\", "/")).lower()
    return set(_NOT_T2I.findall(n))


# ── immagini ─────────────────────────────────────────────────────────────────────────────────────────────
ARCH_FAMILIES = {"FLUX": {"flux1", "flux2"}, "Z-Image": {"zimage"}, "Ernie": {"ernie"}}


def is_t2i_for_arch(arch_label: str, rel: str) -> bool:
    """`rel` è un modello di generazione (t2i) dell'arch del selettore «Genera immagine»."""
    c = classify(rel)
    if c["family"] not in ARCH_FAMILIES.get(arch_label, set()):
        return False
    if arch_label == "FLUX":                     # i non-generativi vivono nei pannelli di modifica
        if c["role"] in ("inpaint", "adapter") or c["variant"] in ("fill", "kontext", "redux", "klein"):
            return False
        if _flags(rel):
            return False
    return True


def is_flux_family(rel: str, family_name: str | None = None) -> bool:
    c = classify(rel)
    return c["family"] in ({family_name} if family_name else {"flux1", "flux2"})


def is_kontext(rel: str) -> bool:
    """FLUX.1 Kontext: header FLUX.1 (stesse chiavi del dev) + «kontext» (variante rilevata o nome)."""
    c = classify(rel)
    return c["family"] == "flux1" and (c["variant"] == "kontext" or "kontext" in _flags(rel))


def is_flux2_edit_pool(rel: str) -> bool:
    """Modelli per il motore «FLUX.2» di Integra/Colore: arch `flux2_kontext` = Flux2KleinPipeline → SOLO Klein
    (header FLUX.2 senza guidance_in; il Dev dà errore chiaro nel backend, Nunchaku non ha loader per Klein)."""
    return inpaint_kind(rel) == "klein"


def is_flux2_refine_pool(rel: str) -> bool:
    """Rifinitura dell'upscale: FLUX.2 (dev e Klein) in formato caricabile (no Fill, no Nunchaku)."""
    c = classify(rel)
    return c["family"] == "flux2" and "fill" not in _flags(rel) and not c.get("packed")


# ── video ────────────────────────────────────────────────────────────────────────────────────────────────
def is_ltx_transformer(rel: str, catalog_arch: str = "") -> bool:
    """Transformer LTX per il menu video «LTX»: l'header/la cartella decidono; il catalogo esterno e il nome
    solo se il file non è riconoscibile. Gli ausiliari (text projection, aux, vae, LoRA…) e i DiT di altre
    architetture non sono checkpoint LTX."""
    from core.ltx_release import _NOT_A_CKPT
    low = os.path.basename((rel or "").replace("\\", "/")).lower()
    if any(x in low for x in _NOT_A_CKPT):
        return False
    c = classify(rel)
    if c["source"] in ("header", "folder") and c["family"] != UNKNOWN:
        return c["family"] == "ltxv"
    if (catalog_arch or "").upper() == "LTX" or "ltx" in low or "eros" in low:
        return True
    return c["family"] == "ltxv"


# ── componenti (text encoder / VAE) per architettura immagine ────────────────────────────────────────────
# Fonte: core/model_detect.compatible_components (flux1: clip_l+t5xxl, zimage: qwen3_4b, ernie: ministral-3-3b,
# flux2: Mistral-3 / Qwen3 → core/flux2_encoder), core/qwen21_spec (Qwen3-VL 8B), Krea 2 (Qwen3-VL 4B) e la
# struttura delle cartelle in E:\AI\Models. Confronto su nomi normalizzati (senza «_ - .»): «qwen_3_4b» = «qwen3_4b».
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_TE_RULES = {
    "flux1": (("t5xxl", "clipl", "vitl14"), ("umt5", "hidream")),
    "zimage": (("qwen34b",), ()),
    "ernie": (("ministral",), ()),
    "krea2": (("qwen3vl4b",), ()),
    "qwen_image": (("qwen3vl8b",), ("pei2i", "pet2i")),
}
_VAE_RULES = {                         # nomi base esatti (normalizzati)
    "flux1": {"aesafetensors"},
    "zimage": {"aesafetensors"},
    "flux2": {"flux2vaesafetensors"},
    "ernie": {"flux2vaesafetensors"},
    "krea2": set(),                    # Krea 2: VAE (Qwen-Image) dentro lo snapshot
    "qwen_image": {"qwenimage21vaebf16safetensors"},
}


def image_arch_key(arch_label: str, model: str = "") -> str:
    """Chiave di famiglia per i componenti: FLUX → flux1/flux2 secondo il file scelto ('' se Auto/segnaposto)."""
    base = {"Krea 2": "krea2", "Qwen-Image 2.1": "qwen_image", "Z-Image": "zimage", "Ernie": "ernie"}
    if arch_label in base:
        return base[arch_label]
    if arch_label == "FLUX":
        fam = family(model) if model and resolve(model) else UNKNOWN
        return fam if fam in ("flux1", "flux2") else ""
    return ""


def compatible_text_encoder(arch_key: str, name: str, model: str = "") -> bool:
    """`name` (voce dell'inventario text_encoders) serve all'architettura. arch_key '' (FLUX con Auto) = flux1 o flux2."""
    if not arch_key:
        return compatible_text_encoder("flux1", name) or compatible_text_encoder("flux2", name)
    if arch_key == "flux2":
        from core.flux2_encoder import INDIM_KEYWORDS, is_compatible_encoder, txt_in_indim
        mp = resolve(model) if model else None
        if mp and txt_in_indim(mp) is not None:
            return is_compatible_encoder(mp, name)
        n = _norm(name)                                   # file ignoto/GGUF: i soli encoder FLUX.2 noti (Mistral-3, Qwen3 4B/8B)
        return any(_norm(k) in n for kws in INDIM_KEYWORDS.values() for k in kws)
    inc, exc = _TE_RULES.get(arch_key, ((), ()))
    n = _norm(name)
    return any(i in n for i in inc) and not any(x in n for x in exc)


def compatible_vae(arch_key: str, name: str) -> bool:
    if not arch_key:
        return compatible_vae("flux1", name) or compatible_vae("flux2", name)
    base = _norm(os.path.basename((name or "").replace("\\", "/")))
    return base in _VAE_RULES.get(arch_key, set())


# ── inpaint: cosa sa FARE davvero il backend per ogni file ───────────────────────────────────────────────
# Dal codice (ui/main_window._start_native_inpaint + inference_server/backends/image_flux): arch «inpaint» =
# FluxFillPipeline (Fill, 384 canali in ingresso) o FluxInpaintPipeline (FLUX.1 base, 64 canali); «flux2_inpaint» =
# Flux2KleinInpaintPipeline (SOLO Klein: 128 canali e NESSUN guidance_in; FLUX.2 Dev ha guidance_in → nessuna pipeline
# inpaint, KeyError reale il 16/10). I checkpoint (CheckpointLoaderSimple) restano su ComfyUI (dismesso); i pesi
# Nunchaku/SVDQuant non hanno loader.
def inpaint_kind(rel: str) -> str | None:
    """'fill' | 'base' | 'klein' se il backend sa fare inpaint con questo file, altrimenti None. Dall'HEADER (canali in
    ingresso, vector_in/guidance_in); per i GGUF (header non standard) solo se il nome dice «fill»."""
    p = resolve(rel)
    if not p:
        return None
    low = os.path.basename(p).lower()
    if "svdq" in low or "nunchaku" in p.lower():
        return None
    if low.endswith(".gguf"):
        return "fill" if "fill" in low else None
    if "kontext" in low or "redux" in low:
        return None                                  # stesse chiavi del dev: lo dice solo il nome
    try:
        from core import model_detect
        tens, _meta = model_detect.read_header(p)
    except Exception:                                # noqa: BLE001
        return None
    if not tens or any(k.endswith("qweight") for k in tens):
        return None
    shp = None
    for k, v in tens.items():
        if k.endswith("img_in.weight") or k.endswith("x_embedder.weight"):
            shp = v.get("shape")
            break
    if not shp or len(shp) < 2 or not any("double_blocks" in k for k in tens):
        return None
    ch = int(shp[1])
    has_vec = any("vector_in" in k for k in tens)
    has_guid = any("guidance_in" in k for k in tens)
    if has_vec:                                      # FLUX.1
        return "fill" if ch == 384 else ("base" if ch == 64 else None)
    if ch == 128 and not has_guid:                   # FLUX.2 senza guidance_in = Klein
        return "klein"
    return None


def inpaint_defaults(kind: str, name: str = "") -> tuple[int, float]:
    """(passi, guidance) del PROPRIO modello (stessi valori di image_flux._auto_settings, che la GUI può importare
    senza torch): Fill dev 40/30, FLUX.1 base 28/3.5, Klein 28/4; distillati (hyper/turbo/schnell…) 8 passi, ≤3.5."""
    from inference_server.backends.image_flux import _auto_settings
    arch = {"fill": "inpaint", "base": "flux", "klein": "flux2_inpaint"}[kind]
    steps, guid, _h = _auto_settings(arch, name)
    return int(steps), float(guid)
