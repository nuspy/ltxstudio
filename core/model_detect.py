"""Rilevamento AUTOMATICO dell'architettura di un modello dal FILE (no liste hardcoded).

Idea (come ComfyUI `comfy/model_detection.py`): leggere SOLO l'header del file — chiavi, forme,
dtype e `__metadata__` — e dedurne famiglia/variante/ruolo/precisione. Così l'utente aggiunge o
toglie un file dalla cartella e l'app lo riconosce e lo mostra nel posto giusto, senza mapping manuali.

Tre segnali, in ordine di affidabilità:
 1. `__metadata__["modelspec.architecture"]` (standard SAI, presente in molti checkpoint) → mappa diretta.
 2. **Fingerprint chiavi+forme** dei tensori (autoritativo, sempre disponibile): es. `double_blocks`+
    `single_blocks`+`vector_in` → FLUX.1; senza `vector_in` → FLUX.2; `input_blocks`+`cond_stage_model`
    → SD1.5; `conditioner.embedders.1` → SDXL; `patch_embedding` → Wan; ecc.
 3. cartella/nome file (ultima spiaggia, es. per i .gguf).

La **precisione** si legge dai dtype dell'header (autoritativo: niente più "FP8 nel nome nasconde i bf16"):
F8_E4M3/E5M2 → fp8; U8+`*.comfy_quant`/nvfp4 → fp4; estensione .gguf → gguf; BF16/F16 → fp16.

CACHE su file (`data/model_detect.json`): per ogni path salviamo mtime+size+detection. A ogni scan si
rileva solo ciò che è NUOVO o CAMBIATO; gli scomparsi vengono rimossi. Nessun riprocesso dei file noti.

API:
 - `detect_one(path) -> dict`            rileva un singolo file (no cache).
 - `scan(roots=None, force=False) -> list[dict]`   scansione cache-aware di tutte le cartelle modelli.
 - `get(path) -> dict`                   detection cache-aware di un singolo path.
 - `compatible_components(det) -> dict`  componenti (vae/text_encoder/scheduler) consigliati per l'arch.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

# radici di scansione (immagini + video). Allineate a core/offline_inventory + layout E:\AI\Models.
try:
    from core.offline_inventory import MODELS_ROOT as _MR  # type: ignore
    _MODELS_ROOT = Path(_MR)
except Exception:                                          # noqa: BLE001
    _MODELS_ROOT = Path(r"E:\AI\Models")

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "model_detect.json"
_SCAN_DIRS = ("diffusion_models", "Checkpoints", "checkpoints")
_EXTS = (".safetensors", ".gguf", ".ckpt", ".sft")

# ──────────────────────────────────────────────────────────────────────────────
# lettura header (no pesi)
# ──────────────────────────────────────────────────────────────────────────────
def read_header(path: str) -> tuple[dict, dict]:
    """Ritorna (tensors, metadata) leggendo SOLO l'header safetensors. tensors = {nome:{dtype,shape}}.
    Per file non-safetensors ritorna ({}, {})."""
    p = str(path)
    if not p.lower().endswith((".safetensors", ".sft")):
        return {}, {}
    try:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 100_000_000:
                return {}, {}
            head = json.loads(f.read(n))
        meta = head.pop("__metadata__", {}) or {}
        return head, meta
    except Exception:                                     # noqa: BLE001
        return {}, {}


def _shape(tensors: dict, *suffixes):
    """Forma del primo tensore la cui chiave termina con uno dei suffissi (ignora prefissi)."""
    for k, v in tensors.items():
        for s in suffixes:
            if k.endswith(s):
                return v.get("shape")
    return None


def _has(keys, *subs) -> bool:
    return any(any(s in k for k in keys) for s in subs)


def _count(keys, prefix_sub: str) -> int:
    """Conta gli indici di blocco distinti per chiavi tipo '...{prefix_sub}{N}....'."""
    idxs = set()
    for k in keys:
        i = k.find(prefix_sub)
        if i < 0:
            continue
        rest = k[i + len(prefix_sub):]
        num = rest.split(".", 1)[0]
        if num.isdigit():
            idxs.add(int(num))
    return len(idxs)


# ──────────────────────────────────────────────────────────────────────────────
# precisione dai dtype
# ──────────────────────────────────────────────────────────────────────────────
def precision_from_header(path: str, tensors: dict) -> str:
    p = str(path).lower()
    if p.endswith(".gguf"):
        return "gguf"
    if not tensors:
        return "unknown"
    dts = {}
    for v in tensors.values():
        if isinstance(v, dict):
            dts[v.get("dtype")] = dts.get(v.get("dtype"), 0) + 1
    keys = list(dts)
    # nvfp4 (ComfyUI): pesi U8 packed + marcatore .comfy_quant
    if any(k == "U8" for k in keys) and ("comfy_quant" in " ".join(list(tensors)[:500]) or dts.get("U8", 0) > 50):
        return "fp4"
    if any(k in ("F8_E4M3", "F8_E5M2", "F8_E4M3FN") for k in keys):
        return "fp8"
    if any(k in ("F16", "BF16", "F32") for k in keys):
        return "fp16"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# mappa da modelspec.architecture (standard SAI) → (family, variant, role, kind)
# ──────────────────────────────────────────────────────────────────────────────
def _from_arch_str(a: str):
    s = (a or "").strip().lower().replace(" ", "-")
    if not s or s == "-":
        return None
    # FLUX.2
    if "flux.2" in s or "flux-2" in s or "flux2" in s:
        return ("flux2", "dev", "t2i", "image")
    # FLUX.1 varianti
    if s.startswith("flux"):
        if "redux" in s:
            return ("flux1", "redux", "adapter", "image")
        if "inpaint" in s or "fill" in s:
            return ("flux1", "fill", "inpaint", "image")
        if "kontext" in s:
            return ("flux1", "kontext", "edit", "image")
        if "schnell" in s:
            return ("flux1", "schnell", "t2i", "image")
        return ("flux1", "dev", "t2i", "image")
    if "stable-diffusion-xl" in s or s.startswith("sdxl"):
        return ("sdxl", "base", "t2i", "image")
    if "stable-diffusion-v1" in s or "stable-diffusion-1" in s or s in ("sd1", "sd15"):
        return ("sd15", "base", "t2i", "image")
    if "stable-diffusion-3" in s or "sd3" in s:
        return ("sd3", "base", "t2i", "image")
    if "z-image" in s or "zimage" in s:
        return ("zimage", "base", "t2i", "image")
    if "ernie" in s:
        return ("ernie", "base", "t2i", "image")
    if "hidream" in s or "hi-dream" in s:
        return ("hidream", "base", "t2i", "image")
    if "qwen" in s and "image" in s:
        return ("qwen_image", "edit" if "edit" in s else "base", "edit" if "edit" in s else "t2i", "image")
    if "krea-2" in s or "krea2" in s:
        var = "raw" if "raw" in s else "turbo"
        return ("krea2", var, "t2i", "image")
    # VIDEO
    if "krea-realtime" in s or "krea_rt" in s or "krea-rt" in s:
        return ("krea_rt", "base", "video", "video")
    if "lingbot" in s:
        var = "moe" if ("moe" in s or "30b" in s or "a3b" in s) else "dense"
        return ("lingbot", var, "video", "video")
    if "ltx" in s or "lightricks" in s:
        return ("ltxv", "base", "video", "video")
    if "wan" in s:
        v = "wan22" if ("2_2" in s or "2.2" in s) else "wan21"
        role = "i2v" if "image2video" in s or "i2v" in s else ("s2v" if "s2v" in s else "t2v")
        return (v, role, role, "video")
    if "hunyuan" in s and "video" in s:
        return ("hunyuan_video", "base", "video", "video")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# fingerprint da chiavi+forme (fallback quando manca modelspec.architecture)
# ──────────────────────────────────────────────────────────────────────────────
def _from_keys(tensors: dict, name: str):
    keys = list(tensors.keys())
    if not keys:
        return None
    nl = name.lower()

    # --- VIDEO prima di FLUX (Hunyuan Video condivide double/single/vector/guidance) ---
    # LingBot-Video: normalmente uno snapshot diffusers (rilevato da cartella/config); ma se si punta al
    # solo transformer .safetensors, riconoscilo dal nome + eventuali chiavi MoE (experts/gate).
    if "lingbot" in nl:
        var = "moe" if (_has(keys, "experts") or _has(keys, ".gate.") or "moe" in nl or "30b" in nl) else "dense"
        return ("lingbot", var, "video", "video")
    if _has(keys, "individual_token_refiner"):
        return ("hunyuan_video", "base", "video", "video")
    if _has(keys, "patch_embedding"):                     # Wan (2.1/2.2)
        v = "wan22" if ("2.2" in nl or "2_2" in nl or "wan2.2" in nl) else "wan21"
        role = "i2v" if ("i2v" in nl or "image" in nl) else ("s2v" if "s2v" in nl else "t2v")
        return (v, role, role, "video")

    # --- FLUX family: double_blocks + single_blocks ---
    if _has(keys, "double_blocks") and _has(keys, "single_blocks"):
        # redux = solo proiettore (poche chiavi)
        if len(keys) <= 8 and _has(keys, "redux"):
            return ("flux1", "redux", "adapter", "image")
        img = _shape(tensors, "img_in.weight")
        hidden = (img[0] if img else None)
        in_ch = (img[1] if img and len(img) > 1 else None)
        has_vec = _has(keys, "vector_in")
        n_single = _count(keys, "single_blocks.")
        if has_vec:
            family = "flux1"
            if in_ch and in_ch >= 256:                    # Fill: img_in in-channels grandi (es. 384)
                return (family, "fill", "inpaint", "image")
            if "kontext" in nl:
                return (family, "kontext", "edit", "image")
            if "redux" in nl:
                return (family, "redux", "adapter", "image")
            var = "schnell" if "schnell" in nl else "dev"
            return (family, var, "t2i", "image")
        # niente vector_in → FLUX.2 (Dev) o Klein, oppure DreamO (editing flux-based)
        if "dreamo" in nl:
            return ("dreamo", "base", "edit", "image")
        if hidden and hidden >= 6144 and _has(keys, "guidance_in"):
            return ("flux2", "dev", "t2i", "image")
        # hidden ~4096, niente guidance → Klein (edit/inpaint)
        return ("flux2", "klein", "edit", "image")

    # --- SD/SDXL ---
    if _has(keys, "conditioner.embedders.1") or _has(keys, "label_emb.0.0.weight"):
        return ("sdxl", "base", "t2i", "image")
    if _has(keys, "input_blocks") and _has(keys, "cond_stage_model"):
        return ("sd15", "base", "t2i", "image")
    if _has(keys, "input_blocks.0.0"):                    # unet-only SD
        return ("sd15", "base", "t2i", "image")

    # --- DiT vari ---
    if _has(keys, "joint_blocks"):                        # SD3 / mmdit
        return ("sd3", "base", "t2i", "image")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# fallback cartella/nome
# ──────────────────────────────────────────────────────────────────────────────
_FOLDER_FAMILY = {
    "z-image": ("zimage", "base", "t2i", "image"), "ernie": ("ernie", "base", "t2i", "image"),
    "hidream": ("hidream", "base", "t2i", "image"), "qwen": ("qwen_image", "base", "t2i", "image"),
    "ltx": ("ltxv", "base", "video", "video"), "ltx-2": ("ltxv", "base", "video", "video"),
    "wan2.1": ("wan21", "t2v", "t2v", "video"), "wan2.2": ("wan22", "t2v", "t2v", "video"),
    "wan": ("wan21", "t2v", "t2v", "video"), "hunyuanvideo": ("hunyuan_video", "base", "video", "video"),
    "dreamo": ("dreamo", "base", "edit", "image"), "flux": ("flux1", "dev", "t2i", "image"),
    "sd": ("sd15", "base", "t2i", "image"), "supir": ("sdxl", "base", "t2i", "image"),
}


def _vid_role(nl: str, default: str) -> str:
    if "i2v" in nl or "image_to_video" in nl or "image2video" in nl or "img2video" in nl:
        return "i2v"
    if "s2v" in nl or "sound" in nl or "audio2video" in nl:
        return "s2v"
    if "t2v" in nl or "text_to_video" in nl or "text2video" in nl:
        return "t2v"
    return default


def _refine(fam, variant, role, kind, nl: str):
    """Affina versione/ruolo dal nome file (es. .gguf via fallback cartella: i2v/t2v/s2v, wan 2.1↔2.2)."""
    if fam in ("wan21", "wan22") or kind == "video":
        if fam.startswith("wan"):
            fam = "wan22" if ("2.2" in nl or "2_2" in nl or "wan2.2" in nl) else ("wan21" if ("2.1" in nl or "2_1" in nl or "wan2.1" in nl) else fam)
        r = _vid_role(nl, role if role in ("t2v", "i2v", "s2v") else "t2v")
        if fam.startswith("wan"):
            role = variant = r
    return fam, variant, role, kind


def _from_folder(path: str):
    nl = Path(path).name.lower()
    parts = [p.lower() for p in Path(path).parts]
    for seg in reversed(parts[:-1]):                      # cartelle (esclude il filename)
        if seg in _FOLDER_FAMILY:
            return _refine(*_FOLDER_FAMILY[seg], nl)
    for key, val in _FOLDER_FAMILY.items():
        if key in nl:
            return _refine(*val, nl)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# detection di un singolo file
# ──────────────────────────────────────────────────────────────────────────────
def _from_config_file(path: str):
    """Classifica i modelli MULTI-FILE diffusers (shard 0000X-of-0000Y) leggendo config.json /
    model_index.json nella stessa cartella: un singolo shard ha chiavi parziali e non si riconosce
    dai tensori. Mappa _class_name/architectures → famiglia. None se nessuna config utile."""
    d = Path(path).parent
    blob = ""
    for fn in ("config.json", "model_index.json", "../model_index.json"):
        f = d / fn
        try:
            if f.exists():
                blob += " " + f.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:                                    # noqa: BLE001
            pass
    if not blob:
        return None
    # firme per famiglia (sottostringhe in _class_name/architectures/model_type)
    sig = [
        ("flux2", ["flux2", "flux.2"]), ("flux1", ["flux"]),
        ("ltxv", ["ltx"]), ("wan22", ["wan2.2", "wan22"]), ("wan21", ["wan"]),
        ("hunyuan_video", ["hunyuanvideo", "hunyuan_video"]), ("qwen_image", ["qwenimage", "qwen_image"]),
        ("hidream", ["hidream"]), ("sd3", ["sd3", "stablediffusion3"]),
        ("sdxl", ["xlpipeline", "stablediffusionxl"]), ("zimage", ["zimage", "z-image", "lumina"]),
        ("ernie", ["ernie"]),
    ]
    for fam, keys in sig:
        if any(k in blob for k in keys):
            kind = "video" if fam in ("ltxv", "wan21", "wan22", "hunyuan_video") else "image"
            return (fam, "", "unknown", kind)
    return None


def _from_adapter_keys(tensors: dict):
    """Adattatori che NON sono modelli generativi (ControlNet, LoRA di controllo MagicQuill): il loro header somiglia
    a quello del modello base (modelspec «Flux.1-dev/controlnet», cartella «Flux») e finiva fra i t2i. Si riconoscono
    dalle chiavi: ruolo `controlnet` / `control_lora`, mai `t2i`."""
    keys = tensors.keys() if tensors else ()
    if not keys:
        return None
    ks = list(keys)
    if any(k.startswith("controlnet_blocks.") for k in ks) and "controlnet_x_embedder.weight" in tensors:
        shp = (tensors["controlnet_x_embedder.weight"] or {}).get("shape") or [0, 64]
        return ("flux1", "controlnet_inpaint" if int(shp[1]) > 64 else "controlnet_union", "controlnet", "image")
    if any(k.startswith("control_layers.") for k in ks) and any(k.startswith("control_all_x_embedder.") for k in ks):
        return ("zimage", "controlnet", "controlnet", "image")
    if any(".attn.processor.q_loras." in k for k in ks[:2000]):
        return ("flux1", "control_lora", "control_lora", "image")
    return None


DET_VERSION = 2          # 2: ControlNet e LoRA di controllo riconosciuti dalle chiavi (ruolo controlnet/control_lora)


def detect_one(path: str) -> dict:
    path = str(path)
    name = Path(path).name
    tensors, meta = read_header(path)
    arch_meta = (meta.get("modelspec.architecture") or meta.get("general.architecture") or "")
    res = _from_adapter_keys(tensors)
    source = "keys"
    if res is None:
        res = _from_arch_str(arch_meta)
        source = "metadata"
    if res is None:
        res = _from_keys(tensors, name)
        source = "keys"
    if res is None:
        res = _from_folder(path)
        source = "folder"
    if res is None:                                          # shard diffusers / file parziale → config cartella
        res = _from_config_file(path)
        source = "config" if res else source
    if res is None:
        res = ("unknown", "", "unknown", "unknown")
        source = "none"
    family, variant, role, kind = res
    img = _shape(tensors, "img_in.weight", "x_embedder.weight")
    out = {
        "path": path, "name": name, "folder": Path(path).parent.name,
        "family": family, "variant": variant, "role": role, "kind": kind,
        "precision": precision_from_header(path, tensors),
        "arch_meta": arch_meta, "source": source,
        "hidden": (img[0] if img else None),
        "n_double": _count(list(tensors), "double_blocks."),
        "n_single": _count(list(tensors), "single_blocks."),
    }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# componenti compatibili per arch (per la voce "auto")
# ──────────────────────────────────────────────────────────────────────────────
def compatible_components(det: dict) -> dict:
    """VAE / text-encoder / scheduler consigliati per l'architettura rilevata. Usato dalla voce 'auto':
    l'utente sceglie il modello principale e i componenti vengono impostati di conseguenza (override libero)."""
    fam = det.get("family")
    var = (det.get("variant") or "").lower()
    if fam == "flux2":
        # Dev → encoder Mistral-3; Klein (distillato) → encoder Qwen3 (4B/8B secondo la dim testo).
        if "klein" in var:
            return {"vae": "Flux/flux2-vae.safetensors", "text_encoders": ["qwen3_8b", "qwen3_4b"]}
        return {"vae": "Flux/flux2-vae.safetensors", "text_encoders": ["mistral_3_small_flux2"]}
    M = {
        "flux1":  {"vae": "Flux/ae.safetensors", "text_encoders": ["clip_l", "t5xxl"]},
        "zimage": {"vae": "Flux/ae.safetensors", "text_encoders": ["qwen3_4b"]},
        "ernie":  {"vae": "Flux/flux2-vae.safetensors", "text_encoders": ["ministral-3-3b"]},
        "sd15":   {"vae": "built-in", "text_encoders": ["clip_l"]},
        "sdxl":   {"vae": "sdxl_vae", "text_encoders": ["clip_l", "clip_g"]},
        "ltxv":   {"vae": "ltx-2 builtin", "text_encoders": ["t5xxl"]},
        "wan21":  {"vae": "wan_vae", "text_encoders": ["umt5_xxl"]},
        "wan22":  {"vae": "wan_vae", "text_encoders": ["umt5_xxl"]},
        # LingBot: pipeline diffusers self-contained (vae+text_encoder VLM Qwen dentro lo snapshot).
        "lingbot": {"vae": "builtin (diffusers)", "text_encoders": ["builtin (Qwen VLM)"]},
        # Krea: snapshot diffusers self-contained. Krea2 = Qwen3-VL + VAE Qwen-Image; Krea RT = base Wan 2.1.
        "krea2": {"vae": "builtin (Qwen-Image)", "text_encoders": ["builtin (Qwen3-VL)"]},
        "krea_rt": {"vae": "builtin (Wan)", "text_encoders": ["builtin (UMT5, base Wan 2.1)"]},
    }
    return M.get(fam, {"vae": "auto", "text_encoders": []})


def components_for_path(path: str) -> dict:
    """Convenience GUI: {family, variant, vae, text_encoders} per un file modello (cache-aware).
    Usato dai dropdown per mostrare/risolvere la voce 'auto' in base all'architettura del modello scelto."""
    try:
        det = get(path) or {}
    except Exception:                                          # noqa: BLE001
        det = {}
    comp = compatible_components(det)
    return {"family": det.get("family"), "variant": det.get("variant"),
            "vae": comp.get("vae"), "text_encoders": comp.get("text_encoders", [])}


# ──────────────────────────────────────────────────────────────────────────────
# cache su file
# ──────────────────────────────────────────────────────────────────────────────
# Memoizzazione IN-MEMORY della cache (loading veloce dei dropdown): il file viene letto da disco
# UNA volta e riusato; si ricarica solo se il file su disco cambia (mtime). Senza, ogni get() per
# ogni voce di ogni dropdown rileggeva+parsava l'intero JSON → lentezza al refresh.
_MEM = {"cache": None, "mtime": None}


def _load_cache() -> dict:
    try:
        mt = CACHE_PATH.stat().st_mtime_ns if CACHE_PATH.exists() else None
    except OSError:
        mt = None
    if _MEM["cache"] is not None and _MEM["mtime"] == mt:
        return _MEM["cache"]                              # in-memory valido → niente disco
    cache = {}
    try:
        if CACHE_PATH.exists():
            d = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            cache = d if isinstance(d, dict) else {}
    except Exception:                                     # noqa: BLE001
        cache = {}
    _MEM["cache"] = cache
    _MEM["mtime"] = mt
    return cache


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CACHE_PATH)
        # aggiorna lo stato in-memory (cache + mtime) → i get() successivi non rileggono il disco
        _MEM["cache"] = cache
        try:
            _MEM["mtime"] = CACHE_PATH.stat().st_mtime_ns
        except OSError:
            _MEM["mtime"] = None
    except Exception:                                     # noqa: BLE001
        pass


def _iter_model_files(roots):
    seen = set()                                          # dedup case-insensitive (Windows: Checkpoints==checkpoints)
    for root in roots:
        base = _MODELS_ROOT / root
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in _EXTS:
                key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                yield str(p)


def scan(roots=None, force: bool = False) -> list[dict]:
    """Scansione cache-aware: rileva SOLO i file nuovi/cambiati, rimuove gli scomparsi, riusa il resto.
    Ritorna la lista delle detection (dict). `force=True` ignora la cache e riprocessa tutto."""
    roots = roots or _SCAN_DIRS
    cache = {} if force else _load_cache()
    present = set()
    out = []
    changed = False
    for path in _iter_model_files(roots):
        present.add(path)
        try:
            st = os.stat(path)
            sig = [st.st_mtime_ns, st.st_size]
        except OSError:
            continue
        ent = cache.get(path)
        if ent and ent.get("_sig") == sig and "family" in ent and ent.get("_dv") == DET_VERSION:
            out.append(ent)
            continue
        det = detect_one(path)
        det["_sig"] = sig
        det["_dv"] = DET_VERSION
        cache[path] = det
        out.append(det)
        changed = True
    # rimuovi gli scomparsi
    for gone in [k for k in cache if k not in present]:
        cache.pop(gone, None)
        changed = True
    if changed:
        _save_cache(cache)
    return out


def get(path: str) -> dict:
    """Detection cache-aware di un singolo path (usa/aggiorna la cache)."""
    path = str(path)
    cache = _load_cache()
    try:
        st = os.stat(path)
        sig = [st.st_mtime_ns, st.st_size]
    except OSError:
        return detect_one(path)
    ent = cache.get(path)
    if ent and ent.get("_sig") == sig and "family" in ent and ent.get("_dv") == DET_VERSION:
        return ent
    det = detect_one(path)
    det["_sig"] = sig
    det["_dv"] = DET_VERSION
    cache[path] = det
    _save_cache(cache)
    return det


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(detect_one(sys.argv[1]), indent=2, ensure_ascii=False))
    else:
        for d in sorted(scan(force=True), key=lambda x: (x["kind"], x["family"], x["name"])):
            print(f"{d['name'][:46]:48} {d['kind']:6} {d['family']:13} {d['variant']:8} "
                  f"{d['role']:8} {d['precision']:7} src={d['source']}")
