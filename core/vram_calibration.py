"""Calibrazione VRAM CONDIVISA (immagini + video): enumera i modelli locali, ne calcola/misura il footprint
reale e lo salva nella cache su file (core.vram_estimate). Incrementale (ricalcola solo i nuovi), invalidabile.

Due livelli di precisione, entrambi cache-ati con la STESSA chiave (path+size):
  1. STIMA da header (istantanea, byte reali per dtype) — pre-popola la cache per TUTTI i modelli, subito.
  2. Footprint REALE misurato al 1° caricamento vero (server immagini via _measure_pipe_mb; video via
     post-load VRAM) — RAFFINA la voce quando il modello viene davvero usato.
La calibrazione qui fa il livello 1 (veloce, nessun load, sicuro cross-venv); il livello 2 arriva da solo
con l'uso. + planner COMBINATORIO per decidere cosa tenere residente e cosa offloadare data la VRAM libera.
"""
from __future__ import annotations

import glob
import os

from core import vram_estimate as _ve

# Cartelle dei pesi (immagini + video + AUX F8: "modello" = tutto ciò che va in VRAM).
_MODEL_ROOTS = [r"E:\AI\Models\diffusion_models", r"E:\AI\Models\Checkpoints",
                r"E:\AI\Models\TextEncoders", r"E:\AI\Models\VAE",
                r"E:\AI\Models\latent_upscale_models"]
_EXTS = (".safetensors", ".gguf")

# Footprint NOTI dei componenti CONDIVISI (bf16 in memoria) — base del planner combinatorio. Il transformer
# è per-modello (dalla cache/header); questi sono gli "altri" componenti che si sommano/condividono.
COMPONENT_MB = {
    "t5xxl": 9100,        # T5-XXL (FLUX.1) text-encoder
    "clip_l": 250,        # CLIP-L (FLUX.1)
    "vae_flux": 320,      # VAE FLUX
    "gemma": 9000,        # Gemma text-encoder (LTX-2 video)
    "vae_video": 800,     # VAE video LTX-2
    "vocoder": 200,       # vocoder audio LTX-2
    "mistral": 9000,      # Mistral text-encoder (FLUX.2)
}
# Componenti richiesti per famiglia d'architettura (per il planner combinatorio).
ARCH_COMPONENTS = {
    "flux": ["t5xxl", "clip_l", "vae_flux"],
    "flux2": ["mistral", "vae_flux"],
    "ltx2": ["gemma", "vae_video", "vocoder"],
}


def _is_fp8(path: str) -> bool:
    try:
        h = _ve._read_header(path)                           # noqa: SLF001 (helper condiviso)
        for k, t in h.items():
            if k != "__metadata__" and str(t.get("dtype", "")).startswith("F8"):
                return True
    except Exception:                                        # noqa: BLE001
        pass
    return False


def enumerate_models() -> list[dict]:
    """Tutti i checkpoint/GGUF locali (immagini + video). Ritorna [{path,name,size_mb,is_gguf,cached}]."""
    seen, out = set(), []
    for root in _MODEL_ROOTS:
        if not os.path.isdir(root):
            continue
        for ext in _EXTS:
            for p in glob.glob(os.path.join(root, "**", "*" + ext), recursive=True):
                try:
                    st = os.stat(p)
                    sz = st.st_size // (1024 * 1024)
                except OSError:
                    continue
                # dedup per inode: gli HARDLINK (stesso file in diffusion_models/ e Checkpoints/)
                # contano UNA volta — per path apparivano come due modelli distinti da misurare.
                key = (st.st_dev, st.st_ino) if st.st_ino else os.path.normpath(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                if sz < 100:                                 # salta LoRA/piccoli (i VAE ~160MB restano dentro)
                    continue
                out.append({"path": p, "name": os.path.basename(p), "size_mb": sz,
                            "is_gguf": p.lower().endswith(".gguf"),
                            "cached": (_ve.cache_get(p) > 0 or _ve.cache_get(p, "fp8") > 0)})
    out.sort(key=lambda d: d["name"].lower())
    return out


def calibrate_all(force: bool = False, on_progress=None) -> dict:
    """Calibra (livello-1, header) tutti i modelli non ancora in cache. force=True ricalcola tutti.
    on_progress(i, n, name, mb). Ritorna {calibrated, skipped, total, results:[{name,mb,source}]}."""
    models = enumerate_models()
    n = len(models)
    res, done, skip = [], 0, 0
    for i, m in enumerate(models):
        if m["cached"] and not force:
            skip += 1
            _mb = _ve.cache_get(m["path"]) or _ve.cache_get(m["path"], "fp8")
            res.append({"name": m["name"], "mb": _mb, "source": "cache"})
            if on_progress:
                on_progress(i + 1, n, m["name"], _mb)
            continue
        _var = ""
        if m["is_gguf"]:
            mb = m["size_mb"]                                # GGUF: resta quantizzato → footprint ≈ dim file
            src = "gguf"
        else:
            _fp8 = _is_fp8(m["path"])
            _var = "fp8" if _fp8 else ""
            mb = _ve.header_mb(m["path"], fp8_storage=_fp8)
            src = "header"
        if mb > 0:
            _cat = ""
            try:                                             # F2: categoria dal detector (kind image/video)
                from core import model_detect
                _cat = str((model_detect.get(m["path"]) or {}).get("kind") or "") if not m["is_gguf"] else ""
            except Exception:                                # noqa: BLE001
                pass
            _ve.cache_put(m["path"], mb, variant=_var, source=src, category=_cat)
            done += 1
        res.append({"name": m["name"], "mb": mb, "source": src})
        if on_progress:
            on_progress(i + 1, n, m["name"], mb)
    return {"calibrated": done, "skipped": skip, "total": n, "results": res}


def invalidate(path: str | None = None) -> int:
    """Invalida la cache: un modello (path) o TUTTA (path=None). Ritorna il numero di voci rimosse."""
    cache = _ve._load_cache()                                # noqa: SLF001
    if not cache:
        return 0
    if path is None:
        removed = len(cache)
        cache = {}
    else:
        key = _ve._key(path)                                 # noqa: SLF001
        removed = 1 if key in cache else 0
        cache.pop(key, None)
    try:
        import json
        os.makedirs(os.path.dirname(_ve._CACHE), exist_ok=True)   # noqa: SLF001
        with open(_ve._CACHE, "w", encoding="utf-8") as fh:       # noqa: SLF001
            json.dump(cache, fh, indent=1)
    except Exception:                                        # noqa: BLE001
        pass
    return removed


def plan_placement(transformer_mb: int, arch: str, free_mb: int, working_mb: int = 3200) -> dict:
    """Planner COMBINATORIO: dato il transformer (dalla cache), l'arch (→ componenti richiesti) e la VRAM
    LIBERA, decide cosa tenere RESIDENTE e cosa OFFLOADARE per minimizzare l'offload (= massima velocità).
    Ritorna {resident:[...], offload:[...], resident_mb, fits_all, mode}."""
    comps = ARCH_COMPONENTS.get(arch, [])
    # F6: dimensioni componenti dal DB canonico (misure reali via comp_put) con fallback alla
    # tabella statica — la tabella resta l'ultima risorsa, mai la prima scelta.
    comp_sizes = [(c, _ve.comp_get(c) or COMPONENT_MB.get(c, 0)) for c in comps]
    total_comp = sum(s for _, s in comp_sizes)
    # 1) tutto residente (transformer + tutti i componenti + working)?
    if transformer_mb + total_comp + working_mb <= free_mb:
        return {"resident": ["transformer"] + comps, "offload": [], "mode": "none",
                "resident_mb": transformer_mb + total_comp, "fits_all": True}
    # 2) transformer residente + offload dei componenti più grossi finché entra (greedy dal più grande).
    budget = free_mb - transformer_mb - working_mb
    if budget < 0:                                           # il transformer DA SOLO non entra → tutto offload
        return {"resident": [], "offload": ["transformer"] + comps, "mode": "cpu",
                "resident_mb": 0, "fits_all": False}
    resident, offload = ["transformer"], []
    for c, s in sorted(comp_sizes, key=lambda x: x[1]):      # tieni residenti i PIÙ PICCOLI (più riuso/veloci)
        if s <= budget:
            resident.append(c); budget -= s
        else:
            offload.append(c)
    mode = "cpu" if offload else "none"
    _sz = dict(comp_sizes)
    return {"resident": resident, "offload": offload, "mode": mode,
            "resident_mb": transformer_mb + (total_comp - sum(_sz.get(c, 0) for c in offload)),
            "fits_all": not offload}
