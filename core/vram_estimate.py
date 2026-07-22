"""Registro CANONICO delle dimensioni modelli (immagini + video + aux) — unified-model-manager.

Byte reali dell'header safetensors per dtype + cache su file del footprint reale misurato.
stdlib-only → usabile da QUALSIASI venv (server immagini, worker video ltx2, 3D…).

Ordine di precisione: cache_get (footprint REALE misurato al 1° load) → header_mb (byte reali per
dtype) → 0 (il chiamante ripiega sulla dimensione file grezza). estimate_mb() somma anche il
working-set (attivazioni).

DB su disco: data/model_vram_cache.json. Entry checkpoint (chiave "path|size" o "path|size|fp8"):
  {"footprint_mb": int, "name": str, "source": "measured|header|gguf", "category": "image|video|aux",
   "arch": str, "dtype": str, "transformer_mb": int, "components": {nome: mb}, "ts": int}
Entry componenti condivisi (chiave "comp:<store_key>"): {"footprint_mb", "category", "dtype", "ts"}.
I campi extra sono opzionali: i lettori vecchi che leggono solo footprint_mb continuano a funzionare.
"""
from __future__ import annotations

import json
import os
import struct
import time

# path derivato dal repo (non hardcoded): core/ → root → data/. Override per test: LTX_VRAM_DB.
_CACHE = os.environ.get("LTX_VRAM_DB") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model_vram_cache.json")
_LEGACY_FOOTPRINT = os.path.join(os.path.dirname(_CACHE), "model_footprint.json")
# byte in MEMORIA per dtype safetensors (fp8 = 1B se resta quantizzato, 2B se upcastato a bf16).
_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
_TE_KEYS = ("text_encoder", "t5xxl", "t5.", "cond_stage", "clip_l", "text_model", "gemma", "mistral")
# prefissi TOP-LEVEL dei componenti NON-transformer nei bundle all-in-one (LTX: model.* è il
# transformer; questi sono vae/proiezioni/vocoder) — esclusi da transformer_mb, non da header_mb.
_AUX_PREFIXES = ("vae", "audio_vae", "video_vae", "vocoder", "text_embedding_projection",
                 "text_embeddings_projection", "audio_decoder", "latent_upsampler", "first_stage_model")


def _read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _scan_header(path: str, *, fp8_storage: bool) -> tuple[float, float, bool]:
    """(mb_totali, mb_transformer_only, has_te) dai dtype reali dell'header. (0,0,False) se illeggibile."""
    if not path or path.lower().endswith(".gguf"):
        return 0.0, 0.0, False
    try:
        h = _read_header(path)
    except Exception:                                       # noqa: BLE001
        return 0.0, 0.0, False
    total, tr_only, has_te = 0.0, 0.0, False
    for k, t in h.items():
        if k == "__metadata__":
            continue
        kl = k.lower()
        is_te = any(s in kl for s in _TE_KEYS)
        has_te = has_te or is_te
        is_aux = is_te or kl.split(".")[0] in _AUX_PREFIXES
        n = 1
        for s in t.get("shape", []):
            n *= int(s)
        dt = str(t.get("dtype", "F16"))
        b = _DTYPE_BYTES.get(dt, 2)
        if dt.startswith("F8") and not fp8_storage:
            b = 2
        total += n * b
        if not is_aux:
            tr_only += n * b
    return total / 1024**2, tr_only / 1024**2, has_te


def header_mb(path: str, *, fp8_storage: bool = False) -> int:
    """MB reali in memoria di un checkpoint safetensors (0 se illeggibile / non-safetensors). fp8_storage=True
    → i pesi F8 restano 1B (layerwise casting); altrimenti diffusers li upcasta a bf16 (2B). Se il file NON
    include il text-encoder (transformer-only) → aggiunge T5/CLIP/VAE (~9.7GB)."""
    mb, tr, has_te = _scan_header(path, fp8_storage=fp8_storage)
    if not mb:
        return 0
    # il supplemento TE esterni (~9.7GB) vale solo per i checkpoint transformer-only VERI: un file
    # AUX (VAE 160MB, text-projection 2GB…) non ha TE ma non ne trascina nessuno al load (F8).
    if not has_te and tr > 5120:
        mb += 9700
    return int(mb)


def transformer_mb(path: str, *, fp8_storage: bool = False) -> int:
    """MB del SOLO transformer/diffusion-model (esclude text-encoder/gemma/clip già nel file, e NON
    aggiunge il +9700 dei TE esterni — a differenza di header_mb). Preferisce il valore MISURATO in
    cache se presente; altrimenti header. Elimina la correzione artigianale `file_gb-10` dei worker."""
    e = _entry(path, "fp8" if fp8_storage else "")
    if e and int(e.get("transformer_mb", 0) or 0) > 0:
        return int(e["transformer_mb"])
    _mb, tr, _te = _scan_header(path, fp8_storage=fp8_storage)
    return int(tr)


def _key(path: str, variant: str = "") -> str:
    try:
        base = f"{os.path.normpath(path).lower()}|{os.stat(path).st_size}"   # invalida se il file cambia
    except Exception:                                       # noqa: BLE001
        base = os.path.normpath(path).lower()
    return f"{base}|{variant}" if variant else base


def _load_cache() -> dict:
    try:
        with open(_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                       # noqa: BLE001
        return {}


def _atomic_update(mutate) -> None:
    """Scrittura ATOMICA con merge: rilegge il DB, applica `mutate(db)`, scrive su .tmp + os.replace.
    Retry breve su PermissionError (Windows: replace mentre un altro processo legge). Due processi
    concorrenti al peggio fanno last-writer-wins su UNA voce — mai un file corrotto/troncato."""
    for attempt in range(3):
        try:
            db = _load_cache()
            mutate(db)
            os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
            tmp = _CACHE + f".tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(db, fh, indent=1)
            os.replace(tmp, _CACHE)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
        except Exception:                                   # noqa: BLE001
            return


def _entry(path: str, variant: str = "") -> dict:
    """Entry completa dalla cache: prova prima path|size|variant, poi path|size (compat voci vecchie)."""
    db = _load_cache()
    if variant:
        e = db.get(_key(path, variant))
        if e:
            return e
    return db.get(_key(path)) or {}


def cache_get(path: str, variant: str = "") -> int:
    return int(_entry(path, variant).get("footprint_mb", 0) or 0)


def cache_put(path: str, footprint_mb: int, *, variant: str = "", source: str = "",
              category: str = "", arch: str = "", dtype: str = "",
              transformer_mb: int = 0, components: dict | None = None) -> None:
    """Registra il footprint (idealmente MISURATO). I metadati sono opzionali e additivi: una put
    senza metadati NON cancella quelli già registrati per la stessa voce (merge per-campo)."""
    if footprint_mb <= 0:
        return
    key = _key(path, variant)
    extra = {"source": source, "category": category, "arch": arch, "dtype": dtype,
             "transformer_mb": int(transformer_mb or 0), "components": components}

    def _mut(db: dict) -> None:
        e = db.get(key) or {}
        e["footprint_mb"] = int(footprint_mb)
        e["name"] = os.path.basename(path)
        e["ts"] = int(time.time())
        for k, v in extra.items():
            if v:
                e[k] = v
        db[key] = e

    _atomic_update(_mut)


# ── componenti condivisi (T5, CLIP, VAE, gemma… — sostituisce model_footprint.json) ──────────

def comp_get(key: str) -> int:
    """MB di un componente condiviso registrato (0 se sconosciuto). key = chiave dello store
    (es. 'te:t5xxl:bf16', 'vae_flux')."""
    return int(_load_cache().get(f"comp:{key}", {}).get("footprint_mb", 0) or 0)


def comp_put(key: str, mb: float, *, category: str = "aux", dtype: str = "") -> None:
    """Registra il footprint MISURATO di un componente condiviso (idempotente, atomico)."""
    if not key or mb <= 0:
        return

    def _mut(db: dict) -> None:
        db[f"comp:{key}"] = {"footprint_mb": int(mb), "category": category or "aux",
                             "dtype": dtype, "ts": int(time.time())}

    _atomic_update(_mut)


def all_entries() -> dict:
    """Copia dell'intero DB (per audit/planner/UI)."""
    return dict(_load_cache())


def migrate_legacy() -> int:
    """Importa il vecchio data/model_footprint.json (registro write-only di model_registry) come
    voci comp:* del DB canonico, poi lo rinomina .migrated. Idempotente. Ritorna quante importate."""
    if not os.path.exists(_LEGACY_FOOTPRINT):
        return 0
    try:
        with open(_LEGACY_FOOTPRINT, encoding="utf-8") as fh:
            legacy = json.load(fh)
    except Exception:                                       # noqa: BLE001
        return 0
    n = 0
    for key, e in (legacy or {}).items():
        mb = float(e.get("vram_mb", 0) or 0)
        if mb > 0 and not comp_get(key):
            comp_put(key, mb, category=e.get("category") or "aux", dtype=e.get("dtype") or "")
            n += 1
    try:
        os.replace(_LEGACY_FOOTPRINT, _LEGACY_FOOTPRINT + ".migrated")
    except Exception:                                       # noqa: BLE001
        pass
    return n


def estimate_mb(path: str, *, fp8_storage: bool = False, working_mb: int = 3200,
                variant: str = "") -> tuple[int, str]:
    """(stima_MB, fonte) — cache reale → header preciso → 0. working_mb = attivazioni/VAE/workspace
    (immagini ~3200; video molto più alto, ~9000: attivazioni temporali su molti frame)."""
    c = cache_get(path, variant or ("fp8" if fp8_storage else ""))
    if c:
        return c + working_mb, "cache"
    h = header_mb(path, fp8_storage=fp8_storage)
    if h:
        return h + working_mb, "header"
    return 0, "none"
