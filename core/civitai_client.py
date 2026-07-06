"""Client Civitai — puro urllib (nessuna dipendenza pesante), per l'importer modelli (#17).

Copre ciò che gli script del catalogo NON fanno: risoluzione di URL/ID, ricerca (browse in-app) e DOWNLOAD.
L'enrichment by-hash dei file locali resta a `enrich_civitai.py` (via CatalogTooling). Nessuna chiamata di
rete a import: la rete avviene SOLO quando l'utente invoca una funzione.

API pubblica (no auth per lettura). Domini: civitai.com / civitai.green (filtrati SFW) / civitai.red (mature)
condividono la stessa REST API su civitai.com. Un API-key opzionale (settings) serve solo per scaricare
modelli early-access/loggati.

Rif.: developer.civitai.com/site/reference — hash by-hash accetta SHA256 completo o AutoV2 (primi 10 hex).
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://civitai.com/api/v1"
_UA = "LTXStudio-ModelImporter/1.0 (+local)"
_APP_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = _APP_ROOT / "data" / "civitai_cache.json"

# NsfwLevel Civitai (bitmask): PG=1, PG13=2, R=4, X=8, XXX=16, Blocked=32. SFW = PG|PG13 = 3.
NSFW_PG, NSFW_PG13, NSFW_R, NSFW_X, NSFW_XXX, NSFW_BLOCKED = 1, 2, 4, 8, 16, 32
SFW_MASK = NSFW_PG | NSFW_PG13                                # 1|2 = 3


# ---------------- hash ----------------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA256 esadecimale minuscolo dell'intero file (come Civitai Helper / Lora-Manager)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest().lower()


def autov2(sha256: str) -> str:
    """AutoV2 = primi 10 caratteri hex dello SHA256 (minuscolo). Non è un hash separato."""
    return sha256[:10].lower()


# ---------------- cache ----------------
def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
    except Exception:                                        # noqa: BLE001
        pass


# ---------------- HTTP ----------------
def _req(url: str, api_key: str | None, timeout: float):
    headers = {"User-Agent": _UA, "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(url, headers=headers)


def _get_json(url: str, api_key: str | None = None, timeout: float = 30.0):
    with urllib.request.urlopen(_req(url, api_key, timeout), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------- lookup ----------------
def by_hash(sha256: str, api_key: str | None = None, use_cache: bool = True) -> dict | None:
    """GET /model-versions/by-hash/{sha}. Ritorna l'oggetto version o None (404). Cache su disco."""
    key = f"hash:{sha256.lower()}"
    cache = _load_cache() if use_cache else {}
    if use_cache and key in cache:
        return cache[key] or None
    try:
        data = _get_json(f"{API}/model-versions/by-hash/{urllib.parse.quote(sha256)}", api_key)
    except urllib.error.HTTPError as e:                       # noqa: PERF203
        if e.code == 404:
            cache[key] = None; _save_cache(cache); return None
        raise
    if use_cache:
        cache[key] = data; _save_cache(cache)
    return data


def version(version_id: int | str, api_key: str | None = None) -> dict:
    """GET /model-versions/{id}."""
    return _get_json(f"{API}/model-versions/{version_id}", api_key)


def model(model_id: int | str, api_key: str | None = None) -> dict:
    """GET /models/{id} (contiene modelVersions[])."""
    return _get_json(f"{API}/models/{model_id}", api_key)


def search(query: str = "", types: list[str] | None = None, base_models: list[str] | None = None,
           nsfw: bool | None = None, limit: int = 24, page: int = 1,
           api_key: str | None = None) -> dict:
    """GET /models — ricerca per il browse in-app. types es. ['LORA','Checkpoint']."""
    params: list[tuple[str, str]] = [("limit", str(limit)), ("page", str(page))]
    if query:
        params.append(("query", query))
    for t in types or []:
        params.append(("types", t))
    for bm in base_models or []:
        params.append(("baseModels", bm))
    if nsfw is not None:
        params.append(("nsfw", "true" if nsfw else "false"))
    return _get_json(f"{API}/models?{urllib.parse.urlencode(params)}", api_key)


# ---------------- URL/ID ----------------
_URL_MODEL = re.compile(r"/models/(\d+)")
_URL_VER = re.compile(r"[?&]modelVersionId=(\d+)")
_URL_APIVER = re.compile(r"/model-versions/(\d+)")


def parse_url(text: str) -> dict:
    """Estrae {model_id, version_id} da un URL Civitai (.com/.green/.red) o da un ID nudo.
    Ritorna almeno una delle due chiavi (None l'altra). Solleva ValueError se non riconosciuto."""
    s = (text or "").strip()
    if s.isdigit():
        return {"model_id": int(s), "version_id": None}
    out: dict = {"model_id": None, "version_id": None}
    m = _URL_APIVER.search(s)
    if m:
        out["version_id"] = int(m.group(1)); return out
    m = _URL_VER.search(s)
    if m:
        out["version_id"] = int(m.group(1))
    m = _URL_MODEL.search(s)
    if m:
        out["model_id"] = int(m.group(1))
    if out["model_id"] is None and out["version_id"] is None:
        raise ValueError(f"URL/ID Civitai non riconosciuto: {text!r}")
    return out


# ---------------- download ----------------
def pick_primary_file(ver: dict) -> dict | None:
    """Sceglie il file principale (safetensors) di una version. Preferisce files[].primary."""
    files = ver.get("files") or []
    if not files:
        return None
    for f in files:
        if f.get("primary"):
            return f
    # fallback: il più grande fra .safetensors/.ckpt/.pt
    cand = [f for f in files if str(f.get("name", "")).lower().endswith((".safetensors", ".ckpt", ".pt"))]
    return (max(cand, key=lambda f: f.get("sizeKB", 0)) if cand else files[0])


def download(url: str, dest: str | Path, progress_cb=None, cancel_evt=None,
             api_key: str | None = None, timeout: float = 60.0) -> Path:
    """Scarica un file con RESUME (Range) su <dest>.part → rename. progress_cb(done, total).
    cancel_evt: threading.Event → se set, interrompe e lascia il .part (ripartibile).
    Serializzato dal chiamante (un download alla volta — connessione metered)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": _UA}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        done = have
        with open(part, mode) as f:
            while True:
                if cancel_evt is not None and cancel_evt.is_set():
                    raise InterruptedError("download annullato")
                buf = r.read(1 << 16)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress_cb:
                    progress_cb(done, total)
    part.replace(dest)
    return dest


def api_key_from_prefs() -> str | None:
    """API-key opzionale salvato nelle preferenze (per modelli early-access/loggati)."""
    try:
        from core import user_prefs
        return (user_prefs.get("civitai_api_key", "") or "").strip() or None
    except Exception:                                        # noqa: BLE001
        return None
