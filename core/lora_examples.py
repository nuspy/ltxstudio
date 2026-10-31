"""Pagina e immagini d'esempio delle LoRA, CERCATE online (Civitai per hash, poi un agente di ricerca) e tenute in cache locale
`data/lora_examples/` (index.json + immagini): mai nel repo, mai generate dall'app (punto 6, 31/10). Nessuna Qt."""
from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "lora_examples"
_cache: dict = {"mtime": None, "idx": {}}


def _index() -> dict:
    f = DIR / "index.json"
    try:
        m = f.stat().st_mtime
    except OSError:
        return {}
    if _cache["mtime"] != m:
        try:
            _cache["idx"] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            _cache["idx"] = {}
        _cache["mtime"] = m
    return _cache["idx"]


def info_for(sha256: str) -> dict:
    """{page, description, images[...], confidence, source} o {}. Le voci a bassa confidenza (< 0,7) non si mostrano."""
    e = _index().get((sha256 or "").lower()) or {}
    return e if float(e.get("confidence", 0) or 0) >= 0.7 else {}


def best_image(sha256: str, allow_nsfw: bool) -> str | None:
    """Percorso di un'immagine d'esempio in cache; in modalita SFW solo quelle con nsfwLevel <= 1."""
    for im in info_for(sha256).get("images", []):
        if allow_nsfw or int(im.get("nsfw", 1)) <= 1:
            p = DIR / im.get("file", "")
            if p.is_file():
                return str(p)
    return None
