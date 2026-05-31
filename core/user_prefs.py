"""Preferenze utente persistenti per LTX Studio (JSON, stdlib-only).

Conserva:
- `fav_loras`: set di LoRA preferite (per comfy_name).
- `lora_strength`: ultima strength impostata a mano per ogni LoRA → diventa il default.
- `fav_models`: set di modelli/checkpoint preferiti (per nome file).
- `presets`: preset video personali { nome -> dict impostazioni }.

File: E:\\AI\\ltxstudio\\data\\user_prefs.json. Scrittura atomica. Nessuna dipendenza
da Qt/ComfyUI → testabile in isolamento. Singleton di modulo (load lazy, save on write).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PREFS_PATH = Path(__file__).resolve().parent.parent / "data" / "user_prefs.json"

_DEFAULT = {
    "fav_loras": [],          # list[str] comfy_name
    "lora_strength": {},      # comfy_name -> float
    "fav_models": [],         # list[str] nome file modello
    "presets": {},            # nome -> dict impostazioni
}

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    data = dict(_DEFAULT)
    try:
        if PREFS_PATH.exists():
            loaded = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for k, v in _DEFAULT.items():
                    data[k] = loaded.get(k, v)
    except Exception:
        pass
    # normalizza: liste → set-like (manteniamo liste per il JSON)
    _cache = data
    return _cache


def _save() -> None:
    data = _load()
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, PREFS_PATH)
    except Exception:
        pass


def _norm(name: str | None) -> str:
    """Normalizza un nome file/lora per confronto stabile (case + separatori)."""
    return (name or "").replace("/", "\\").strip().lower()


# ────────────────────────────────────────────────────────────────────────────
# LoRA preferite + strength default
# ────────────────────────────────────────────────────────────────────────────
def is_fav_lora(comfy_name: str) -> bool:
    n = _norm(comfy_name)
    return any(_norm(x) == n for x in _load()["fav_loras"])


def toggle_fav_lora(comfy_name: str, strength: float | None = None) -> bool:
    """Inverte lo stato preferito. Se diventa preferita e `strength` è dato, lo salva come
    default. Ritorna il nuovo stato (True = preferita)."""
    d = _load()
    n = _norm(comfy_name)
    favs = [x for x in d["fav_loras"] if _norm(x) != n]
    now_fav = len(favs) == len(d["fav_loras"])     # non c'era → ora aggiunta
    if now_fav:
        favs.append(comfy_name)
        if strength is not None:
            d["lora_strength"][comfy_name] = float(strength)
    d["fav_loras"] = favs
    _save()
    return now_fav


def set_lora_strength(comfy_name: str, strength: float) -> None:
    """Memorizza l'ultima strength impostata a mano → nuovo default per quella LoRA."""
    d = _load()
    # rimuovi eventuali chiavi equivalenti con case/sep diversi, poi setta
    for k in [k for k in d["lora_strength"] if _norm(k) == _norm(comfy_name)]:
        d["lora_strength"].pop(k, None)
    d["lora_strength"][comfy_name] = float(strength)
    _save()


def get_lora_strength(comfy_name: str, default: float = 1.0) -> float:
    d = _load()
    n = _norm(comfy_name)
    for k, v in d["lora_strength"].items():
        if _norm(k) == n:
            try:
                return float(v)
            except Exception:
                return default
    return default


# ────────────────────────────────────────────────────────────────────────────
# Modelli / checkpoint preferiti
# ────────────────────────────────────────────────────────────────────────────
def is_fav_model(name: str) -> bool:
    n = _norm(name)
    return any(_norm(x) == n for x in _load()["fav_models"])


def toggle_fav_model(name: str) -> bool:
    d = _load()
    n = _norm(name)
    favs = [x for x in d["fav_models"] if _norm(x) != n]
    now_fav = len(favs) == len(d["fav_models"])
    if now_fav:
        favs.append(name)
    d["fav_models"] = favs
    _save()
    return now_fav


# ────────────────────────────────────────────────────────────────────────────
# Preset video personali
# ────────────────────────────────────────────────────────────────────────────
def custom_presets() -> dict:
    return dict(_load()["presets"])


def save_preset(name: str, cfg: dict) -> None:
    d = _load()
    d["presets"][name] = dict(cfg)
    _save()


def delete_preset(name: str) -> bool:
    d = _load()
    if name in d["presets"]:
        d["presets"].pop(name, None)
        _save()
        return True
    return False
