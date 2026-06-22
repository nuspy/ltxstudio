"""Contatore generazioni (per il limite della demo: cap.gen_images.params.gen_limit).

Persistito in data/usage.json. Limite 0 = illimitato.
"""
from __future__ import annotations

import json
from pathlib import Path

_FILE = Path(__file__).resolve().parent.parent / "data" / "usage.json"


def _load() -> dict:
    try:
        return json.loads(_FILE.read_text("utf-8"))
    except Exception:
        return {}


def count() -> int:
    return int(_load().get("gen_count", 0))


def increment() -> int:
    d = _load()
    d["gen_count"] = int(d.get("gen_count", 0)) + 1
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(json.dumps(d), "utf-8")
    return d["gen_count"]


def limit_reached(limit: int) -> bool:
    return bool(limit) and count() >= int(limit)
