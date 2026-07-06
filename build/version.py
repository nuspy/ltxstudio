"""Versioning delle release: per-edizione R_<major>.<minor> + build_number globale.

Stato in build/versions.json. `minor` si auto-incrementa a ogni build; `--major N`
lo override (e azzera minor). `build_number` è un contatore globale monotòno (usato
dalle finestre di licenza dell'eventuale layer anti-pirateria).
"""
from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

VERSIONS = Path(__file__).resolve().parent / "versions.json"


def rand6() -> str:
    """6 caratteri alfanumerici casuali (maiuscoli+cifre), diversi a ogni chiamata."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _clean(s: str) -> str:
    """Componente-nome sicuro: sequenze non alfanumeriche → '-'."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-") or "app"


def artifact_name(app_name: str, profile: str, build_number: int, ext: str) -> str:
    """Nome file release: <App>_<Profilo>_v<NNNNN>_<RAND6>.<ext>, separatori '_'.
    Esempio: AI-Creative-Center_Uncensored_v00010_AB12E5.apk"""
    return f"{_clean(app_name)}_{_clean(profile)}_v{int(build_number):05d}_{rand6()}.{ext.lstrip('.')}"


def _load() -> dict:
    try:
        return json.loads(VERSIONS.read_text("utf-8"))
    except Exception:
        return {"build_number": 0, "editions": {}}


def _save(d: dict) -> None:
    VERSIONS.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")


def next_version(edition: str, major_override: int | None = None) -> dict:
    """Calcola e PERSISTE la prossima versione per l'edizione. Ritorna
    {major, minor, build_number, tag}."""
    d = _load()
    eds = d.setdefault("editions", {})
    cur = eds.get(edition, {"major": 1, "minor": -1})
    if major_override is not None:
        major, minor = int(major_override), 0
    else:
        major, minor = int(cur["major"]), int(cur["minor"]) + 1
    d["build_number"] = int(d.get("build_number", 0)) + 1
    eds[edition] = {"major": major, "minor": minor}
    _save(d)
    return {"major": major, "minor": minor, "build_number": d["build_number"],
            "tag": f"V_{edition}_R_{major}.{minor:02d}"}


def peek(edition: str, major_override: int | None = None) -> dict:
    """Come next_version ma SENZA persistere (per --dry-run)."""
    d = _load()
    cur = d.get("editions", {}).get(edition, {"major": 1, "minor": -1})
    if major_override is not None:
        major, minor = int(major_override), 0
    else:
        major, minor = int(cur["major"]), int(cur["minor"]) + 1
    bn = int(d.get("build_number", 0)) + 1
    return {"major": major, "minor": minor, "build_number": bn,
            "tag": f"V_{edition}_R_{major}.{minor:02d}"}
