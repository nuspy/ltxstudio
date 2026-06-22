"""Catalogo pesi del wizard: lettura, destinazione type/family, presenza, scan pre-download.

Puro Python (no torch/hf). La destinazione è derivata dalla CARTELLA BASE scelta dall'utente:
  <base_dir>/<type>/<family>/<repo_basename>      (snapshot)
  <base_dir>/<type>/<family>/<filename>           (file)
I pesi tool-specifici hanno `tool_dest` (percorso fisso del modulo, ignora base_dir).
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "data" / "weights_catalog.json"
CONFIG = Path(__file__).resolve().parents[1] / "data" / "weights_config.json"
DEFAULT_BASE = r"E:\AI\Models"

IMG_EXT = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".gguf", ".onnx")


def load() -> list[dict]:
    try:
        return json.loads(CATALOG.read_text("utf-8")).get("weights", [])
    except Exception:                                   # noqa: BLE001
        return []


# ── cartella base configurabile ──────────────────────────────────────────────
def get_base_dir() -> str:
    try:
        return json.loads(CONFIG.read_text("utf-8")).get("base_dir", DEFAULT_BASE)
    except Exception:                                   # noqa: BLE001
        return DEFAULT_BASE


def set_base_dir(path: str) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    try:
        cfg = json.loads(CONFIG.read_text("utf-8"))
    except Exception:                                   # noqa: BLE001
        pass
    cfg["base_dir"] = path
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", "utf-8")


# ── destinazione coerente type/family ────────────────────────────────────────
def repo_basename(entry: dict) -> str:
    return entry.get("repo_id", "").split("/")[-1]


def dest_for(entry: dict, base: str | None = None) -> Path:
    if entry.get("tool_dest"):
        return Path(entry["tool_dest"])
    base = base or get_base_dir()
    d = Path(base) / entry.get("type", "misc") / entry.get("family", "misc")
    if entry.get("kind") == "file":
        return d
    return d / repo_basename(entry)                     # snapshot → sottocartella col nome repo


def is_present(entry: dict, base: str | None = None) -> bool:
    d = dest_for(entry, base)
    if entry.get("kind") == "file":
        fn = entry.get("filename", "")
        return (d / fn).exists() if fn else False
    return d.is_dir() and any(d.iterdir())


# ── scan pre-download: cerca se esiste già in una directory dell'utente ───────
def scan_existing(entry: dict, search_dir: str, max_depth: int = 6) -> str | None:
    """Cerca ricorsivamente in `search_dir` una cartella/file che corrisponda agli scan_names
    del peso. Ritorna il primo path trovato, altrimenti None."""
    root = Path(search_dir)
    if not root.is_dir():
        return None
    names = [n.lower() for n in entry.get("scan_names", []) if n]
    if not names:
        names = [repo_basename(entry).lower()]
    base_len = len(root.parts)
    for p in root.rglob("*"):
        try:
            if len(p.parts) - base_len > max_depth:
                continue
            pl = p.name.lower()
            if p.is_dir() and any(n in pl for n in names) and any(p.iterdir()):
                return str(p)
            if p.is_file() and p.suffix.lower() in IMG_EXT and any(n in pl for n in names):
                return str(p)
        except Exception:                               # noqa: BLE001
            continue
    return None


def by_module() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for e in load():
        groups.setdefault(e.get("module", "Altro"), []).append(e)
    return groups
