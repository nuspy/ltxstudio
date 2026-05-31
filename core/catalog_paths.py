"""Configurazione dei percorsi per gli strumenti del catalogo AI (E:\\AI\\dictionary\\tools).

Sorgente unica di verità (lato LTX Studio) per:
- tools_dir    : cartella degli script del catalogo (place_files, scan, enrich, build_html…)
- models       : radice modelli (dove place_files smista per tipo/architettura)
- workflows    : radice workflow ComfyUI (.json/.png)
- dict_root    : radice del dizionario (catalog.json, index.html, entries/, assets/)
- inbox        : cartella di staging dei file in ingresso
- hashes_jsonl : cache hash (opzionale) prodotta dal reorg, velocizza lo scan

Al salvataggio sincronizza `<tools_dir>/paths.json`, che `catalog_lib.py` legge per far
funzionare gli script (anche da CLI) con i percorsi scelti. I default coincidono con
l'installazione attuale, quindi senza configurazione tutto continua a funzionare.

Nota installer: l'installer futuro userà `suggest_from_comfy()` per proporre percorsi in
base all'installazione di ComfyUI/app, ma la decisione finale resta all'utente.
"""
from __future__ import annotations

import json
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent          # E:\AI\ltxstudio
CONFIG_PATH = APP_ROOT / "data" / "catalog_paths.json"

# Default = installazione attuale.
DEFAULTS: dict[str, str] = {
    "comfy_root":   r"E:\AI\Comfy_Nexus\ComfyUI",
    "tools_dir":    r"E:\AI\dictionary\tools",
    "models":       r"E:\AI\Models",
    "workflows":    r"E:\AI\workflows",
    "dict_root":    r"E:\AI\dictionary",
    "inbox":        r"E:\AI\inbox",
    "hashes_jsonl": r"E:\AI\_reorg_log\hashes.jsonl",
}

# Chiavi scritte in <tools_dir>/paths.json (lette da catalog_lib._resolve_paths).
_TOOL_KEYS = ("models", "workflows", "dict_root", "inbox", "hashes_jsonl")

# Etichetta breve + descrizione, per la UI. Ordine di visualizzazione.
LABELS: list[tuple[str, str, str]] = [
    ("comfy_root",   "Cartella ComfyUI",
     "Radice di ComfyUI (contiene main.py e custom_nodes/). Usata per installare la patch "
     "di compatibilità LTX-2.3."),
    ("tools_dir",    "Cartella strumenti",
     "Script del catalogo: place_files, scan, enrich_civitai, build_html…"),
    ("inbox",        "Inbox (staging)",
     "Cartella dove i file in ingresso attendono di essere smistati da «Sposta»."),
    ("models",       "Radice modelli",
     "Dove i modelli vengono smistati per tipo/architettura (LoRA, Checkpoints, VAE, …)."),
    ("workflows",    "Radice workflow",
     "Dove finiscono i workflow ComfyUI (.json / .png)."),
    ("dict_root",    "Radice dizionario",
     "catalog.json, index.html, pagine entries/ e assets/ del catalogo."),
    ("hashes_jsonl", "Cache hash (opzionale)",
     "File hashes.jsonl del reorg: se presente velocizza lo scan, altrimenti gli hash "
     "vengono calcolati al volo."),
]


def load() -> dict:
    """Config corrente: default sovrascritti da quanto salvato in CONFIG_PATH."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS and v})
        except Exception:
            pass
    return cfg


def save(cfg: dict) -> Path:
    """Salva la config di LTX Studio e sincronizza <tools_dir>/paths.json. Ritorna il path
    del paths.json scritto (o None se la cartella tools non esiste)."""
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in cfg.items() if k in DEFAULTS and v})
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return _sync_tools_paths(merged)


def _sync_tools_paths(cfg: dict) -> Path | None:
    tools = Path(cfg.get("tools_dir") or DEFAULTS["tools_dir"])
    if not tools.exists():
        return None
    payload = {k: cfg[k] for k in _TOOL_KEYS if cfg.get(k)}
    out = tools / "paths.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def tools_dir() -> Path:
    return Path(load()["tools_dir"])


def is_default(cfg: dict) -> bool:
    return all((cfg.get(k) or DEFAULTS[k]) == DEFAULTS[k] for k in DEFAULTS)


# ---------------------------------------------------------------- suggerimenti installer
def detect_comfy_root() -> Path | None:
    """Best-effort: individua la radice di ComfyUI cercando main.py nei percorsi tipici."""
    candidates = [
        r"E:\AI\Comfy_Nexus\ComfyUI",
        r"C:\ComfyUI", r"D:\ComfyUI", r"E:\ComfyUI",
        str(Path.home() / "ComfyUI"),
    ]
    for c in candidates:
        if (Path(c) / "main.py").exists():
            return Path(c)
    return None


def suggest_from_comfy(comfy_root: str | Path) -> dict:
    """Percorsi suggeriti in base a un'installazione ComfyUI. L'utente decide se accettarli.
    Restituisce solo le chiavi che ha senso derivare da ComfyUI (models/workflows)."""
    root = Path(comfy_root)
    return {
        "comfy_root": str(root),
        "models": str(root / "models"),
        "workflows": str(root / "user" / "default" / "workflows"),
    }
