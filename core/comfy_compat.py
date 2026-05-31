"""Installazione/gestione dello shim di compatibilità ComfyUI (LTX-2.3).

Lo shim è il pacchetto `comfy_compat/ltxstudio_comfy_compat/` del repo. Viene copiato in
`<ComfyUI>/custom_nodes/ltxstudio_comfy_compat/`, dove ComfyUI lo carica all'avvio e applica
i monkey-patch per LTX-2.3 (senza toccare i file core). Vedi il pacchetto per i dettagli.

La cartella di ComfyUI viene letta da core.catalog_paths (chiave 'comfy_root').
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from core import catalog_paths

# Sorgente dello shim nel repo LTX Studio
SHIM_SRC = Path(__file__).resolve().parent.parent / "comfy_compat" / "ltxstudio_comfy_compat"
SHIM_NAME = "ltxstudio_comfy_compat"


def _comfy_root() -> Path:
    return Path(catalog_paths.load().get("comfy_root") or catalog_paths.DEFAULTS["comfy_root"])


def custom_nodes_dir() -> Path:
    return _comfy_root() / "custom_nodes"


def _installed_dir() -> Path:
    return custom_nodes_dir() / SHIM_NAME


def _read_version(version_py: Path) -> str:
    try:
        for line in version_py.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("COMPAT_VERSION"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "?"


def repo_version() -> str:
    return _read_version(SHIM_SRC / "version.py")


def status() -> dict:
    """Stato dello shim: presente nel repo? installato? versioni? comfy valido?"""
    root = _comfy_root()
    cn = custom_nodes_dir()
    dest = _installed_dir()
    installed = dest.exists() and (dest / "__init__.py").exists()
    cfg = {}
    cfg_file = dest / "compat_config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    return {
        "comfy_root": str(root),
        "comfy_valid": (root / "main.py").exists(),
        "custom_nodes": str(cn),
        "repo_shim_present": SHIM_SRC.exists(),
        "repo_version": repo_version(),
        "installed": installed,
        "installed_version": _read_version(dest / "version.py") if installed else None,
        "up_to_date": installed and _read_version(dest / "version.py") == repo_version(),
        "prompt_unlock": bool(cfg.get("prompt_unlock", True)),
    }


def install(prompt_unlock: bool | None = None) -> dict:
    """Copia/aggiorna lo shim in <ComfyUI>/custom_nodes/. Preserva/scrive compat_config.json.
    Ritorna lo status aggiornato. Solleva RuntimeError se i prerequisiti mancano."""
    if not SHIM_SRC.exists():
        raise RuntimeError(f"Pacchetto shim non trovato nel repo: {SHIM_SRC}")
    cn = custom_nodes_dir()
    if not cn.exists():
        raise RuntimeError(
            f"Cartella custom_nodes inesistente: {cn}\n"
            "Verifica «Cartella ComfyUI» nei percorsi (deve contenere main.py e custom_nodes/).")
    dest = _installed_dir()
    # preserva il flag prompt_unlock già presente, se non specificato
    prev_cfg = {}
    if (dest / "compat_config.json").exists():
        try:
            prev_cfg = json.loads((dest / "compat_config.json").read_text(encoding="utf-8")) or {}
        except Exception:
            prev_cfg = {}
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SHIM_SRC, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # scrivi compat_config.json (priorità: argomento esplicito > config precedente > default True)
    unlock = prompt_unlock if prompt_unlock is not None else bool(prev_cfg.get("prompt_unlock", True))
    (dest / "compat_config.json").write_text(
        json.dumps({"prompt_unlock": unlock}, indent=2), encoding="utf-8")
    return status()


def set_prompt_unlock(enabled: bool) -> dict:
    """Aggiorna il flag prompt_unlock dello shim installato (effettivo al riavvio di ComfyUI)."""
    dest = _installed_dir()
    if not dest.exists():
        raise RuntimeError("Shim non installato.")
    (dest / "compat_config.json").write_text(
        json.dumps({"prompt_unlock": bool(enabled)}, indent=2), encoding="utf-8")
    return status()


def uninstall() -> dict:
    """Rimuove lo shim dai custom_nodes (effettivo al riavvio di ComfyUI)."""
    dest = _installed_dir()
    if dest.exists():
        shutil.rmtree(dest)
    return status()
