"""Prompt Rewriter proprietario di LingBot-Video (#17) — prosa → caption JSON strutturata + auto-negative.

LingBot raccomanda un rewriter a 2 stadi (Qwen3.6-27B base → espansione; + rewriter-LoRA → JSON) seguito
da un passo Auto-Negative. È OPZIONALE e i pesi sono download-gated (grandi). Questo modulo:
  - `available()`  → True se i pesi del rewriter (base + LoRA) sono presenti E il venv LingBot esiste.
  - `rewrite(prompt, first_frame, nsfw)` → (json_prompt, negative). Se il rewriter proprietario è
    disponibile gira nel venv LingBot (workers/lingbot_rewriter.py); ALTRIMENTI fallback sul composer
    locale `core/prompt_struct.to_lingbot_json` (LLM Gemma locale). Non solleva MAI: in caso di errore
    ritorna il prompt originale.

Stessa filosofia di core/prompt_enhance.py: opzionale, fallback-safe, pilotato da una checkbox UI.
Vedi [[prompt-enhancer-and-encoders]].
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINGBOT_PY = r"E:\AI\Tools\lingbot_venv\Scripts\python.exe"
REWRITER_WORKER = str(REPO / "workers" / "lingbot_rewriter.py")


def _catalog_present(cat_id: str) -> bool:
    try:
        from core import weights_catalog as wc
        e = {x["id"]: x for x in wc.load()}.get(cat_id)
        return bool(e and wc.is_present(e))
    except Exception:                                        # noqa: BLE001
        return False


def _rewriter_dirs() -> tuple[str, str]:
    """(base_dir, lora_dir) del rewriter dal catalogo, o ('','') se assenti."""
    try:
        from core import weights_catalog as wc
        ent = {x["id"]: x for x in wc.load()}
        base = ent.get("lingbot_rewriter_base")
        lora = ent.get("lingbot_rewriter_lora")
        b = str(wc.dest_for(base)) if base and wc.is_present(base) else ""
        l = str(wc.dest_for(lora)) if lora and wc.is_present(lora) else ""
        return b, l
    except Exception:                                        # noqa: BLE001
        return "", ""


def available() -> bool:
    """True se il rewriter PROPRIETARIO è utilizzabile (venv + pesi base + LoRA presenti)."""
    if not os.path.exists(LINGBOT_PY):
        return False
    base, lora = _rewriter_dirs()
    return bool(base and lora)


def rewrite(prompt: str, *, first_frame: str = "", nsfw: bool = False, mode: str = "t2v") -> tuple[str, str]:
    """Ritorna (json_prompt, negative). Proprietario se disponibile, altrimenti composer locale.
    Non solleva mai: in errore ritorna (prompt, "")."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "", ""
    if available():
        try:
            return _rewrite_proprietary(prompt, first_frame=first_frame, mode=mode)
        except Exception as ex:                             # noqa: BLE001
            print(f"[lingbot-rewriter] proprietario fallito ({ex}) → composer locale", flush=True)
    # fallback: composer JSON locale (Gemma)
    try:
        from core import prompt_struct
        return prompt_struct.to_lingbot_json(prompt, nsfw=nsfw, first_frame=first_frame)
    except Exception:                                        # noqa: BLE001
        return prompt, ""


def _rewrite_proprietary(prompt: str, *, first_frame: str, mode: str) -> tuple[str, str]:
    """Esegue il rewriter a 2 stadi + auto-negative nel venv LingBot (subprocess)."""
    base, lora = _rewriter_dirs()
    import tempfile
    out = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    out.close()
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["LINGBOT_REWRITER_BASE"] = base
    env["LINGBOT_REWRITER_LORA"] = lora
    args = [LINGBOT_PY, "-u", REWRITER_WORKER, "--prompt", prompt, "--mode", mode, "--out", out.name]
    if first_frame:
        args += ["--first-frame", first_frame]
    subprocess.run(args, env=env, capture_output=True, text=True, timeout=600)
    with open(out.name, encoding="utf-8") as f:
        res = json.load(f)
    try:
        os.remove(out.name)
    except OSError:
        pass
    return res.get("prompt_json", prompt), res.get("negative", "")
