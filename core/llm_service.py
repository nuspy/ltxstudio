"""Layer LLM unificato per l'AI Film Studio (e riusabile altrove).

Due provider selezionabili da GUI PRIMA di generare:
  - "gemma":    LLM locale Gemma-3 già nell'app (core/lyrics_llm) — 100% offline, zero dipendenze.
  - "lmstudio": server locale OpenAI-compatibile (http://localhost:<porta>/v1, default 1234,
                modificabile). Auto-launch dell'app/CLI LM Studio se non risponde; dopo connessione
                la lista modelli viene da /v1/models e l'utente sceglie quale usare.

Tutte le chiamate del Film Studio (espansione storia, breakdown scene/shot in JSON, rigenerazioni,
captioning) passano da qui. Config persistita in data/llm_config.json.

API principali:
  get_config() / set_config(**kw)
  set_provider(name) / set_lmstudio(port=, model=)
  lmstudio_up(port=None) -> bool
  ensure_lmstudio(port=None, timeout=40) -> bool      # auto-launch + attesa
  list_models(port=None) -> list[str]                  # modelli LM Studio caricati/disponibili
  available() -> bool                                  # provider corrente pronto
  complete(system, user, *, json=False, temperature=, max_tokens=) -> str
  complete_json(system, user, **kw) -> dict|list       # parse robusto del JSON
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

_CFG_PATH = Path(__file__).resolve().parents[1] / "data" / "llm_config.json"
_DEFAULT = {"provider": "gemma", "lmstudio_port": 1234, "lmstudio_model": "", "temperature": 0.8}


# ─────────────────────────── config ───────────────────────────
def get_config() -> dict:
    try:
        d = json.loads(_CFG_PATH.read_text("utf-8"))
    except Exception:                                       # noqa: BLE001
        d = {}
    return {**_DEFAULT, **d}


def set_config(**kw) -> dict:
    d = get_config()
    d.update({k: v for k, v in kw.items() if v is not None})
    _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CFG_PATH.write_text(json.dumps(d, indent=2), "utf-8")
    return d


def set_provider(name: str) -> None:
    set_config(provider=name)


def set_lmstudio(port: int | None = None, model: str | None = None) -> None:
    set_config(lmstudio_port=port, lmstudio_model=model)


# ─────────────────────────── LM Studio ───────────────────────────
def _base(port: int | None = None) -> str:
    return f"http://localhost:{port or get_config()['lmstudio_port']}/v1"


def lmstudio_up(port: int | None = None) -> bool:
    try:
        with urllib.request.urlopen(_base(port) + "/models", timeout=3) as r:
            return r.status == 200
    except Exception:                                       # noqa: BLE001
        return False


def _lms_cli() -> str | None:
    """Trova la CLI 'lms' di LM Studio (per avviare il server headless)."""
    exe = shutil.which("lms")
    if exe:
        return exe
    for c in (Path(os.environ.get("USERPROFILE", "")) / ".lmstudio" / "bin" / "lms.exe",
              Path(os.environ.get("LOCALAPPDATA", "")) / "LM Studio" / "lms.exe"):
        if c.exists():
            return str(c)
    return None


def _lms_app() -> str | None:
    """Trova l'eseguibile dell'app LM Studio (fallback se manca la CLI)."""
    for c in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio" / "LM Studio.exe",
              Path(os.environ.get("LOCALAPPDATA", "")) / "LM Studio" / "LM Studio.exe"):
        if c.exists():
            return str(c)
    return None


def ensure_lmstudio(port: int | None = None, timeout: float = 40) -> bool:
    """Se il server LM Studio non risponde, prova ad avviarlo (CLI 'lms server start', poi app) e attende."""
    if lmstudio_up(port):
        return True
    cli = _lms_cli()
    started = False
    if cli:
        try:
            subprocess.Popen([cli, "server", "start", "--port", str(port or get_config()["lmstudio_port"])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            started = True
        except Exception:                                   # noqa: BLE001
            started = False
    if not started:
        app = _lms_app()
        if app:
            try:
                subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                started = True
            except Exception:                               # noqa: BLE001
                started = False
    if not started:
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        if lmstudio_up(port):
            return True
        time.sleep(2)
    return False


def list_models(port: int | None = None) -> list[str]:
    try:
        with urllib.request.urlopen(_base(port) + "/models", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:                                       # noqa: BLE001
        return []


def _lmstudio_chat(system: str, user: str, *, want_json: bool, temperature: float, max_tokens: int) -> str:
    cfg = get_config()
    body = {
        "model": cfg.get("lmstudio_model") or (list_models() or [""])[0],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": float(temperature), "max_tokens": int(max_tokens), "stream": False,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(_base() + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"].strip()


# ─────────────────────────── API unificata ───────────────────────────
def available() -> bool:
    cfg = get_config()
    if cfg["provider"] == "lmstudio":
        return lmstudio_up()
    from core import lyrics_llm
    return lyrics_llm.available()


def unload() -> None:
    """Libera l'LLM dalla VRAM (provider gemma → scarica Gemma-3). DA CHIAMARE prima di generare
    IMMAGINI/VIDEO: altrimenti Gemma (~24GB con device_map=auto) resta in VRAM nel processo GUI e i
    subprocess pesanti (Krea2, LTX) vanno in contesa → OOM/crash. No-op per il provider lmstudio (out-of-process)."""
    try:
        if get_config().get("provider") == "gemma":
            from core import lyrics_llm
            lyrics_llm.unload()
    except Exception:                                          # noqa: BLE001
        pass


def complete(system: str, user: str, *, json: bool = False, temperature: float | None = None,
             max_tokens: int = 1200) -> str:
    """Completion con il provider corrente. `json=True` chiede output JSON (best-effort)."""
    cfg = get_config()
    temp = cfg["temperature"] if temperature is None else temperature
    if cfg["provider"] == "lmstudio":
        if not ensure_lmstudio():
            raise RuntimeError("LM Studio non raggiungibile (avvialo o cambia provider/porta).")
        return _lmstudio_chat(system, user, want_json=json, temperature=temp, max_tokens=max_tokens)
    # default: Gemma locale via lyrics_llm (riusa il modello già caricato)
    import torch
    from core import lyrics_llm
    model, proc = lyrics_llm._load()
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=True,
                             temperature=float(temp), top_p=0.95)
    return proc.decode(out[0][n_in:], skip_special_tokens=True).strip()


def complete_json(system: str, user: str, *, retries: int = 2, **kw):
    """Come complete(json=True) ma fa il parse robusto (estrae l'oggetto/array anche se circondato
    da testo). Se il parse fallisce (tipico: output TRONCATO da max_tokens su breakdown lunghi)
    RIPROVA fino a `retries` volte alzando il budget del 50% a giro — l'LLM è stocastico e il
    film automatico non deve morire per un JSON monco (successo al primo run, crash al secondo)."""
    last = ""
    mt = int(kw.pop("max_tokens", 1200))
    for attempt in range(1 + max(0, retries)):
        txt = complete(system, user, json=True, max_tokens=mt, **kw)
        last = txt
        for pat in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pat, txt, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:                           # noqa: BLE001
                    continue
        mt = int(mt * 1.5)
        print(f"[llm] JSON non valido (tentativo {attempt + 1}, probabile troncamento) → "
              f"riprovo con max_tokens={mt}", flush=True)
    raise ValueError("output LLM non è JSON valido:\n" + last[:500])
