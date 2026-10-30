"""MultiTalk NATIVO (Wan2.1 I2V 14B + adattatore audio) — preflight e job, SENZA ComfyUI.

Il motore vive in `workers/wan_mt/` (torch) e gira nel subprocess `workers/wan_multitalk.py` col venv consolidato
(#61: torch cu128, transformers, sageattention). Questo modulo e' solo stdlib: lo importa la GUI (nessun
`torch.cuda` nel processo dell'app). Pesi: id di catalogo `wan21-i2v-14b-480p-fp8` (le dipendenze — adattatore audio,
CLIP vision, distillazione lightx2v, T5/VAE di Wan2.1, wav2vec2 — arrivano da `deps`).

API: preflight(params=None) -> list[str] · missing_weights() -> list[str] · worker_params(gui_params) -> dict ·
     run_job(params, on_log=None, on_proc=None, timeout_s=None) -> str (path mp4) — solleva RuntimeError.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
WORKER = _ROOT / "workers" / "wan_multitalk.py"
_CONS = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")           # venv consolidato (#61)
VENV_PY = Path(os.environ.get("LTX_MULTITALK_PY", str(_CONS)))
WEIGHT_ID = "wan21-i2v-14b-480p-fp8"                                            # voce principale: tira le deps
_PATHS_PY = _ROOT / "workers" / "wan_mt" / "paths.py"

_LABELS = {"dit": "Wan2.1 I2V 14B 480P fp8", "multitalk": "adattatore audio MultiTalk", "vae": "VAE Wan2.1",
           "t5": "T5 umt5-xxl", "t5_tokenizer": "tokenizer umt5-xxl", "clip": "CLIP vision H",
           "wav2vec": "wav2vec2 cinese"}


def _paths_mod():
    """workers/wan_mt/paths.py (solo stdlib) caricato per file: stessa logica di percorsi del worker."""
    spec = importlib.util.spec_from_file_location("_wan_mt_paths", _PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def missing_weights() -> list[str]:
    pm = _paths_mod()
    p = pm.resolve_paths()
    return [_LABELS.get(k, k) for k in pm.missing_required(p)]


def weight_ids() -> list[str]:
    return [WEIGHT_ID]


def is_installed() -> bool:
    return VENV_PY.exists() and WORKER.exists() and _PATHS_PY.exists()


def preflight(params: dict | None = None) -> list[str]:
    errs = []
    if not VENV_PY.exists():
        errs.append(f"Python del venv consolidato assente: {VENV_PY}.")
    if not WORKER.exists():
        errs.append(f"Worker MultiTalk assente: {WORKER}.")
        return errs
    miss = missing_weights()
    if miss:
        errs.append("Pesi MultiTalk mancanti: " + ", ".join(miss) + ". Scaricali dal tab «Pesi modelli» "
                    f"(id {WEIGHT_ID}: le altre parti arrivano come dipendenze).")
    if params is not None:
        if not (params.get("image_path") or params.get("source_video")):
            errs.append("MultiTalk: serve una foto (Frame 1) o un video sorgente.")
        ap = params.get("audio_path")
        if not ap or not Path(str(ap)).exists():
            errs.append("MultiTalk: serve un file audio reale (parlato o canzone).")
    return errs


def worker_params(g: dict) -> dict:
    """Parametri della GUI (main_window._maybe_multitalk) -> JSON del worker. `guidance` della GUI e' quello di LTX:
    MultiTalk con la distillazione lightx2v lavora a cfg 1 (un solo passaggio per step), quindi si usa `cfg`."""
    keep = ("image_path", "source_video", "audio_path", "prompt", "negative", "width", "height", "num_frames", "fps",
            "steps", "cfg", "audio_cfg", "audio_scale", "shift", "seed", "scheduler", "blocks_to_swap", "lora",
            "lora_strength", "loras", "separate_vocals", "clip_frames", "motion_frames", "ref_anchor_strength",
            "resume", "attention", "output", "keep_chunks")
    out = {k: g[k] for k in keep if k in g and g[k] is not None}
    out.setdefault("blocks_to_swap", "auto")
    out["steps"] = int(out.get("steps", 10) or 10)
    return out


def run_job(params: dict, on_log=None, on_proc=None, timeout_s: float | None = None) -> str:
    """Esegue il worker e ritorna il path del video. Lock GPU inter-processo + liberazione degli altri motori prima
    del load (14B). Solleva RuntimeError con l'ultima riga `ERRORE:` del worker."""
    errs = preflight(params)
    if errs:
        raise RuntimeError("; ".join(errs))
    _gl = None
    try:
        from core import gpu_lock as _gl
        _gl.acquire("multitalk", on_wait=lambda h: on_log and on_log(f"[gpu] occupata da {h.get('tag', '?')} — attendo…"))
    except Exception:                                           # noqa: BLE001
        _gl = None
    try:
        from core import mem_coordinator as _mc
        _mc.free_and_wait("video", need_mb=18000)
    except Exception:                                           # noqa: BLE001
        pass
    if timeout_s is None:
        timeout_s = float(os.environ.get("LTX_MULTITALK_TIMEOUT_S", "0") or 0)
    jf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(worker_params(params), jf)
    jf.close()
    try:
        proc = subprocess.Popen([str(VENV_PY), "-u", str(WORKER), jf.name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except Exception:
        if _gl:
            _gl.release()
        raise
    if on_proc:
        on_proc(proc)                                           # #91: la GUI puo' terminarlo alla chiusura
    killed = {"t": False}
    wd = None
    if timeout_s > 0:
        def _kill():
            killed["t"] = True
            try:
                proc.kill()
            except Exception:                                   # noqa: BLE001
                pass
        wd = threading.Timer(timeout_s, _kill)
        wd.daemon = True
        wd.start()
    out_path, err_line, tail = None, "", []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            del tail[:-15]
            if line.startswith("OK:"):
                out_path = line[3:].strip()
            elif line.startswith("ERRORE:"):
                err_line = line
            if on_log and (line.startswith("[") or line.startswith("OK:") or line.startswith("ERRORE")):
                on_log(line)
        proc.wait()
    finally:
        if wd:
            wd.cancel()
        if _gl:
            _gl.release()
        try:
            os.remove(jf.name)
        except OSError:
            pass
    if killed["t"]:
        raise RuntimeError(f"MultiTalk: timeout ({int(timeout_s)}s) — processo terminato.")
    if proc.returncode == 0 and out_path and os.path.exists(out_path):
        return out_path
    raise RuntimeError(err_line or ("MultiTalk fallito (exit %s): %s" % (proc.returncode, " | ".join(tail[-3:]))))
