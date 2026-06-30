"""Driver del SERVER VIDEO PERSISTENTE LTX-2.3 (workers/ltx2_server.py, porta 8190, venv ltx2).

L'app PyQt chiama generate(); il server tiene la pipeline in RAM (offload CPU) → nessun
ricaricamento da disco tra le generazioni. Avvio automatico del server se non attivo.
ZERO ComfyUI. Pattern: submit asincrono + polling (come comfy_driver/native_driver).
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

import requests

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
REPO = str(Path(__file__).resolve().parent.parent)
BASE = "http://127.0.0.1:8190"
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_CREATE_NEW_CONSOLE = 0x00000010  # Windows
# Motore video DEFAULT = comfy-native (loader+memory-management di ComfyUI come LIBRERIA): VRAM dinamica
# = NIENTE OOM/spill sui 32GB (il pacchetto Lightricks accumula → sborza). Vedi [[native-video-engine]].
COMFY_PY = r"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"   # venv torch2.8+cu128 + SageAttention
_DM_ROOT = r"E:\AI\Models\diffusion_models"


def is_up() -> bool:
    try:
        requests.get(f"{BASE}/system_stats", timeout=2)
        return True
    except Exception:                                       # noqa: BLE001
        return False


def installed() -> bool:
    return os.path.exists(LTX2_PY) and os.path.exists(os.path.join(REPO, "workers", "ltx2_server.py"))


def ensure_server(wait_s: int = 120) -> bool:
    """Avvia il server video persistente se non è attivo (subprocess in console separata)."""
    if is_up():
        return True
    if not installed():
        return False
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.Popen([LTX2_PY, "-m", "workers.ltx2_server"], cwd=REPO, env=env,
                     creationflags=_CREATE_NEW_CONSOLE)
    for _ in range(wait_s // 2):
        if is_up():
            return True
        time.sleep(2)
    return False


def _free_gpu_servers() -> None:
    """Libera la VRAM tenuta da ALTRI server (immagini :8189) prima del video: su single-GPU un solo
    modello pesante alla volta, altrimenti il transformer 22B non entra e sborza in shared-RAM."""
    import urllib.request
    for url in ("http://127.0.0.1:8189/free",):
        try:
            urllib.request.urlopen(urllib.request.Request(url, data=b"{}",
                                   headers={"Content-Type": "application/json"}), timeout=20).read()
        except Exception:                                      # noqa: BLE001
            pass


def _to_dm_relative(model: str) -> str:
    """Converte il path modello al nome RELATIVO sotto diffusion_models (ComfyUI folder_paths)."""
    if not model:
        return "LTXVideo/v2/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
    m = model.replace("/", "\\")
    root = _DM_ROOT.rstrip("\\") + "\\"
    if m.lower().startswith(root.lower()):
        m = m[len(root):]
    return m.replace("\\", "/")


def _auto_calibrate_video_model(model_rel: str) -> str:
    """Sceglie il modello LTX-2.3 22B per FASCIA VRAM (distribuibile, niente hardcode), preferendo il
    DISTILLED (step veloci) dove entra. Solo se il default è il 22B fp8; rispetta scelte esplicite (GGUF/altro).
    Fasce: ≥30GB→fp8 distilled (24GB, qualità) · ≥16GB→Q4_K_M distilled (14GB) · <16GB→Q3_K_M dev (~10GB)."""
    import os as _os, glob
    if not model_rel or model_rel.lower().endswith(".gguf") or "22b" not in model_rel.lower():
        return model_rel
    try:
        from core import model_precision as _mp
        free = float(_mp.gpu_free_gb())
    except Exception:                                       # noqa: BLE001
        return model_rel

    def _first(*cands):
        for c in cands:
            if c and _os.path.exists(c):
                return c
        return ""
    if free >= 30.0:
        return model_rel                                    # fp8 distilled (24GB) — qualità piena
    if free >= 16.0:                                        # Q4_K_M distilled (14GB) — veloce, entra su 32GB/24GB
        g = _first(_os.path.join(_DM_ROOT, "LTXVideo", "LTX-2", "quantstack", "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"),
                   *glob.glob(_os.path.join(_DM_ROOT, "**", "*distilled*Q4_K_M.gguf"), recursive=True))
        if g:
            print(f"[auto-calibra] VRAM~{free:.0f}GB → Q4 distilled (14GB)", flush=True); return _to_dm_relative(g)
    # schede piccole → Q3_K_M dev (~10GB)
    g = _first(_os.path.join(_DM_ROOT, "ltx-2-3-22b-dev-Q3_K_M.gguf"),
               *glob.glob(_os.path.join(_DM_ROOT, "**", "*Q3_K_M.gguf"), recursive=True),
               *glob.glob(_os.path.join(_DM_ROOT, "**", "*Q4_K_M.gguf"), recursive=True))
    if g:
        print(f"[auto-calibra] VRAM~{free:.0f}GB → GGUF piccolo ({_os.path.basename(g)})", flush=True)
        return _to_dm_relative(g)
    return model_rel


def _generate_comfy_native(params: dict, timeout: int) -> dict:
    """Genera via workers/ltx_comfy_native.py (model_management ComfyUI = VRAM dinamica, no OOM).
    One-shot subprocess nel venv ComfyUI: GPU pulita ad ogni run (niente server che accumula)."""
    import json
    _free_gpu_servers()
    p = dict(params)
    p["model"] = _to_dm_relative(p.get("model", ""))
    # AUTO-CALIBRAZIONE per HARDWARE (distribuibile): fp8 (qualità piena) quando la VRAM basta; su schede
    # con poca VRAM libera auto-scelta del GGUF Q4 (footprint ridotto). NIENTE hardcode → si adatta a
    # 4090/5090/8GB/48GB. L'offload fine lo fa comunque il model_management di ComfyUI. Vedi [[dynamic-vram-tuning]].
    p["model"] = _auto_calibrate_video_model(p["model"])
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pj = os.path.join(OUTPUT_DIR, f"_vparams_{uuid.uuid4().hex[:8]}.json")
    Path(pj).write_text(json.dumps(p), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    worker = os.path.join(REPO, "workers", "ltx_comfy_native.py")
    try:
        r = subprocess.run([COMFY_PY, worker, pj], cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "timeout generazione video"}
    finally:
        try:
            os.remove(pj)
        except OSError:
            pass
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("OK:"):
            return {"status": "success", "output": [line[3:].strip()]}
    return {"status": "error", "error": out[-2000:]}


def generate(params: dict, timeout: int = 3600) -> dict:
    """Genera un video. params: prompt, image_path?(i2v), num_frames, fps, steps, guidance,
    seed, model, quant, offload, loras=[{name,strength}], output?. Ritorna {status, output:[mp4]}.

    DEFAULT = comfy-native (model_management ComfyUI, no OOM su 32GB). LTX_ENGINE=lightricks per il
    vecchio server persistente ltx-core (sborza sui 32GB)."""
    if os.environ.get("LTX_ENGINE", "comfy").lower() != "lightricks":
        return _generate_comfy_native(params, timeout)
    if not ensure_server():
        return {"status": "error", "error": "server video LTX-2 non avviato (venv ltx2 mancante?)"}
    p = dict(params)
    p.setdefault("output", os.path.join(OUTPUT_DIR, f"ltx2_{uuid.uuid4().hex[:8]}.mp4"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        jid = requests.post(f"{BASE}/submit", json={"params": p}, timeout=30).json()["prompt_id"]
    except Exception as e:                                  # noqa: BLE001
        return {"status": "error", "error": f"submit fallito: {e}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(f"{BASE}/history/{jid}", timeout=10).json()
        except Exception:                                  # noqa: BLE001
            time.sleep(3); continue
        if h.get("status") == "success":
            return {"status": "success", "output": h.get("output", [])}
        if h.get("status") == "error":
            return {"status": "error", "error": h.get("error", "")[-2000:]}
        time.sleep(3)
    return {"status": "error", "error": "timeout generazione video"}
