"""FlashVSR+ (Turbo / Base) super-risoluzione VIDEO (fino a x4, streaming) — STANDALONE, niente ComfyUI.

Pacchetto OpenImagingLab/FlashVSR_Plus (Apache-2.0; clonato in E:\\AI\\Tools\\FlashVSR_Plus con
scripts/install_flashvsrplus.py) usato come LIBRERIA dal runner `workers/flashvsrplus_run.py` in un
subprocess nel venv consolidato (#61): nessuna dipendenza da diffsynth. Pesi dal registro unico
(flashvsr-plus-turbo / flashvsr-plus-base + TinyDecoder), cartella E:\\AI\\Models\\FlashVSR\\FlashVSR_plus.

Rispetto a FlashVSR v1.1 (core/flashvsr_service.py): attenzione ibrida (fino a 2x piu' veloce nel DiT,
-45% di picco di memoria), nessun tiling a 4K su 24 GB, due preset (Turbo = veloce, Base = qualita') e
«compressione temporale» regolabile (4 default; 2 o 1 per i movimenti veloci, a costo di tempo).
Attenzione: SDPA (ripiego portabile): su Windows/RTX 50 non ci sono i backend sparsi ufficiali. La VRAM e'
tarata dinamicamente dal runner (segmenti di decodifica, offload cache KV, TinyDecoder seriale).

API: is_installed() -> bool · preflight(mode) -> list[str] · upscale_video(inp, out, *, mode="turbo",
     scale=2.0, seed=1, temporal_compression=4, decode_every=None, color_fix=False, max_frames=None,
     timeout_s=None, on_log=None) -> str
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

CODE_DIR = Path(os.environ.get("LTX_FLASHVSRPLUS_CODE", r"E:\AI\Tools\FlashVSR_Plus"))
_CONS = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")   # venv consolidato (#61)
VENV_PY = Path(os.environ.get("LTX_FLASHVSRPLUS_PY", str(_CONS)))
RUNNER = Path(__file__).resolve().parents[1] / "workers" / "flashvsrplus_run.py"
REPO = "https://github.com/OpenImagingLab/FlashVSR_Plus"
MODES = ("turbo", "base")
_IDS = {"turbo": ["flashvsr-plus-turbo", "flashvsr-plus-turbo-lq", "flashvsr-plus-tc"],
        "base": ["flashvsr-plus-base", "flashvsr-plus-base-lq", "flashvsr-plus-tc"]}
_FALLBACK_ROOT = Path(r"E:\AI\Models\FlashVSR\FlashVSR_plus")


def weight_ids(mode: str) -> list[str]:
    """Id di catalogo dei pesi di una variante (per offerta di download / controllo presenza)."""
    return list(_IDS[mode if mode in _IDS else "turbo"])


def ckpt_root() -> Path:
    """Cartella radice dei pesi (contiene turbo/, base/ e TCDecoder.ckpt), derivata dal registro unico."""
    try:
        from core import model_paths as _mp
        p = _mp.path_of("flashvsr-plus-tc")
        if p:
            return Path(str(p)).parent
    except Exception:                                      # noqa: BLE001
        pass
    return _FALLBACK_ROOT


def missing_weights(mode: str = "turbo") -> list[str]:
    mode = mode if mode in MODES else "turbo"
    root = ckpt_root()
    need = [root / mode / "dit.safetensors", root / mode / "lq_proj_in.safetensors", root / "TCDecoder.ckpt"]
    return [str(p) for p in need if not p.exists()]


def is_installed() -> bool:
    return VENV_PY.exists() and (CODE_DIR / "inference" / "run_stream.py").exists() and RUNNER.exists()


def preflight(mode: str = "turbo") -> list[str]:
    errs = []
    if not (CODE_DIR / "inference" / "run_stream.py").exists():
        errs.append(f"FlashVSR+ assente: clona {REPO} in {CODE_DIR} (scripts/install_flashvsrplus.py).")
    elif not (CODE_DIR / "prompt_tensor" / "posi_prompt.pth").exists():
        errs.append(f"FlashVSR+: manca prompt_tensor/posi_prompt.pth in {CODE_DIR} (clone incompleto).")
    if not VENV_PY.exists():
        errs.append(f"Python del venv consolidato assente: {VENV_PY}.")
    miss = missing_weights(mode)
    if miss:
        errs.append("Pesi FlashVSR+ (" + mode + ") assenti: " + "; ".join(miss) + ".")
    return errs


def upscale_video(inp: str, out: str, *, mode: str = "turbo", scale: float = 2.0,
                  target_scale: float | None = None, seed: int = 1, temporal_compression: int = 4,
                  decode_every=None, color_fix: bool = False, max_frames: int | None = None,
                  timeout_s: float | None = None, on_log=None) -> str:
    """Super-risolve un video con FlashVSR+. La scala e' clampata a x4 (nativa); l'uscita ha la misura
    esatta sorgente x scala (ingresso riempito ai bordi fino al multiplo di 128 e ritagliato a fine corsa,
    nessun taglio di scena). `decode_every`: None = tarato dal runner sulla VRAM libera. Ritorna l'mp4 (audio
    del sorgente conservato, firma invisibile + watermark come negli altri motori)."""
    mode = mode if mode in MODES else "turbo"
    errs = preflight(mode)
    if errs:
        raise RuntimeError("; ".join(errs))
    run_scale = min(4.0, max(1.0, float(target_scale or scale or 2.0)))
    tc = int(temporal_compression) if int(temporal_compression) in (1, 2, 4) else 4
    # LOCK GPU inter-processo PRIMA del free (come FlashVSR v1.1 / SeedVR2); rilascio nel finally.
    _gl = None
    try:
        from core import gpu_lock as _gl
        _gl.acquire("upscale:flashvsr_plus",
                    on_wait=lambda h: on_log and on_log(f"[gpu] occupata da {h.get('tag', '?')} — attendo…"))
    except Exception:                                       # noqa: BLE001
        _gl = None
    try:
        from core import mem_coordinator as _mc
        _mc.free_and_wait("video", need_mb=10000, wait_s=30)
    except Exception:                                       # noqa: BLE001
        pass
    if timeout_s is None:
        timeout_s = float(os.environ.get("LTX_FLASHVSRPLUS_TIMEOUT_S", "3600") or 0)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = str(Path(out).with_name("_fvsrplus_" + Path(out).name))
    cmd = [str(VENV_PY), "-I", "-u", str(RUNNER), inp, tmp, "--mode", mode, "--scale", f"{run_scale:g}",
           "--seed", str(int(seed)), "--temporal-compression", str(tc),
           "--decode-every", "profile" if decode_every in (None, "", "profile") else str(decode_every),
           "--code-dir", str(CODE_DIR), "--ckpt-root", str(ckpt_root())]
    if color_fix:
        cmd.append("--color-fix")
    if max_frames:
        cmd += ["--max-frames", str(int(max_frames))]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=str(RUNNER.parent), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    except Exception:
        if _gl:
            _gl.release()
        raise
    killed = {"timeout": False}
    wd = None
    if timeout_s > 0:
        import threading

        def _kill():
            killed["timeout"] = True
            try:
                proc.kill()
            except Exception:                              # noqa: BLE001
                pass
        wd = threading.Timer(timeout_s, _kill)
        wd.daemon = True
        wd.start()
    tail: list[str] = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            del tail[:-12]
            if on_log:
                on_log(line)
        proc.wait()
    finally:
        if wd:
            wd.cancel()
        if _gl:
            _gl.release()
    if killed["timeout"]:
        raise RuntimeError(f"FlashVSR+: timeout ({int(timeout_s)}s) — processo terminato.")
    if proc.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"FlashVSR+ fallito (exit {proc.returncode}): " + " | ".join(tail[-3:]))
    os.replace(tmp, out)
    try:  # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark visibile
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception:                                      # noqa: BLE001
        pass
    # #106 watermark obbligatorio a fine job (RuntimeError = errore del job, fail-closed)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except RuntimeError:
        raise
    except Exception:                                      # noqa: BLE001
        pass
    return out
