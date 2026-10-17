"""H3 Video Upsampler — servizio headless (GUI e REST): ingrandisce un video con MiniMax H3, nativo.

Come flashvsr_service: subprocess in venv isolato (diffusers_next), lock GPU inter-processo, coordinamento memoria,
colonna sonora ORIGINALE rimessa con ffmpeg (il worker produce solo i fotogrammi), firma invisibile + watermark.
Nessun download: pesi assenti → WeightsMissingError con id e GB (la GUI offre «Apri tab Pesi modelli»).

API: preflight(p) -> list[str] · upscale_video(inp, out, **cfg) -> str
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from core import h3_upsample_spec as U

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "workers" / "h3_upsample_infer.py"
OUT = REPO / "output" / "h3up"


def _py() -> str:
    from core.ltx_video_job import MINIMAX_H3_PY
    return MINIMAX_H3_PY


def _basic_errors(p: dict) -> list[str]:
    errs = U.validate(p)
    if p.get("video") and not os.path.isfile(p["video"]):
        errs.append(f"Video inesistente: {p['video']}")
    if not os.path.isfile(_py()):
        errs.append("venv diffusers_next assente: usa scripts/install_diffusers_next.py")
    if not WORKER.is_file():
        errs.append(f"worker assente: {WORKER}")
    return errs


def preflight(p: dict) -> list[str]:
    """Errori che impediscono di partire (valori, file sorgente, venv, pesi). Lista vuota = si può lanciare."""
    errs = _basic_errors(p)
    if not errs:
        try:
            miss = U.missing_weights(p)
        except Exception as e:                                       # noqa: BLE001
            return [f"controllo pesi fallito: {e}"]
        if miss:
            errs.append("Pesi mancanti per H3 Video Upsampler: " + U.missing_text(miss)
                        + ". Scaricali dal tab «Pesi modelli» (nessun download automatico).")
    return errs


def _mux_audio(src_with_audio: str, video_no_audio: str, out: str) -> str:
    """La colonna sonora ORIGINALE, senza ricodificarla se il contenitore la accetta (il worker produce solo i
    fotogrammi); sorgente muta → copia."""
    import shutil
    for acodec in (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "256k"]):
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_no_audio, "-i", src_with_audio,
                            "-c:v", "copy", *acodec, "-map", "0:v:0", "-map", "1:a:0?", out], capture_output=True)
        if r.returncode == 0 and os.path.exists(out):
            return out
    shutil.copyfile(video_no_audio, out)
    return out


def upscale_video(inp: str, out: str, *, on_log=None, timeout_s: float | None = None, **cfg) -> str:
    """cfg: h3_profile (obbligatorio), h3_te, megapixels, denoise, guide, lora_strength, tile_tokens (U.build_params).
    Ritorna il path dell'mp4 (audio originale rimesso)."""
    p = U.build_params({**cfg, "video": inp, "output": out})
    errs = _basic_errors(p)
    if errs:
        raise RuntimeError("; ".join(errs))
    miss = U.missing_weights(p)
    if miss:
        from core import ensure_weights as ew
        raise ew.WeightsMissingError(miss, "H3 Video Upsampler")
    _gl = None
    try:
        from core import gpu_lock as _gl
        _gl.acquire("upscale:h3_upsampler",
                    on_wait=lambda h: on_log and on_log(f"[gpu] occupata da {h.get('tag', '?')} — attendo…"))
    except Exception:                                                # noqa: BLE001
        _gl = None
    try:
        from core import mem_coordinator as _mc
        _mc.free_and_wait("video", need_mb=24000, wait_s=30)
    except Exception:                                                # noqa: BLE001
        pass
    if timeout_s is None:
        timeout_s = float(os.environ.get("LTX_H3UP_TIMEOUT_S", "14400") or 0)
    OUT.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = str(Path(out).with_name("_h3up_noaudio_" + Path(out).name))
    wp = dict(p, output=tmp, parent_pid=os.getpid(), workdir=str(OUT))
    pj = OUT / f"_params_h3up_{int(time.time() * 1000)}.json"
    pj.write_text(json.dumps(wp, ensure_ascii=False, indent=1), "utf-8")
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.Popen([_py(), "-u", str(WORKER), str(pj)], cwd=str(REPO), env=env, stdout=subprocess.PIPE,
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
            except Exception:                                        # noqa: BLE001
                pass
        wd = threading.Timer(timeout_s, _kill)
        wd.daemon = True
        wd.start()
    tail: list[str] = []
    try:
        for line in proc.stdout:
            tail.append(line.rstrip())
            tail[:] = tail[-40:]
            if on_log:
                on_log(line.rstrip())
        proc.wait()
    finally:
        if wd:
            wd.cancel()
        if _gl:
            _gl.release()
        try:
            pj.unlink()
        except OSError:
            pass
    if killed["timeout"]:
        raise RuntimeError(f"H3 Video Upsampler: timeout ({int(timeout_s)}s) — processo terminato.")
    if proc.returncode != 0 or not os.path.exists(tmp):
        err = next((l for l in reversed(tail) if l.startswith("ERRORE:")), "")
        raise RuntimeError(f"H3 Video Upsampler fallito (exit {proc.returncode}). {err}".strip())
    res = _mux_audio(inp, tmp, out)
    try:
        os.remove(tmp)
    except OSError:
        pass
    try:                                                             # #107 firma invisibile di provenienza
        from core import stego
        if stego.enabled():
            stego.sign_file(res)
    except Exception:                                                # noqa: BLE001
        pass
    try:                                                             # #106 watermark: una volta per catena
        from core import watermark
        res = watermark.apply_to_file(res)
    except RuntimeError:
        raise
    except Exception:                                                # noqa: BLE001
        pass
    return res
