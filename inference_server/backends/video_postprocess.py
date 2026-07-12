"""Post-process VIDEO nativo — sostituisce ComfyUI/VHS per l'enhance video.

upscale per-frame (ESRGAN/spandrel, riusa image_upscale): estrae i frame con ffmpeg,
applica il modello a ogni frame, ricompone il video preservando FPS e AUDIO.
Indipendente dal motore di generazione (lavora su un video esistente) → testabile senza Gemma.
RIFE (FPS) / SeedVR2 / face-restore: handler separati (richiedono i loro pesi).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

VRAM_MB = 4000


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or r"C:\Users\atain\scoop\shims\ffmpeg.exe"


def _ffprobe() -> str:
    return shutil.which("ffprobe") or r"C:\Users\atain\scoop\shims\ffprobe.exe"


def _fps(video: str) -> float:
    try:
        out = subprocess.run([_ffprobe(), "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate", video],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        n, d = out.split("/") if "/" in out else (out, "1")
        return float(n) / float(d) if float(d) else 24.0
    except Exception:                                       # noqa: BLE001
        return 24.0


def _frames_io(video: str):
    """Estrae i frame di un video in una tmpdir; ritorna (tmpdir, lista_path, fps, ffmpeg)."""
    import tempfile
    td = tempfile.mkdtemp(prefix="vpp_")
    ind = os.path.join(td, "in"); os.makedirs(ind)
    ff = _ffmpeg()
    subprocess.run([ff, "-y", "-i", video, os.path.join(ind, "f_%06d.png")],
                   capture_output=True, timeout=600)
    frames = sorted(Path(ind).glob("f_*.png"))
    return td, ind, frames, _fps(video), ff


def _reassemble(outd: str, fps: float, video_audio_src: str, out_path: str) -> None:
    ff = _ffmpeg()
    subprocess.run([ff, "-y", "-framerate", f"{fps}", "-i", os.path.join(outd, "f_%06d.png"),
                    "-i", video_audio_src, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out_path],
                   capture_output=True, timeout=900)


def run_face(params: dict, mm, output_dir: str) -> dict:
    """Restauro VOLTI video per-frame (GFPGAN) — niente ComfyUI. Richiede `pip install gfpgan`
    nel venv nativo + modello .pth (params['model']). Codice pronto, test a lib+modello presenti."""
    import shutil
    from PIL import Image
    import numpy as np
    try:
        from gfpgan import GFPGANer
    except Exception as e:                                  # noqa: BLE001
        return {"status": "error", "error": f"GFPGAN non installato nel venv nativo ({e}). "
                "pip install gfpgan; poi riprova."}
    video = params.get("image_path") or params.get("video")
    if not video or not os.path.exists(video):
        return {"status": "error", "error": f"video non trovato: {video}"}
    model = params.get("model") or r"E:\AI\Models\facerestore_models\GFPGANv1.4.pth"
    # la GUI passa il NOME file (es. 'codeformer-v0.1.0.pth'), non il path: risolvi nelle dir note
    if not os.path.exists(str(model)):
        _name = os.path.basename(str(model))
        for _d in (r"E:\AI\Models\facerestore_models",
                   r"E:\AI\Comfy_Nexus.bak\ComfyUI\models\facerestore_models"):
            _cand = os.path.join(_d, _name)
            if os.path.exists(_cand):
                model = _cand
                break
        else:
            return {"status": "error", "error": f"modello restauro volti non trovato: {_name} "
                    r"(atteso in E:\AI\Models\facerestore_models)"}
    # ARCH corretto dal nome modello: CodeFormer ha la 'fidelity' (weight), GFPGAN no. Prima si caricava
    # SEMPRE come GFPGAN 'clean' → un checkpoint CodeFormer sbagliava arch e la fidelity era ignorata.
    _fid = float(params.get("fidelity", 0.5) or 0.5)
    _is_cf = "codeformer" in os.path.basename(str(model)).lower()
    _arch = "CodeFormer" if _is_cf else "clean"
    try:
        restorer = GFPGANer(model_path=model, upscale=1, arch=_arch, channel_multiplier=2)
    except Exception:                                        # noqa: BLE001 — gfpgan vecchio senza arch CodeFormer
        restorer = GFPGANer(model_path=model, upscale=1, arch="clean", channel_multiplier=2)
    td, ind, frames, fps, _ = _frames_io(video)
    try:
        if not frames:
            return {"status": "error", "error": "estrazione frame fallita"}
        outd = os.path.join(td, "out"); os.makedirs(outd)
        for fp in frames:
            arr = np.array(Image.open(fp).convert("RGB"))[:, :, ::-1]   # BGR per gfpgan
            try:                                             # 'weight' = fidelity CodeFormer (ignorato da GFPGAN)
                _, _, restored = restorer.enhance(arr, has_aligned=False, paste_back=True, weight=_fid)
            except TypeError:                                # gfpgan senza il parametro weight
                _, _, restored = restorer.enhance(arr, has_aligned=False, paste_back=True)
            Image.fromarray(restored[:, :, ::-1]).save(os.path.join(outd, fp.name))
        os.makedirs(output_dir, exist_ok=True)
        out_path = str(Path(output_dir) / f"native_faces_{params.get('seed', 0)}.mp4")
        _reassemble(outd, fps, video, out_path)
        try:                                                 # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark
            from core import stego
            if stego.enabled():
                stego.sign_file(out_path)
        except Exception:                                    # noqa: BLE001
            pass
        try:                                                 # #106 watermark obbligatorio (gated feature_matrix)
            from core import watermark
            out_path = watermark.apply_to_file(out_path)     # unica applicazione: output finale del job
        except RuntimeError:
            return {"status": "error", "error": "watermark obbligatorio non applicabile"}
        except Exception:                                    # noqa: BLE001
            pass
        return {"status": "success", "output": [out_path]}
    finally:
        shutil.rmtree(td, ignore_errors=True)


def run_fps(params: dict, mm, output_dir: str) -> dict:
    """Aumento FPS video — interpolazione movimento ffmpeg (minterpolate). Niente ComfyUI/modelli.
    RIFE (qualità superiore via rete neurale) = upgrade futuro con modello dedicato."""
    video = params.get("image_path") or params.get("video")
    if not video or not os.path.exists(video):
        return {"status": "error", "error": f"video non trovato: {video}"}
    target = int(params.get("target_fps", 60))
    os.makedirs(output_dir, exist_ok=True)
    out_path = str(Path(output_dir) / f"native_fps{target}_{params.get('seed', 0)}.mp4")
    r = subprocess.run([_ffmpeg(), "-y", "-i", video, "-filter:v",
                        f"minterpolate=fps={target}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir",
                        "-c:a", "copy", out_path], capture_output=True, timeout=1800)
    if not os.path.exists(out_path):
        return {"status": "error", "error": f"ffmpeg minterpolate fallito: {r.stderr[-400:]!r}"}
    try:                                                     # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(out_path)
    except Exception:                                        # noqa: BLE001
        pass
    try:                                                     # #106 watermark obbligatorio (gated feature_matrix)
        from core import watermark
        out_path = watermark.apply_to_file(out_path)         # unica applicazione: output finale del job
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                        # noqa: BLE001
        pass
    return {"status": "success", "output": [out_path]}


def run(params: dict, mm, output_dir: str) -> dict:
    """params: image_path (= video), model (ESRGAN), target_scale, seed."""
    import numpy as np
    import torch
    from PIL import Image
    from inference_server.backends.image_upscale import _resolve, _load, _upscale_tiled, _run_once

    video = params.get("image_path") or params.get("video")   # robusto a entrambe le chiavi
    if not video or not os.path.exists(video):
        return {"status": "error", "error": f"video non trovato: {video}"}
    mp = _resolve(params["model"])
    if not os.path.exists(mp):
        return {"status": "error", "error": f"modello upscale non trovato: {mp}"}
    key = f"upscale:{mp}"
    mm.register(key, lambda: _load(mp))
    model = mm.ensure(key, need_mb=VRAM_MB)
    fps = _fps(video)
    ff = _ffmpeg()

    with tempfile.TemporaryDirectory() as td:
        ind = os.path.join(td, "in"); outd = os.path.join(td, "out")
        os.makedirs(ind); os.makedirs(outd)
        # 1) estrai i frame
        subprocess.run([ff, "-y", "-i", video, os.path.join(ind, "f_%06d.png")],
                       capture_output=True, timeout=600)
        frames = sorted(Path(ind).glob("f_*.png"))
        if not frames:
            return {"status": "error", "error": "estrazione frame fallita"}
        # 2) upscale per-frame → poi porta al FATTORE RICHIESTO (target_scale). Prima il target_scale era
        # ignorato: l'ESRGAN esce alla sua scala nativa (es. 4×) anche se l'utente chiedeva 2× → output
        # sempre 4× (come nel path IMMAGINE, image_upscale.run() righe 78-80, che invece lo applica).
        _tgt = params.get("target_scale") or params.get("scale")   # robusto a entrambe le chiavi
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            if torch.cuda.is_available():
                t = t.cuda()
            out = _upscale_tiled(model, t) if max(img.size) > 1024 else _run_once(model, t)
            arr = (out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
            res = Image.fromarray(arr)
            if _tgt:                                          # scala finale = dimensioni ORIGINALI × target_scale
                res = res.resize((int(img.width * float(_tgt)), int(img.height * float(_tgt))), Image.LANCZOS)
            res.save(os.path.join(outd, fp.name))
        # 3) ricomponi video + riporta l'audio originale
        os.makedirs(output_dir, exist_ok=True)
        path = str(Path(output_dir) / f"native_videoup_{params.get('seed', 0)}.mp4")
        subprocess.run([ff, "-y", "-framerate", f"{fps}", "-i", os.path.join(outd, "f_%06d.png"),
                        "-i", video, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", path],
                       capture_output=True, timeout=900)
    try:                                                     # #107 firma invisibile di provenienza (default ON) — PRIMA del watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                        # noqa: BLE001
        pass
    try:                                                     # #106 watermark obbligatorio (gated feature_matrix)
        from core import watermark
        path = watermark.apply_to_file(path)                 # unica applicazione: output finale del job
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                        # noqa: BLE001
        pass
    return {"status": "success", "output": [path]}
