"""Runner FlashVSR+ (Turbo / Base) — super-risoluzione video streaming, NATIVO (niente diffsynth, niente ComfyUI).

Usa il pacchetto del repo OpenImagingLab/FlashVSR_Plus (Apache-2.0, clonato in E:\\AI\\Tools\\FlashVSR_Plus) come
LIBRERIA con pesi/input/output PARAMETRICI (la CLI ufficiale vuole i pesi sotto <repo>/ckpts). Attenzione:
  * attenzione = PyTorch SDPA («generic»): su Windows/RTX 5090 (sm120) né Block-Sparse-Attention né il backend
    Triton (richiede flash-attn 2.7.4 su Linux) sono disponibili; SDPA è il ripiego dichiarato dal progetto.
  * la CLI ufficiale ritaglia al centro il video per farlo combaciare con altezza×larghezza (multipli di 128):
    qui l'ingresso viene invece ingrandito alla scala richiesta e RIEMPITO ai bordi (replica del bordo) fino al
    multiplo di 128 successivo; a fine corsa l'uscita è ritagliata alla misura esatta → nessun taglio di scena.

Uso: python flashvsrplus_run.py <input> <output> --mode turbo|base --scale 2.0 [--seed 1]
       [--temporal-compression 4|2|1] [--decode-every 32|auto] [--color-fix] [--max-frames N]
       [--code-dir E:\\AI\\Tools\\FlashVSR_Plus] [--ckpt-root E:\\AI\\Models\\FlashVSR\\FlashVSR_plus]
       [--stats-json PATH]
Stampa a fine corsa `[flashvsrplus] STATS {json}` (tempo, s/frame, picco VRAM/RAM).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CODE_DIR = r"E:\AI\Tools\FlashVSR_Plus"
DEFAULT_CKPT_ROOT = r"E:\AI\Models\FlashVSR\FlashVSR_plus"
MAX_SCALE = 4.0


def target_dims(w: int, h: int, scale: float) -> tuple[int, int, int, int]:
    """(sw, sh, tw, th): misura esatta dell'uscita (pari) e misura di lavoro (multiplo di 128)."""
    scale = min(MAX_SCALE, max(1.0, float(scale)))
    sw = max(2, int(round(w * scale / 2.0)) * 2)
    sh = max(2, int(round(h * scale / 2.0)) * 2)
    up = lambda v: ((v + 127) // 128) * 128          # noqa: E731
    return sw, sh, up(sw), up(sh)


def model_paths(code_dir: str, ckpt_root: str, mode: str):
    from inference.loading import ModelPaths
    d = Path(ckpt_root) / mode
    return ModelPaths(
        dit=str(d / "dit.safetensors"),
        lq_projection=str(d / "lq_proj_in.safetensors"),
        context=str(Path(code_dir) / "prompt_tensor" / "posi_prompt.pth"),
        tiny_decoder=str(Path(ckpt_root) / "TCDecoder.ckpt"),
    )


def missing_files(code_dir: str, ckpt_root: str, mode: str) -> list[str]:
    p = model_paths(code_dir, ckpt_root, mode)
    return [x for x in (p.dit, p.lq_projection, p.context, p.tiny_decoder) if not os.path.exists(x)]


def install_pad_resize(sw_sh_fn):
    """Sostituisce il ritaglio centrale del repo con: resize bicubico a (sw, sh) + riempimento a (th, tw)."""
    import numpy as np
    from PIL import Image
    import inference.preprocessing.input as inp

    def _pad_resize(frames, target_height, target_width, *, resample=Image.Resampling.BICUBIC):
        out = []
        for fr in frames:
            im = Image.fromarray(fr, mode="RGB")
            sw, sh = sw_sh_fn(im.size[0], im.size[1])
            if im.size != (sw, sh):
                im = im.resize((sw, sh), resample=resample)
            a = np.asarray(im)
            if (sh, sw) != (target_height, target_width):
                a = np.pad(a, ((0, target_height - sh), (0, target_width - sw), (0, 0)), mode="edge")
            out.append(a)
        return np.ascontiguousarray(np.stack(out))

    inp.pil_center_crop_resize = _pad_resize          # usato da load_lq_video / iter_lq_video (lookup a runtime)


def vram_profile(total_gb: float, free_gb: float, width: int, height: int) -> dict:
    """Taratura DINAMICA sulla scheda (l'app e' distribuita: 5090 ottimizzata, ma niente e' presupposto).
    decode_every = quanti frame restano in RAM/VRAM per segmento di decodifica; offload della cache KV
    delle attention originali (solo Turbo) e TinyDecoder seriale per le schede piccole. Pixel grandi = piu' cauti."""
    budget = min(total_gb, free_gb + 1.0)
    mp = width * height / 1e6
    de = 32 if budget >= 28 else 24 if budget >= 20 else 16 if budget >= 14 else 8
    if mp > 6.0 and budget < 28:                      # oltre ~3K su schede da 24 GB o meno: segmenti piu' corti
        de = max(8, de // 2)
    return {"decode_every": de, "offload_original_cache": budget < 20.0,
            "tiny_decoder_mode": "serial" if budget < 20.0 else "auto",
            "cache_offload_mode": "always" if budget < 14.0 else "auto"}


def crop_and_finish(raw: str, out: str, sw: int, sh: int, tw: int, th: int) -> None:
    if (sw, sh) == (tw, th):
        shutil.move(raw, out)
        return
    ff = shutil.which("ffmpeg")
    if not ff:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ff, "-v", "error", "-y", "-i", raw, "-vf", f"crop={sw}:{sh}:0:0",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p",
                        "-c:a", "copy", "-map", "0:v:0", "-map", "0:a:0?", out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"ritaglio finale fallito: {r.stderr[-400:]}")
    os.remove(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--mode", choices=["turbo", "base"], default="turbo")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temporal-compression", type=int, choices=[1, 2, 4], default=4)
    ap.add_argument("--decode-every", default="profile",
                    help="multiplo di 4, 'auto' (carica tutto il video) o 'profile' (tarato sulla VRAM libera)")
    ap.add_argument("--color-fix", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--code-dir", default=os.environ.get("LTX_FLASHVSRPLUS_CODE", DEFAULT_CODE_DIR))
    ap.add_argument("--ckpt-root", default=os.environ.get("LTX_FLASHVSRPLUS_CKPT", DEFAULT_CKPT_ROOT))
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "auto"], help="sdpa = ripiego portabile (default)")
    ap.add_argument("--stats-json", default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, a.code_dir)
    miss = missing_files(a.code_dir, a.ckpt_root, a.mode)
    if miss:
        print("[flashvsrplus] file mancanti: " + "; ".join(miss), flush=True)
        return 3
    import cv2
    import torch
    if not torch.cuda.is_available():
        print("[flashvsrplus] CUDA non disponibile (il motore e' solo-GPU)", flush=True)
        return 4
    from inference import InferenceRequest, build_inference_pipeline
    from inference.presets import stream_config_for_mode

    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        print(f"[flashvsrplus] video non apribile: {a.input}", flush=True)
        return 2
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nsrc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    sw, sh, tw, th = target_dims(w, h, a.scale)
    install_pad_resize(lambda ww, hh: target_dims(ww, hh, a.scale)[:2])
    print(f"[flashvsrplus] {a.mode} {w}x{h} -> {sw}x{sh} (lavoro {tw}x{th}), {nsrc} frame, "
          f"tc={a.temporal_compression}, attn=sdpa", flush=True)

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    prof = vram_profile(total_b / 2 ** 30, free_b / 2 ** 30, tw, th)
    if a.decode_every == "profile":
        de = prof["decode_every"]
    else:
        de = a.decode_every if a.decode_every == "auto" else int(a.decode_every)
    print(f"[flashvsrplus] GPU {torch.cuda.get_device_name(dev)}: {total_b / 2**30:.0f} GB tot, "
          f"{free_b / 2**30:.0f} GB liberi -> {prof} (decode_every={de})", flush=True)
    config = stream_config_for_mode(a.mode, device="cuda:0", seed=a.seed, decode_every=de,
                                    temporal_compression=a.temporal_compression,
                                    hardware_profile="generic", color_correction=a.color_fix,
                                    offload_original_cache=prof["offload_original_cache"],
                                    cache_offload_mode=prof["cache_offload_mode"],
                                    tiny_decoder_mode=prof["tiny_decoder_mode"])
    t0 = time.time()
    pipe = build_inference_pipeline(config, model_paths(a.code_dir, a.ckpt_root, a.mode))
    torch.cuda.synchronize(dev)
    t_load = time.time() - t0
    print(f"[flashvsrplus] modello caricato in {t_load:.0f}s", flush=True)
    torch.cuda.reset_peak_memory_stats(dev)

    out = os.path.abspath(a.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    raw = out + ".fvp_raw.mp4"
    t1 = time.time()
    res = pipe.infer(InferenceRequest(input_path=a.input, output_path=raw, height=th, width=tw,
                                      max_frames=a.max_frames),
                     progress=lambda m: print(f"[flashvsrplus] {m}", flush=True))
    torch.cuda.synchronize(dev)
    t_inf = time.time() - t1
    vram_alloc = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    vram_res = torch.cuda.max_memory_reserved(dev) / 2 ** 20
    crop_and_finish(raw, out, sw, sh, tw, th)
    ram_peak = None
    try:
        import psutil
        mi = psutil.Process().memory_info()
        ram_peak = getattr(mi, "peak_wset", mi.rss) / 2 ** 20
    except Exception:                                   # noqa: BLE001
        pass
    frames = int(res.frames or nsrc or 0)
    stats = {"mode": a.mode, "src": [w, h], "out": [sw, sh], "work": [tw, th], "frames": frames,
             "tempo_carico_s": round(t_load, 1), "tempo_inferenza_s": round(t_inf, 1),
             "s_per_frame": round(t_inf / frames, 3) if frames else None,
             "picco_vram_alloc_mb": round(vram_alloc), "picco_vram_riservata_mb": round(vram_res),
             "picco_ram_mb": round(ram_peak) if ram_peak else None,
             "temporal_compression": a.temporal_compression, "decode_every": de, "gpu": torch.cuda.get_device_name(dev),
             "vram_totale_mb": round(total_b / 2 ** 20), "profilo": prof}
    print("[flashvsrplus] STATS " + json.dumps(stats), flush=True)
    if a.stats_json:
        Path(a.stats_json).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[flashvsrplus] OK -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
