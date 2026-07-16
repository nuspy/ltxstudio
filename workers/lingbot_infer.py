"""Worker inference LingBot-Video — 2ª architettura video (accanto a LTX-2).

Eseguito nel venv dedicato `E:\\AI\\Tools\\lingbot_venv` (torch 2.12 cu130, diffusers 0.39,
transformers 5.8, peft 0.19) via subprocess da `core/lingbot_video_job.py`. ZERO ComfyUI, ZERO
ltx_core: usa la pipeline diffusers `LingBotVideoPipeline` col modello LOCALE.

Protocollo (identico a workers/ltx2_infer.py, così il dispatch è simmetrico):
  - legge i parametri da un file JSON (argv[1]); `output` = path mp4 da scrivere
  - genera video (T2V / I2V / TI2V), refiner opzionale, extend a chunk, mux audio opzionale
  - firma stego (#107) + watermark (#106) sul file finale
  - stampa `OK:<path>` su stdout a fine corsa

⚠️ LIVE-TEST PENDENTE: la firma esatta di `LingBotVideoPipeline.__call__` (nomi argomenti
frames/height/width/…) va confermata contro il pacchetto ufficiale. I punti da validare sono
marcati con `# VALIDATE`. Struttura completa; taratura fine al primo run reale (prassi del progetto).

Pesi: cartelle diffusers scaricate dal wizard 'Pesi' (modulo Video):
  robbyant/lingbot-video-dense-1.3b  |  robbyant/lingbot-video-moe-30b-a3b (+refiner)
  quant consumer: WaveCut ...uint4-static, ALXOPENSOURCE ...1.3b-fp8
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root repo (import core.*)

import torch

# Inferenza pura: niente autograd (come nel worker LTX). Evita grafi trattenuti = picchi RAM inutili.
torch.set_grad_enabled(False)

NEG_DEFAULT = ("worst quality, inconsistent motion, blurry, jittery, distorted, "
               "low quality, deformed, static")


# ── risoluzione modello / variante ───────────────────────────────────────────
def _resolve_model_dir(p: dict) -> tuple[str, str]:
    """Ritorna (model_dir, variant). variant ∈ {dense, moe}. Preferisce p['model'] (path esplicito),
    altrimenti risolve dal catalogo pesi in base a p['lingbot_variant']."""
    variant = str(p.get("lingbot_variant", "dense")).lower()
    m = p.get("model") or ""
    if m and os.path.isdir(m):
        low = m.lower()
        if "moe" in low or "30b" in low:
            variant = "moe"
        elif "dense" in low or "1.3b" in low:
            variant = "dense"
        return m, variant
    # risoluzione dal catalogo (dest_for) — sceglie il quant consumer se il bf16 non è presente
    try:
        from core import weights_catalog as wc
        pref = {
            "dense": ["lingbot_dense", "lingbot_dense_fp8"],
            "moe": ["lingbot_moe", "lingbot_moe_uint4"],
        }.get(variant, ["lingbot_dense"])
        entries = {e["id"]: e for e in wc.load()}
        # prima il modello pieno se presente, poi il quant consumer
        order = pref if variant != "moe" else ["lingbot_moe_uint4", "lingbot_moe"]  # MoE: quant di default
        for eid in order:
            e = entries.get(eid)
            if e and wc.is_present(e):
                return str(wc.dest_for(e)), variant
        # nessuno presente: ritorna la dest del primo (il worker fallirà con errore chiaro)
        e = entries.get(pref[0])
        if e:
            return str(wc.dest_for(e)), variant
    except Exception as ex:                                   # noqa: BLE001
        print(f"[lingbot] risoluzione catalogo fallita: {ex}", flush=True)
    raise RuntimeError(
        "Modello LingBot non trovato su disco. Scaricalo dal wizard 'Pesi' (modulo Video) — "
        "es. 'LingBot-Video Dense 1.3B' — poi riprova.")


def _free_vram_mb() -> int:
    try:
        from core import gpu_mem
        return int(gpu_mem.free_vram_mb())
    except Exception:                                        # noqa: BLE001
        return 0


# ── caricamento pipeline ─────────────────────────────────────────────────────
_PIPE_CACHE: dict = {}


def _load_pipeline(model_dir: str, variant: str, i2v: bool = False):
    """Costruisce (e cacha) la pipeline LingBot. Sceglie la CLASSE giusta (T2V vs I2V), dtype + strategia
    offload per fascia VRAM.

    La classe pipeline vive nel pacchetto `lingbot_video` (NON è registrata in diffusers né nello snapshot):
      - T2V: lingbot_video.pipeline_lingbot_video.LingBotVideoPipeline
      - I2V/TI2V: lingbot_video.pipeline_lingbot_video_i2v.LingBotVideoImageToVideoPipeline

    MoE: pesi residenti enormi (~60GB transformer + ~60GB refiner bf16) ma attivazioni per-forward piccole
    (3B attivi) → quant (uint4/gguf) + offload sequenziale. Dense 1.3B: ~12GB, resta residente in VRAM."""
    key = (model_dir, variant, i2v)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    dtype = torch.bfloat16                                    # transformer/text_encoder bf16, vae fp32
    print(f"[lingbot] carico pipeline: {model_dir} (variant={variant}, i2v={i2v}, dtype=bf16)", flush=True)
    try:
        if i2v:
            from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline as _Pipe
        else:
            from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline as _Pipe
        pipe = _Pipe.from_pretrained(model_dir, torch_dtype=dtype)
    except Exception as ex:                                   # noqa: BLE001
        # fallback: prova via diffusers/trust_remote_code (se un domani lo registrano nello snapshot)
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=dtype, trust_remote_code=True)
        except Exception:                                    # noqa: BLE001
            raise RuntimeError(f"Caricamento pipeline LingBot fallito da {model_dir}: {ex}")

    # NB VAE = Wan VAE (conv3d): tenerla in bf16 (come il resto). NON castarla a fp32 → il conv3d fp32 su
    # CUDA non ha kernel (aten::slow_conv3d_forward è solo CPU). E NON usare model_cpu_offload con questa
    # VAE → l'hook di accelerate le dà input fp32 mentre i pesi restano bf16 ("Input type float / bias
    # BFloat16"). Conclusione: LingBot va in RESIDENTE (bf16); per i video lunghi si limita la VRAM con la
    # risoluzione o il chunking, non con l'offload dei pesi.
    free = _free_vram_mb()
    force_offload = os.environ.get("NATIVE_FORCE_OFFLOAD") == "1"
    # LINGBOT_OFFLOAD: forza la strategia (utile per video lunghi dove le ATTIVAZIONI, non i pesi, saturano
    # la VRAM → i pesi residenti andrebbero in OOM; con model_cpu_offload i pesi si streammano e le
    # attivazioni hanno spazio). 'model' | 'sequential' | 'resident' | 'auto' (default).
    mode = os.environ.get("LINGBOT_OFFLOAD", "auto").lower()
    if mode == "model":
        pipe.enable_model_cpu_offload()
        print(f"[lingbot] offload model_cpu (forzato, VRAM libera {free}MB)", flush=True)
        try:
            if hasattr(pipe, "vae"):
                pipe.vae.enable_tiling()
        except Exception:                                    # noqa: BLE001
            pass
        _PIPE_CACHE[key] = pipe
        return pipe
    if mode == "resident":
        pipe = pipe.to("cuda")
        print(f"[lingbot] pipeline RESIDENTE (forzato, VRAM libera {free}MB)", flush=True)
        _PIPE_CACHE[key] = pipe
        return pipe
    heavy = variant == "moe"
    # Ingombro APPROSSIMATIVO dei pesi residenti per variante (transformer+VLM text-encoder+VAE):
    # Dense ~15GB, MoE ~60GB+ (serve comunque quant+offload). Le ATTIVAZIONI (dipendono da frame×risoluzione)
    # sono a parte: per i video lunghi si limitano col chunking, non con l'offload dei pesi.
    need_gb = 60000 if heavy else 15000
    if force_offload or (heavy and free < 30000):
        try:
            pipe.enable_sequential_cpu_offload()             # esperti MoE streammati: VRAM minima
            print(f"[lingbot] offload SEQUENZIALE (VRAM libera {free}MB, variant={variant})", flush=True)
        except Exception:                                    # noqa: BLE001
            pipe.enable_model_cpu_offload()
            print("[lingbot] offload model_cpu (fallback)", flush=True)
    elif free < need_gb:
        pipe.enable_model_cpu_offload()                      # non entra residente → offload dei moduli
        print(f"[lingbot] offload model_cpu (VRAM libera {free}MB < ~{need_gb}MB)", flush=True)
    else:
        pipe = pipe.to("cuda")                               # RESIDENTE = veloce (caso Dense su 24/32GB)
        print(f"[lingbot] pipeline RESIDENTE in VRAM (libera {free}MB)", flush=True)

    try:
        if hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()                          # VAE tiling: riduce il picco al decode
    except Exception:                                        # noqa: BLE001
        pass

    _PIPE_CACHE[key] = pipe
    return pipe


# ── generazione singola ──────────────────────────────────────────────────────
def _to_multiple(v: int, m: int = 16) -> int:
    return max(m, (int(v) // m) * m)


def _snap_frames(n: int) -> int:
    """LingBot esige num_frames = 1 oppure 4n+1 (vae_scale_factor_temporal=4)."""
    n = int(n)
    if n <= 1:
        return 1
    return ((n - 1) // 4) * 4 + 1


def _gen_once(pipe, p: dict, *, num_frames: int, image_path: str = "") -> list:
    """Un passaggio di generazione. Ritorna la lista di frame (np.uint8 HxWx3).

    VALIDATE: i nomi degli argomenti di __call__ vanno confermati contro il pacchetto ufficiale
    (qui si usano le convenzioni diffusers per pipeline video: prompt/negative_prompt/height/width/
    num_frames/num_inference_steps/guidance_scale)."""
    from PIL import Image

    w = _to_multiple(p.get("width", 832))
    h = _to_multiple(p.get("height", 480))
    kwargs = dict(
        prompt=p.get("prompt", ""),
        negative_prompt=p.get("negative", NEG_DEFAULT),
        height=h, width=w,
        num_frames=int(num_frames),
        num_inference_steps=int(p.get("steps", 40)),
        guidance_scale=float(p.get("guidance", 3.0)),
        shift=float(p.get("shift", 3.0)),                    # default ufficiale LingBot (README)
        generator=torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(int(p.get("seed", 7))),
    )
    if image_path and os.path.exists(image_path):
        kwargs["image"] = Image.open(image_path).convert("RGB")   # VALIDATE: nome arg TI2V (image / conditioning)

    result = pipe(**kwargs)
    frames = getattr(result, "frames", result)
    # LingBot (output_type="np"): frames = list[ ndarray (num_frames, H, W, C) in [0,1] ], una per batch.
    if isinstance(frames, (list, tuple)) and frames and not isinstance(frames[0], Image.Image):
        frames = frames[0]                                    # array video del batch 0: (F,H,W,C) o (F,C,H,W)
    return _to_uint8_list(frames)


def _to_uint8_list(vid) -> list:
    """Normalizza l'output della pipeline in una LISTA di frame (H,W,C) uint8.
    Gestisce: ndarray/tensor (B,F,H,W,C)/(F,H,W,C)/(F,C,H,W), o lista di frame PIL/HWC."""
    import numpy as np
    from PIL import Image
    if isinstance(vid, (list, tuple)):                        # già una lista di frame (PIL o HWC)
        return _as_uint8_frames(vid)
    if torch.is_tensor(vid):
        vid = vid.detach().cpu().float().numpy()
    vid = np.asarray(vid)
    if vid.ndim == 5:                                         # (B, ...) → batch 0
        vid = vid[0]
    if vid.ndim == 4:
        if vid.shape[-1] in (1, 3, 4):                        # (F,H,W,C)
            pass
        elif vid.shape[1] in (1, 3, 4):                       # (F,C,H,W) → (F,H,W,C)
            vid = np.transpose(vid, (0, 2, 3, 1))
        frames = [vid[i] for i in range(vid.shape[0])]
    elif vid.ndim == 3:                                       # singolo frame
        frames = [vid]
    else:
        raise RuntimeError(f"formato frame LingBot inatteso: shape {vid.shape}")
    return _as_uint8_frames(frames)


def _as_uint8_frames(frames) -> list:
    import numpy as np
    from PIL import Image
    out = []
    for f in frames:
        if isinstance(f, Image.Image):
            out.append(np.asarray(f.convert("RGB")))
        elif torch.is_tensor(f):
            a = f.detach().cpu().float().numpy()
            if a.ndim == 3 and a.shape[0] in (1, 3):         # CHW → HWC
                a = a.transpose(1, 2, 0)
            if a.max() <= 1.01:
                a = a * 255.0
            out.append(a.clip(0, 255).astype("uint8"))
        else:
            a = np.asarray(f)
            if a.dtype != np.uint8:
                if a.max() <= 1.01:
                    a = a * 255.0
                a = a.clip(0, 255).astype("uint8")
            out.append(a)
    return out


def _refine(pipe, p: dict, frames: list) -> list:
    """Passaggio refiner (solo MoE): il 2° DiT raffina il video base con pochi step (~8).
    VALIDATE: l'API del refiner (componente `refiner` della pipeline o metodo dedicato)."""
    try:
        if not hasattr(pipe, "refiner") and not hasattr(pipe, "refine"):
            print("[lingbot] refiner non disponibile nella pipeline → salto", flush=True)
            return frames
        refine_fn = getattr(pipe, "refine", None)
        if refine_fn is None:
            print("[lingbot] refiner presente ma API non nota → salto (VALIDATE)", flush=True)
            return frames
        r = refine_fn(
            frames=frames, prompt=p.get("prompt", ""), negative_prompt=p.get("negative", NEG_DEFAULT),
            num_inference_steps=int(p.get("refiner_steps", 8)),
            guidance_scale=float(p.get("refiner_guidance", 3.0)),
        )
        return _as_uint8_frames(getattr(r, "frames", r))
    except Exception as ex:                                   # noqa: BLE001
        print(f"[lingbot] refiner fallito ({ex}) → uso il video base", flush=True)
        return frames


# ── extend a chunk (autoregressivo) ──────────────────────────────────────────
def _run_chunked(pipe, p: dict, out: str, fps: int) -> str:
    """Extend autoregressivo: genera a chunk, l'ultimo frame di un chunk = start del successivo.
    Replica il pattern di workers/ltx2_infer.py:_run_chunked (LingBot non ha long-video nativo)."""
    import numpy as np
    from PIL import Image

    total = _snap_frames(p.get("num_frames", 121))
    chunk = _snap_frames(p.get("chunk_frames", 121))
    overlap = max(0, (int(p.get("chunk_overlap", 8)) // 4) * 4)   # multiplo di 4 (compat 4n+1)
    seg_dir = out + "_segs"
    os.makedirs(seg_dir, exist_ok=True)

    all_frames: list = []
    start_img = p.get("image_path", "") or ""
    produced = 0
    idx = 0
    while produced < total:
        n = _snap_frames(min(chunk, total - produced + (overlap if produced else 0)))
        n = max(5, n)                                        # minimo valido 4n+1 = 5
        print(f"[lingbot] chunk {idx} — {n} frame (prodotti {produced}/{total})", flush=True)
        fr = _gen_once(pipe, p, num_frames=n, image_path=start_img)
        if p.get("lingbot_variant") == "moe" and p.get("run_refiner"):
            fr = _refine(pipe, p, fr)
        if produced and overlap and len(all_frames) >= overlap:
            fr = _crossfade(all_frames, fr, overlap)          # blend sull'overlap
            all_frames.extend(fr[overlap:])
        else:
            all_frames.extend(fr)
        produced = len(all_frames)
        # ultimo frame → start del prossimo chunk (anti-drift: si potrebbe re-ancorare alla foto originale)
        last = all_frames[-1]
        start_path = os.path.join(seg_dir, f"anchor_{idx}.png")
        Image.fromarray(last).save(start_path)
        start_img = start_path
        idx += 1
        if idx > 200:                                        # backstop
            break

    _encode_frames(all_frames[:total], out, fps, p)
    return out


def _crossfade(prev: list, cur: list, n: int) -> list:
    """Crossfade smootherstep sui primi n frame di `cur` con gli ultimi n di `prev`."""
    import numpy as np
    tail = prev[-n:]
    out = list(cur)
    for i in range(min(n, len(cur), len(tail))):
        t = (i + 1) / (n + 1)
        w = t * t * t * (t * (t * 6 - 15) + 10)               # smootherstep
        out[i] = (tail[i].astype("float32") * (1 - w) + cur[i].astype("float32") * w).clip(0, 255).astype("uint8")
    return out


# ── encode + audio ───────────────────────────────────────────────────────────
def _encode_frames(frames: list, out: str, fps: int, p: dict) -> None:
    import numpy as np
    # GUARDIA anti-NaN: pesi corrotti → frame NaN → video verde consegnato come "successo". Meglio errore.
    for f in frames:
        arr = np.asarray(f)
        if arr.dtype.kind == "f" and (np.isnan(arr).any() or np.isinf(arr).any()):
            raise RuntimeError("Frame NaN/inf dal denoise: pesi LingBot corrotti/incompleti. Riscarica il modello.")
    import imageio
    tmp = out if not p.get("audio_path") else out + ".noaudio.mp4"
    imageio.mimsave(tmp, [np.asarray(f) for f in frames], fps=int(fps),
                    codec="libx264", quality=8, macro_block_size=16)
    if p.get("audio_path") and os.path.exists(p["audio_path"]):
        _mux_audio(tmp, p["audio_path"], out, fps=fps, nframes=len(frames),
                   start=float(p.get("audio_start_time", 0.0) or 0.0))
        try:
            os.remove(tmp)
        except OSError:
            pass


def _mux_audio(video: str, audio: str, out: str, *, fps: int, nframes: int, start: float) -> None:
    """Mux audio esterno sul video (LingBot non genera audio; l'utente può fornirne uno di sottofondo)."""
    import subprocess
    dur = nframes / float(fps)
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                        "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                        "-shortest", out], capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        # fallback: nessun audio, tieni il video muto
        import shutil
        shutil.copyfile(video, out)
        print(f"[lingbot] mux audio fallito ({r.stderr[-200:] if r.stderr else '?'}) → video muto", flush=True)


# ── entry ────────────────────────────────────────────────────────────────────
def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fps = int(p.get("fps", 24))
    model_dir, variant = _resolve_model_dir(p)
    p["lingbot_variant"] = variant
    i2v = bool(p.get("image_path"))                          # I2V/TI2V se c'è un frame di partenza
    pipe = _load_pipeline(model_dir, variant, i2v=i2v)

    if int(p.get("chunk_frames", 0)) and int(p.get("num_frames", 0)) > int(p.get("chunk_frames", 0)):
        out = _run_chunked(pipe, p, out, fps)
    else:
        nf = _snap_frames(p.get("num_frames", 121))          # LingBot: 1 o 4n+1
        frames = _gen_once(pipe, p, num_frames=nf, image_path=p.get("image_path", "") or "")
        if variant == "moe" and p.get("run_refiner"):
            frames = _refine(pipe, p, frames)
        _encode_frames(frames, out, fps, p)

    # #107 firma invisibile di provenienza (default ON)
    try:
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                  # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    # #106 watermark (gated feature_matrix)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                  # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
