"""Worker MultiTalk (Wan2.1) NATIVO via WanVideoWrapper-as-library — lip-sync su audio REALE.

foto + audio.mp3 -> video con bocca/movimento sincronizzati all'audio. Usa i nodi kijai
WanVideoWrapper importati come libreria (shim `server` per girare headless, niente app ComfyUI),
con WanVideoContextOptions per i video lunghi (memory-bounded). Pesi tutti locali.

Param JSON (argv[1]): image_path, audio_path, prompt, negative, width, height, num_frames, fps,
steps, cfg, shift, seed, blocks_to_swap, lora_strength, context (bool), output.
"""
from __future__ import annotations
import json, os, sys, types, subprocess, importlib

COMFY = r"E:\AI\Comfy_Nexus\ComfyUI"


def _shim_and_import():
    os.chdir(COMFY)
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    # modulo `server` fittizio: evita app.frontend_management -> utils.install_util in headless
    fs = types.ModuleType("server")

    class _PS:
        instance = None
        def __init__(self, *a, **k): pass
        def add_routes(self, *a, **k): pass
        def send_sync(self, *a, **k): pass
        def send_progress_text(self, *a, **k): pass
    fs.PromptServer = _PS
    _PS.instance = _PS()
    sys.modules["server"] = fs

    import folder_paths
    M = r"E:\AI\Models"
    # registra i path locali dove stanno i pesi Wan/MultiTalk
    add = folder_paths.add_model_folder_path
    for sub in ("diffusion_models\\Wan2.1", "diffusion_models\\Wan2.2",
                "Checkpoints\\Wan", "Checkpoints\\Wan2.1", "Checkpoints\\Wan2.2"):
        add("diffusion_models", os.path.join(M, sub))
    add("vae", os.path.join(M, "VAE", "Wan"))
    add("vae", os.path.join(M, "VAE"))
    add("text_encoders", os.path.join(M, "TextEncoders"))
    add("clip_vision", os.path.join(M, "VisionEncoders"))
    for sub in ("LoRA\\Wan2.1", "LoRA\\Wan", "LoRA\\Wan2.2"):
        add("loras", os.path.join(M, sub))

    sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
    wm = importlib.import_module("ComfyUI-WanVideoWrapper")
    return wm.NODE_CLASS_MAPPINGS, folder_paths


def _v(x):
    """Estrae il valore utile dal ritorno di un nodo (tuple / NodeOutput)."""
    if hasattr(x, "result"):
        x = x.result
    return x


def _first(x):
    x = _v(x)
    return x[0] if isinstance(x, (tuple, list)) else x


def _call(node_cls, **kwargs):
    """Istanzia il nodo, chiama il suo FUNCTION iniettando gli hidden input (unique_id/prompt/
    extra_pnginfo) se la firma li richiede, e ritorna il primo valore del risultato."""
    import inspect
    inst = node_cls()
    fn = getattr(inst, node_cls.FUNCTION)
    params = inspect.signature(fn).parameters
    for hid, val in (("unique_id", "0"), ("prompt", {}), ("extra_pnginfo", {})):
        if hid in params and hid not in kwargs:
            kwargs[hid] = val
    return _first(fn(**kwargs))


def _load_audio(path):
    import torchaudio
    wav, sr = torchaudio.load(path)              # [C, N]
    return {"waveform": wav.unsqueeze(0), "sample_rate": sr}   # [1, C, N]


def _first_frame(video_path):
    """Estrae il primo frame di un video come PNG (per applicare MultiTalk a un video esistente)."""
    out = os.path.splitext(video_path)[0] + "_frame0.png"
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _vocals_audio(path):
    """Isola la VOCE dalla traccia (demucs htdemucs via CLI, robusto a ogni versione): MultiTalk è
    addestrato sul parlato/canto, la musica di sottofondo peggiora il lip-sync. Ritorna un AUDIO dict
    coi soli vocals (cache su disco)."""
    out = os.path.join(os.path.dirname(path) or ".", "_demucs")
    name = os.path.splitext(os.path.basename(path))[0]
    voc = os.path.join(out, "htdemucs", name, "vocals.wav")
    if not os.path.exists(voc):
        os.makedirs(out, exist_ok=True)
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", out, path], check=True)
    return _load_audio(voc)


def _slice_audio(audio: dict, t0: float, t1: float) -> dict:
    """Estrae la finestra [t0,t1] secondi da un AUDIO dict (per i segmenti dei video lunghi)."""
    sr = int(audio["sample_rate"]); w = audio["waveform"]
    return {"waveform": w[:, :, int(t0 * sr):int(t1 * sr)], "sample_rate": sr}


def _auto_blocks_to_swap(fp, model_name, total_blocks=40, reserve_gb=12.0):
    """Tara block_swap sulla VRAM del computer: tiene più blocchi possibile residenti in VRAM,
    lasciando `reserve_gb` per attivazioni/VAE/altri modelli. Generico per ogni GPU (5090→0 swap,
    4090→pochi, 3060→molti). reserve_gb scala con la richiesta di attivazioni (passata da run())."""
    import torch
    if not torch.cuda.is_available():
        return total_blocks
    free, total = torch.cuda.mem_get_info()           # VRAM REALMENTE libera ORA (esclude altri modelli)
    free_gb, total_gb = free / 1e9, total / 1e9
    try:
        model_gb = os.path.getsize(fp.get_full_path("diffusion_models", model_name)) / 1e9
    except Exception:
        model_gb = 16.0
    per_block = (model_gb / total_blocks) or 0.4
    resident = int(max(0.0, free_gb - reserve_gb) / per_block)
    swap = max(0, min(total_blocks, total_blocks - resident))
    print(f"[auto-vram] {free_gb:.0f}GB liberi / {total_gb:.0f}GB tot "
          f"(occupati da altro: {total_gb - free_gb:.0f}GB), modello {model_gb:.0f}GB "
          f"({per_block:.2f}GB/blk), reserve {reserve_gb:.0f}GB -> blocks_to_swap={swap}/{total_blocks}", flush=True)
    return swap


def run(p: dict) -> str:
    import time, numpy as np, torch
    from PIL import Image
    NM, fp = _shim_and_import()
    import nodes  # core ComfyUI (CLIPVisionLoader) — dopo lo shim/sys.path

    W = int(p.get("width", 512)); H = int(p.get("height", 512))
    F = int(p.get("num_frames", 81)); FPS = float(p.get("fps", 25.0))
    t0 = time.time()

    # ---- loaders (tutti i nodi via _call: kwargs + hidden inputs auto) ----
    model_name = p.get("model", "Wan2.1_14b_Image_to_Video_720p_GGUF_Q8.gguf")
    # block_swap tarato DINAMICAMENTE sulla VRAM del computer ("auto") o forzato (numero)
    bts = p.get("blocks_to_swap", "auto")
    if bts in ("auto", None, ""):
        # riserva = base + attivazioni(∝risoluzione) + latent Wan reale(∝frame, ma piccolo) + margine
        lat_gb = 16 * (F / 4.0) * (H / 8.0) * (W / 8.0) * 2 / 1e9   # latent bf16 [16, T/4, H/8, W/8]
        reserve = float(p.get("vram_reserve_gb") or 0) or (9.0 + 6.0 * (W * H) / (640.0 * 640.0) + 6.0 * lat_gb + 3.0)
        bts = _auto_blocks_to_swap(fp, model_name, reserve_gb=reserve)
    else:
        bts = int(bts)
    mt = _call(NM["MultiTalkModelLoader"],
               model=p.get("multitalk", "WanVideo_2_1_Multitalk_14B_fp32.safetensors"))
    bs = _call(NM["WanVideoBlockSwap"],
               blocks_to_swap=bts, offload_img_emb=False, offload_txt_emb=False)
    lora = _call(NM["WanVideoLoraSelect"],
                 lora=p.get("lora", "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64_fixed.safetensors"),
                 strength=float(p.get("lora_strength", 0.6)), merge_loras=False)
    model = _call(NM["WanVideoModelLoader"],
                  model=model_name,
                  base_precision="fp16_fast", quantization="disabled", load_device="offload_device",
                  attention_mode="sageattn", block_swap_args=bs, lora=lora, multitalk_model=mt)
    vae = _call(NM["WanVideoVAELoader"],
                model_name=p.get("vae", "Wan2_1_VAE_bf16.safetensors"), precision="bf16")
    t5 = _call(NM["LoadWanVideoT5TextEncoder"],
               model_name=p.get("t5", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"), precision="bf16",
               load_device="offload_device", quantization="fp8_e4m3fn")
    clipv = _call(nodes.CLIPVisionLoader, clip_name=p.get("clip_vision", "clip_vision_h.safetensors"))
    w2v = _call(NM["DownloadAndLoadWav2VecModel"],
                model="TencentGameMate/chinese-wav2vec2-base", base_precision="fp16", load_device="main_device")
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    # ---- conditioning testo (una volta sola, condiviso da tutti i segmenti) ----
    txt = _call(NM["WanVideoTextEncode"],
                positive_prompt=p.get("prompt", "a woman singing, expressive performance, close-up"),
                negative_prompt=p.get("negative", "static, blurry, deformed, extra fingers, bad hands, low quality"),
                t5=t5)

    # sorgente volto: foto (Frame 1) oppure primo frame di un video esistente
    img_path = p.get("image_path")
    if p.get("source_video") and not img_path:
        img_path = _first_frame(p["source_video"])
        print(f"[src] primo frame del video -> {os.path.basename(img_path)}", flush=True)
    im = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
    img_t = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)[None]    # [1,H,W,3]

    # audio completo (vocals isolati opz.) — verrà affettato per segmento
    if p.get("separate_vocals", True):
        try:
            audio_full = _vocals_audio(p["audio_path"])
            print("[audio] vocals isolati (demucs) -> lip-sync su voce pulita", flush=True)
        except Exception as _e:                          # noqa: BLE001
            print(f"[audio] separazione vocals fallita ({_e}); uso audio intero", flush=True)
            audio_full = _load_audio(p["audio_path"])
    else:
        audio_full = _load_audio(p["audio_path"])

    steps = int(p.get("steps", 10)); cfg = float(p.get("cfg", 1.0))
    shift = float(p.get("shift", 5.0)); seed = int(p.get("seed", 8))
    sched = p.get("scheduler", "dpm++_sde"); ascale = float(p.get("audio_scale", 2.0))
    acfg = float(p.get("audio_cfg_scale", 1.0)); use_ctx = bool(p.get("context", False))

    def _gen_chunk(start_t, audio_d, nf, seed_c):
        """Genera UN segmento da `nf` frame partendo da start_t (continuità) + finestra audio."""
        clipemb = _call(NM["WanVideoClipVisionEncode"],
                        clip_vision=clipv, image_1=start_t, strength_1=1.0, strength_2=1.0,
                        crop="center", combine_embeds="average", force_offload=True)
        imgemb = _call(NM["WanVideoImageToVideoEncode"],
                       width=W, height=H, num_frames=nf, noise_aug_strength=0.0, start_latent_strength=1.0,
                       end_latent_strength=1.0, force_offload=True, vae=vae, clip_embeds=clipemb, start_image=start_t)
        mtemb = _call(NM["MultiTalkWav2VecEmbeds"],
                      wav2vec_model=w2v, audio_1=audio_d, normalize_loudness=True, num_frames=nf, fps=FPS,
                      audio_scale=ascale, audio_cfg_scale=acfg, multi_audio_type="para")
        kw = dict(text_embeds=txt, multitalk_embeds=mtemb)
        if use_ctx and nf > 81:
            kw["context_options"] = _call(NM["WanVideoContextOptions"],
                                          context_schedule="static_standard", context_frames=81,
                                          context_stride=4, context_overlap=32, freenoise=True, verbose=False)
        samples = _call(NM["WanVideoSampler"],
                        model=model, image_embeds=imgemb, steps=steps, cfg=cfg, shift=shift,
                        seed=seed_c, force_offload=True, scheduler=sched, riflex_freq_index=0, **kw)
        return _call(NM["WanVideoDecode"], vae=vae, samples=samples, enable_vae_tiling=True,
                     tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128)

    # ---- generazione a SEGMENTI in STREAMING su disco (memory-bounded in VRAM **e** RAM) ----
    # Scrive ogni chunk subito via ffmpeg, tenendo in RAM solo 1 chunk + l'overlap (niente torch.cat
    # di tutti i frame → su 3:30 erano migliaia di frame che saturavano la RAM e crashavano il sistema).
    SEG = int(p.get("seg_frames", 0) or 0) or int(15 * FPS)   # default 15s/segmento (override da GUI)
    OVERLAP = int(p.get("overlap_frames", 0) or 0) or 12       # frame di crossfade (override da GUI: lv_overlap)
    anchor_every = int(p.get("anchor_every", 0) or 0)         # 0=continuità; N=ri-ancora la FOTO ogni N chunk (anti-drift volto)
    tg = time.time()
    os.makedirs(os.path.dirname(p["output"]) or ".", exist_ok=True)
    tmp = p["output"] + ".noaudio.mp4"
    writer = {"proc": None}

    def _emit(frames):
        if frames is None or frames.shape[0] == 0:
            return
        arr = (frames.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        n, h, w, _ = arr.shape
        if writer["proc"] is None:
            writer["proc"] = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
                 "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
        writer["proc"].stdin.write(arr.tobytes())

    def _xfade(prev_tail, imgs):
        ov = min(OVERLAP, prev_tail.shape[0], imgs.shape[0])
        if ov <= 0:
            return imgs, 0
        t = torch.linspace(0.0, 1.0, ov, device=imgs.device, dtype=torch.float32)
        t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)        # smootherstep
        wgt = t.view(ov, 1, 1, 1).to(imgs.dtype)
        _emit(prev_tail[-ov:] * (1.0 - wgt) + imgs[:ov] * wgt)
        return imgs[ov:], ov

    nframes = 0
    if F <= SEG:
        imgs = _gen_chunk(img_t, audio_full, F, seed)
        _emit(imgs); nframes = imgs.shape[0]
    else:
        stride = max(1, SEG - OVERLAP)
        n0 = 0; cur = img_t; ci = 0; prev_tail = None
        while n0 < F:
            nf = min(SEG, F - n0)
            if nf < 25:
                break
            a0, a1 = n0 / FPS, (n0 + nf) / FPS
            print(f"[chunk {ci}] frame {n0}..{n0 + nf} ({a0:.1f}-{a1:.1f}s)", flush=True)
            start = img_t if (anchor_every and ci % anchor_every == 0) else cur   # anti-drift opzionale
            imgs = _gen_chunk(start, _slice_audio(audio_full, a0, a1), nf, seed + ci)
            body = imgs
            if ci > 0:                                   # fonde l'overlap col chunk precedente (crossfade)
                body, ov = _xfade(prev_tail, imgs); nframes += ov
            head = body[:-OVERLAP] if OVERLAP > 0 else body   # trattieni la coda per il prossimo crossfade
            _emit(head); nframes += head.shape[0]
            prev_tail = imgs[-OVERLAP:].clone() if OVERLAP > 0 else None
            cur = imgs[-1:].clone()
            n0 += stride; ci += 1
            del imgs, body
        if prev_tail is not None:                        # coda finale (nessun chunk successivo)
            _emit(prev_tail); nframes += prev_tail.shape[0]

    if writer["proc"] is not None:
        writer["proc"].stdin.close(); writer["proc"].wait()
    print(f"[gen] {time.time()-tg:.0f}s, {nframes} frame totali (streaming)", flush=True)
    # mux audio reale (durata = nframes/FPS) → resta in sync
    dur = nframes / FPS
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", p["audio_path"], "-t", f"{dur:.3f}",
                    "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", p["output"]],
                   check=False)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return p["output"]


def _save(images, out_path, fps, audio_path):
    import numpy as np
    arr = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    n, h, w, _ = arr.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".noaudio.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    proc.stdin.write(arr.tobytes()); proc.stdin.close(); proc.wait()
    # mux audio reale (tagliato alla durata del video)
    dur = n / fps
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", audio_path, "-t", f"{dur:.3f}",
                    "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest", out_path], check=False)
    try: os.remove(tmp)
    except OSError: pass
    return out_path


def main():
    import torch
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    with torch.inference_mode():
        print("OK:" + run(p), flush=True)


if __name__ == "__main__":
    main()
