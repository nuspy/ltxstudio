r"""Motore video LTX-2.3 che usa il LOADER + MEMORY-MANAGEMENT di ComfyUI come LIBRERIA (no server/UI).

Il memory-management di ComfyUI regola la VRAM DINAMICAMENTE: su 32GB tiene il transformer in VRAM
(massima velocità + qualità), offloada per-blocco solo se necessario. fp8-native, niente upcast.
Pesi dal workflow reale Pluto_AllInOne (transformer-only fp8 + Gemma + projection ltxv + VAE).

Eseguito nel venv ComfyUI: E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe
Uso: python workers/ltx_comfy_native.py params.json
"""
import asyncio
import json
import os
import subprocess
import sys

COMFY = r"E:\AI\Comfy_Nexus.bak\ComfyUI"   # repo allineato al venv attivo (torch 2.8+cu128 + SageAttention)
sys.path.insert(0, COMFY)
os.chdir(COMFY)                                          # cwd corretto: import relativi (utils) + extra paths

import torch  # noqa: E402
import folder_paths  # noqa: E402

# SAGEATTENTION: acceleratore attention (fp8) — ComfyUI la usa se args.use_sage_attention=True. Senza,
# l'attention LTX gira lenta. La attiviamo se sageattention è installata (env LTX_SAGE=0 per disattivarla).
if os.environ.get("LTX_SAGE", "1") != "0":
    try:
        import sageattention  # noqa: F401
        from comfy.cli_args import args as _comfy_args
        _comfy_args.use_sage_attention = True
        print("[sage] SageAttention ATTIVA (ComfyUI)", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[sage] non attivata: {_e}", flush=True)

# Registra i percorsi modelli reali (E:\AI\Models) oltre ai default di ComfyUI.
for _folder, _path in (("diffusion_models", r"E:\AI\Models\diffusion_models"),
                       ("text_encoders", r"E:\AI\Models\TextEncoders"),
                       ("clip", r"E:\AI\Models\TextEncoders"),
                       ("loras", r"E:\AI\Models\LoRA"),
                       ("vae", r"E:\AI\Models\VAE"),
                       ("checkpoints", r"E:\AI\Models\Checkpoints")):
    try:
        folder_paths.add_model_folder_path(_folder, _path)
    except Exception:  # noqa: BLE001
        pass

# repo LTX Studio in sys.path → import di core.preview_file (canale anteprima video su file)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.append(_REPO)


_LTX_RGB_W = None                                           # tensore fattori (3,128) memoizzato
_LTX_RGB_B = None                                           # tensore bias (3,)
_LTX_VIDEO_SHAPE = None                                     # shape latente VIDEO (B,C,T,H,W) per spacchettare x0


def _decode_ltx_preview(x0, step, total):
    """#92: anteprima-immagine del latente denoised LTX-2.3 (tokenizzato + AV NestedTensor). Replica
    WrappedPreviewer.decode_latent_to_preview di ComfyUI-KJNodes coi fattori corretti a 128 canali
    (is_23=True): F.linear(movedim(1,-1))+sigmoid. Niente import di KJNodes (che rompe su 'server').
    COMPLETAMENTE guardato: qualunque errore NON deve rompere la generazione."""
    global _LTX_RGB_W, _LTX_RGB_B
    import io
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from core import preview_file
    if _LTX_RGB_W is None:
        from workers._ltx_rgb_factors import LTX23_RGB_FACTORS, LTX23_RGB_BIAS
        _LTX_RGB_W = torch.tensor(LTX23_RGB_FACTORS).transpose(0, 1)   # (3,128)
        _LTX_RGB_B = torch.tensor(LTX23_RGB_BIAS)                      # (3,)
    lat = x0
    # AV NestedTensor(video,audio) → estrai il latente VIDEO
    try:
        from comfy.nested_tensor import NestedTensor
        if isinstance(lat, NestedTensor):
            lat = lat.unbind()[0]
    except Exception:                                       # noqa: BLE001
        pass
    with torch.no_grad():
        # x0 packed (B,1,seq): il video+audio è una sequenza piatta. Prendo la parte VIDEO (primi
        # prod(shape[1:]) elementi) e la reshape a (B,C,T,H,W) usando la shape del latente video.
        if getattr(lat, "ndim", 0) == 3 and _LTX_VIDEO_SHAPE is not None:
            import math
            vs = _LTX_VIDEO_SHAPE
            cut = int(math.prod(vs[1:]))
            lat = lat[:, :, :cut].reshape((lat.shape[0],) + tuple(vs[1:]))
        # 5D (B,C,T,H,W) → (frames, C, H, W) come WrappedPreviewer.decode_latent_to_preview_image.
        # NB: due statement separati (come KJNodes): dopo movedim shape[-3:] deve essere (C,H,W).
        if getattr(lat, "ndim", 0) == 5:
            lat = lat.movedim(2, 1)                          # (B,T,C,H,W)
            lat = lat.reshape((-1,) + lat.shape[-3:])        # (B*T, C, H, W)
        W = _LTX_RGB_W.to(dtype=lat.dtype, device=lat.device)
        B = _LTX_RGB_B.to(dtype=lat.dtype, device=lat.device)
        img = torch.sigmoid(F.linear(lat.movedim(1, -1), W, bias=B))   # (N,H,W,3) [0,1]
        frame = img[0] if img.ndim == 4 else img
        a = (frame.clamp(0, 1).mul(255)).to("cpu", torch.uint8).numpy()
        pim = Image.fromarray(a, "RGB"); pim.thumbnail((384, 384))
        buf = io.BytesIO(); pim.save(buf, "PNG")
        preview_file.write_preview(buf.getvalue(), step, total)


def _setup_preview_hook():
    """PREVIEW LIVE per-step (#90/#92): barra via hook globale ProgressBar + anteprima-IMMAGINE video
    decodificata col WrappedPreviewer di KJNodes (fattori LTX-2.3 corretti). Il previewer core di ComfyUI
    è incompatibile col latente-token LTX-2.3 → disattivato. Best-effort: NON può MAI rompere la gen."""
    try:
        import comfy.utils
        from comfy.cli_args import args as _ca, LatentPreviewMethod
        from core import preview_file
        try:
            _ca.preview_method = LatentPreviewMethod.NoPreviews   # niente previewer core (crasha su LTX-2.3)
        except Exception:                                   # noqa: BLE001
            pass

        def _hook(current, total, preview=None, node_id=None):
            try:
                preview_file.write_progress(current, total)
            except Exception:                               # noqa: BLE001
                pass
        comfy.utils.set_progress_bar_global_hook(_hook)

        # anteprima-immagine: wrap di prepare_callback (usato da SamplerCustomAdvanced) per il latente x0
        import latent_preview
        if not getattr(latent_preview, "_ltx_prev_wrapped", False):
            _orig_prep = latent_preview.prepare_callback

            def _wrap(model, steps, x0_output_dict=None, _orig=_orig_prep):
                base = _orig(model, steps, x0_output_dict)
                every = max(1, int(steps) // 6)

                def cb(step, x0, x, total_steps):
                    try:
                        base(step, x0, x, total_steps)      # progress (best-effort)
                    except Exception:                       # noqa: BLE001
                        pass
                    try:
                        if step % every == 0 or step == int(total_steps) - 1:
                            _decode_ltx_preview(x0, step + 1, total_steps)
                    except Exception:                       # noqa: BLE001
                        pass
                return cb

            latent_preview.prepare_callback = _wrap
            latent_preview._ltx_prev_wrapped = True
    except Exception as _e:                                 # noqa: BLE001
        print(f"[preview] hook non attivato: {_e}", flush=True)


def _vals(x):
    """Tutti i valori di ritorno (tuple o io.NodeOutput)."""
    if isinstance(x, tuple):
        return list(x)
    items = getattr(x, "result", None) or getattr(x, "items", None)
    return list(items) if items is not None else [x]


def _unwrap(x):
    """Primo valore di ritorno (tuple o io.NodeOutput)."""
    return _vals(x)[0]


def _load_audio_vae(name: str):
    """Carica l'Audio VAE LTXV (replica LTXVAudioVAELoader: prefix replace + comfy.sd.VAE)."""
    import comfy.utils
    import comfy.sd
    import folder_paths
    path = name if os.path.isabs(name) else (folder_paths.get_full_path("vae", name) or
                                             os.path.join(r"E:\AI\Models\VAE", name))
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    sd = comfy.utils.state_dict_prefix_replace(
        sd, {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _load_audio(path: str):
    """Carica un file audio → dict {waveform:[1,C,N], sample_rate} (formato ComfyUI Audio)."""
    import torchaudio
    wav, sr = torchaudio.load(path)            # [C,N]
    return {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}


async def _init():
    import nodes
    await nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)


def run(p: dict) -> str:
    import time
    import nodes
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider, CFGGuider, KSamplerSelect, RandomNoise, SamplerCustomAdvanced)
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVScheduler

    # PREVIEW LIVE per-step (task #90): azzera lo stato e aggancia l'hook (progresso + anteprima → file
    # che la GUI polla). Disattivabile con preview=False. Best-effort.
    if p.get("preview", True):
        try:
            from core import preview_file
            preview_file.reset()
            preview_file.write_message("Caricamento componenti…")
            _setup_preview_hook()
        except Exception:                                   # noqa: BLE001
            pass
    t0 = time.time()
    _mname = p.get("model", "LTXVideo/v2/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors")
    if str(_mname).lower().endswith(".gguf"):            # GGUF (schede piccole): loader ComfyUI-GGUF
        _GGUF = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if _GGUF is None:
            raise RuntimeError("UnetLoaderGGUF non disponibile (ComfyUI-GGUF non caricato)")
        model = _unwrap(_GGUF().load_unet(_mname))
        print(f"[model] GGUF (ComfyUI-GGUF) {os.path.basename(_mname)}", flush=True)
    else:                                                # safetensors (fp8/bf16): UNETLoader standard
        model = _unwrap(nodes.UNETLoader().load_unet(_mname, "default"))
    clip = _unwrap(nodes.DualCLIPLoader().load_clip(
        p.get("clip1", "gemma_3_12B_it.safetensors"),
        p.get("clip2", "ltx-2.3_text_projection_bf16.safetensors"), "ltxv"))
    vae = _unwrap(nodes.VAELoader().load_vae(p.get("vae", "LTX23_video_vae_bf16.safetensors")))
    # LoRA video (p['loras']=[{name,strength}]): applicate al modello via LoraLoaderModelOnly (ComfyUI).
    for _lo in (p.get("loras") or []):
        _ln = _lo.get("name") if isinstance(_lo, dict) else _lo
        if not _ln:
            continue
        _ls = float(_lo.get("strength", 1.0)) if isinstance(_lo, dict) else 1.0
        try:
            model = _unwrap(nodes.LoraLoaderModelOnly().load_lora_model_only(model, _ln.replace("\\", "/"), _ls))
            print(f"[lora] {os.path.basename(str(_ln))} @ {_ls}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[lora] '{_ln}' non applicata: {_e}", flush=True)
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    if torch.cuda.is_available():
        print(f"[vram] post-load {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    enc = nodes.CLIPTextEncode()
    pos = _unwrap(enc.encode(clip, p.get("prompt", "")))
    neg = _unwrap(enc.encode(clip, p.get("negative", "worst quality, blurry")))
    W, H, F = int(p.get("width", 768)), int(p.get("height", 512)), int(p.get("num_frames", 97))
    # Forza immagine: la GUI manda ref_strength (fallback img_strength). 1.0=guida piena.
    _strength = float(p.get("ref_strength", p.get("img_strength", 1.0)))
    import numpy as np
    from PIL import Image as _PILImage

    def _img_tensor(path):
        im = _PILImage.open(path).convert("RGB")
        return torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)[None]      # [1,H,W,3]

    def _video_tensor(path, w, h, nmax):
        """Carica i frame di un video di controllo (canny/depth/pose) → [N,h,w,3] float [0,1]."""
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < nmax:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB)
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError(f"nessun frame nel control_video: {path}")
        a = np.asarray(frames, dtype="float32") / 255.0                            # [N,h,w,3]
        return torch.from_numpy(a)

    if p.get("control_video"):                           # ControlNet: IC-LoRA control (in loras) + video-guida
        from comfy_extras.nodes_lt import LTXVAddGuide
        latent = _unwrap(EmptyLTXVLatentVideo().execute(W, H, F))
        ctrl = _video_tensor(p["control_video"], W, H, F)
        _cs = float(p.get("control_strength", 0.8))
        # reference_downscale_factor dai metadata della control-LoRA (es. union-control=2): la guida va
        # ridimensionata di conseguenza, altrimenti LTXVAddGuide produce dimensioni errate.
        _iclora = None
        _clora = p.get("control_lora") or (p["loras"][0].get("name") if p.get("loras") else None)
        if _clora:
            try:
                from safetensors import safe_open as _sopen
                _cp = _clora if os.path.isabs(_clora) else os.path.join(r"E:\AI\Models\LoRA", _clora)
                with _sopen(_cp.replace("/", "\\"), "pt") as _f:
                    _fac = max(1, round(float((_f.metadata() or {}).get("reference_downscale_factor", 1))))
                if _fac > 1:
                    _iclora = {"reference_downscale_factor": _fac}
                    print(f"[controlnet] reference_downscale_factor={_fac}", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[controlnet] meta factor non letto: {_e}", flush=True)
        v = _vals(LTXVAddGuide().execute(pos, neg, vae, latent, ctrl, 0, _cs, iclora_parameters=_iclora))
        pos, neg, latent = v[0], v[1], v[2]
        print(f"[controlnet] guida da {os.path.basename(p['control_video'])} ({ctrl.shape[0]} frame) @ {_cs}", flush=True)
    elif p.get("keyframes"):                             # storyboard / multi-keyframe: foto a frame diversi
        from comfy_extras.nodes_lt import LTXVAddGuide
        latent = _unwrap(EmptyLTXVLatentVideo().execute(W, H, F))
        for kf in p["keyframes"]:
            if not kf.get("path"):
                continue
            fr = max(0, min(F - 1, int(kf.get("frame", 0))))
            v = _vals(LTXVAddGuide().execute(pos, neg, vae, latent, _img_tensor(kf["path"]),
                                             fr, float(kf.get("strength", 1.0))))
            pos, neg, latent = v[0], v[1], v[2]
        print(f"[keyframes] {len(p['keyframes'])} guide a tempo (storyboard)", flush=True)
    elif p.get("image_path"):                            # i2v: condiziona sul primo frame
        import numpy as np
        from PIL import Image
        im = Image.open(p["image_path"]).convert("RGB")
        img_t = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)[None]   # [1,H,W,3]
        if p.get("i2v_method") == "inplace":            # inplace-SOFT: il modello reinterpreta (noise_mask)
            from comfy_extras.nodes_lt import LTXVImgToVideoInplace
            latent = _unwrap(EmptyLTXVLatentVideo().execute(W, H, F))
            latent = _unwrap(LTXVImgToVideoInplace().execute(vae, img_t, latent, _strength))
            print("[i2v] metodo inplace-soft", flush=True)
        else:                                            # addguide: guida piena sul primo frame
            from comfy_extras.nodes_lt import LTXVImgToVideo
            v = _vals(LTXVImgToVideo().execute(pos, neg, img_t, vae, W, H, F, 1, _strength))
            pos, neg, latent = v[0], v[1], v[2]
    else:
        latent = _unwrap(EmptyLTXVLatentVideo().execute(W, H, F))
    if not isinstance(latent, dict):
        latent = {"samples": getattr(latent, "samples", latent)}
    print(f"[cond] ok, latent={tuple(latent['samples'].shape)}", flush=True)

    # ── AUDIO (modello AV congiunto LTX-2.3). Senza condizionamento (t2v/i2v puri) l'audio LTX-2 è RUMORE
    # non condizionato ("rumore grigio", vedi fix b6073a9 in ltx2_infer.py): default OFF, opt-in con
    # generate_audio=True. Audio fornito = ENCODE+KEEP (lip-sync/canto), video generato. ──
    av_mode = False
    audio_vae = None
    _has_audio_file = bool(p.get("audio_path")) and os.path.isfile(str(p.get("audio_path")))
    _want_audio = bool(p.get("generate_audio", False))   # opt-in: senza file l'audio generato è rumore
    if _has_audio_file or _want_audio:
        from comfy_extras.nodes_lt import LTXVConcatAVLatent
        from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode, LTXVEmptyLatentAudio
        audio_vae = _load_audio_vae(p.get("audio_vae", "LTX23_audio_vae_bf16.safetensors"))
        if _has_audio_file:                              # audio dato → encode e MANTIENI (noise_mask 0)
            a_in = _load_audio(p["audio_path"])
            audio_latent = _unwrap(LTXVAudioVAEEncode().execute(a_in, audio_vae))
            audio_latent = dict(audio_latent)
            audio_latent["noise_mask"] = torch.zeros_like(audio_latent["samples"])
            print(f"[av] audio encode+keep {os.path.basename(p['audio_path'])} (lip-sync/canto)", flush=True)
        else:                                            # nessun file → audio GENERATIVO (t2v/i2v con suono)
            audio_latent = _unwrap(LTXVEmptyLatentAudio().execute(F, int(p.get("fps", 24)), 1, audio_vae))
            print("[av] audio generativo (empty latent) — t2v/i2v con suono", flush=True)
        global _LTX_VIDEO_SHAPE                          # #92: shape del latente VIDEO (pre-concat) per l'anteprima
        try:
            _LTX_VIDEO_SHAPE = tuple(latent["samples"].shape)
        except Exception:                                # noqa: BLE001
            pass
        latent = _unwrap(LTXVConcatAVLatent().execute(latent, audio_latent))
        av_mode = True
        print("[av] AV latent concatenato", flush=True)

    cfg = float(p.get("guidance", 1.0))
    if cfg <= 1.0:
        guider = _unwrap(BasicGuider().get_guider(model, pos))
    else:
        guider = _unwrap(CFGGuider().get_guider(model, pos, neg, cfg))
    sampler = _unwrap(KSamplerSelect().execute(p.get("sampler", "euler")))
    sigmas = _unwrap(LTXVScheduler().execute(int(p.get("steps", 30)), 2.05, 0.95, True, 0.1, latent))
    noise = _unwrap(RandomNoise().execute(int(p.get("seed", 42))))

    try:
        from core import preview_file as _pf
        _pf.write_message("Generazione…")
    except Exception:                                       # noqa: BLE001
        _pf = None
    if not av_mode:                                      # #92: senza audio il latente è già solo-video
        try:
            globals()["_LTX_VIDEO_SHAPE"] = tuple(latent["samples"].shape)
        except Exception:                                # noqa: BLE001
            pass
    tg = time.time()
    out = SamplerCustomAdvanced().execute(noise, guider, sampler, sigmas, latent)
    denoised = out[1] if isinstance(out, tuple) and len(out) > 1 else _unwrap(out)
    if not isinstance(denoised, dict):
        denoised = {"samples": getattr(denoised, "samples", denoised)}
    print(f"[gen] {time.time()-tg:.0f}s", flush=True)
    if _pf is not None:
        try:
            _pf.write_message("Decodifica e salvataggio…")
        except Exception:                                   # noqa: BLE001
            pass

    if av_mode:                                          # a2v: separa video/audio, decodifica entrambi, muxa
        from comfy_extras.nodes_lt import LTXVSeparateAVLatent
        from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode
        sep = _vals(LTXVSeparateAVLatent().execute(denoised))
        vid_lat, aud_lat = sep[0], sep[1]
        images = _unwrap(nodes.VAEDecode().decode(vae, vid_lat))
        a_out = _unwrap(LTXVAudioVAEDecode().execute(aud_lat, audio_vae))
        print("[a2v] decode video+audio ok", flush=True)
        return _save_mp4(images, p["output"], float(p.get("fps", 24.0)),
                         audio=a_out.get("waveform"), audio_sr=int(a_out.get("sample_rate", 44100)))

    # IC-LoRA/guide in-context (controlnet, keyframe, V2V): ritaglia i frame-guida dopo il sampling.
    # No-op se non ci sono guide nel positive (t2v/i2v semplici). Come i workflow LTX ufficiali.
    from comfy_extras.nodes_lt import LTXVCropGuides
    try:
        _cg = _vals(LTXVCropGuides().execute(pos, neg, denoised))
        denoised = _cg[2] if not isinstance(_cg[2], dict) or "samples" in _cg[2] else denoised
        if not isinstance(denoised, dict):
            denoised = {"samples": getattr(denoised, "samples", denoised)}
    except Exception as _e:  # noqa: BLE001
        print(f"[cropguides] saltato: {_e}", flush=True)

    images = _unwrap(nodes.VAEDecode().decode(vae, denoised))     # (frames,H,W,3) [0,1]
    return _save_mp4(images, p["output"], float(p.get("fps", 24.0)))


def _save_mp4(images, out_path: str, fps: float, audio=None, audio_sr: int = 44100) -> str:
    import numpy as np
    # GUARDIA anti-NaN: pesi corrotti/incompatibili producono frame NaN → video VERDE uniforme
    # consegnato come successo. Errore esplicito invece del silenzio.
    if torch.isnan(images).any() or torch.isinf(images).any():
        raise RuntimeError("Frame NaN/inf dal denoise: pesi del modello corrotti o incompatibili "
                           "(es. GGUF con download incompleto). Verifica/riscarica il checkpoint.")
    if audio is not None and hasattr(audio, "isnan") and torch.isnan(audio).any():
        print("[save] audio generato NaN → scartato (video muto)", flush=True)
        audio = None
    arr = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (N,H,W,3)
    n, h, w, _ = arr.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if audio is not None:                                # a2v: scrivi WAV temporaneo e muxa con l'audio
        import torchaudio
        wav = audio[0] if hasattr(audio, "dim") and audio.dim() == 3 else audio   # [C,N]
        wpath = out_path + ".audio.wav"
        torchaudio.save(wpath, wav.cpu().float(), audio_sr)
        ok = False
        try:
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                 "-i", "-", "-i", wpath, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 "-shortest", out_path], stdin=subprocess.PIPE)
            proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            proc.wait()
            ok = proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
        except OSError:
            # ffmpeg morto a metà stream (es. audio invalido per aac): il write su stdin fallisce.
            ok = False
        finally:
            try:
                os.remove(wpath)
            except OSError:
                pass
        if ok:
            return out_path
        # fallback: NON perdere la generazione per un mux fallito → salva il video MUTO
        print("[save] mux audio fallito → salvo il video senza audio", flush=True)
        # ricade nel ramo solo-video qui sotto
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)
    proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_path


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    asyncio.run(_init())
    with torch.inference_mode():                          # come ComfyUI: niente grafo autograd
        print("OK:" + run(p), flush=True)


if __name__ == "__main__":
    main()
