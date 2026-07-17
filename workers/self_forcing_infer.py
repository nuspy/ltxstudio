"""Worker Self-Forcing 1.3B — variante VIDEO **realtime** di krea_rt (autoregressivo, leggero per la 5090).

Self-Forcing (github.com/guandeh17/Self-Forcing, Apache-2.0) distilla Wan 2.1 1.3B in un generatore
autoregressivo (Self-Forcing/DMD, 4 step) → t2v/i2v/estensione in streaming, ~realtime su una singola 4090.
NON è diffusers: si usa il runner del repo come libreria (come LingBot). Girato nel venv `krea_rt_venv`
(torch + FA2 + deps leggere), con cwd = repo (i wrapper cercano `wan_models/Wan2.1-T2V-1.3B/` RELATIVO).

Pesi (catalogo): `self_forcing_dmd` (checkpoints/self_forcing_dmd.pt, 5.6GB) + base **formato Wan nativo**
`wan21_13b_original` (Wan-AI/Wan2.1-T2V-1.3B: models_t5_umt5-xxl-enc-bf16.pth + Wan2.1_VAE.pth + WanModel).

VRAM: `DynamicSwapInstaller` (block-swap nativo del repo) sotto i 40GB → entra comodo sui 32GB.
⚠️ LIVE-TEST PENDENTE: forma del noise per risoluzioni ≠480×832, ritorno di inference(), Windows-compat
(# VALIDATE) — attesa iterazione come per gli altri.
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF_REPO = r"E:\AI\Tools\self-forcing"
if _ROOT not in sys.path:                       # core.* importabile fin da subito (prima del chdir sul repo)
    sys.path.insert(0, _ROOT)

# Anti-frammentazione VRAM: con la KV-cache stesa la memoria si frammenta e alloc da ~550MB falliscono
# pur avendo GB liberi (l'utente ha misurato: muore a ~27GB con VRAM libera). Windows NON supporta
# expandable_segments → uso garbage_collection_threshold (libera aggressivo i blocchi cache-ati) +
# max_split_size_mb (non spezzetta i blocchi grandi). DEVE stare prima dell'init CUDA di torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.9,max_split_size_mb:256")


def _resolve_paths(p: dict) -> tuple[str, str]:
    """(checkpoint_dmd.pt, wan_native_dir) dal catalogo pesi."""
    from core import weights_catalog as wc
    ent = {e["id"]: e for e in wc.load()}

    def dmd():
        e = ent.get("self_forcing_dmd")
        if not e:
            return ""
        d = wc.dest_for(e)
        return str(d / e.get("filename", "")) if e.get("kind") == "file" else str(d)

    def wan():
        e = ent.get("wan21_13b_original")
        return str(wc.dest_for(e)) if e else ""

    ckpt = p.get("checkpoint") or dmd()
    wan_dir = p.get("wan_dir") or wan()
    if not (ckpt and os.path.isfile(ckpt)):
        raise RuntimeError("Checkpoint Self-Forcing (self_forcing_dmd.pt) non trovato. Scaricalo dal wizard 'Pesi'.")
    if not (wan_dir and os.path.isdir(wan_dir)):
        raise RuntimeError("Base Wan 2.1 1.3B (formato nativo) non trovata. Scaricala dal wizard 'Pesi'.")
    return ckpt, wan_dir


def _ensure_wan_link(wan_dir: str) -> None:
    """I wrapper cercano `wan_models/Wan2.1-T2V-1.3B/` RELATIVO al repo → creo una junction verso la base scaricata."""
    link = os.path.join(SF_REPO, "wan_models", "Wan2.1-T2V-1.3B")
    if os.path.isdir(link) and os.path.isfile(os.path.join(link, "Wan2.1_VAE.pth")):
        return
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.exists(link):
        try:
            os.remove(link)
        except OSError:
            pass
    try:
        os.symlink(wan_dir, link, target_is_directory=True)          # richiede privilegio/developer-mode
    except OSError:
        import subprocess                                            # fallback: junction Windows (no admin)
        subprocess.run(["cmd", "/c", "mklink", "/J", link, wan_dir], check=False)


def _snap_lat(px: int) -> int:
    """spaziale latente = pixel / 8 (VAE Wan)."""
    return max(16, int(px) // 8)


def _repair_seams(frames, trained_lat: int = 21, px_per_lat: int = 4, search: int = 8, win: int = 3):
    """Ripara i pochi frame rumorosi al CONFINE dell'orizzonte addestrato (ogni `trained_lat` latenti ≈ 5s):
    quando il rollout supera per la prima volta la lunghezza di training il modello "singhiozza" per ~5 frame.
    NON è stitching. Rilevo il picco di differenza inter-frame vicino al confine atteso e sostituisco quella
    manciata di frame con un blend lineare dei frame puliti prima/dopo (impercettibile: il moto è lento)."""
    import numpy as np
    T = int(frames.shape[0])
    if T < 3 * win + 6:
        return frames
    d = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=tuple(range(1, frames.ndim)))  # diff per coppia
    lat_total = (T - 1) // px_per_lat + 1
    for sl in range(trained_lat, lat_total, trained_lat):
        c = sl * px_per_lat                                       # posizione pixel attesa del confine
        lo, hi = max(1, c - search), min(T - 1, c + search)
        if hi <= lo:
            continue
        peak = lo + int(np.argmax(d[lo - 1:hi - 1]))             # il frame col salto di rumore = il seam
        a, b = max(0, peak - win - 1), min(T - 1, peak + win + 1)
        if b - a < 2:
            continue
        fa, fb = frames[a].astype(np.float32), frames[b].astype(np.float32)
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            frames[i] = (fa * (1 - t) + fb * t).clip(0, 255).astype(np.uint8)
    return frames


def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ckpt, wan_dir = _resolve_paths(p)
    _ensure_wan_link(wan_dir)

    # cwd + sys.path del repo (i moduli pipeline/utils/wan sono locali) + root app (core.stego/watermark)
    os.chdir(SF_REPO)
    for pth in (SF_REPO, _ROOT):
        if pth not in sys.path:
            sys.path.insert(0, pth)

    import time
    import torch
    from omegaconf import OmegaConf
    from einops import rearrange
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed
    from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    set_seed(int(p.get("seed", 0)))
    _t0 = time.perf_counter()                                     # cronometro: distingue overhead FISSO vs generazione

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipeline = CausalInferencePipeline(cfg, device=device)          # few-step (denoising_step_list) → causale
    sd = torch.load(ckpt, map_location="cpu")
    key = "generator_ema" if "generator_ema" in sd else "generator"
    pipeline.generator.load_state_dict(sd[key])                    # VALIDATE: chiave stato (ema vs no)
    pipeline = pipeline.to(dtype=torch.bfloat16)

    # VIDEO LUNGO a memoria COSTANTE (finestra scorrevole con SFRATTO). Il modello HA la logica di eviction
    # (causal_model: shift a sinistra dei token vecchi quando la cache si riempie), ma solo se
    # `local_attn_size != -1` sui MODULI di attenzione (non basta sul pipeline/model: ogni blocco ha il suo,
    # fissato alla costruzione). Propagando local_attn_size + max_attention_size a TUTTI i moduli:
    #  - la KV-cache è piccola (es. 21 frame ≈ 6GB) e COSTANTE a qualsiasi durata,
    #  - lo sfratto tiene solo gli ultimi `local_attn_size` frame (finestra in distribuzione → qualità piena),
    #  → durata ILLIMITATA senza OOM. Questa è la strada giusta (intuizione utente: sfrattare la KV inutile).
    _la = p.get("kv_local_attn")
    if _la is not None:
        _la = int(_la)
        pipeline.local_attn_size = _la
        n_set = 0
        for _m in pipeline.generator.model.modules():
            if hasattr(_m, "local_attn_size"):
                _m.local_attn_size = _la
                n_set += 1
            if hasattr(_m, "max_attention_size"):
                _m.max_attention_size = _la * 1560
        print(f"[self_forcing] local_attn_size={_la} propagato a {n_set} moduli → finestra scorrevole con "
              f"sfratto (cache ~costante, durata illimitata)", flush=True)

    low = get_cuda_free_memory_gb(gpu) < 40                        # 32GB → block-swap del text-encoder
    if low:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    # noise autoregressivo: [1, num_output_frames(latenti), 16, H/8, W/8]. Default 21 latenti ≈ 81px @16fps ≈ 5s.
    # Estensione/long-video: aumentare num_output_frames (rolling KV-cache) → memory-bounded.
    W = int(p.get("width", 832)); H = int(p.get("height", 480))
    lat_h, lat_w = _snap_lat(H), _snap_lat(W)
    n_out = int(p.get("num_output_frames") or 0)
    if n_out <= 0:                                                 # da durata pixel: ~4 px per latente
        n_out = ((int(p.get("num_frames", 81)) - 1) // 4) + 1
    # il modello richiede num_frames latenti multipli di num_frame_per_block (=3) → arrotondo
    _fpb = int(getattr(cfg, "num_frame_per_block", 3) or 3)
    n_out = max(_fpb, round(n_out / _fpb) * _fpb)
    fps = int(p.get("fps", 16))
    prompt = p.get("prompt", "")

    initial_latent = None                                         # i2v/estensione: VALIDATE (encode_to_latent)
    if p.get("image_path") and os.path.exists(str(p["image_path"])):
        from PIL import Image
        import numpy as np
        img = Image.open(p["image_path"]).convert("RGB").resize((W, H))
        t = torch.from_numpy(np.asarray(img)).float().div(127.5).sub(1.0)     # [-1,1]
        t = t.permute(2, 0, 1)[None, :, None].to(device=device, dtype=torch.bfloat16)   # (1,C,1,H,W)
        initial_latent = pipeline.vae.encode_to_latent(t).to(device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    _t_load = time.perf_counter() - _t0                          # FISSO: costruzione pipeline + ckpt + to-device

    # Tetto per-chiamata: la KV-cache del checkpoint `dmd` è fissa a 32760 token = 21 latenti (~5s) → oltre
    # va in overflow. Il VIDEO LUNGO (>5s) richiede il chaining a finestre (in lavorazione) o il checkpoint
    # `self_forcing_10s`. Qui limito a una finestra per garantire un output valido.
    MAX_WIN = 21 // _fpb * _fpb                                   # 21 latenti (~5s) — cap sicuro del dmd
    if _la is None and n_out > MAX_WIN:                           # senza rolling: clip a 5s per un output valido
        print(f"[self_forcing] durata richiesta {n_out} latenti > cap per-chiamata {MAX_WIN} (~5s): "
              f"clip a {MAX_WIN} (usa kv_local_attn per il rolling)", flush=True)
        n_out = MAX_WIN
    noise = torch.randn([1, n_out, 16, lat_h, lat_w], device=device, dtype=torch.bfloat16)
    print(f"[self_forcing] t2v realtime {W}x{H} ~{n_out} latenti (~{(n_out-1)*4+1}px @ {fps}fps), 4 step", flush=True)
    _tg = time.perf_counter()
    video, _lat = pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                                     initial_latent=initial_latent, low_memory=low)
    torch.cuda.synchronize()
    _t_gen = time.perf_counter() - _tg                           # SCALA con la durata: encode+denoise autoregressivo+VAE
    # (b,t,c,h,w) in [0,1] → (t,h,w,c) uint8. Conversione SU GPU: il video è già in VRAM; farla in float32
    # su CPU creava 3-4 array da ~2GB (477×480×832×3) → MemoryError in RAM di sistema sui video lunghi.
    # Qui: check finito + clamp + *255 + uint8 sulla GPU, poi porto in RAM solo l'uint8 (~0.5GB).
    v = rearrange(video, "b t c h w -> b t h w c")[0]
    if not torch.isfinite(v).all():
        raise RuntimeError("Frame NaN/inf: pesi Self-Forcing/Wan corrotti o incompleti.")
    frames = (v.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    del v, video
    if not p.get("no_seam_repair"):                              # ripara il grain al confine dell'orizzonte (~5s)
        frames = _repair_seams(frames)

    _ts = time.perf_counter()
    import imageio
    imageio.mimsave(out, list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        pipeline.vae.model.clear_cache()
    except Exception:                                             # noqa: BLE001
        pass
    _t_save = time.perf_counter() - _ts
    n_px = int(frames.shape[0])
    # Riepilogo tempi: LOAD/SAVE sono ~FISSI (uguali per 5s o 10min); GEN è la generazione VERA (scala).
    gen_rate = _t_gen / max(1, n_px)
    print(f"[timing] load={_t_load:.1f}s (fisso) | GEN={_t_gen:.1f}s ({gen_rate*1000:.0f}ms/frame, "
          f"{n_px/max(0.001,_t_gen):.2f} fps gen) | save={_t_save:.1f}s | frame={n_px} @ {fps}fps "
          f"= {n_px/fps:.1f}s video | TOT={_t_load+_t_gen+_t_save:.1f}s", flush=True)

    try:                                                          # #107 stego + #106 watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                      # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                     # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
