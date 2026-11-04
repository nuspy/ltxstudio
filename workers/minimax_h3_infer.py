"""Worker MiniMax H3 — video + audio stereo generati INSIEME (omni-transformer 33B, pruned).

Pipeline ufficiale diffusers modular (MiniMaxH3Blocks) eseguita A FASI, così in VRAM c'è un solo modello
grande alla volta e tutto sta in 32 GB:
  1. before_encode + text_encoder  → Qwen3-VL-32B (nvfp4-AWQ 15,7 GB / fp8 26,5 GB / int8-ConvRot) → liberato
  2. vae_encoder (keyframe/riferimenti) → VAE video (fp32) [+ audio VAE per Ref2VA]              → liberato
  3. denoise → DiT pruned (nvfp4 12,5 GB / fp8 / int8-ConvRot) + LoRA turbo/utente               → liberato
  4. decode → VAE video + audio VAE → mp4 con audio (PyAV, encode_video)
Pesi SOLO in formato nativo (core/native_convert: i single-file ComfyUI vengono convertiti al download).
Il transformer pruned è il MiniMaxH3PrunedTransformer3DModel ufficiale (workers/minimax_h3_pruned.py) con
qkv e gate/up FUSI come nel file quantizzato (vedi workers/minimax_h3_convert.py).

Nessuna scelta automatica dei pesi: profilo, TE e turbo arrivano espliciti dai params (core/minimax_h3_spec).
Uso:  python minimax_h3_infer.py params.json   → stampa "OK:<mp4>"
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# frammentazione: expandable_segments esiste solo su Linux (su Windows PyTorch lo ignora con un avviso: li' si
# compensa con empty_cache + offload dinamico in workers/h3_chunk). Va impostato PRIMA di importare torch.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import minimax_h3_spec as S                                  # noqa: E402

DIT_FP32 = ("time_embedder", "proj_in", "proj_out", "audio_proj_in", "audio_proj_out", "folded_bias",
            "adaln_basis", "adaln_mean")


def _log(m: str) -> None:
    print(f"[h3] {m}", flush=True)


def _gc():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram() -> str:
    import torch
    return f"{torch.cuda.memory_allocated() / 2**30:.1f} GB (picco {torch.cuda.max_memory_allocated() / 2**30:.1f})"


def _free_gpu_servers() -> None:
    """Libera la VRAM dei server immagini/video dell'app prima di caricare (stdlib)."""
    import urllib.request as u
    for port in (8189, 8190, 8191, 8192):
        try:
            u.urlopen(u.Request(f"http://127.0.0.1:{port}/free", data=b'{"mode": "destroy"}',
                                headers={"Content-Type": "application/json"}), timeout=15).read()
        except Exception:                                            # noqa: BLE001 (server giù)
            pass


# ── pesi ───────────────────────────────────────────────────────────────────────────────────────
def resolve(p: dict) -> dict:
    from core import model_paths as mp
    from core import weights_catalog as wc
    from core import native_convert as nc
    ids = S.weight_ids(p)
    out, missing = {}, []
    for k, i in ids.items():
        e = mp.entry(i)
        if not e or not wc.is_present(e, str(mp.models_root())):
            missing.append(i)
            continue
        path = mp.p(i, "")
        if e.get("convert") and not nc.is_native(path):     # file ComfyUI non ancora convertito
            raise RuntimeError(f"{os.path.basename(path)} non è in formato nativo: riscaricalo dal tab «Pesi»")
        out[k] = path
    if missing:
        raise RuntimeError("Pesi MiniMax H3 mancanti: " + ", ".join(missing) +
                           ". Scaricali dal tab «Pesi modelli» (profilo " + str(p.get("h3_profile") or "nvfp4") + ").")
    out["ids"] = ids
    return out


def _read(path: str, device: str = "cuda", skip=()):
    import _qlinear as Q
    return Q.read_safetensors(path, device, skip=skip)


def _header_keys(path: str) -> list:
    """Nomi dei tensori di un .safetensors, dal solo header (nessun dato letto)."""
    import json
    import struct
    with open(path, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    header.pop("__metadata__", None)
    return list(header)


VAE_DECODER_PREFIXES = ("decoder.", "post_quant_conv.")


def load_text_encoder(W: dict):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    import _qlinear as Q
    from minimax_h3_convert import PassthroughDecoderLayer
    cfg = AutoConfig.from_pretrained(str(Path(W["configs"]) / "text_encoder"))
    sd = _read(W["te"])
    n_file = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith("model.language_model.layers."))
    # il pipeline legge hidden_states[50] e vuole più di 50 layer: 50 veri + 1 passante (vedi PassthroughDecoderLayer)
    cfg.text_config.num_hidden_layers = n_file + 1
    if hasattr(cfg.text_config, "layer_types") and cfg.text_config.layer_types:
        cfg.text_config.layer_types = list(cfg.text_config.layer_types)[: n_file + 1]
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration._from_config(cfg, torch_dtype=torch.bfloat16)
    layers = te.model.language_model.layers
    layers[n_file] = PassthroughDecoderLayer(getattr(layers[n_file], "attention_type", "full_attention"))
    te.lm_head = torch.nn.Identity()
    sd = Q.quantize_from_state_dict(te, sd)
    miss, unex = Q.load_into(te, sd)
    del sd
    lm = te.model.language_model
    if "model.language_model.norm.weight" in miss:        # mai usata (si legge hidden_states[50]) ma deve esistere
        lm.norm.weight = torch.nn.Parameter(torch.ones(lm.norm.weight.shape, dtype=torch.bfloat16, device="cuda"))
        miss.remove("model.language_model.norm.weight")
    miss = [m for m in miss if not m.endswith("inv_freq")]
    if miss or unex:
        raise RuntimeError(f"text encoder: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    return te.to("cuda").eval()


def lora_specs(W: dict, p: dict) -> list:
    """[(percorso, forza)] delle LoRA da applicare al DiT: turbo + LoRA dell'utente (risolte e verificate)."""
    loras = []
    if W.get("turbo"):
        loras.append((W["turbo"], 1.0))
    for lo in p.get("loras") or []:
        name = lo.get("path") or lo.get("name") or ""
        if not name:
            continue
        # dalla GUI arriva il nome relativo alla cartella LoRA (LoraEntry.comfy_name): risolverlo, mai scartarlo
        from core import model_paths as mp
        path = name if os.path.isabs(name) else os.path.join(str(mp.models_root()), "LoRA", name)
        if not os.path.isfile(path):
            raise RuntimeError(f"LoRA non trovata su disco: {name}")
        loras.append((path, float(lo.get("strength", 1.0))))
    return loras


def load_transformer(W: dict, mode: str, p: dict):
    from accelerate import init_empty_weights
    import _qlinear as Q
    from minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
    from minimax_h3_convert import fuse_dit_projections
    sub = "transformer_ref" if mode == "ref2va" else "transformer"
    base = json.loads((Path(W["configs"]) / sub / "config.json").read_text("utf-8"))
    cfg = {k: v for k, v in base.items() if not k.startswith("_")}
    cfg.update(adaln_rank=8, time_table_size=1025)
    with init_empty_weights():
        tr = MiniMaxH3PrunedTransformer3DModel(**cfg)
    fuse_dit_projections(tr)
    sd = Q.quantize_from_state_dict(tr, _read(W["dit"]))
    miss, unex = Q.load_into(tr, sd, keep_fp32=DIT_FP32)
    del sd
    miss = [m for m in miss if m not in ("adaln_basis", "adaln_mean")]   # servono solo per LoRA «released»
    if miss or unex:
        raise RuntimeError(f"transformer: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    tr = tr.to("cuda").eval()
    loras = lora_specs(W, p)
    for path, s in loras:
        from core import native_convert as nc
        if not nc.is_native(path):
            raise RuntimeError(f"LoRA {os.path.basename(path)} non in formato nativo (convertila dal tab «Pesi»)")
        n = Q.apply_lora(tr, _read(path), s)
        _log(f"LoRA {os.path.basename(path)} ×{s}: {n} moduli")
        if n == 0:
            raise RuntimeError(f"LoRA {os.path.basename(path)}: nessun modulo agganciato (non è una LoRA H3?)")
    backend = str(p.get("attention_backend") or "")
    if backend:
        try:
            tr.set_attention_backend(backend)
        except Exception as e:                                       # noqa: BLE001
            _log(f"backend attenzione {backend} non disponibile ({e}): uso quello di default")
    if p.get("h3_int8_attention"):                     # attenzione int8 di comfy_kitchen (casella della GUI, spenta di default): al posto del backend native
        import h3_int8_attention
        h3_int8_attention.install()
    if W.get("veda"):                                  # DOPO il backend: i processor originali restano il ripiego
        import veda_h3
        if not (W.get("turbo") or S.CUSTOM_DITS.get(str(p.get("h3_dit") or ""), {}).get("turbo_fused")):
            _log("Veda: il predittore è addestrato col Turbo 8 passi; senza Turbo la qualità va verificata")
        veda_h3.enable(tr, W["veda"], generated=S.VEDA_BUDGET, reference=S.VEDA_BUDGET, log=_log)
    return tr


def load_vae(W: dict, encoder_only: bool = False, device: str = "cuda", dtype=None):
    """VAE video in fp32 (`dtype` None) o nel dtype dato (il decode lo vuole fp16, vedi `decode_dtype`). `encoder_only`: solo encoder e quant_conv (0,7 GB invece di 9,7: il decoder e 4,5 dei 4,85 GB del file fp16), per le codifiche di riferimenti,
    guida e clip sorgente. I pesi dell'encoder sono gli stessi fp16 del file portati a fp32: nessuna differenza numerica rispetto al VAE intero.
    A 736p il VAE intero (9,7 GB) piu il testo non liberato mandava in OOM la codifica del riferimento (turno 40 del 04/11)."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLMiniMaxH3
    import _qlinear as Q
    import h3_lightvae as LV
    cfg = AutoencoderKLMiniMaxH3.load_config(str(Path(W["configs"]) / "vae"))
    if W.get("vae_dec_lite") and not encoder_only:     # LightVAE: decoder con 12 dei 36 blocchi (rinumerati)
        cfg = {**cfg, "decoder_num_layers": LV.decoder_layers(W["vae_dec_lite"])}
    with init_empty_weights():
        vae = AutoencoderKLMiniMaxH3.from_config(cfg)
    if W.get("vae_enc_lite"):
        LV.prune_encoder(vae, LV.kept_residuals(W["vae_enc_lite"]))
    skip = frozenset(key for key in _header_keys(W["vae"]) if key.startswith(VAE_DECODER_PREFIXES)) if encoder_only else frozenset()
    use_decoder_lite = bool(W.get("vae_dec_lite")) and not encoder_only
    sd = LV.merge_state(_read(W["vae"], device, skip),
                        dec=_read(W["vae_dec_lite"], device) if use_decoder_lite else None,
                        enc=_read(W["vae_enc_lite"], device) if W.get("vae_enc_lite") else None)
    if W.get("vae_dec_lite") or W.get("vae_enc_lite"):
        _log("VAE video LightVAE Lite: " + " + ".join(n for n, k in (("decoder", "vae_dec_lite"),
                                                                    ("encoder", "vae_enc_lite")) if W.get(k)))
    miss, unex = Q.load_into(vae, sd, device=device, dtype=dtype or torch.float32)
    del sd
    if encoder_only:                                   # il decoder resta su meta: si sostituisce, cosi .to(device) non lo tocca e nessuno lo usa per sbaglio
        miss = [key for key in miss if not key.startswith(VAE_DECODER_PREFIXES)]
        vae.decoder, vae.post_quant_conv = torch.nn.Identity(), torch.nn.Identity()
    if miss or unex:
        raise RuntimeError(f"VAE video: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    return vae.to(device).eval()


def _drop_module_weights(m) -> None:
    """Libera DAVVERO i pesi di un modulo anche se qualche riferimento Python lo tiene ancora vivo: parametri e buffer diventano vuoti."""
    import torch
    freed = 0
    for t in list(m.parameters(recurse=True)) + list(m.buffers(recurse=True)):
        try:
            if t.is_cuda:
                freed += t.numel() * t.element_size()
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        except Exception:                                            # noqa: BLE001
            pass
    _log(f"{type(m).__name__}: pesi liberati ({freed / 2**30:.2f} GB su GPU)")


def purge_big_quantized_tensors(min_bytes: int = 32 << 20, device_type: str = "cuda", log=None) -> float:
    """Dopo il denoise il DiT non serve piu e il decode ha bisogno di ogni GB: i pesi quantizzati (int8/fp8/uint8) ancora vivi sulla scheda si svuotano SUL POSTO
    qualunque sia l'oggetto che li tiene. Giro del 03/11 (video_minimax_h3_04, 736x1376, int8): dopo `_drop_module_weights(tr)` (14,86 GB liberati) restavano vivi
    14,3 GB di int8 dei blocchi del DiT (37 blocchi x 4 matrici) — 29,15 GB allocati al decode e OOM per 164 MiB. Chi li tenesse (offload/chunk/hook) non e stato
    individuato: qui si agisce sui tensori, non sui proprietari. Il VAE e in fp32/bf16, quindi non viene toccato. Ritorna i GB liberati."""
    import gc
    import torch
    freed = 0
    seen = set()
    if log:                                                          # diagnosi (una sola riga): chi tiene il tensore quantizzato piu grande ancora vivo
        try:
            _big = max((o for o in gc.get_objects() if isinstance(o, torch.Tensor) and o.device.type == device_type
                        and o.dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2) and o.numel() * o.element_size() >= min_bytes),
                       key=lambda t: t.numel() * t.element_size(), default=None)
            if _big is not None:
                log("proprietari del tensore quantizzato piu grande ancora vivo: " + ", ".join(sorted({type(r).__name__ for r in gc.get_referrers(_big)})[:8]))
        except Exception:                                            # noqa: BLE001
            pass
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.device.type == device_type and o.dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
                nb = o.numel() * o.element_size()
                if nb >= min_bytes and o.untyped_storage().data_ptr() not in seen:
                    seen.add(o.untyped_storage().data_ptr())
                    freed += nb
                    o.data = torch.empty(0, dtype=o.dtype, device=o.device)
        except Exception:                                            # noqa: BLE001
            pass
    if log:
        log(f"tensori quantizzati ancora vivi dopo il DiT: {freed / 2**30:.2f} GB liberati sul posto")
    return freed / 2**30


def load_audio_vae(W: dict):
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLMiniMaxH3Audio
    import _qlinear as Q
    with init_empty_weights():
        av = AutoencoderKLMiniMaxH3Audio.from_config(
            AutoencoderKLMiniMaxH3Audio.load_config(str(Path(W["configs"]) / "audio_vae")))
    miss, unex = Q.load_into(av, _read(W["audio_vae"]), dtype=torch.float32)
    miss = [m for m in miss if not m.endswith(("filter", "zero_k_bias"))]
    if miss or unex:
        raise RuntimeError(f"audio VAE: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    return av.to("cuda").eval()


# ── prompt ─────────────────────────────────────────────────────────────────────────────────────
def build_prompt(p: dict, duration_s: float) -> str:
    """Prompt nel formato ufficiale H3 (core/h3_prompt): campi strutturati, speaker (S1), <d>[Lingua] …</d>."""
    from core import h3_prompt
    if not _guided(p):
        return h3_prompt.compose(p, duration_s)
    # Con la guida allineata: il prompt e quello che la LoRA di guida vuole (una frase, trigger davanti, caption fissa per LMS), non la struttura
    # «integrated_multimodal_description»; un prompt GIA strutturato (l'edit con maschera) passa invariato e riceve solo il trigger.
    from core import h3_guide as CG
    info = CG.guide_of_loras(p.get("loras"))
    prompt = (p.get("prompt") or "").strip() if (info or not p.get("references")) else h3_prompt.compose(p, duration_s)
    return CG.apply_trigger(prompt, info)


def _guided(p: dict) -> bool:
    """La richiesta porta una guida di latenti allineata (video sorgente da codificare o latenti gia pronti dell'edit)."""
    return bool(p.get("h3_guide")) or p.get("_h3_guide_latents") is not None


def _guide_setup(p: dict, edit) -> dict | None:
    """Piano della guida: None se la richiesta non ne ha. `latents` pronti (edit: sono gia quelli del clip) oppure `video` da leggere e codificare dopo il text encoder.
    Il video generato ha i fotogrammi della guida (17n+5, tetto `max_s`); la tela ha il rapporto del video guida."""
    if not _guided(p):
        return None
    from core import h3_guide as CG
    spec = dict(p.get("h3_guide") or {})
    fi = int(spec.get("frame_idx") or 0)
    if p.get("h3_veda"):
        raise RuntimeError("La guida allineata non funziona con l'attenzione sparsa Veda (il predittore e addestrato senza righe di guida): spegni Veda.")
    lat = p.get("_h3_guide_latents")
    if lat is not None:
        return {"latents": lat, "frame_idx": fi, "steps": int(lat.shape[2]), "video": None, "frames": None, "info": None, "spec": spec}
    video = str(spec.get("video") or "")
    if not os.path.isfile(video):
        raise RuntimeError(f"video guida non trovato: {video}")
    import h3_guide as HG
    info = HG.probe(video)
    plan = CG.plan_guide(info["n24"], frame_idx=fi, max_frames=int(round(float(spec.get("max_s") or 0) * S.FPS)),
                         target_frames=int(spec.get("target_frames") or 0))
    return {"latents": None, "frame_idx": plan["frame_idx"], "steps": plan["steps"], "video": video, "frames": plan["frames"],
            "guide_frames": plan["guide_frames"], "info": info, "spec": spec}


AUDIO_VAE_SR = 32000                     # sample rate dell'audio VAE di H3
REF_SHORT_EDGE = 1024                    # lato corto delle immagini di riferimento Ref2VA (ufficiale 2048)


def _decode_audio_av(path: str, sr: int = AUDIO_VAE_SR):
    """Traccia audio → tensore (canali, campioni) float32 GIÀ a `sr`, ricampionato con PyAV (libswresample).
    diffusers ricampionerebbe con torchaudio, che nel venv non c'è (e non serve aggiungerlo)."""
    import av
    import numpy as np
    import torch
    chunks = []
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        st = c.streams.audio[0]
        layout = "stereo" if (st.channels or 1) >= 2 else "mono"
        rs = av.AudioResampler(format="fltp", layout=layout, rate=sr)
        for frame in c.decode(st):
            for f in rs.resample(frame):
                chunks.append(f.to_ndarray())
        for f in rs.resample(None):                      # svuota il ricampionatore
            chunks.append(f.to_ndarray())
    if not chunks:
        return None
    return torch.from_numpy(np.concatenate(chunks, axis=1).astype("float32"))


def build_references(p: dict) -> list:
    from diffusers.modular_pipelines.minimax_h3 import (MiniMaxH3AudioReference, MiniMaxH3ImageReference,
                                                        MiniMaxH3VideoReference)
    cls = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
    out = []
    for r in p.get("references") or []:
        if r.get("role") == "background" and not r.get("token"):
            continue
        k = r.get("kind") or "image"
        if k not in cls or not os.path.isfile(r.get("path") or ""):
            raise RuntimeError(f"riferimento non valido: {r.get('path')} ({k})")
        if k == "audio":
            wav = _decode_audio_av(r["path"])
            if wav is None:
                raise RuntimeError(f"riferimento audio senza traccia audio: {r['path']}")
            out.append(cls[k](audio=wav, sample_rate=AUDIO_VAE_SR))
        elif k == "video":
            ref = cls[k].from_file(r["path"])
            if getattr(ref, "audio", None) is not None and ref.sample_rate != AUDIO_VAE_SR:
                ref = cls[k](frames=ref.frames, fps=ref.fps, audio=_decode_audio_av(r["path"]),
                             sample_rate=AUDIO_VAE_SR)
            out.append(ref)
        else:
            out.append(cls[k].from_file(r["path"]))
    return out


# ── fasi ───────────────────────────────────────────────────────────────────────────────────────
def _stage(blocks, names: list, components: dict):
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from collections import OrderedDict
    sub = OrderedDict((n, blocks.sub_blocks[n]) for n in names)
    pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline()
    pipe.update_components(**components)
    return pipe


def _inputs_for(stage, kw: dict) -> dict:
    """Gli ingressi della richiesta che QUESTA fase dichiara. Ogni fase va chiamata con i suoi: il denoise ha
    bisogno di num_frames/height/width anche se la prima fase li aveva già ricevuti (altrimenti usa i default,
    124 fotogrammi — visto il 27/09: chiesti 158, generati 124)."""
    names = {i.name for i in stage.blocks.inputs if i.name}
    return {k: v for k, v in kw.items() if k in names}


def _weights_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / 2**30
    except OSError:
        return 0.0


def reference_video_canvas_for(p: dict, height: int, width: int) -> dict:
    """Tela del video di movimento di una richiesta: negli edit (`_h3_edit`) lato lungo `h3_ref_max_side` (default S.REF_VIDEO_MAX_SIDE, 0 = tela piena);
    altrimenti la tela del bersaglio, come il pipeline di default."""
    max_side = int(p.get("h3_ref_max_side", S.REF_VIDEO_MAX_SIDE)) if p.get("_h3_edit") else 0
    return S.reference_video_canvas(height, width, max_side)


def _canvas_hw(kw: dict, canvas: tuple) -> tuple:
    """(altezza, larghezza) che avrebbe la tela 544p con lo stesso rapporto della richiesta."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    return resolve_canvas_size(int(kw["width"]), int(kw["height"]), 32, canvas[0], canvas[1])


def _preflight(W: dict, p: dict, mode: str, kw: dict, free_gb: float, frames: int):
    """Stima il picco di VRAM (pesi + LoRA + attivazioni per la lunghezza reale della sequenza) e confronta con la
    VRAM libera di ADESSO. Se nemmeno con parte dei blocchi in RAM ci sta: RuntimeError con numeri e rimedi."""
    from core import h3_vram_plan as HP
    sub = "transformer_ref" if mode == "ref2va" else "transformer"
    try:
        dims = HP.dims_from_config(json.loads((Path(W["configs"]) / sub / "config.json").read_text("utf-8")))
    except (OSError, ValueError):
        dims = dict(HP.DEFAULT_DIMS)
    h, w = int(kw["height"]), int(kw["width"])
    keyframes = int(bool(p.get("image_path"))) + int(bool(p.get("last_image_path")))
    video_canvas = reference_video_canvas_for(p, h, w)
    refs = (HP.ref_rows_estimate(p.get("references"), int(p.get("h3_ref_short_edge") or REF_SHORT_EDGE), h, w, (video_canvas["height"], video_canvas["width"]))
            if mode == "ref2va" else 0)
    gsteps = int(kw.get("_guide_steps") or 0)
    if gsteps:                                       # la guida e un blocco di righe condizione in piu, come un riferimento video
        from core import h3_guide as CG
        refs += CG.guide_rows(gsteps, h, w)
    tokens = HP.sequence_tokens(frames, h, w, keyframes=keyframes, ref_rows=refs)
    w_gb = _weights_gb(W["dit"])
    # TUTTE le LoRA (turbo + utente); +25 %: le LoRA «diff» tengono in VRAM anche una matrice identità per modulo
    l_gb = 1.25 * sum(_weights_gb(path) for path, _ in lora_specs(W, p))
    te_gb = _weights_gb(W["te"])
    avail = free_gb - 0.4                          # lo stesso margine del tetto per-processo
    if te_gb + 2.0 > avail:                        # fase 1: il text encoder è l'unico modello in VRAM
        raise RuntimeError(f"VRAM insufficiente per il text encoder di MiniMax H3 ({os.path.basename(W['te'])}: "
                           f"{te_gb:.1f} GB + 2 di lavoro, liberi {avail:.1f} GB). Scegli un text encoder più "
                           f"leggero (nvfp4) o chiudi i programmi che usano la GPU.")
    plan0 = HP.plan(avail, tokens, dims, weights_gb=w_gb, lora_gb=l_gb, chunked=_chunked(),
                    act_factor=HP.VIDEO_REF_ACT_FACTOR if (mode == "ref2va" and (HP.has_video_refs(p.get("references")) or gsteps)) else 1.0)
    modo = "tutto residente" if plan0["offload_gb"] <= 0 else f"blocchi in RAM: {plan0['offload_gb']:.1f} GB"
    _log(f"piano VRAM: {tokens // 1000}k token · pesi {w_gb:.1f} + LoRA {l_gb:.1f} + attivazioni "
         f"{plan0['act_gb']:.1f} + riserva = {plan0['need_gb']:.1f} GB · liberi {avail:.1f} GB → {modo}")
    if not plan0["fits"]:
        raise RuntimeError(HP.explain_failure(plan0, height=h, width=w, num_frames=frames, free_gb=avail, dims=dims,
                                              extra_rows=refs, canvas_544=_canvas_hw(kw, S.CANVAS["544"])))
    return dims, plan0


def _chunked() -> bool:
    return os.environ.get("H3_BLOCK_CHUNK") != "0"


def _fit_after_load(tr, dims: dict, plan0, kw: dict, frames: int) -> None:
    """DiT e LoRA sono in VRAM: rimisura la VRAM libera VERA e, se le attivazioni non ci stanno, tiene in RAM
    gli ultimi blocchi quanto basta (workers/h3_offload). Se non basta nemmeno così: errore chiaro."""
    import h3_offload as O
    import h3_chunk as K
    from core import h3_vram_plan as HP
    K.enable_chunked_blocks(tr, _log)               # qkv / norme / MLP a blocchi di token (picco da blocco, non da sequenza)
    free = O.free_gb_now() - 0.4
    # margine incluso in RESERVE_GB: se la VRAM libera non lo copre si tengono blocchi in RAM, non si spera
    need = plan0["act_gb"] + HP.RESERVE_GB
    blocks = getattr(tr, "transformer_blocks", [])
    n = O.blocks_for_deficit(blocks, need - free) if free < need else 0
    if os.environ.get("H3_STREAM_ALL") == "1":          # ripiego dopo un OOM: tutti i blocchi del DiT che si possono spostare restano in RAM (streaming)
        n = max(n, int(HP.OFFLOADABLE * len(blocks)))
        _log(f"memoria insufficiente al primo tentativo: streaming completo del modello, {n}/{len(blocks)} blocchi del DiT in RAM (piu lento)")
    # l'offloader esiste SEMPRE (anche con n = 0): se a runtime la VRAM non basta, workers/h3_chunk libera altri
    # blocchi al volo invece di andare in OOM (la prova del 30/10 aveva ~11 GB residenti non previsti)
    O.enable_block_offload(blocks, n, "cuda", _log)
    free2 = O.free_gb_now() - 0.4
    _log(f"VRAM dopo il piano: liberi {free2:.1f} GB (servono ~{need:.1f}); "
         f"{'tutto residente' if n == 0 else f'{n} blocchi in RAM'}; offload dinamico attivo")


def cap_fraction(free_gb: float, total_gb: float, reserved_gb: float = 0.0) -> float:
    """Frazione di `torch.cuda.set_per_process_memory_fraction`: (VRAM libera + gia riservata dal processo - margine 0,4 GB) / totale, 0,2..1,0."""
    return max(0.2, min(1.0, (free_gb + reserved_gb - 0.4) / total_gb))


def round_keyframe_conditions(state, turbo: str, mode: str) -> None:
    """Pika H3 Flex FL2VA (`bf16_conditioning_v1` del manifest, diffusers d0b9b1b5 encoders.py:~373): i latenti di condizionamento dei keyframe si arrotondano a bf16 prima
    del denoise. Solo con una turbo che lo dichiara (S.TURBO_SAMPLER[...]["keyframe_bf16"]) e solo in FL2VA: Ref2VA usa `fp32_conditioning_v1`, cioe quello di sempre."""
    import torch
    sampler = S.TURBO_SAMPLER.get(turbo or "") or {}
    latents = state.get("condition_latents")
    if mode != "fl2va" or not sampler.get("keyframe_bf16") or not latents:
        return
    state.set("condition_latents", [item.to(torch.bfloat16).to(torch.float32) for item in latents])
    _log(f"condizionamento dei keyframe arrotondato a bf16 ({len(latents)} latenti): politica bf16_conditioning_v1 della turbo {turbo}")


def scheduler_classes(turbo: str, seed: int) -> tuple:
    """(classe dello scheduler video, classe dello scheduler audio): l'Euler di diffusers, o per le turbo con campionatore proprio (S.TURBO_SAMPLER, es. Pika a 6 passi)
    la sottoclasse «renoise» con la griglia dei nodi dell'adattatore. Video e audio hanno semi di rumore diversi."""
    from diffusers import MiniMaxH3Scheduler
    sampler = S.TURBO_SAMPLER.get(turbo or "")
    if not sampler:
        return MiniMaxH3Scheduler, MiniMaxH3Scheduler
    if sampler["method"] != "renoise":
        raise RuntimeError(f"campionatore sconosciuto per la turbo {turbo}: {sampler['method']}")
    import h3_renoise_scheduler
    knots = tuple(sampler["knots"])
    return (h3_renoise_scheduler.make_renoise_scheduler_class(MiniMaxH3Scheduler, knots, seed),
            h3_renoise_scheduler.make_renoise_scheduler_class(MiniMaxH3Scheduler, knots, seed + 1))


def decode_dtype():
    """Dtype dei pesi del VAE video al decode: fp16 (default) o fp32 (H3_VAE_DECODE_DTYPE=fp32). Il pipeline decodifica gia dentro un `torch.autocast(float16)` (diffusers
    modular_pipelines/minimax_h3/decoders.py:~187) e il file del VAE e fp16: con i pesi fp16 i valori sono gli stessi dell'autocast (fp32 -> fp16 e esatto), meta memoria
    (4,9 invece di 9,7 GB) e meta tempo di caricamento. Come ComfyUI (sd.py: working_dtypes del VAE H3 = fp16, fp32). Verificato su CPU: fp16 contro fp32 PSNR 63,5 dB."""
    import torch
    return torch.float32 if os.environ.get("H3_VAE_DECODE_DTYPE", "fp16").lower() == "fp32" else torch.float16


def _decode_and_free(blocks, g_dec, W, state, *, load_vae_fn=None, load_avae_fn=None, stage_fn=None):
    """Decodifica col VAE video (fp16, ~4,9 GB; fp32 con H3_VAE_DECODE_DTYPE=fp32) e il VAE audio e LI SVUOTA SUBITO. Il `del st` da solo non li liberava (giro 02/11, edit a clip in
    catena: dopo il decode del clip 1 restavano ~10 GB di VAE in VRAM e il DiT del clip 2 moriva per OOM a 27,7 GB allocati). Con un solo clip il
    problema non si vedeva perche il processo finiva."""
    lv, la, sf = load_vae_fn or load_vae, load_avae_fn or load_audio_vae, stage_fn or _stage
    v, a = (lv(W, dtype=decode_dtype()) if lv is load_vae else lv(W)), la(W)
    st = sf(blocks, g_dec, {"vae": v, "audio_vae": a})
    try:
        return st(state=state, output=["videos", "audio", "sampling_rate"])
    finally:
        _drop_module_weights(v)
        _drop_module_weights(a)
        del st, v, a
        _gc()


def _encode_text(W: dict, p: dict, mode: str, kw: dict, edit, blocks_text, g_text: list):
    """Fase del testo: carica il text encoder, codifica prompt e riferimenti, lo libera e ritorna lo stato del pipeline."""
    import torch
    from transformers import AutoTokenizer, Qwen3VLProcessor
    cfgdir = Path(W["configs"])
    t0 = time.time()
    te = load_text_encoder(W)
    _log(f"text encoder pronto in {time.time() - t0:.0f}s — VRAM {_vram()}")
    comps = dict(text_encoder=te, tokenizer=AutoTokenizer.from_pretrained(str(cfgdir / "tokenizer")),
                 processor=Qwen3VLProcessor.from_pretrained(str(cfgdir / "processor")))
    st = _stage(blocks_text, g_text, comps)
    if mode == "ref2va" and kw.get("references"):
        # il pipeline porta ogni immagine di riferimento a 2048 px di lato corto (anche ingrandendola): con 32 GB
        # l'attenzione del Qwen3-VL su quei token va in OOM (visto 27/09). 1024 = 4× meno token, identità intatta.
        st.update_components(reference_image_short_edge=int(p.get("h3_ref_short_edge") or REF_SHORT_EDGE))
    if edit:
        # I VIDEO di riferimento vengono portati sulla tela di `canvas_short_edge` (default del pacchetto 768, max 768x1344): il riferimento dell'edit
        # va alla STESSA tela del bersaglio (480p), altrimenti ha 2,5x i token del previsto (giro 02/11: righe di riferimento 26k invece di 10k,
        # OOM nel DiT; sequenza del TE ~9k token, OOM da 9 GB nell'attenzione). Dal 04/11 il riferimento sta a ~512 px sul lato lungo come nell'originale
        # (S.REF_VIDEO_MAX_SIDE; h3_ref_max_side=0 = tela del bersaglio): posa e tempi non vogliono piu pixel.
        _th, _tw = edit["canvas"]
        video_canvas = reference_video_canvas_for(p, _th, _tw)
        st.update_components(canvas_short_edge=int(video_canvas["short_edge"]), canvas_max_pixels=int(video_canvas["max_pixels"]))
        _log(f"video di riferimento (movimento): tela {video_canvas['width']}x{video_canvas['height']} (bersaglio {_tw}x{_th}, lato lungo {int(p.get('h3_ref_max_side', S.REF_VIDEO_MAX_SIDE)) or 'pieno'})")
        # Anche le FOTO di riferimento alla tela del bersaglio: nel workflow MIT di Openjutsu (simple.api.json, nodo MiniMaxH3ReferenceToVideo) l'input
        # `ref_image_size` e «match». A 1024 di lato corto una foto vale ~1,7k token nel TE e nel DiT, alla tela 480 sono ~0,4k.
        st.update_components(reference_image_short_edge=int(min(_th, _tw)))
    import h3_te_attn as _TA                   # attenzione del TE a blocchi di query (esatta): niente matrice L x L in un colpo solo
    # Cronometro per parte (03/11: ~100 s per clip nell'Edit H3 senza sapere dove): visione vs linguaggio, chiamate e lunghezza della sequenza
    _tt, _hooks = {}, []

    def _timed(name, mod):
        def pre(_m, args, kwargs):
            torch.cuda.synchronize()
            _tt.setdefault(name, [0.0, 0, 0])[2] = time.time()
        def post(_m, args, kwargs, out):
            torch.cuda.synchronize()
            r = _tt[name]
            r[0] += time.time() - r[2]
            r[1] += 1
            x = kwargs.get("inputs_embeds") if isinstance(kwargs, dict) else None
            if x is not None:
                _tt[name + "_len"] = int(x.shape[-2])
        _hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        _hooks.append(mod.register_forward_hook(post, with_kwargs=True))
    inner = getattr(te, "model", te)
    for _n in ("visual", "language_model"):
        if getattr(inner, _n, None) is not None:
            _timed(_n, getattr(inner, _n))
    t1 = time.time()
    with _TA.query_chunked_attention():
        state = st(**_inputs_for(st, kw))
    for h in _hooks:
        h.remove()
    _log("codifica TE: " + f"totale {time.time() - t1:.0f}s · " + " · ".join(
        f"{k} {v[0]:.0f}s in {v[1]} chiamate" for k, v in _tt.items() if isinstance(v, list))
        + (f" · sequenza {_tt['language_model_len']} token" if "language_model_len" in _tt else ""))
    _drop_module_weights(te)                   # il `del` da solo non basta (visto sul DiT e sul VAE): i pesi del TE escono dalla VRAM comunque
    del st, te, comps                          # comps teneva ancora il TE: senza, resta in VRAM
    # Dopo `_drop_module_weights` restavano vivi ~12 GB di pesi quantizzati del TE (150 tensori uint8 da 65 MB, 100 da 21 MB, scale fp8): turno 40 del 04/11, 736p,
    # OOM nella codifica del VAE con 28,75 GB allocati; e a 480p «transformer pronto» a 26,1 GB con DiT+LoRA da 12,6. Il DiT non e ancora caricato, quindi ogni tensore
    # quantizzato grande ancora su GPU e del TE e si svuota sul posto.
    _log(f"pesi quantizzati del testo ancora vivi: {purge_big_quantized_tensors(log=_log):.2f} GB liberati")
    _gc()
    _log(f"testo codificato in {time.time() - t0:.0f}s")

    return state


def _text_state(W: dict, p: dict, mode: str, kw: dict, edit, blocks_text, g_text: list):
    """Stato del testo del clip: dalla cache del job se c'e (riavvio dopo OOM: niente ricodifica), altrimenti lo codifica e lo salva."""
    from diffusers.modular_pipelines import PipelineState
    import h3_te_cache
    cache = h3_te_cache.TextStateCache.for_request(p, _log)
    payload = cache.load() if cache else None
    if payload is not None:
        _log("testo: ripreso dalla cache del clip (nessuna ricodifica)")
        return h3_te_cache.TextStateCache.restore(payload, kw, "cuda", PipelineState())
    state = _encode_text(W, p, mode, kw, edit, blocks_text, g_text)
    if cache:
        cache.save(state.values, kw, state.kwargs_mapping)
    return state


def run(p: dict) -> str:
    from core import resources
    # nessuna soglia fissa sui GB: la VRAM necessaria si calcola più sotto da pesi, LoRA, durata e tela
    resources.require(commit_gb=float(p.get("need_commit_gb") or 8), vram_gb=float(p.get("need_vram_gb") or 0),
                      what="MiniMax H3")
    resources.start_commit_guard(2.0, what="MiniMax H3")
    import torch
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    from diffusers.utils.export_utils import encode_video
    from PIL import Image
    torch.set_grad_enabled(False)
    t_all = time.time()
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from _watchdog import start_watchdog
        start_watchdog(int(p.get("parent_pid") or 0))
    except Exception:                                                # noqa: BLE001
        pass
    W = resolve(p)
    mode = S.mode_for(p)
    _free_gpu_servers()
    free0, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    # TETTO: meglio un OOM chiaro che il driver WDDM che riversa la VRAM nella RAM di sistema.
    # Il tetto vale per il TOTALE riservato da questo processo, non per la VRAM libera: se il processo ha gia memoria riservata (clip precedente
    # nello stesso processo) quella va SOMMATA alla libera, altrimenti si conta due volte (giro 02/11: «21,29 GiB allowed» con 6,8 GiB liberi).
    reserved0 = torch.cuda.memory_reserved() / 2**30
    frac = cap_fraction(free0, total, reserved0)
    torch.cuda.set_per_process_memory_fraction(frac)
    import h3_chunk as _K
    _K.CAP_BYTES = frac * total * 2**30                 # il tetto serve al controllo di margine per blocco
    _log(f"{mode} · {p.get('h3_dit') or 'profilo ' + str(p.get('h3_profile') or 'nvfp4')} · "
         f"VRAM libera {free0:.1f}/{total:.1f} GB")

    wf = "ref2va" if mode == "ref2va" else ("fl2va" if (p.get("image_path") or p.get("last_image_path")) else "t2va")
    blocks = MiniMaxH3Blocks().get_workflow(wf)
    edit = p.get("_h3_edit")                                     # edit con maschera a clip (workers/h3_edit_run.py): lunghezza e tela le decide lui
    guide = _guide_setup(p, edit)                                # None senza guida: tutto quel che segue e come sempre
    guide_only = bool(guide) and not p.get("references")           # solo guida (LMS, VFX Edit, Style Transfer a parole): niente riferimenti nativi
    blocks_text = MiniMaxH3Blocks().get_workflow("t2va") if guide_only else blocks
    if guide or p.get("h3_ref_clock_align"):
        import h3_guide as HG
        if p.get("h3_ref_clock_align"):
            if not p.get("references"):
                raise RuntimeError("h3_ref_clock_align: serve un riferimento VIDEO nativo (e quello che si allinea)")
        HG.install(blocks)                                         # layout + condizioni del denoise Ref2VA conoscono la guida; senza `guide_latents` restano quelli di diffusers
    names = list(blocks.sub_blocks.keys())
    g_text = [n for n in blocks_text.sub_blocks.keys() if n in ("before_encode", "text_encoder")]
    g_venc = [] if guide_only else [n for n in names if n == "vae_encoder"]
    g_den = [n for n in names if n.startswith("denoise")]
    g_dec = [n for n in names if n.startswith("decode")]

    # ingressi della richiesta (edit e guida sono definiti piu sopra: servono gia alla scelta dei blocchi)
    frames = (int(edit["frames"]) if edit else
              (guide["frames"] if guide and guide["frames"] else S.snap_frames(float(p.get("duration_s") or (int(p.get("num_frames") or 121) / S.FPS)))))
    kw = {"prompt": build_prompt(p, S.frames_seconds(frames)), "num_frames": frames}
    if p.get("h3_ref_clock_align"):
        kw["ref_clock_align"] = True                               # orologio rotary del riferimento video = quello del video generato (variante sperimentale)
    if p.get("image_path"):
        kw["image"] = Image.open(p["image_path"]).convert("RGB")
    if p.get("last_image_path"):
        kw["last_image"] = Image.open(p["last_image_path"]).convert("RGB")
    # tela: il RAPPORTO viene dal keyframe (come fa il pipeline) o dalla risoluzione scelta nel tab; i lati
    # esatti li decide la griglia di H3 (lato corto 768 o 544, area massima, multipli di 32)
    short, maxpx = S.CANVAS.get(str(p.get("h3_canvas") or "768"), S.CANVAS["768"])
    key = kw.get("image") or kw.get("last_image")
    if key is not None:
        ratio = key.size                                             # (w, h)
    elif guide and guide.get("info"):
        ratio = (guide["info"]["w"], guide["info"]["h"])             # la guida occupa la STESSA griglia del video generato: il rapporto e quello del sorgente
    elif str(p.get("h3_aspect") or "") in S.ASPECTS:
        ratio = S.ASPECTS[p["h3_aspect"]]
    elif int(p.get("width") or 0) and int(p.get("height") or 0):
        ratio = (int(p["width"]), int(p["height"]))
    else:
        ratio = (16, 9)
    kw["height"], kw["width"] = tuple(edit["canvas"]) if edit else resolve_canvas_size(ratio[0], ratio[1], 32, short, maxpx)
    if mode == "ref2va":
        if p.get("references") or not guide:
            kw["references"] = build_references(p)
            if not kw["references"]:
                raise RuntimeError("Ref2VA senza riferimenti validi")
    fused = S.CUSTOM_DITS.get(str(p.get("h3_dit") or ""), {})
    turbo = "" if fused.get("turbo_fused") else str(p.get("h3_turbo") or "")
    if fused.get("turbo_fused"):                       # DiT col turbo nei pesi (10Eros-Max): passi e shift suoi
        nfe, vshift = int(fused["nfe"]), float(fused["video_shift"])
    else:
        nfe = S.TURBO[turbo][1] if turbo else int(p.get("steps") or S.DEFAULT_NFE)
        vshift = S.TURBO[turbo][2] if turbo else float(p.get("h3_video_shift") or S.VIDEO_SHIFT)
    seed = int(p.get("seed") or 0)
    _log(f"workflow {wf}: {frames} fotogrammi ({S.frames_seconds(frames):.2f}s) · "
         f"{kw.get('width', 'auto')}x{kw.get('height', 'auto')} · {nfe} passi"
         f"{' turbo' if turbo else ''} · shift video {vshift} · seme {seed}")
    _log("prompt:\n" + kw["prompt"])

    # pianificazione VRAM PRIMA di caricare qualunque cosa: errore chiaro invece di un OOM a metà denoise
    if guide:
        kw["_guide_steps"] = guide["steps"]
    dims, plan0 = _preflight(W, p, mode, kw, free0, frames)

    # 1) text encoder (con cache nel job del clip)
    state = _text_state(W, p, mode, kw, edit, blocks_text, g_text)

    # 2) VAE encoder (keyframe / riferimenti)
    if guide and guide["latents"] is not None:                   # edit: la guida e il latente del clip, gia codificato dal chiamante
        kw["guide_latents"], kw["guide_frame_idx"] = guide["latents"], guide["frame_idx"]
    if guide_only:                                               # nessun riferimento nativo: il denoise Ref2VA parte con liste vuote (layout = testo + guida + video)
        kw.update(normalized_references=[], condition_latents=[], audio_condition_latents=[])
    if g_venc or (guide and guide["latents"] is None):
        t0 = time.time()
        comps = {"vae": load_vae(W, encoder_only=True)}            # qui il VAE codifica soltanto (riferimenti, guida); il decode lo ricarica intero
        if mode == "ref2va" and g_venc:
            comps["audio_vae"] = load_audio_vae(W)
        if g_venc:
            st = _stage(blocks, g_venc, comps)
            state = st(state=state, **_inputs_for(st, kw))
            del st
            round_keyframe_conditions(state, turbo, mode)
        if guide and guide["latents"] is None:
            t2 = time.time()
            fr = HG.read_guide_frames(guide["video"], int(kw["width"]), int(kw["height"]), guide["guide_frames"])
            kw["guide_latents"], kw["guide_frame_idx"] = HG.encode_guide(comps["vae"], fr), guide["frame_idx"]
            _log(f"guida allineata: {guide['guide_frames']} fotogrammi -> {tuple(kw['guide_latents'].shape)} in {time.time() - t2:.0f}s (frame_idx {guide['frame_idx']})")
            del fr
        _enc_modules = list(comps.values())
        del comps
        # Il VAE fp32 (9,7 GB) restava vivo in VRAM durante il denoise: lo teneva ancora qualcosa dentro `blocks`/`state` (il `del` non basta;
        # per il TE funziona, per il VAE no). Giro 31/10 22:27, 10Eros int8: dump «per RADICE» = AutoencoderKLMiniMaxH3 9,70 GB fp32 -> OOM.
        # Si svuotano i pesi sul posto: la VRAM torna libera comunque, e al decode il VAE viene ricaricato da load_vae.
        for _m in _enc_modules:
            _drop_module_weights(_m)
        del _enc_modules
        _gc()
        _log(f"condizionamenti VAE in {time.time() - t0:.0f}s")

    # 3) denoise
    t0 = time.time()
    tr = load_transformer(W, mode, p)
    _log(f"transformer pronto in {time.time() - t0:.0f}s — VRAM {_vram()}")
    _fit_after_load(tr, dims, plan0, kw, frames)
    tname = "transformer_ref" if mode == "ref2va" else "transformer"
    if edit:                                                     # righe tenute legate alla sorgente
        import h3_edit_hold as _EH
        video_class, audio_class = scheduler_classes(turbo, seed)
        _vs = _EH.make_hold_scheduler_class(video_class)(shift=vshift)
        _as = _EH.make_hold_scheduler_class(audio_class)(shift=S.AUDIO_SHIFT)
        _ps = tuple(tr.config.patch_size)
        _vs.set_hold(*edit["video_hold"](_ps))
        if edit.get("audio_hold") is not None:
            _as.set_hold(*edit["audio_hold"]())
    else:
        video_class, audio_class = scheduler_classes(turbo, seed)
        _vs, _as = video_class(shift=vshift), audio_class(shift=S.AUDIO_SHIFT)
    st = _stage(blocks, g_den, {tname: tr, "scheduler": _vs, "audio_scheduler": _as})
    t1 = time.time()
    torch.cuda.synchronize()
    base_gb = torch.cuda.memory_allocated() / 2**30            # pesi + LoRA residenti prima del denoise
    torch.cuda.reset_peak_memory_stats()
    _den_kw = _inputs_for(st, kw)
    if edit:                                                     # il rumore iniziale lo scegliamo noi: lo stesso rumore serve a rumorizzare la sorgente
        _names = {i.name for i in st.blocks.inputs if i.name}
        for _k, _v in (("latents", edit["noise"]), ("audio_latents", edit.get("audio_noise"))):
            if _v is None:
                continue
            if _k not in _names:
                raise RuntimeError(f"edit: la fase di denoise non dichiara l'ingresso «{_k}» (diffusers cambiato?)")
            _den_kw[_k] = _v
    state = st(state=state, **_den_kw, num_inference_steps=nfe + 1,
               generator=torch.Generator("cpu").manual_seed(seed))
    _log(f"denoise {nfe} passi in {time.time() - t1:.0f}s ({(time.time() - t1) / max(1, nfe):.1f}s/passo) — "
         f"VRAM {_vram()}")
    _log(f"calibrazione: attivazioni misurate {torch.cuda.max_memory_allocated() / 2**30 - base_gb:.1f} GB "
         f"(stimate {plan0['act_gb']:.1f}) su {plan0['tokens'] // 1000}k token")
    if getattr(tr, "_veda_state", None) is not None:
        _log(f"Veda: {tr._veda_state.summary()}")
    # Il DiT NON viene davvero liberato dal solo `del` (giro 31/10 22:27, 10Eros int8 al decode: 12,6 GB di int8 + 9 GB di fp32 dei blocchi ancora
    # vivi in VRAM — i blocchi risultano radici orfane nel dump — piu il VAE fp32 da 9,7 GB caricato per il decode = OOM). Si svuotano i pesi
    # sul posto prima di caricare il VAE: la VRAM torna libera anche se qualcosa (h3_chunk/offload/Veda) tiene ancora un riferimento.
    _drop_module_weights(tr)
    del st, tr
    _gc()
    try:
        purge_big_quantized_tensors(log=_log)                 # sicurezza: vedi la funzione (14,3 GB di int8 sopravvissuti al giro del 03/11)
        _gc()
    except Exception as _pe:                                  # noqa: BLE001 — mai far fallire il job per la pulizia
        _log(f"pulizia dei pesi quantizzati saltata ({_pe})")

    # 4) decode
    t0 = time.time()
    res = _decode_and_free(blocks, g_dec, W, state)
    _log(f"decode in {time.time() - t0:.0f}s")
    if edit:
        return res
    out = p.get("output") or str(REPO / "output" / f"minimax_h3_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with_audio = p.get("generate_audio", True) is not False
    a_out, a_sr = (res["audio"][0].float().cpu(), int(res["sampling_rate"])) if with_audio else (None, None)
    if guide and guide["video"] and guide["info"]["has_audio"] and (guide["spec"] or {}).get("keep_audio", True):
        a_out, a_sr = HG.trim_audio(_decode_audio_av(guide["video"]), frames), AUDIO_VAE_SR       # il suono e quello del sorgente (come nel «tieni l'audio» dell'edit)
    encode_video(res["videos"][0], fps=S.FPS, output_path=out, audio=a_out, audio_sample_rate=a_sr)
    _log(f"fatto in {time.time() - t_all:.0f}s → {out}")
    return out


def friendly_error(e: BaseException) -> str:
    """Un OOM di CUDA non deve arrivare alla GUI come «nessun output»: l'ultima riga dice cosa è successo e cosa fare."""
    if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
        return ("memoria GPU (VRAM) esaurita durante la generazione di MiniMax H3. Riduci la durata o scegli la tela "
                "544p, usa meno LoRA o un profilo più leggero (int4), e chiudi i programmi che usano la GPU. "
                f"Dettaglio: {str(e).splitlines()[0][:200]}")
    return str(e)


def dump_gpu_state(log=print, top: int = 10) -> None:
    """Allo OOM: riepilogo dell'allocatore + i tensori CUDA vivi piu' grandi (per sapere cosa occupa la VRAM)."""
    import gc
    import torch
    try:
        log(torch.cuda.memory_summary(abbreviated=True))
        storages = {}
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    st = o.untyped_storage()
                    storages.setdefault(st.data_ptr(), (st.nbytes(), tuple(o.shape), str(o.dtype)))
            except Exception:                                        # noqa: BLE001
                continue
        tot = sum(v[0] for v in storages.values())
        log(f"[oom] tensori CUDA vivi: {len(storages)} storage, {tot / 2**30:.2f} GB")
        for nb, shape, dt in sorted(storages.values(), key=lambda v: -v[0])[:top]:
            log(f"[oom]   {nb / 2**30:6.2f} GB  {dt}  {list(shape)}")
        groups = {}
        for nb, shape, dt in storages.values():
            g = groups.setdefault((shape, dt), [0, 0])
            g[0] += 1
            g[1] += nb
        # PROPRIETARIO dei tensori (31/10): a quale modulo di primo livello appartengono (DiT, VAE, TE, …), cosi si vede CHI occupa i 24,7 GB
        # (nel giro 30-31/10: 14 GB di int8 del DiT + 9 GB di fp32 con layer da 2048 canali che nessun piano contava).
        try:
            own = {}
            for o in gc.get_objects():
                if isinstance(o, torch.nn.Module):
                    for nm, t in list(o.named_parameters(recurse=True)) + list(o.named_buffers(recurse=True)):
                        if t.is_cuda:
                            key = t.untyped_storage().data_ptr()
                            prev = own.get(key)
                            if prev is None or len(nm) < len(prev[1]):          # il nome piu corto = il modulo piu vicino alla radice
                                own[key] = (type(o).__name__, nm)
            by = {}
            for ptr, (nb, shape, dt) in storages.items():
                cls, nm = own.get(ptr, ("(senza modulo)", ""))
                g = by.setdefault((cls, dt), [0, 0, nm])
                g[0] += 1
                g[1] += nb
            log("[oom] per proprietario (classe del modulo, dtype):")
            for (cls, dt), (n, nb, nm) in sorted(by.items(), key=lambda kv: -kv[1][1])[:8]:
                log(f"[oom]   {nb / 2**30:6.2f} GB  {n} x {dt}  {cls}  es. {nm[:60]}")
        except Exception as e2:                                      # noqa: BLE001
            log(f"[oom] attribuzione non riuscita: {e2}")
        # 2ª attribuzione, per RADICE: la prima (sopra) si fermava alla classe del Linear foglia («Linear … weight») e non diceva di quale
        # modello fossero i 9 GB fp32 con layer da 2048 canali. Qui si risale alle radici (moduli che nessun altro modulo contiene).
        try:
            mods = [o for o in gc.get_objects() if isinstance(o, torch.nn.Module)]
            child = set()
            for o in mods:
                for c in o.children():
                    child.add(id(c))
            rows = []
            for o in mods:
                if id(o) in child:
                    continue
                by = {}
                for t in list(o.parameters(recurse=True)) + list(o.buffers(recurse=True)):
                    if t.is_cuda:
                        by[str(t.dtype)] = by.get(str(t.dtype), 0) + t.untyped_storage().nbytes()
                tot = sum(by.values())
                if tot > 2**28:
                    nl = sum(1 for m in o.modules() if m.__class__.__name__ == "Linear")
                    rows.append((tot, type(o).__name__, by, nl))
            log("[oom] per RADICE (modello intero che possiede i pesi su GPU):")
            for tot, cls, by, nl in sorted(rows, key=lambda r: -r[0])[:8]:
                det = ", ".join(f"{k.replace('torch.', '')} {v / 2**30:.2f}" for k, v in sorted(by.items(), key=lambda kv: -kv[1]))
                log(f"[oom]   {tot / 2**30:6.2f} GB  {cls}  ({nl} Linear)  [{det}]")
        except Exception as e3:                                      # noqa: BLE001
            log(f"[oom] attribuzione per radice non riuscita: {e3}")
        log("[oom] gruppi (forma, dtype) per byte totali:")
        for (shape, dt), (n, nb) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:top]:
            log(f"[oom]   {nb / 2**30:6.2f} GB  {n} x {dt} {list(shape)}")
        _oom_table(log, gc, torch, storages, groups)
    except Exception as e:                                           # noqa: BLE001
        log(f"[oom] dump non riuscito: {e}")


def _oom_table(log, gc, torch, storages: dict, groups: dict) -> None:
    """TABELLA per categoria al momento dell'OOM (giro 03/11: 27,1 GB allocati, il piano ne prevedeva 24,4): pesi residenti del DiT, blocchi offloadati,
    LoRA, VAE, token di sequenza (flusso residuo / Q-K-V / riferimenti), buffer h3_chunk, altro. E CHI TIENE IN VITA il gruppo piu grande di tensori
    non-peso (referrer: classe e funzione del frame): e cosi che si e trovato il ciclo di riferimenti di each_chunk (50 flussi residui da 0,38 GB vivi)."""
    try:
        wptr = set()
        for o in gc.get_objects():
            if isinstance(o, torch.nn.Module):
                for t in list(o.parameters(recurse=True)) + list(o.buffers(recurse=True)):
                    if t.is_cuda:
                        wptr.add(t.untyped_storage().data_ptr())
        weights = sum(v[0] for k, v in storages.items() if k in wptr)
        seq = other = 0
        for k, (nb, shape, dt) in storages.items():
            if k in wptr:
                continue
            if len(shape) >= 3 and shape[1] >= 4096:
                seq += nb
            else:
                other += nb
        try:
            import h3_offload as _O
            off = _O.CURRENT
            n_off = len(getattr(off, "hosted", ()) or ()) if off is not None else 0
        except Exception:                                            # noqa: BLE001
            n_off = 0
        log("[oom] TABELLA per categoria:")
        log(f"[oom]   pesi sulla GPU (DiT residente + LoRA + VAE + encoder): {weights / 2**30:6.2f} GB  (blocchi del DiT in RAM: {n_off})")
        log(f"[oom]   attivazioni di sequenza (flusso residuo, Q/K/V, token dei riferimenti video+immagine, latente sorgente): {seq / 2**30:6.2f} GB")
        log(f"[oom]   altro (condizionamenti, tabelle rotary, buffer piccoli): {other / 2**30:6.2f} GB")
        # i non-pesi piu grandi: chi li tiene vivi
        big = [(g[1], k) for k, g in groups.items() if len(k[0]) >= 3 and k[0][1] >= 4096 and g[0] >= 2]
        if big:
            big.sort(reverse=True)
            (nb, (shape, dt)) = big[0]
            cnt = groups[(shape, dt)][0]
            log(f"[oom] il gruppo piu grande di attivazioni: {cnt} x {dt} {list(shape)} = {nb / 2**30:.2f} GB; chi li tiene vivi (primi 3):")
            done = 0
            for o in gc.get_objects():
                try:
                    if torch.is_tensor(o) and o.is_cuda and tuple(o.shape) == tuple(shape) and str(o.dtype) == dt:
                        refs = []
                        for r in gc.get_referrers(o)[:6]:
                            if type(r).__name__ == "frame":
                                refs.append(f"frame {r.f_code.co_name}")
                            elif type(r).__name__ == "cell":
                                refs.append("cella-chiusura")
                            else:
                                refs.append(type(r).__name__)
                        log(f"[oom]   referrer: {', '.join(refs)}")
                        done += 1
                        if done >= 3:
                            break
                except Exception:                                    # noqa: BLE001
                    continue
    except Exception as ex:                                          # noqa: BLE001
        log(f"[oom] tabella non riuscita: {ex}")


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: minimax_h3_infer.py params.json")
        return 2
    try:
        if sys.argv[1] == "--h3-edit-clip":                      # figlio dell'edit a clip: UN clip per processo (VRAM pulita)
            import h3_edit_run as _ER
            _ER.generate_clip(sys.argv[2], sys.modules[__name__])
            print(f"CLIP_PRONTO:{sys.argv[2]}", flush=True)          # NON «OK:»: l'interfaccia prende l'ultima riga «OK:» come file finale del job
            return 0
        _p = json.loads(Path(sys.argv[1]).read_text("utf-8"))
        if _p.get("h3_edit"):                                    # cambia sfondo / sostituisci personaggio con maschera (clip in catena)
            import h3_edit_run as _ER
            out = _ER.run_edit(_p, sys.modules[__name__])
        else:
            out = run(_p)
        print(f"OK:{out}", flush=True)
        return 0
    except Exception as e:                                           # noqa: BLE001
        traceback.print_exc()
        if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
            dump_gpu_state(lambda m: print(m, flush=True))            # prima dell'ERRORE: finale (ultima riga = causa)
        print(f"ERRORE: {friendly_error(e)}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
