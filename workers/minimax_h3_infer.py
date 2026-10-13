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


def _read(path: str, device: str = "cuda"):
    import _qlinear as Q
    return Q.read_safetensors(path, device)


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


def load_transformer(W: dict, mode: str, p: dict):
    import torch
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
    loras = []
    if W.get("turbo"):
        loras.append((W["turbo"], 1.0))
    for lo in p.get("loras") or []:
        path = lo.get("path") or lo.get("name") or ""
        if path and os.path.isfile(path):
            loras.append((path, float(lo.get("strength", 1.0))))
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
    return tr


def load_vae(W: dict):
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLMiniMaxH3
    import _qlinear as Q
    with init_empty_weights():
        vae = AutoencoderKLMiniMaxH3.from_config(AutoencoderKLMiniMaxH3.load_config(str(Path(W["configs"]) / "vae")))
    miss, unex = Q.load_into(vae, _read(W["vae"]), dtype=torch.float32)
    if miss or unex:
        raise RuntimeError(f"VAE video: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    return vae.to("cuda").eval()


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
    return h3_prompt.compose(p, duration_s)


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


def run(p: dict) -> str:
    from core import resources
    resources.require(commit_gb=float(p.get("need_commit_gb") or 8), vram_gb=float(p.get("need_vram_gb") or 26),
                      what="MiniMax H3")
    resources.start_commit_guard(2.0, what="MiniMax H3")
    import torch
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    from diffusers.utils.export_utils import encode_video
    from PIL import Image
    from transformers import AutoTokenizer, Qwen3VLProcessor
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
    # TETTO: meglio un OOM chiaro che il driver WDDM che riversa la VRAM nella RAM di sistema
    torch.cuda.set_per_process_memory_fraction(max(0.2, min(1.0, (free0 - 0.4) / total)))
    _log(f"{mode} · {p.get('h3_dit') or 'profilo ' + str(p.get('h3_profile') or 'nvfp4')} · "
         f"VRAM libera {free0:.1f}/{total:.1f} GB")

    cfgdir = Path(W["configs"])
    wf = "ref2va" if mode == "ref2va" else ("fl2va" if (p.get("image_path") or p.get("last_image_path")) else "t2va")
    blocks = MiniMaxH3Blocks().get_workflow(wf)
    names = list(blocks.sub_blocks.keys())
    g_text = [n for n in names if n in ("before_encode", "text_encoder")]
    g_venc = [n for n in names if n == "vae_encoder"]
    g_den = [n for n in names if n.startswith("denoise")]
    g_dec = [n for n in names if n.startswith("decode")]

    # ingressi della richiesta
    frames = S.snap_frames(float(p.get("duration_s") or (int(p.get("num_frames") or 121) / S.FPS)))
    kw = {"prompt": build_prompt(p, S.frames_seconds(frames)), "num_frames": frames}
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
    elif str(p.get("h3_aspect") or "") in S.ASPECTS:
        ratio = S.ASPECTS[p["h3_aspect"]]
    elif int(p.get("width") or 0) and int(p.get("height") or 0):
        ratio = (int(p["width"]), int(p["height"]))
    else:
        ratio = (16, 9)
    kw["height"], kw["width"] = resolve_canvas_size(ratio[0], ratio[1], 32, short, maxpx)
    if mode == "ref2va":
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

    # 1) text encoder
    t0 = time.time()
    te = load_text_encoder(W)
    _log(f"text encoder pronto in {time.time() - t0:.0f}s — VRAM {_vram()}")
    comps = dict(text_encoder=te, tokenizer=AutoTokenizer.from_pretrained(str(cfgdir / "tokenizer")),
                 processor=Qwen3VLProcessor.from_pretrained(str(cfgdir / "processor")))
    st = _stage(blocks, g_text, comps)
    if mode == "ref2va":
        # il pipeline porta ogni immagine di riferimento a 2048 px di lato corto (anche ingrandendola): con 32 GB
        # l'attenzione del Qwen3-VL su quei token va in OOM (visto 27/09). 1024 = 4× meno token, identità intatta.
        st.update_components(reference_image_short_edge=int(p.get("h3_ref_short_edge") or REF_SHORT_EDGE))
    state = st(**_inputs_for(st, kw))
    del st, te, comps                          # comps teneva ancora il TE: senza, resta in VRAM
    _gc()
    _log(f"testo codificato in {time.time() - t0:.0f}s")

    # 2) VAE encoder (keyframe / riferimenti)
    if g_venc:
        t0 = time.time()
        comps = {"vae": load_vae(W)}
        if mode == "ref2va":
            comps["audio_vae"] = load_audio_vae(W)
        st = _stage(blocks, g_venc, comps)
        state = st(state=state, **_inputs_for(st, kw))
        del st, comps
        _gc()
        _log(f"condizionamenti VAE in {time.time() - t0:.0f}s")

    # 3) denoise
    t0 = time.time()
    tr = load_transformer(W, mode, p)
    _log(f"transformer pronto in {time.time() - t0:.0f}s — VRAM {_vram()}")
    tname = "transformer_ref" if mode == "ref2va" else "transformer"
    st = _stage(blocks, g_den, {tname: tr, "scheduler": MiniMaxH3Scheduler(shift=vshift),
                                "audio_scheduler": MiniMaxH3Scheduler(shift=S.AUDIO_SHIFT)})
    t1 = time.time()
    state = st(state=state, **_inputs_for(st, kw), num_inference_steps=nfe + 1,
               generator=torch.Generator("cpu").manual_seed(seed))
    _log(f"denoise {nfe} passi in {time.time() - t1:.0f}s ({(time.time() - t1) / max(1, nfe):.1f}s/passo) — "
         f"VRAM {_vram()}")
    del st, tr
    _gc()

    # 4) decode
    t0 = time.time()
    st = _stage(blocks, g_dec, {"vae": load_vae(W), "audio_vae": load_audio_vae(W)})
    res = st(state=state, output=["videos", "audio", "sampling_rate"])
    del st
    _gc()
    _log(f"decode in {time.time() - t0:.0f}s")
    out = p.get("output") or str(REPO / "output" / f"minimax_h3_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with_audio = p.get("generate_audio", True) is not False
    encode_video(res["videos"][0], fps=S.FPS, output_path=out,
                 audio=res["audio"][0].float().cpu() if with_audio else None,
                 audio_sample_rate=int(res["sampling_rate"]) if with_audio else None)
    _log(f"fatto in {time.time() - t_all:.0f}s → {out}")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: minimax_h3_infer.py params.json")
        return 2
    try:
        out = run(json.loads(Path(sys.argv[1]).read_text("utf-8")))
        print(f"OK:{out}", flush=True)
        return 0
    except Exception as e:                                           # noqa: BLE001
        traceback.print_exc()
        print(f"ERRORE: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
