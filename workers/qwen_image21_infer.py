"""Worker Qwen-Image 2.1 — generazione e editing immagini (fino a 10 riferimenti, trasparenza RGBA nativa).

Gira nel venv condiviso `E:\\AI\\Tools\\diffusers_next_venv` (scripts/install_diffusers_next.py: diffusers main
con QwenImage21Pipeline, transformers ≥ 5.17). Invocato da core/img_gen_job.py (arch `qwen_image21`).

Modalità (dai parametri): t2i · editing con riferimenti (`ref_images`, ≤10, token «image N») · img2img
(`init_image` + `strength`: coda della schedule a partire dall'originale rumorato) · inpaint (`init_image` +
`mask_path`: originale annotato in rosso come riferimento 1 + latenti fuori maschera bloccati sull'originale a
ogni passo + incollaggio finale al pixel). Combinabili con trasparenza RGBA e LoRA.

PESI piccoli che stanno in 32 GB (DiT scelto ESPLICITAMENTE, mai sostituito in silenzio — core/qwen21_spec.DITS):
  · nvfp4 — DiT 4.2 GB + TE Qwen3-VL-8B 7.5 GB (BennyDaBall, formato ComfyUI «Prism», Nvfp4Linear)
  · fp8   — DiT 7.1 GB + TE 9.4 GB (unsloth, torchao Float8 per riga)
  · int8  — DiT 7.3 GB (unsloth, torchao Int8 per riga: LPIPS migliore, va anche senza FP8 nativo) + TE fp8
  · Turbo ufficiale (8 passi, schedule TURBO_SIGMAS, CFG 1) nvfp4/int8 — stesso TE del base
  · 18+: Uncensored (abenzerps) nvfp4/int8 e Noct-Q V4 / Anime int8 — single-file ComfyUI convertiti in nativo
    all'arrivo (core/native_convert, int8 ConvRot: il worker ruota l'ingresso)
Config/processor/scheduler dal repo ufficiale (solo json), VAE bf16 (stesse chiavi dell'ufficiale fp32).
Modelli costruiti con accelerate.init_empty_weights (solo i PARAMETRI su meta: il transformer ha buffer non
persistenti e frequenze RoPE che devono restare reali) e pesi caricati senza mmap direttamente in GPU.

⚠ LICENZA «Qwen Research»: uso NON commerciale. Forniamo solo il codice; i pesi li scarica l'utente.
⚠ VALIDATE al primo run: inpaint a maschera (annotazione rossa «painted annotations» + blocco dei latenti),
  img2img (coda della schedule), checkpoint Turbo e 18+ convertiti.

Uso: python qwen_image21_infer.py params.json → stampa "OK:<png>"
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32":
    _pac = ",".join(x for x in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
                    if x and "expandable_segments" not in x)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _pac
    if not _pac:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from core.qwen21_spec import (DITS, TE_VARIANTS, TURBO, TURBO_SIGMAS, ASPECTS_2K, MAX_REFS,  # noqa: E402
                              RGBA_TEMPLATE, MASK_HINT, DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_STRENGTH,
                              dit_for, is_turbo)


def _log(m: str) -> None:
    print(f"[qwen21] {m}", flush=True)


def resolve(p: dict) -> dict:
    """Path dei componenti dal registro unico; errore chiaro con gli ID mancanti (nessun ripiego)."""
    from core import model_paths as mp
    from core import weights_catalog as wc
    try:
        dit_id = dit_for(p)
    except ValueError as e:
        raise RuntimeError(str(e)) from None
    te_id = TE_VARIANTS.get(str(p.get("qwen21_te") or ""), DITS[dit_id]["te"])
    prof = dit_id.removeprefix("qwen21-")
    ids = {"configs": "qwen21-configs", "vae": "qwen21-vae", "dit": dit_id, "te": te_id}
    turbo = str(p.get("qwen21_turbo") or "")
    if turbo in TURBO and not is_turbo(dit_id):           # il Turbo distillato non vuole la LoRA turbo
        ids["turbo"] = TURBO[turbo][0]
    out, missing = {}, []
    for k, i in ids.items():
        e = mp.entry(i)
        if not e or not wc.is_present(e, str(mp.models_root())):
            missing.append(i)
        else:
            out[k] = mp.p(i, "")
    if missing:
        raise RuntimeError("Pesi Qwen-Image 2.1 mancanti: " + ", ".join(missing) +
                           ". Scaricali dal tab «Pesi modelli» (profilo " + prof + ").")
    out["profile"] = prof
    out["turbo_ckpt"] = is_turbo(dit_id)
    return out


def _read_sd(path: str, device: str = "cuda", skip=()) -> dict:
    """Un tensore alla volta, senza mmap (vedi _qlinear.read_safetensors): commit di sistema minimo."""
    import _qlinear as Q
    return Q.read_safetensors(path, device, skip=skip)


def build_pipe(W: dict):
    import torch
    from accelerate import init_empty_weights
    from diffusers import (QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21,
                           FlowMatchEulerDiscreteScheduler)
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoConfig
    import _qlinear as Q
    cfg = Path(W["configs"])
    t0 = time.time()
    _log(f"profilo {W['profile']}: transformer")
    with init_empty_weights():
        tr = QwenImage21Transformer2DModel.from_config(QwenImage21Transformer2DModel.load_config(str(cfg / "transformer")))
    sd = Q.quantize_from_state_dict(tr, _read_sd(W["dit"]))
    miss, unex = Q.load_into(tr, sd)
    if miss or unex:
        raise RuntimeError(f"transformer: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    tr = tr.to("cuda").eval()          # buffer non persistenti (init_empty_weights li lascia in CPU)
    _log("text encoder Qwen3-VL")
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(str(cfg / "text_encoder")),
                                                          torch_dtype=torch.bfloat16)
    # lm_head (1.2 GB) non serve: la pipeline legge solo gli stati nascosti → identità (il file nativo non
    # lo contiene; nei TE fp8 unsloth c'è e si salta in lettura)
    te.lm_head = torch.nn.Identity()
    sd = Q.quantize_from_state_dict(te, _read_sd(W["te"], skip=("lm_head.weight",)))
    miss, unex = Q.load_into(te, sd)
    miss = [m for m in miss if not m.endswith("rotary_emb.inv_freq")]
    if miss or unex:
        raise RuntimeError(f"text encoder: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    te = te.to("cuda").eval()          # rotary_emb.inv_freq resta in CPU altrimenti
    _log("VAE + processor + scheduler")
    with init_empty_weights():
        vae = AutoencoderKLQwenImage21.from_config(AutoencoderKLQwenImage21.load_config(str(cfg / "vae")))
    miss, unex = Q.load_into(vae, _read_sd(W["vae"]))
    if miss or unex:
        raise RuntimeError(f"VAE: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    vae = vae.to("cuda").eval()
    proc = Qwen3VLProcessor.from_pretrained(str(cfg / "processor"))
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(str(cfg / "scheduler"))
    if W.get("turbo_ckpt"):            # scheduler del repo Turbo: niente shift dinamico né terminale
        sch = FlowMatchEulerDiscreteScheduler.from_config(sch.config, use_dynamic_shifting=False,
                                                          shift_terminal=None, shift=1.0)
    pipe = QwenImage21Pipeline(scheduler=sch, vae=vae, text_encoder=te, processor=proc, transformer=tr)
    _log(f"pronto in {time.time() - t0:.0f}s — VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    return pipe


def _annotate(img, mask_path: str):
    """Evidenzia in rosso semitrasparente la zona della maschera (editing locale «painted annotation»)."""
    from PIL import Image
    m = Image.open(mask_path).convert("L").resize(img.size)
    red = Image.new("RGBA", img.size, (255, 0, 0, 0))
    red.putalpha(m.point(lambda v: 150 if v > 127 else 0))
    base = img.convert("RGBA")
    base.alpha_composite(red)
    return base


def _lora_path(name: str) -> str:
    """La GUI manda il nome relativo alla cartella LoRA (es. «Qwen21\\x.safetensors»): lo risolve sulla
    radice dei modelli. "" se il file non esiste."""
    if not name:
        return ""
    if os.path.isfile(name):
        return name
    from core import model_paths as mp
    cand = Path(mp.models_root()) / "LoRA" / name
    return str(cand) if cand.is_file() else ""


def _sigmas(pipe, steps: int, turbo: bool, seq_len: int, strength: float):
    """Schedule effettiva: (sigmas da passare alla pipeline, sigma di partenza dopo lo shift dello scheduler).
    img2img = coda della schedule completa (lo shift è elemento per elemento, quindi tagliare prima o dopo
    è lo stesso): con forza s si saltano i primi (1-s)·N passi, come il denoise <1 di un KSampler."""
    import numpy as np
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    full = list(TURBO_SIGMAS) if turbo else [float(x) for x in np.linspace(1.0, 1 / steps, steps)]
    start = min(len(full) - 1, int(round(len(full) * (1.0 - max(0.0, min(1.0, strength))))))
    tail = full[start:]
    c = pipe.scheduler.config
    mu = calculate_shift(seq_len, c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    pipe.scheduler.set_timesteps(sigmas=tail, device="cuda", mu=mu)
    return tail, float(pipe.scheduler.sigmas[0])


def _encode_init(pipe, img, w: int, h: int, gen):
    """Immagine iniziale → latenti impacchettati (1, lh·lw, C), gli stessi che la pipeline denoisa."""
    import torch
    x = pipe.image_processor.preprocess(img.convert("RGBA"), width=w, height=h).unsqueeze(2)
    enc = pipe._encode_vae_image(x.to("cuda", torch.bfloat16), gen)
    _, c, _, lh, lw = enc.shape
    return pipe._pack_latents(enc, 1, c, lh, lw), (lh, lw)


def _latent_mask(mask_path: str, lh: int, lw: int):
    """Maschera (bianco = da rifare) sulla griglia dei latenti, impacchettata come i latenti (1, lh·lw, 1)."""
    import numpy as np
    import torch
    from PIL import Image
    m = Image.open(mask_path).convert("L").resize((lw, lh), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)
    return t.reshape(1, lh * lw, 1).to("cuda", torch.bfloat16)


def _paste_back(gen_img, init_img, mask_path: str, feather: int = 8):
    """Risultato alla risoluzione dell'ORIGINALE (il canvas lo rimette nel suo riquadro) e, fuori dalla
    maschera, l'originale al pixel (il giro nel VAE ammorbidisce i dettagli)."""
    from PIL import Image, ImageFilter
    size = init_img.size
    m = Image.open(mask_path).convert("L").resize(size, Image.BILINEAR)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    gen = gen_img.resize(size, Image.LANCZOS)
    return Image.composite(gen, init_img.convert(gen.mode), m)


def run(p: dict) -> str:
    from core import resources
    resources.require(commit_gb=float(p.get("need_commit_gb") or 6), vram_gb=float(p.get("need_vram_gb") or 14),
                      what="Qwen-Image 2.1")
    resources.start_commit_guard(2.0, what="Qwen-Image 2.1")
    import torch
    from PIL import Image
    import _qlinear as Q
    torch.set_grad_enabled(False)
    W = resolve(p)
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from _watchdog import start_watchdog
        start_watchdog(int(p.get("parent_pid") or 0))
    except Exception:                                                # noqa: BLE001
        pass
    pipe = build_pipe(W)
    turbo_ckpt = bool(W.get("turbo_ckpt"))
    steps = len(TURBO_SIGMAS) if turbo_ckpt else int(p.get("steps") or DEFAULT_STEPS)
    if "turbo" in W:
        from safetensors.torch import load_file
        n = Q.apply_lora(pipe.transformer, load_file(W["turbo"]), 1.0)
        steps = TURBO[str(p["qwen21_turbo"])][1]
        _log(f"LoRA turbo: {n} moduli, {steps} passi")
    for lo in p.get("loras") or []:
        path = _lora_path(lo.get("path") or lo.get("name") or "")
        if not path:
            raise RuntimeError(f"LoRA non trovata: {lo.get('path') or lo.get('name')}")
        from safetensors.torch import load_file
        n = Q.apply_lora(pipe.transformer, load_file(path), float(lo.get("strength", 1.0)))
        _log(f"LoRA {os.path.basename(path)}: {n} moduli")
        if not n:
            raise RuntimeError(f"LoRA {os.path.basename(path)}: nessun modulo riconosciuto "
                               "(non è una LoRA per Qwen-Image 2.1?)")

    # ── ingressi: riferimenti (editing), immagine iniziale (img2img / inpaint), maschera ──
    refs = [x for x in (p.get("ref_images") or []) if x]
    init_path = p.get("init_image") or ""
    mask_path = p.get("mask_path") or ""
    if not init_path and p.get("image_path"):
        # compatibilità: image_path = riferimento singolo (editing); con la maschera è l'immagine da ritoccare
        if mask_path:
            init_path = p["image_path"]
        else:
            refs = [p["image_path"]] + refs
    init_img = Image.open(init_path) if init_path else None
    if mask_path and init_img is None:
        raise RuntimeError("Inpaint Qwen-Image 2.1: manca l'immagine da ritoccare")
    images = [Image.open(x) for x in refs]
    prompt = str(p.get("prompt") or "")
    if init_img is not None and mask_path:
        # inpaint: l'originale annotato in rosso è il riferimento 1 (il modello capisce COSA cambiare)
        # e fuori dalla maschera i latenti restano quelli dell'originale (DOVE cambiare lo decide la maschera)
        images = [_annotate(init_img, mask_path)] + images
        prompt = f"{prompt} {MASK_HINT}"
    images = images[:MAX_REFS]
    transparent = bool(p.get("transparent"))
    if transparent and "RGBA image" not in prompt:
        prompt = RGBA_TEMPLATE.format(p=prompt)

    out_res = int(p.get("output_resolution") or 1024)
    w, h = int(p.get("width") or 0), int(p.get("height") or 0)
    if p.get("aspect_2k") in ASPECTS_2K:
        w, h = ASPECTS_2K[p["aspect_2k"]]
    if init_img is not None and not (w and h):       # img2img / inpaint: proporzioni dell'immagine iniziale
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
        w, h, _ = calculate_dimensions(out_res * out_res, init_img.size[0] / init_img.size[1])
    kw = {}
    if w and h:
        kw.update(width=w // 32 * 32, height=h // 32 * 32)
    guidance = 1.0 if turbo_ckpt else float(p.get("guidance") if p.get("guidance") is not None
                                             else DEFAULT_GUIDANCE)
    gen = torch.Generator("cuda").manual_seed(int(p.get("seed") or 0))

    mode = "edit" if images else "t2i"
    if init_img is not None:
        strength = float(p["strength"]) if p.get("strength") is not None else \
            (1.0 if mask_path else DEFAULT_STRENGTH)
        x0, (lh, lw) = _encode_init(pipe, init_img, kw["width"], kw["height"], gen)
        sig, s0 = _sigmas(pipe, steps, turbo_ckpt, x0.shape[1], strength)
        noise = torch.randn(x0.shape, generator=gen, device="cuda", dtype=x0.dtype)
        kw["latents"] = s0 * noise + (1.0 - s0) * x0
        kw["sigmas"] = sig
        if mask_path:
            mask = _latent_mask(mask_path, lh, lw)
            sch_sig = pipe.scheduler.sigmas.clone()   # la pipeline le ricalcola identiche dagli stessi sigmas

            def _keep_outside(_pipe, i, _t, cb):
                # dopo il passo i i latenti stanno a sigma[i+1]: fuori maschera = originale rumorato a quel livello
                s = float(sch_sig[i + 1]) if i + 1 < len(sch_sig) else 0.0
                orig = s * noise + (1.0 - s) * x0
                return {"latents": mask * cb["latents"] + (1.0 - mask) * orig}
            kw["callback_on_step_end"] = _keep_outside
        mode = f"{'inpaint' if mask_path else 'img2img'} forza {strength:.2f} ({len(sig)}/{steps} passi)"
    elif turbo_ckpt:
        kw["sigmas"] = list(TURBO_SIGMAS)
    _log(f"{mode} · {len(images)} rif. · {kw.get('width', '?')}x{kw.get('height', '?')} · "
         f"{'Turbo ' if turbo_ckpt else ''}{steps} passi · cfg {guidance}")
    img = pipe(prompt=prompt, image=images or None, negative_prompt=p.get("negative") or None,
               true_cfg_scale=guidance, num_inference_steps=steps, generator=gen,
               output_resolution=out_res, **kw).images[0]
    if not transparent:
        img = img.convert("RGB")
    if init_img is not None and mask_path and p.get("paste_back", True):
        img = _paste_back(img, init_img, mask_path,          # sfumatura: 0 = auto, proporzionale alla foto
                          feather=int(p.get("feather") or 0) or max(4, min(init_img.size) // 128))
    out = p.get("output") or str(REPO / "output" / f"qwen21_{int(time.time())}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: qwen_image21_infer.py params.json"); return 2
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
