"""Worker Qwen-Image 2.1 — generazione e editing immagini (fino a 10 riferimenti, trasparenza RGBA nativa).

Gira nel venv condiviso `E:\\AI\\Tools\\diffusers_next_venv` (scripts/install_diffusers_next.py: diffusers main
con QwenImage21Pipeline, transformers ≥ 5.17). Invocato da core/img_gen_job.py (arch `qwen_image21`).

PESI piccoli che stanno in 32 GB (profilo scelto ESPLICITAMENTE, mai sostituito in silenzio):
  · nvfp4 — DiT 4.2 GB + TE Qwen3-VL-8B 7.5 GB (BennyDaBall, formato ComfyUI «Prism», Nvfp4Linear)
  · fp8   — DiT 7.1 GB + TE 9.4 GB (unsloth, torchao Float8 per riga)
  · int8  — DiT 7.3 GB (unsloth, torchao Int8 per riga: LPIPS migliore, va anche senza FP8 nativo) + TE fp8
Config/processor/scheduler dal repo ufficiale (solo json), VAE bf16 (stesse chiavi dell'ufficiale fp32).
Modelli costruiti con accelerate.init_empty_weights (solo i PARAMETRI su meta: il transformer ha buffer non
persistenti e frequenze RoPE che devono restare reali) e pesi caricati senza mmap direttamente in GPU.

⚠ LICENZA «Qwen Research»: uso NON commerciale. Forniamo solo il codice; i pesi li scarica l'utente.
⚠ VALIDATE al primo run: editing locale con maschera (l'API non ha un parametro mask: usiamo l'annotazione
  in rosso sull'immagine, come indicato dal modello «painted annotations»), formato LoRA turbo Viggle.

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

from core.qwen21_spec import (PROFILES, TE_VARIANTS, TURBO, ASPECTS_2K, MAX_REFS,  # noqa: E402
                              RGBA_TEMPLATE, MASK_HINT, DEFAULT_STEPS, DEFAULT_GUIDANCE)


def _log(m: str) -> None:
    print(f"[qwen21] {m}", flush=True)


def resolve(p: dict) -> dict:
    """Path dei componenti dal registro unico; errore chiaro con gli ID mancanti (nessun ripiego)."""
    from core import model_paths as mp
    from core import weights_catalog as wc
    prof = str(p.get("qwen21_profile") or "fp8").lower()
    if prof not in PROFILES:
        raise RuntimeError(f"Profilo Qwen-Image 2.1 sconosciuto: {prof} (nvfp4 | fp8 | int8)")
    dit_id, te_id = PROFILES[prof]
    te_id = TE_VARIANTS.get(str(p.get("qwen21_te") or ""), te_id)
    ids = {"configs": "qwen21-configs", "vae": "qwen21-vae", "dit": dit_id, "te": te_id}
    turbo = str(p.get("qwen21_turbo") or "")
    if turbo in TURBO:
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
    return out


def _read_sd(path: str, device: str = "cuda") -> dict:
    from safetensors.torch import load as st_load
    with open(path, "rb") as f:
        data = f.read()
    sd = st_load(data)
    del data
    return {k: v.to(device) for k, v in sd.items()}


def _te_keys(sd: dict) -> dict:
    """Nomi ComfyUI del Qwen3-VL (model.layers.*, model.embed_tokens, model.norm) → transformers
    (model.language_model.*). La torre visiva (model.visual.*) e lm_head hanno già i nomi giusti."""
    out = {}
    for k, v in sd.items():
        for a, b in (("model.layers.", "model.language_model.layers."),
                     ("model.embed_tokens.", "model.language_model.embed_tokens."),
                     ("model.norm.", "model.language_model.norm.")):
            if k.startswith(a):
                k = b + k[len(a):]
                break
        out[k] = v
    return out


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
    tr.eval()
    _log("text encoder Qwen3-VL")
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(str(cfg / "text_encoder")),
                                                          torch_dtype=torch.bfloat16)
    sd = Q.quantize_from_state_dict(te, _te_keys(_read_sd(W["te"])))
    miss, unex = Q.load_into(te, sd)
    miss = [m for m in miss if not m.endswith("rotary_emb.inv_freq")]
    if miss or unex:
        raise RuntimeError(f"text encoder: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    te.eval()
    _log("VAE + processor + scheduler")
    with init_empty_weights():
        vae = AutoencoderKLQwenImage21.from_config(AutoencoderKLQwenImage21.load_config(str(cfg / "vae")))
    miss, unex = Q.load_into(vae, _read_sd(W["vae"]))
    if miss or unex:
        raise RuntimeError(f"VAE: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    vae.eval()
    proc = Qwen3VLProcessor.from_pretrained(str(cfg / "processor"))
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(str(cfg / "scheduler"))
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


def run(p: dict) -> str:
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
    steps = int(p.get("steps") or DEFAULT_STEPS)
    if "turbo" in W:
        from safetensors.torch import load_file
        n = Q.apply_lora(pipe.transformer, load_file(W["turbo"]), 1.0)
        steps = TURBO[str(p["qwen21_turbo"])][1]
        _log(f"LoRA turbo: {n} moduli, {steps} passi")
    for lo in p.get("loras") or []:
        path = lo.get("path") or lo.get("name") or ""
        if path and os.path.isfile(path):
            from safetensors.torch import load_file
            n = Q.apply_lora(pipe.transformer, load_file(path), float(lo.get("strength", 1.0)))
            _log(f"LoRA {os.path.basename(path)}: {n} moduli")
    refs = [x for x in (p.get("ref_images") or ([p["image_path"]] if p.get("image_path") else [])) if x]
    images = [Image.open(x) for x in refs[:MAX_REFS]]
    prompt = str(p.get("prompt") or "")
    if p.get("mask_path") and images:
        images[0] = _annotate(images[0], p["mask_path"])
        prompt = f"{prompt} {MASK_HINT}"
    transparent = bool(p.get("transparent"))
    if transparent and "RGBA image" not in prompt:
        prompt = RGBA_TEMPLATE.format(p=prompt)
    w, h = int(p.get("width") or 0), int(p.get("height") or 0)
    if p.get("aspect_2k") in ASPECTS_2K:
        w, h = ASPECTS_2K[p["aspect_2k"]]
    kw = {}
    if w and h:
        kw.update(width=w // 32 * 32, height=h // 32 * 32)
    guidance = float(p.get("guidance") if p.get("guidance") is not None else DEFAULT_GUIDANCE)
    gen = torch.Generator("cuda").manual_seed(int(p.get("seed") or 0))
    _log(f"{'edit' if images else 't2i'} · {len(images)} rif. · {kw or 'dimensioni dal riferimento'} · "
         f"{steps} passi · cfg {guidance}")
    img = pipe(prompt=prompt, image=images or None, negative_prompt=p.get("negative") or None,
               true_cfg_scale=guidance, num_inference_steps=steps, generator=gen,
               output_resolution=int(p.get("output_resolution") or 1024), **kw).images[0]
    out = p.get("output") or str(REPO / "output" / f"qwen21_{int(time.time())}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (img if transparent else img.convert("RGB")).save(out)
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
