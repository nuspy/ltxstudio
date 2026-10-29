"""Worker FLUX.3 Image Klein — creazione, modifica per istruzione, multi-riferimento (≤10) e riquadri (layout/edit).

Gira nel venv condiviso `E:\\AI\\Tools\\diffusers_next_venv` (torch + transformers ≥ 5.17 + diffusers per il VAE): il DiT è
NATIVO (workers/flux3_dit.py), quindi nessun diffusers più recente. Invocato da core/img_gen_job.py (arch `flux3`).

Sequenza (un modello grande alla volta, la VRAM si libera fra una fase e l'altra — `core/flux3_spec.estimate_vram_gb`):
  1. upsampler (Qwen3.5-9B, opzionale ma di default: BFL «do not use the model without it») → caption completo;
  2. text encoder Qwen3-VL-4B → 8 strati impilati (20480 canali) → token;
  3. VAE (fp32): codifica i riferimenti (token puliti `image_cond`, ids[:,3] = -(1+i), RoPE ×0,5);
  4. DiT fp8r (6,3 GB): 16 passi, flow map, SENZA CFG né negative;
  5. VAE: decodifica (tile oltre 2 K) → PNG.
Inpaint con maschera NON è nativo di FLUX.3: se arrivano `init_image` + `mask_path` si fa una modifica per istruzione sulla
foto e il risultato si ricompone con la maschera binarizzata (> 127) come per gli altri motori — dichiarato nel log.

⚠ LICENZA FLUX Kommunity v1.0: l'app non distribuisce i pesi, li scarica l'utente dal tab «Pesi modelli» (avviso + conferma).
Uso: python flux3_infer.py params.json → stampa "OK:<png>"
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
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

from core import flux3_caption as C  # noqa: E402
from core import flux3_spec as S  # noqa: E402


DEV = os.environ.get("FLUX3_DEVICE", "cuda")        # solo i test su CPU (modello minuscolo) lo cambiano


def _log(m: str) -> None:
    print(f"[flux3] {m}", flush=True)


# ── pesi ─────────────────────────────────────────────────────────────────────────────────────────────────
def resolve(p: dict, need_pup: bool) -> dict:
    """Path dei componenti dal registro unico; errore chiaro con gli ID mancanti (nessun ripiego automatico)."""
    from core import model_paths as mp
    from core import weights_catalog as wc
    dit_id = str(p.get("flux3_dit") or S.DEFAULT_DIT)
    if dit_id not in S.DITS:
        raise RuntimeError(f"DiT FLUX.3 sconosciuto: {dit_id} ({', '.join(S.DITS)})")
    te_id = S.te_id(str(p.get("flux3_te") or ""),
                    lambda i: bool(mp.entry(i)) and wc.is_present(mp.entry(i), str(mp.models_root())))
    ids = {"dit": dit_id, "te": te_id, "te_cfg": S.TE_CONFIGS, "vae": S.VAE}
    if te_id == S.TE:                      # il TE ufficiale è uno snapshot con config e tokenizer dentro
        ids.pop("te_cfg")
    if need_pup:
        pup_id = str(p.get("flux3_pup_id") or S.PUP_DEFAULT)
        ids["pup"] = pup_id
        ids["pup_cfg"] = "flux3-pup-configs"
    out, missing = {}, []
    for k, i in ids.items():
        e = mp.entry(i)
        if not e or not wc.is_present(e, str(mp.models_root())):
            missing.append(i)
        else:
            out[k] = mp.p(i, "")
    if missing:
        raise RuntimeError("Pesi FLUX.3 mancanti: " + ", ".join(missing) +
                           ". Scaricali dal tab «Pesi modelli» (FLUX.3 Klein, licenza FLUX Kommunity).")
    out["te_id"] = te_id
    if te_id == S.TE:                      # snapshot ufficiale: shard + config + tokenizer nella stessa cartella
        te = Path(out["te"])
        out["te_src"] = out["te_cfg"] = str(te / "text_encoder") if (te / "text_encoder").is_dir() else str(te)
    else:                                  # TE a file singolo: i json/tokenizer vengono dallo snapshot di configurazione
        cfg = Path(out["te_cfg"])
        out["te_src"] = out["te"]
        out["te_cfg"] = str(cfg / "text_encoder") if (cfg / "text_encoder").is_dir() else str(cfg)
    return out


def _vram(peak: bool = False) -> float:
    """GB allocati (o picco) sul device del job; 0 su CPU (test) — mai un contesto CUDA creato per leggere un numero."""
    if not DEV.startswith("cuda"):
        return 0.0
    import torch
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 2 ** 30


def _free() -> None:
    import torch
    gc.collect()
    if DEV.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── riferimenti ──────────────────────────────────────────────────────────────────────────────────────────
def prepare_reference(image):
    """Riferimento SINGOLO: RGB, adattato a `reference_canvas`, centrato con padding riflesso fino al canvas.
    Ritorna (immagine, riquadro (l,t,r,b) dell'originale dentro il canvas o None)."""
    import numpy as np
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = C.reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    pad = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > S.GRID else "symmetric"
    return (Image.fromarray(np.pad(np.asarray(image), pad, mode=mode)),
            (left, top, left + image.width, top + image.height))


def prepare_multi_reference(image):
    """Ogni riferimento di un edit MULTIPLO: RGB, ≤ 1 MP, ritaglio centrato sul multiplo di 16."""
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image).convert("RGB")
    w0, h0 = C.multi_reference_size(*image.size)
    if (w0, h0) != image.size:
        image = image.resize((w0, h0), Image.Resampling.LANCZOS)
    w, h = w0 // S.GRID * S.GRID, h0 // S.GRID * S.GRID
    left, top = (w0 - w) // 2, (h0 - h) // 2
    return image.crop((left, top, left + w, top + h))


# ── text encoder ─────────────────────────────────────────────────────────────────────────────────────────
def remap_te_key(k: str):
    """Nome HF di un tensore del text encoder, o None se non serve (torre visiva, lm_head). Il TE Krea2 «-native» ha i nomi
    senza il prefisso `model.` (`language_model.layers…`): `model.language_model.layers…` come nel checkpoint ufficiale."""
    if k == "lm_head.weight" or k.startswith(("model.visual.", "visual.")):
        return None
    if k.startswith("language_model."):
        return "model." + k
    return k


def _header_keys(path: str) -> list:
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


def build_te_model(te_src: str, cfg_dir: str):
    """Qwen3-VL-4B solo-linguaggio sul device. `te_src` = cartella con gli shard bf16 ufficiali OPPURE un file singolo (Krea2
    uncensored: fp8 per riga con `weight_scale`, nomi senza `model.`). Con fp8 i pesi restano fp8 in VRAM (4,8 GB) e ogni Linear
    si dequantizza al volo (`_qlinear.RowFp8Linear`). Nessuna torre visiva, nessun lm_head."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    import _qlinear as Q
    cfg = AutoConfig.from_pretrained(cfg_dir)
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    te.lm_head = torch.nn.Identity()                  # servono solo gli stati nascosti
    te.model.visual = torch.nn.Identity()             # il TE legge solo testo: -0,8 GB e tempo di carico
    files = [te_src] if os.path.isfile(te_src) else sorted(str(x) for x in Path(te_src).glob("*.safetensors"))
    sd = {}
    for fpath in files:
        skip = [k for k in _header_keys(fpath) if remap_te_key(k) is None]
        part = Q.read_safetensors(fpath, DEV, skip=tuple(skip))
        sd.update({remap_te_key(k): v for k, v in part.items()})
    if os.path.isfile(te_src):
        sd = Q.quantize_from_state_dict(te, sd)        # fp8 per riga → RowFp8Linear (sul meta-device)
    miss, unex = Q.load_into(te, sd)
    miss = [m for m in miss if not m.endswith("inv_freq")]
    if miss or unex:
        raise RuntimeError(f"text encoder: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    return te.to(DEV).eval()


def load_text_encoder(te_src: str, cfg_dir: str | None = None):
    from transformers import AutoProcessor
    t0 = time.time()
    cfg_dir = cfg_dir or te_src
    te = build_te_model(te_src, cfg_dir)
    proc = AutoProcessor.from_pretrained(cfg_dir)
    proc.tokenizer.padding_side = "right"
    _log(f"text encoder pronto in {time.time() - t0:.0f}s — VRAM {_vram():.1f} GB")
    return te, proc


def encode_text(te, proc, prompt: str):
    """prompt → (token (L, 20480) bf16, ids (L, 4) = (0,0,0,pos)); L multiplo di 80, max 4096 (chat template del TE)."""
    import torch
    with torch.inference_mode():
        return _encode_text(te, proc, prompt)


def _encode_text(te, proc, prompt: str):
    import torch
    text = proc.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    tok = proc.tokenizer
    real = tok(text, return_tensors="pt", truncation=True, max_length=S.TE_MAX_TOKENS)["input_ids"].shape[1]
    length = min(math.ceil(real / S.TE_PAD_MULTIPLE) * S.TE_PAD_MULTIPLE, S.TE_MAX_TOKENS)
    inputs = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
    mask = inputs["attention_mask"].to(DEV)
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    out = te(input_ids=inputs["input_ids"].to(DEV), attention_mask=mask,
             position_ids=positions[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False)
    stacked = torch.stack([out.hidden_states[k] for k in S.TE_LAYERS], dim=1)           # (1, 8, L, 2560)
    tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
    pos = torch.arange(length, device=DEV)
    ids = torch.stack([torch.zeros_like(pos)] * 3 + [pos], dim=-1)
    return tokens, ids


# ── VAE ──────────────────────────────────────────────────────────────────────────────────────────────────
def load_vae(path: str):
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLFlux2
    import _qlinear as Q
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(S.VAE_CONFIG)
    miss, unex = Q.load_into(vae, Q.read_safetensors(path, DEV))
    if miss or unex:
        raise RuntimeError(f"VAE: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    return vae.to(DEV, torch.float32).eval()


def latent_ids(h: int, w: int, device):
    import torch
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    z = torch.zeros_like(rows)
    return torch.stack([z, rows, cols, z], dim=-1).reshape(h * w, 4)


def scaled_ids(ids):
    ids = ids.float()
    ids[:, 1:3] *= S.IMAGE_ROPE_SCALE
    return ids


def vae_encode(vae, pil):
    """Immagine PIL (lati multipli di 16) → (token (h·w,128), ids (h·w,4)): media, patchify 2×2, BatchNorm del VAE."""
    import numpy as np
    import torch
    x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div(127.5).sub(1.0).to(DEV)[None]
    with torch.inference_mode():
        mean = vae.encode(x).latent_dist.mean                                         # (1, 32, H/8, W/8)
        hh, ww = mean.shape[2] // 2, mean.shape[3] // 2
        z = mean.reshape(1, 32, hh, 2, ww, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 128, hh, ww)
        z = vae.bn(z)
    return z[0].permute(1, 2, 0).reshape(hh * ww, 128), latent_ids(hh, ww, DEV)


def vae_decode(vae, tokens, ids):
    """Token (h·w,128) in ordine di id → PIL RGB. Oltre 2 K il VAE lavora a tile (altrimenti i conv fp32 non ci stanno)."""
    import torch
    from PIL import Image
    h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
    with torch.inference_mode():
        z = tokens.float().reshape(h, w, 128).permute(2, 0, 1)[None]
        z = z * torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.bn.eps) + vae.bn.running_mean.view(1, -1, 1, 1)
        z = z.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * h, 2 * w)
        if max(h, w) * S.GRID > 2048:
            vae.enable_tiling()
        img = vae.decode(z).sample[0]
    arr = ((img.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ── DiT ──────────────────────────────────────────────────────────────────────────────────────────────────
def load_dit(path: str):
    import flux3_dit as D
    import _qlinear as Q
    t0 = time.time()
    state = Q.read_safetensors(path, DEV)
    flow = D.is_flow_map_keys(state)
    model = D.build_meta(D.Config(flow_map=flow))
    model = D.load_state(model, state, DEV)
    import torch
    _log(f"DiT {'distillato (flow map)' if flow else 'Base (CFG)'}: {model._n_fp8} Linear fp8 · pronto in "
         f"{time.time() - t0:.0f}s — VRAM {_vram():.1f} GB")
    return model


# ── LoRA ─────────────────────────────────────────────────────────────────────────────────────────────────
_DIFF_TO_NATIVE = (
    ("single_transformer_blocks.", "single_blocks."),
    (".attn.to_q", ".q_proj"), (".attn.to_k", ".k_proj"), (".attn.to_v", ".v_proj"), (".attn.to_out.0", ".attn_out"),
    (".ff.linear_in", ".mlp_in"), (".ff.linear_out", ".mlp_out"),
    ("x_embedders.", "emb_in."), ("context_embedder", "txt_in"),
    ("timestep_embedder.linear_1", "time_in.in_layer"), ("timestep_embedder.linear_2", "time_in.out_layer"),
    ("target_timestep_embedder.linear_1", "ref_time_in.in_layer"),
    ("target_timestep_embedder.linear_2", "ref_time_in.out_layer"), ("proj_out", "final_layer.image.linear"),
)
_LORA_PREFIXES = ("diffusion_model.", "base_model.model.transformer.", "base_model.model.", "transformer.")
_MODE_BLOCKS = __import__("re").compile(r"(?<![A-Za-z_])mode_blocks\.")      # NON dentro «txt_mode_blocks» né «content_mode_blocks»


def _native_name(name: str) -> str:
    """Nome di modulo diffusers/SimpleTuner → nome nativo BFL (i nomi già nativi passano invariati)."""
    name = _MODE_BLOCKS.sub("content_mode_blocks.", name)
    for a, b in _DIFF_TO_NATIVE:
        name = name.replace(a, b)
    return name



def lora_pairs(sd: dict) -> dict:
    """{nome modulo nativo: (A, B, alpha|None)} da una LoRA ai-toolkit (chiavi native `diffusion_model.…`) o diffusers/
    SimpleTuner (`transformer.mode_blocks…to_q…`). Le chiavi non riconosciute si ignorano (il chiamante conta i moduli)."""
    out: dict = {}
    for k, v in sd.items():
        for pre in _LORA_PREFIXES:
            if k.startswith(pre):
                k = k[len(pre):]
                break
        for tag_a, tag_b in ((".lora_A.weight", ".lora_B.weight"), (".lora_A.default.weight", ".lora_B.default.weight"),
                             (".lora_down.weight", ".lora_up.weight")):
            if k.endswith(tag_a):
                name, slot = k[:-len(tag_a)], "A"
            elif k.endswith(tag_b):
                name, slot = k[:-len(tag_b)], "B"
            else:
                continue
            name = _native_name(name)
            ent = out.setdefault(name, {})
            ent[slot] = v
            break
        else:
            if k.endswith(".alpha"):
                name = _native_name(k[:-len(".alpha")])
                out.setdefault(name, {})["alpha"] = float(v)
    return {n: (e["A"], e["B"], e.get("alpha")) for n, e in out.items() if "A" in e and "B" in e}


def merge_lora(model, sd: dict, strength: float) -> int:
    """Fonde la LoRA nei pesi (nn.Linear bf16 oppure FP8Linear: dequantizza la riga, somma, riquantizza per riga)."""
    import torch
    import flux3_dit as D
    n = 0
    mods = dict(model.named_modules())
    for name, (A, B, alpha) in lora_pairs(sd).items():
        m = mods.get(name)
        if m is None or not hasattr(m, "weight"):
            continue
        rank = A.shape[0]
        scale = strength * (alpha / rank if alpha else 1.0)
        delta = (B.to(DEV, torch.float32) @ A.to(DEV, torch.float32)) * scale
        if delta.shape != (m.out_features if hasattr(m, "out_features") else m.weight.shape[0],
                           m.in_features if hasattr(m, "in_features") else m.weight.shape[1]):
            continue
        with torch.no_grad():
            if isinstance(m, D.FP8Linear):
                w = m.weight.float() * m.weight_scale[:, None] + delta
                q, s = D.quantize_rows(w)
                m.weight.copy_(q)
                m.weight_scale.copy_(s)
            else:
                m.weight.copy_((m.weight.float() + delta).to(m.weight.dtype))
        n += 1
    return n


def _lora_path(name: str) -> str:
    if not name:
        return ""
    if os.path.isfile(name):
        return name
    from core import model_paths as mp
    cand = Path(mp.models_root()) / "LoRA" / name
    return str(cand) if cand.is_file() else ""


# ── composizione finale per inpaint (nostro, non nativo) ─────────────────────────────────────────────────
def _paste_back(gen, init_img, mask_path: str, feather: int):
    """Maschera BINARIZZATA (> 127: con alpha/luminanza < 255 l'edit usciva al 60%, visto dal vivo 17/10) e sfumatura."""
    from PIL import Image, ImageFilter
    size = init_img.size
    m = Image.open(mask_path).convert("L").resize(size, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(gen.resize(size, Image.LANCZOS), init_img.convert("RGB"), m)


# ── job ──────────────────────────────────────────────────────────────────────────────────────────────────
def run(p: dict) -> str:
    from core import resources
    refs_in = [x for x in (p.get("ref_images") or []) if x]
    init_path, mask_path = p.get("init_image") or "", p.get("mask_path") or ""
    if not init_path and p.get("image_path"):
        if mask_path:
            init_path = p["image_path"]
        else:
            refs_in = [p["image_path"]] + refs_in
    if mask_path and not init_path:
        raise RuntimeError("Inpaint FLUX.3: manca l'immagine da ritoccare")
    if init_path:
        refs_in = [init_path] + refs_in
    if len(refs_in) > S.MAX_REFS:
        raise RuntimeError(f"FLUX.3 accetta al massimo {S.MAX_REFS} riferimenti (ne arrivano {len(refs_in)})")
    n_refs = len(refs_in)

    boxes = p.get("boxes") or []
    errs = C.validate(boxes, n_refs)
    if errs:
        raise RuntimeError("Riquadri non validi: " + "; ".join(errs))
    rows = C.to_rows(boxes) if boxes else None

    prompt = str(p.get("prompt") or "").strip()
    if not prompt and not rows:
        raise RuntimeError("FLUX.3: serve un prompt")
    use_pup = bool(p.get("flux3_pup", True))

    w_req, h_req = int(p.get("width") or 0), int(p.get("height") or 0)
    want = (w_req, h_req) if (w_req and h_req) else (S.DEFAULT_SIDE, S.DEFAULT_SIDE)
    est = S.estimate_vram_gb(*S.clamp_size(*want), n_refs, use_pup)
    resources.require(commit_gb=float(p.get("need_commit_gb") or 8), vram_gb=float(p.get("need_vram_gb") or est["peak"]),
                      what="FLUX.3 Klein")
    resources.start_commit_guard(2.0, what="FLUX.3 Klein")
    import torch
    from PIL import Image
    torch.set_grad_enabled(False)
    W = resolve(p, use_pup)
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from _watchdog import start_watchdog
        start_watchdog(int(p.get("parent_pid") or 0))
    except Exception:                                                # noqa: BLE001
        pass
    lora_paths = []
    for lo in p.get("loras") or []:                        # prima del caricamento: errore subito se manca
        path = _lora_path(lo.get("path") or lo.get("name") or "")
        if not path:
            raise RuntimeError(f"LoRA non trovata: {lo.get('path') or lo.get('name')}")
        lora_paths.append((path, float(lo.get("strength", 1.0))))

    images = [Image.open(x) for x in refs_in]
    seed = p.get("seed")

    # 1) upsampler
    pup_note = "senza upsampler"
    caption = C.caption_without_upsampler(prompt, rows)
    finished = bool(rows) is False and C.is_finished(prompt)
    if use_pup and finished:
        pup_note = "prompt già completo (box citati): upsampler saltato"
        caption = prompt
    elif use_pup:
        import flux3_pup
        try:
            caption = flux3_pup.run(W["pup"], W["pup_cfg"], prompt, images, rows)
            pup_note = "con upsampler"
        except ValueError as e:                                       # es. riferimento con rapporto oltre 200:1
            _log(f"upsampler saltato: {e} — uso il prompt com'è")
        _free()
    _log(f"caption ({pup_note}): {caption[:500]}")
    if p.get("caption_out"):                                          # la GUI/API può mostrare il caption usato
        Path(p["caption_out"]).write_text(caption, "utf-8")
    if seed is None:
        seed = int(hashlib.sha256(caption.encode()).hexdigest(), 16) % (S.SEED_MAX + 1)
    seed = int(seed)

    # 2) text encoder
    te, proc = load_text_encoder(W["te_src"], W["te_cfg"])
    try:
        txt_tokens, txt_ids = encode_text(te, proc, caption)
    finally:
        del te, proc
        _free()

    # 3) riferimenti → VAE
    if n_refs > 1:
        refs = [prepare_multi_reference(im) for im in images]
        crop = None
    else:
        prepped = [prepare_reference(im) for im in images]
        refs = [im for im, _ in prepped]
        crop = prepped[0][1] if prepped and not (w_req or h_req) else None
    if w_req and h_req:
        width, height = S.clamp_size(w_req, h_req)
    elif refs:
        width, height = refs[0].size
    else:
        width = height = S.DEFAULT_SIDE
    if width % S.GRID or height % S.GRID:
        width, height = S.clamp_size(width, height)
    vae = load_vae(W["vae"])
    cond = None
    if refs:
        toks, idss = [], []
        for i, im in enumerate(refs):
            t, ids = vae_encode(vae, im)
            ids = scaled_ids(ids)
            ids[:, 3] = -(1 + i)
            toks.append(t.to(torch.bfloat16))
            idss.append(ids)
        cond = (torch.cat(toks), torch.cat(idss))

    # 4) DiT
    import flux3_dit as D
    dit = load_dit(W["dit"])
    for path, strength in lora_paths:
        from safetensors.torch import load_file
        n = merge_lora(dit, load_file(path), strength)
        _log(f"LoRA {os.path.basename(path)}: {n} moduli")
        if not n:
            raise RuntimeError(f"LoRA {os.path.basename(path)}: nessun modulo riconosciuto (non è una LoRA FLUX.3?)")
    flow = dit.flow_map
    steps = int(p.get("steps") or (S.DEFAULT_STEPS if flow else S.BASE_STEPS))
    steps = max(S.MIN_STEPS, min(S.MAX_STEPS, steps)) if flow else max(1, steps)
    if flow and p.get("guidance") not in (None, 0, 0.0, 1, 1.0):
        _log(f"il distillato campiona SENZA CFG: guidance {p['guidance']} ignorata")
    guidance = float(p.get("guidance") if (not flow and p.get("guidance")) else S.BASE_GUIDANCE)
    h16, w16 = height // S.GRID, width // S.GRID
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn((h16 * w16, 128), generator=gen).to(torch.bfloat16).float().to(DEV)
    image = (noise, scaled_ids(latent_ids(h16, w16, DEV)))
    sched = D.schedule(steps, S.SHIFT_ALPHA)
    _log(f"{'edit' if n_refs else 't2i'} · {n_refs} rif. · {len(boxes)} riquadri · {width}x{height} · "
         f"{steps} passi · seed {seed}" + ("" if flow else f" · cfg {guidance}"))
    t0 = time.time()
    last = [0.0]

    def tick(i, n):
        if time.time() - last[0] > 5 or i == n:
            last[0] = time.time()
            _log(f"passo {i}/{n}")
    if flow:
        packed = D.make_packed([[(txt_tokens, txt_ids.float()), cond, image]], DEV)
        tokens = D.denoise_flow_map(dit, packed, sched, on_step=tick)
    else:
        raise RuntimeError("FLUX.3 Base: il worker supporta oggi solo il distillato (il Base serve all'allenamento LoRA)")
    peak = _vram(peak=True)
    _log(f"denoise {time.time() - t0:.0f}s — picco VRAM {peak:.1f} GB")
    del dit, packed
    _free()

    # 5) decodifica
    out_img = vae_decode(vae, tokens, latent_ids(h16, w16, DEV))
    del vae
    _free()
    if crop is not None:
        out_img = out_img.crop(crop)
    if mask_path and init_path:                                       # inpaint = composito nostro con maschera binarizzata
        orig = Image.open(init_path)
        out_img = _paste_back(out_img, orig, mask_path, int(p.get("feather") or 0) or max(4, min(orig.size) // 128))
        _log("inpaint: modifica per istruzione + composito con maschera binarizzata (non è un inpaint nativo)")
    out = p.get("output") or str(REPO / "output" / f"flux3_{int(time.time())}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: flux3_infer.py params.json")
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
