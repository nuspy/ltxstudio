"""MultiTalk (Wan2.1 I2V 14B + adattatore audio) NATIVO: nessun ComfyUI, nessun WanVideoWrapper a runtime.

Pipeline (logica di wan/multitalk.py di MeiGen-AI, un solo parlante, video lunghi autoregressivi a clip da 81 frame
con 25 «motion frame» di continuita'):
  testo (umt5, in cache su disco)  ->  immagine/CLIP + VAE  ->  DiT fp8 (dequant bf16 per layer, LoRA al volo,
  block-swap tarato sulla VRAM)  ->  campionatore flow-DPM++ (default) o Euler  ->  VAE decode.
Tutto con ripiego automatico per schede diverse dalla 5090: attenzione SDPA se Sage non c'e', T5 a blocchi se la
VRAM e' poca, adattatore audio fp8 sotto i ~26 GB, block-swap dinamico. Mai `torch.cuda` nella GUI: questo modulo
gira solo nel subprocess del worker.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_WORKERS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_WORKERS)
for _p in (_WORKERS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)                        # `import _qlinear` come negli altri worker; `core` dal repo

import _qlinear as Q                                  # noqa: E402  (RowFp8Linear, read_safetensors, apply_lora)
from wan_mt import audio as wa                        # noqa: E402
from wan_mt.model import WanMTModel, BlockSwap, set_attention_mode, effective_attention  # noqa: E402

NEG_DEFAULT = ("bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
               "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
               "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
               "picture, messy background, three legs, many people in the background, walking backwards")
FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


from wan_mt.paths import resolve_paths, missing_required, MODELS  # noqa: E402,F401  (torch-free, condiviso col servizio)


# ----------------------------------------------------------------------------------------- piano VRAM ----
@dataclass
class Plan:
    resident_blocks: int
    audio_fp8: bool
    t5_stream: bool
    attn: str
    total_gb: float
    free_gb: float
    reserve_gb: float
    notes: list = field(default_factory=list)


def make_plan(total_gb: float, free_gb: float, width: int, height: int, *, blocks: int = 40,
              block_gb: float = 0.43, audio_block_gb_bf16: float = 0.12, fixed_gb: float = 1.2,
              override_swap=None, attn: str = "auto") -> Plan:
    """Taratura DINAMICA sulla scheda. block_gb = blocco DiT fp8 (~17 GB/40), audio bf16 ~0,12 GB/blocco (fp8 la meta'),
    fixed_gb = parti non-blocco (embedding/head/time_*/audio_proj); CLIP e VAE sono gia' in VRAM al momento della misura. Riserva attivazioni ∝ pixel (480x832 ~ 9,5 GB)."""
    mp = width * height / (480 * 832)
    reserve = 3.0 + 6.5 * mp
    audio_fp8 = total_gb < 26.0
    per_block = block_gb + (audio_block_gb_bf16 * (0.5 if audio_fp8 else 1.0))
    budget = free_gb - reserve - fixed_gb
    resident = int(max(0.0, budget) / per_block)
    resident = max(0, min(blocks, resident))
    notes = []
    if override_swap not in (None, "", "auto"):
        resident = max(0, min(blocks, blocks - int(override_swap)))
        notes.append(f"blocks_to_swap forzato = {blocks - resident}")
    if resident < 6:
        notes.append("VRAM molto stretta: pochi blocchi residenti, la generazione sara' lenta (streaming dalla RAM)")
    return Plan(resident, audio_fp8, t5_stream=free_gb < 18.0, attn=attn, total_gb=total_gb, free_gb=free_gb,
                reserve_gb=reserve, notes=notes)


# ----------------------------------------------------------------------------------------- utilita' ----
def header_keys(path: str) -> list[str]:
    """Chiavi di un .safetensors dall'header (senza mmap, senza leggere i tensori)."""
    from core import st_direct
    return [k for k in st_direct.read_header(path)[0] if k != "__metadata__"]


def resize_center_crop(img, w: int, h: int):
    """PIL.Image -> tensor [1,3,1,H,W] in [-1,1] (riempie il canvas senza stirare: scala 'cover' + ritaglio centrale)."""
    from PIL import Image
    sw, sh = img.size
    s = max(w / sw, h / sh)
    nw, nh = math.ceil(sw * s), math.ceil(sh * s)
    img = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    l, t = (nw - w) // 2, (nh - h) // 2
    img = img.crop((l, t, l + w, t + h))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - 0.5) * 2.0)[None, :, None]


def add_noise(orig, noise, t):
    t = float(t) / 1000.0
    return (1 - t) * orig + t * noise


# ----------------------------------------------------------------------------------------- engine ----
class MultiTalkEngine:
    def __init__(self, paths: dict, device="cuda", log=print, attn="auto"):
        self.paths, self.dev, self.log = paths, torch.device(device), log
        self.attn = set_attention_mode(attn)
        self.model: WanMTModel | None = None
        self.swap: BlockSwap | None = None
        self.vae = None
        self.clip = None
        self.plan: Plan | None = None
        self.text_ctx = None
        self._text_cache = Path(os.environ.get("LTX_CACHE_DIR", str(Path.home() / ".cache" / "ltxstudio"))) / "wan_mt_text"

    # ---- memoria ----
    def vram(self):
        if self.dev.type != "cuda" or not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info(self.dev)
        return total / 2 ** 30, free / 2 ** 30

    def _empty(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- testo ----
    def encode_text(self, prompts: list[str]) -> list[torch.Tensor]:
        """umt5-xxl: embedding [len,4096] bf16 per ogni prompt. Cache su disco: dal 2° uso il T5 (11 GB) non si carica."""
        self._text_cache.mkdir(parents=True, exist_ok=True)
        tag = str(os.path.getsize(self.paths["t5"]))
        keys = [hashlib.sha256((tag + "|" + p).encode("utf-8")).hexdigest()[:32] for p in prompts]
        out: list = [None] * len(prompts)
        for i, k in enumerate(keys):
            f = self._text_cache / f"{k}.pt"
            if f.exists():
                try:
                    out[i] = torch.load(f, map_location="cpu", weights_only=True)
                except Exception:                      # noqa: BLE001 — cache corrotta: si rigenera
                    out[i] = None
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            from wan_mt.t5 import T5EncoderModel
            total, free = self.vram()
            stream = free < 18.0
            self.log(f"[t5] carico umt5-xxl ({'a blocchi' if stream else 'intero in VRAM'}), {len(todo)} prompt…")
            t5 = T5EncoderModel(512, self.paths["t5"], self.paths["t5_tokenizer"])
            ctx = t5.encode([prompts[i] for i in todo], self.dev, stream=stream)
            for i, c in zip(todo, ctx):
                out[i] = c.cpu()
                torch.save(out[i], self._text_cache / f"{keys[i]}.pt")
            del t5, ctx
            self._empty()
        return [o.to(self.dev, torch.bfloat16) for o in out]

    def prepare_text(self, prompt: str, negative: str = "", need_null: bool = False):
        """Calcola/carica gli embedding di testo e li tiene per i clip (self.text_ctx = (positivo, negativo|None))."""
        ps = [prompt] + ([negative or NEG_DEFAULT] if need_null else [])
        res = self.encode_text(ps)
        self.text_ctx = (res[0], res[1] if need_null else res[0])

    # ---- CLIP (HF CLIPVisionModelWithProjection dal file comfy `clip_vision_h`) ----
    def load_clip(self):
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        cfg = CLIPVisionConfig(hidden_size=1280, intermediate_size=5120, projection_dim=1024, num_hidden_layers=32,
                               num_attention_heads=16, num_channels=3, image_size=224, patch_size=14,
                               hidden_act="gelu", layer_norm_eps=1e-5)
        with torch.device("meta"):
            m = CLIPVisionModelWithProjection(cfg)
        sd = Q.read_safetensors(self.paths["clip"], device="cpu")
        sd.pop("vision_model.embeddings.position_ids", None)
        miss, unexp = m.load_state_dict(sd, strict=False, assign=True)
        miss = [k for k in miss if "position_ids" not in k]
        if miss:
            raise RuntimeError(f"CLIP vision: chiavi mancanti {miss[:4]}")
        if "vision_model.embeddings.position_ids" in dict(m.named_buffers()):
            m.vision_model.embeddings.register_buffer("position_ids", torch.arange(257).expand((1, -1)), persistent=False)
        self.clip = m.eval().requires_grad_(False).to(self.dev, torch.float16)

    @torch.no_grad()
    def clip_features(self, img: torch.Tensor) -> torch.Tensor:
        """img [1,3,1,H,W] in [-1,1] -> [1,257,1280] (penultimo hidden state, come ComfyUI/Wan)."""
        x = (img[:, :, 0].to(self.dev, torch.float32) + 1.0) / 2.0               # [1,3,H,W] 0..1
        h, w = x.shape[-2:]
        s = 224.0 / min(h, w)
        x = F.interpolate(x, size=(max(224, round(h * s)), max(224, round(w * s))), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
        t, l = (x.shape[-2] - 224) // 2, (x.shape[-1] - 224) // 2
        x = x[..., t:t + 224, l:l + 224]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.dev).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.dev).view(1, 3, 1, 1)
        cdt = next(self.clip.parameters()).dtype                                  # fp16 su GPU, fp32 nei test CPU
        out = self.clip(pixel_values=((x - mean) / std).to(cdt), output_hidden_states=True)
        return out.hidden_states[-2].to(torch.bfloat16)

    # ---- VAE ----
    def load_vae(self):
        from wan_mt.vae import WanVAE
        self.vae = WanVAE(z_dim=16, vae_pth=self.paths["vae"], dtype=torch.bfloat16, device=self.dev)

    # ---- DiT ----
    def load_dit(self, width: int, height: int, loras=(), blocks_to_swap="auto", audio_fp8=None):
        """Carica il DiT fp8 + adattatore MultiTalk + LoRA, con il block-swap deciso dal piano VRAM."""
        total, free = self.vram()
        plan = make_plan(total, free, width, height, override_swap=blocks_to_swap, attn=self.attn)
        if audio_fp8 is not None:
            plan.audio_fp8 = bool(audio_fp8)
        self.plan = plan
        self.log(f"[plan] GPU {total:.0f} GB tot / {free:.0f} GB liberi, riserva attivazioni {plan.reserve_gb:.1f} GB -> "
                 f"{plan.resident_blocks}/40 blocchi residenti (swap {40 - plan.resident_blocks}), audio "
                 f"{'fp8' if plan.audio_fp8 else 'bf16'}, attenzione {effective_attention()}")
        for n in plan.notes:
            self.log(f"[plan] {n}")
        with torch.device("meta"):
            model = WanMTModel()
        # 1) pesi: il blocco i >= resident va su CPU, il resto su GPU (UN tensore alla volta, niente mmap; i float32
        #    diventano bf16 subito, tensore per tensore, cosi' il picco di VRAM non include copie fp32)
        keys_swapped = lambda k: k.startswith("blocks.") and int(k.split(".")[1]) >= plan.resident_blocks  # noqa: E731
        sd = {}
        for path in (self.paths["dit"], self.paths["multitalk"]):
            allk = header_keys(path)
            cpu_keys = {k for k in allk if keys_swapped(k)}
            gpu_keys = set(allk) - cpu_keys
            for dev, skip in ((self.dev, cpu_keys), ("cpu", gpu_keys)):
                if len(skip) == len(allk):
                    continue
                sd.update(Q.read_safetensors(path, device=str(dev), skip=skip, cast=torch.bfloat16))
        if plan.audio_fp8:                              # adattatore audio in fp8 pre-cast (come l'fp8 ufficiale Kijai)
            for k in list(sd):
                if ".audio_cross_attn." in k and k.endswith(".weight") and sd[k].ndim == 2:
                    sd[k] = sd[k].to(torch.float8_e4m3fn)
        sd = Q.quantize_from_state_dict(model, sd)      # nn.Linear fp8 -> RowFp8Linear; stato riscritto nei buffer nostri
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if missing:
            raise RuntimeError(f"DiT Wan/MultiTalk: chiavi mancanti ({len(missing)}): {missing[:6]}")
        if unexpected:
            self.log(f"[dit] {len(unexpected)} chiavi extra ignorate: {unexpected[:3]}")
        # 2) LoRA al volo (distill lightx2v + LoRA utente)
        for lp, st in loras:
            self._apply_lora(model, lp, float(st))
        # 3) block-swap
        model.swap = None
        if plan.resident_blocks < len(model.blocks):
            model.swap = BlockSwap(model.blocks, plan.resident_blocks, self.dev, prefetch=True)
        self.model = model.eval().requires_grad_(False)
        self.swap = model.swap
        # le parti non-blocco restano sulla GPU (embedding/head/audio_proj)
        for name, m in model.named_children():
            if name != "blocks":
                m.to(self.dev)
        return plan

    def _apply_lora(self, model, path: str, strength: float):
        lsd = Q.read_safetensors(path, device="cpu")
        # delta 1-D su norme/LayerNorm (`.diff` su pesi di norma): somma diretta al peso
        mods = dict(model.named_modules())
        for k in list(lsd):
            if k.endswith(".diff") and lsd[k].ndim == 1:
                base = k[:-5]
                for pre in ("diffusion_model.", "model.diffusion_model."):
                    if base.startswith(pre):
                        base = base[len(pre):]
                m = mods.get(base)
                d = lsd.pop(k)
                if m is not None and getattr(m, "weight", None) is not None and m.weight.shape == d.shape:
                    with torch.no_grad():
                        m.weight.data = m.weight.data + (strength * d.float()).to(m.weight.device, m.weight.dtype)
        # _qlinear.apply_lora SALTA le coppie lora_down/up di un modulo che ha anche `diff_b` (pensato per i soli delta
        # H3): le lightx2v hanno entrambi -> due passate separate, prima i delta pieni poi le coppie a basso rango
        pairs = {k: v for k, v in lsd.items() if not (k.endswith(".diff") or k.endswith(".diff_b"))}
        diffs = {k: v for k, v in lsd.items() if k.endswith(".diff") or k.endswith(".diff_b")}
        nd = Q.apply_lora(model, diffs, strength) if diffs else 0
        nl = Q.apply_lora(model, pairs, strength) if pairs else 0
        self.log(f"[lora] {os.path.basename(path)} x{strength:g}: {nl} coppie a basso rango + {nd} delta (bias) agganciati")
        if nl == 0:
            raise RuntimeError(f"LoRA {os.path.basename(path)}: nessuna coppia agganciata (chiavi non riconosciute?)")

    def free_dit(self):
        if self.swap is not None:
            self.swap.offload_all()
        self.model = None
        self.swap = None
        import gc
        gc.collect()
        self._empty()

    # ---- campionamento ----
    def _predict(self, lat, t, ctx, clip, y, audio, audio_scale):
        return self.model(lat, t, ctx, clip, y, audio=audio, audio_scale=audio_scale)

    @torch.no_grad()
    def sample_clip(self, *, noise, y, text, text_null, clip, audio, nf: int, steps: int, shift: float, cfg: float,
                    audio_cfg: float, audio_scale: float, scheduler: str, seed: int, motion=None):
        """Un clip: noise [16,T,h,w], y [20,T,h,w], audio [1,nf,5,12,768]. `motion` = latenti di continuita' [16,Tm,h,w]
        (clip successivi): reiniettati (con rumore al sigma corrente) a ogni passo, come nel codice ufficiale."""
        from wan_mt.fm_solvers import FlowDPMSolverMultistepScheduler
        g = torch.Generator(device=self.dev).manual_seed(int(seed))
        if scheduler.startswith("dpm"):
            sch = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type="sde-dpmsolver++" if "sde" in scheduler else "dpmsolver++")
            sch.set_timesteps(steps, device=self.dev)
            ts = sch.timesteps
        else:                                                # euler flow-matching come il pipeline ufficiale
            ts = torch.linspace(1000, 1, steps, device=self.dev)
            ts = shift * (ts / 1000) / (1 + (shift - 1) * (ts / 1000)) * 1000
            ts = torch.cat([ts, torch.zeros(1, device=self.dev)])
            sch = None
        lat = noise.clone()
        n = len(ts) if sch is not None else len(ts) - 1

        def inject(t_cur):
            if motion is not None:
                nz = torch.randn(motion.shape, generator=g, device=self.dev, dtype=motion.dtype)
                lat[:, :motion.shape[1]] = add_noise(motion, nz, t_cur)

        inject(float(ts[0]))
        for i in range(n):
            t = ts[i].reshape(1)
            pc = self._predict(lat, t, text, clip, y, audio, audio_scale)
            if math.isclose(cfg, 1.0) and math.isclose(audio_cfg, 1.0):
                pred = pc
            elif math.isclose(cfg, 1.0):
                pd = self._predict(lat, t, text, clip, y, torch.zeros_like(audio), audio_scale)
                pred = pd + audio_cfg * (pc - pd)
            else:
                pdt = self._predict(lat, t, text_null, clip, y, audio, audio_scale)
                pu = self._predict(lat, t, text_null, clip, y, torch.zeros_like(audio), audio_scale)
                pred = pu + cfg * (pc - pdt) + audio_cfg * (pdt - pu)
            if sch is not None:
                lat = sch.step(pred.unsqueeze(0), ts[i], lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
                t_next = float(ts[i + 1]) if i + 1 < len(ts) else 0.0
            else:
                dt = (ts[i] - ts[i + 1]) / 1000.0
                lat = lat - pred * dt
                t_next = float(ts[i + 1])
            inject(t_next)
        return lat

    # ---- video completo (autoregressivo, un parlante) ----
    @torch.no_grad()
    def generate(self, *, image, audio_emb: torch.Tensor, total_frames: int, width: int, height: int, prompt: str,
                 negative: str = "", steps: int = 10, shift: float = 5.0, cfg: float = 1.0, audio_cfg: float = 1.0,
                 audio_scale: float = 2.0, scheduler: str = "dpm++_sde", seed: int = 8, clip_frames: int = 81,
                 motion_frames: int = 25, ref_anchor: float = 0.0, cache=None, on_progress=None):
        """Generatore: restituisce, per ogni clip, i NUOVI frame come uint8 [n,H,W,3]. `audio_emb` [T,12,768]
        (25 fps). `cache` opz.: oggetto con load(ci, n)->uint8|None e save(ci, uint8) per riprendere dopo un crash."""
        assert width % 16 == 0 and height % 16 == 0
        clip_frames = max(9, (clip_frames - 1) // 4 * 4 + 1)
        motion_frames = max(1, min(motion_frames, clip_frames - 8))
        lat_h, lat_w = height // 8, width // 8
        cond = resize_center_crop(image, width, height).to(self.dev)           # [1,3,1,H,W]
        ref_clip = self.clip_features(cond) if ref_anchor > 0 else None
        ctx, ctx_null = self.text_ctx
        # audio: pad finale con la coda riflessa (come l'originale) cosi' i clip finali hanno finestre vere
        need = total_frames + clip_frames + 8
        emb = audio_emb
        if emb.shape[0] < need:
            miss = need - emb.shape[0]
            tail = torch.flip(emb[-min(miss, emb.shape[0]):], dims=[0])
            while tail.shape[0] < miss:
                tail = torch.cat([tail, tail])
            emb = torch.cat([emb, tail[:miss]], dim=0)
        idx_win = (torch.arange(5) - 2)[None]
        produced, ci = 0, 0
        cur_motion = 1
        last_pix = None
        a_start = 0
        while produced < total_frames:
            n_new_expected = clip_frames if ci == 0 else clip_frames - cur_motion
            if cache is not None:
                got = cache.load(ci, n_new_expected)
                if got is not None:
                    self.log(f"[clip {ci}] RIPRESO da cache ({got.shape[0]} frame)")
                    produced += got.shape[0]
                    last_pix = (torch.from_numpy(got).to(self.dev).float() / 127.5 - 1).permute(3, 0, 1, 2)[None] \
                        if got.shape[0] >= motion_frames else last_pix
                    cur_motion = motion_frames
                    a_start += clip_frames - motion_frames
                    ci += 1
                    yield got
                    continue
            self.log(f"[clip {ci}] frame {produced}..{produced + n_new_expected} (audio da {a_start})")
            ci_idx = (torch.arange(a_start, a_start + clip_frames)[:, None] + idx_win).clamp(0, emb.shape[0] - 1)
            aud = emb[ci_idx][None].to(self.dev, torch.bfloat16)               # [1,nf,5,12,768]
            # condizione immagine/motion
            if ci == 0:
                cimg = cond
            else:
                cimg = last_pix[:, :, -cur_motion:]
            clip_f = self.clip_features(cimg[:, :, -1:])
            if ref_clip is not None and ci > 0:
                clip_f = (clip_f + ref_anchor * ref_clip) / 2.0                # come WanVideoClipVisionEncode(average)
            T = (clip_frames - 1) // 4 + 1
            msk = torch.ones(1, clip_frames, lat_h, lat_w, device=self.dev)
            msk[:, cur_motion:] = 0
            msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], 4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]       # [4,T,h,w]
            pad = torch.zeros(1, 3, clip_frames - cimg.shape[2], height, width, device=self.dev)
            yl = self.vae.encode(torch.cat([cimg.to(self.dev, torch.float32), pad], dim=2))[0]  # [16,T,h,w]
            motion = None
            if ci > 0:
                motion = yl[:, :1 + (cur_motion - 1) // 4].to(torch.bfloat16)
            y = torch.cat([msk.to(torch.bfloat16), yl.to(torch.bfloat16)], dim=0)
            noise = torch.randn(16, T, lat_h, lat_w, generator=torch.Generator(device=self.dev).manual_seed(int(seed) + ci),
                                device=self.dev, dtype=torch.float32)
            lat = self.sample_clip(noise=noise, y=y, text=ctx, text_null=ctx_null, clip=clip_f, audio=aud, nf=clip_frames,
                                   steps=steps, shift=shift, cfg=cfg, audio_cfg=audio_cfg, audio_scale=audio_scale,
                                   scheduler=scheduler, seed=int(seed) + ci, motion=motion)
            vid = self.vae.decode([lat.to(torch.float32)])[0]                  # [3,F,H,W] in [-1,1]
            vid = vid[None]
            new = vid if ci == 0 else vid[:, :, cur_motion:]
            frames = ((new[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
            if cache is not None:
                cache.save(ci, frames)
            last_pix = vid[:, :, -motion_frames:].to(torch.float32)
            cur_motion = motion_frames
            a_start += clip_frames - motion_frames
            produced += frames.shape[0]
            ci += 1
            if on_progress:
                on_progress(min(produced, total_frames), total_frames)
            del lat, vid, new
            self._empty()
            yield frames
