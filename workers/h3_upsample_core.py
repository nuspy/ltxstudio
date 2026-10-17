"""H3 Video Upsampler — il nucleo con i modelli (rete dell'upscaler latente, layout a sequenza unica con la guida,
ri-disegno a tile, codifica/decodifica del VAE a finestre). Gira nel venv diffusers_next; niente ComfyUI.

Porting nativo di tiled_upsampler.py / upscale_guide.py / video_upsampler.py di ComfyUI-H3-Video-Upsampler
(sandpies / dntpi, MIT; rete e statistiche di LBH-123-AI, Apache-2.0; layout della guida di Alissonerdx, MIT — vedi
docs/THIRD_PARTY_NOTICES_H3_UPSAMPLER.md). Nessun codice è copiato dall'altro nodo di bbaudio-2025 né dal fork di
ai-toolkit: la ricetta è quella dei file studiati, la geometria è ricostruita sul pipeline diffusers di H3.

Convenzioni (verificate sul codice di diffusers, vedi test_h3_upsample_layout.py):
  · i latenti qui sono NORMALIZZATI ((z - mean)/std con le statistiche del VAE): sono quelli che legge il DiT e, per
    l'identità mean/std VAE == mean/std dell'upscaler, anche l'upscaler;
  · timestep H3: t = 1 - sigma, t = 1 è pulito; il perno e la guida stanno a t = 0,999 e vi restano per il passo;
  · il ri-disegno è UN passo da sigma_start a 0: x0 = x_t + sigma_start · v (stessa formula di MiniMaxH3Scheduler.step).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from core import h3_upsample_plan as P                                # noqa: E402

AUDIO_CHANNELS = 2
TEXT_TAG, VIDEO_TAG, AUDIO_TAG = 1, 0, 2          # MINIMAX_H3_*_TAG di diffusers (verificati dal test del layout)
PATCH = (1, 2, 2)
# Livello t a cui la colonna sonora (congelata) entra nel ri-disegno. 1.0 = pulita, come i riferimenti audio del
# Ref2VA (condition_audio_timestep). La ricetta ComfyUI non lo dichiara nel sorgente studiato: DECISIONE RICHIESTA.
AUDIO_COND_T = 1.0


def _log(m: str) -> None:
    print(f"[h3up] {m}", flush=True)


# ── la rete dell'upscaler latente (layout dei pesi di LBH-123-AI) ───────────────────────────────────────────────
def _gn(c):
    return nn.GroupNorm(32, c)


class _Res(nn.Module):
    def __init__(self, c, emb):
        super().__init__()
        self.in_layers = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb, 2 * c))
        self.out_norm = _gn(c)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(0.0), nn.Conv3d(c, c, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype)[..., None, None, None].chunk(2, dim=1)
        return x + self.out_layers(self.out_norm(h) * (1 + scale) + shift)


class _Temporal(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.norm = _gn(c)
        self.dwconv = nn.Conv3d(c, c, (k, 1, 1), padding=(k // 2, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentUpscaler(nn.Module):
    """Rete 3D: conv_in → blocchi (residui + conv temporali) → resize trilineare → blocchi → conv_out. Quale slot è
    residuo e quale conv temporale si legge dalle chiavi: un file con un altro ritmo viene rifiutato."""

    def __init__(self, sd):
        super().__init__()
        c = int(sd["conv_in.weight"].shape[0])
        cin = int(sd["conv_in.weight"].shape[1])
        emb = int(sd["embed.0.weight"].shape[0])
        self.conv_in = nn.Conv3d(cin, c, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.in_blocks = self._blocks(sd, "in_blocks", c, emb)
        self.out_blocks = self._blocks(sd, "out_blocks", c, emb)
        self.norm_out = _gn(c)
        self.conv_out = nn.Conv3d(c, cin, 3, padding=1)

    @staticmethod
    def _blocks(sd, name, c, emb):
        slots = {}
        for k in sd:
            if k.startswith(name + "."):
                i = int(k.split(".")[1])
                if k.endswith(".dwconv.weight"):
                    slots[i] = _Temporal(c, int(sd[k].shape[2]))
                elif ".in_layers." in k:
                    slots.setdefault(i, None)
        for i in [i for i, m in slots.items() if m is None]:
            slots[i] = _Res(c, emb)
        if sorted(slots) != list(range(len(slots))):
            raise RuntimeError(f"{name}: buchi nella lista dei blocchi, non è un upscaler che questo codice legge")
        return nn.ModuleList(slots[i] for i in range(len(slots)))

    def forward(self, x, size, scale):
        emb = self.embed(torch.tensor([[float(scale) - 1.0]], device=x.device, dtype=x.dtype))
        emb = emb.expand(x.shape[0], -1)
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb) if isinstance(b, _Res) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb) if isinstance(b, _Res) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def load_upscaler(path: str, dtype=torch.float16) -> LatentUpscaler:
    """La rete da un .safetensors (solo safetensors, niente pickle), sulla CPU in fp16."""
    from safetensors.torch import load_file
    if not str(path).endswith(".safetensors"):
        raise RuntimeError(f"{Path(path).name}: si leggono solo file .safetensors")
    sd = load_file(path, device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    if "conv_in.weight" not in sd or "embed.0.weight" not in sd:
        raise RuntimeError(f"{Path(path).name} non è un upscaler latente H3")
    if int(sd["conv_in.weight"].shape[1]) != 24:
        raise RuntimeError(f"{Path(path).name} non prende latenti H3 a 24 canali")
    net = LatentUpscaler(sd)
    net.load_state_dict({k: (v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v) for k, v in sd.items()},
                        strict=True)
    net = net.eval().requires_grad_(False).to(dtype)
    _log(f"upscaler {Path(path).name}: {sum(p.numel() for p in net.parameters()) / 1e6:.0f}M parametri")
    return net


def upscale_chunk(net: LatentUpscaler, video, w_px: int, h_px: int, device, dtype=None):
    """Il latente (normalizzato) di un segmento portato a (w_px, h_px) dalla rete; la rete sa la scala davvero
    ottenuta (media dei due assi) e gira in fp16 per intero, come nella ricetta."""
    _, _, t, h_in, w_in = video.shape
    scale = (w_px / (w_in * P.SPATIAL) + h_px / (h_in * P.SPATIAL)) / 2.0
    h_out = max(1, int(round(h_px / P.SPATIAL)))
    w_out = max(1, int(round(w_px / P.SPATIAL)))
    if scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise RuntimeError(f"{w_px}x{h_px} è più piccolo del video ({w_in * P.SPATIAL}x{h_in * P.SPATIAL}): "
                           "questa funzione ingrandisce soltanto. Alza i megapixel.")
    if (h_out, w_out) == (h_in, w_in):
        return video
    dtype = dtype or next(net.parameters()).dtype       # fp16 come nella ricetta (la rete vive in fp16)
    net.to(device)
    try:
        with torch.inference_mode():
            x = video.to(device=device, dtype=dtype, copy=True)
            out = net(x, (t, h_out, w_out), scale)
            del x
    finally:
        net.to("cpu")
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return out.to(device="cpu", dtype=video.dtype)


# ── layout a sequenza unica con perno e guida ─────────────────────────────────────────────────────────────────
def _diffusers_layout_helpers():
    from diffusers.modular_pipelines.minimax_h3 import before_denoise as B
    return B


def build_tile_layout(text_token_tags: torch.Tensor, steps: int, lat_h: int, lat_w: int, audio_t: int, *,
                      pin: bool = False, guide_factor: int = 0, guide_steps: int | None = None) -> dict:
    """Layout [testo | perno | guida | audio | video] di UNA forward di tile (Ref2VA senza riferimenti).

    Coordinate come nel pipeline diffusers (griglia fp64 normalizzata sull'area del TILE, tempo che parte da
    n_testo): il perno è un passo latente ancorato al «primo» (stesso tempo del passo 0); la guida occupa
    `guide_steps` passi SULL'OROLOGIO del video e, in ogni passo, solo le patch (i·f, j·f) della griglia del tile
    (`target_frame_grid[::f, ::f]`, target_grid_stride_v1). Perno e guida sono righe «condizione»: stanno davanti al
    video generato, a t = 0,999. Ritorna i tensori che il DiT legge per nome + quante righe condizione ci sono."""
    B = _diffusers_layout_helpers()
    ph, pw = PATCH[1], PATCH[2]
    hp, wp = lat_h // ph, lat_w // pw
    rpf = hp * wp
    n_text = int(text_token_tags.shape[0])
    f = int(guide_factor)
    vt = int(guide_steps if guide_steps is not None else steps) if f > 1 else 0
    if f > 1 and (hp % f or wp % f):
        raise ValueError(f"griglia di patch {hp}x{wp} non divisibile per il fattore di guida {f}")
    gh, gw = (hp // f, wp // f) if f > 1 else (0, 0)
    n_pin = rpf if pin else 0
    n_guide = vt * gh * gw
    n_audio = audio_t * AUDIO_CHANNELS
    n_video = steps * rpf
    seq = n_text + n_pin + n_guide + n_audio + n_video
    pos = torch.zeros(seq, 3, dtype=torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    frame_grid, width_grid = B._frame_position_grid(lat_h, lat_w, ph, pw)
    time_grid = B._temporal_position_grid(steps, float(n_text))

    c = n_text
    if pin:
        pos[c:c + n_pin, 0] = float(n_text)
        pos[c:c + n_pin, 1:] = frame_grid
        c += n_pin
    if vt:
        if vt > steps:
            raise ValueError(f"la guida ha {vt} passi, il video {steps}")
        sub = frame_grid.reshape(hp, wp, 2)[::f, ::f].reshape(-1, 2)
        pos[c:c + n_guide, 0] = time_grid[:vt].repeat_interleave(gh * gw)
        pos[c:c + n_guide, 1:] = sub.repeat(vt, 1)
        c += n_guide
    audio_start = c
    video_start = audio_start + n_audio
    if n_audio:
        B._fill_audio_positions(pos, slice(audio_start, video_start), audio_t, float(n_text), width_grid,
                                AUDIO_CHANNELS)
    pos[video_start:, 0] = time_grid.repeat_interleave(rpf)
    pos[video_start:, 1:] = frame_grid.repeat(steps, 1)

    video_idx = torch.cat([torch.arange(n_text, audio_start), torch.arange(video_start, seq)])
    audio_idx = torch.arange(audio_start, video_start)
    text_idx = torch.arange(n_text)
    tags = torch.empty(seq, dtype=torch.long)
    tags[text_idx] = text_token_tags.to(torch.long)
    tags[audio_idx] = AUDIO_TAG
    tags[video_idx] = VIDEO_TAG
    return {"position_ids": pos, "token_tags": tags, "video_indices": video_idx, "audio_indices": audio_idx,
            "text_indices": text_idx, "n_cond_rows": n_pin + n_guide, "n_pin_rows": n_pin, "n_guide_rows": n_guide,
            "seq_len": seq}


def patchify(lat: torch.Tensor) -> torch.Tensor:
    B = _diffusers_layout_helpers()
    return B.patchify_video_latents(lat, PATCH)


def unpatchify(rows: torch.Tensor, steps: int, lat_h: int, lat_w: int, channels: int = 24) -> torch.Tensor:
    """Inverso di patchify_video_latents (stessa permutazione dello step di decodifica di diffusers)."""
    pt, ph, pw = PATCH
    r = rows.reshape(-1, steps // pt, lat_h // ph, lat_w // pw, channels, pt, ph, pw)
    r = r.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return r.reshape(-1, channels, steps, lat_h, lat_w).contiguous()


def audio_to_rows(audio: torch.Tensor) -> torch.Tensor:
    """(1, 32, 2, Ta) → righe channel-major (2·Ta, 32) come il layout del pipeline."""
    return audio[0].permute(1, 2, 0).reshape(-1, audio.shape[1]).contiguous()


def row_timesteps(layout: dict, n_text: int, t_video: float, t_cond: float, t_audio: float):
    """(timestep distinti, indice per riga) con i livelli: video generato, righe condizione, audio, testo = video."""
    B = _diffusers_layout_helpers()
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep as S
    n_c = layout["n_cond_rows"]
    return S.build_row_timesteps(layout["video_indices"], layout["audio_indices"], n_c, 0, n_text,
                                 float(t_video), float(t_audio), float(t_cond), 1.0)


# ── il ri-disegno a tile ──────────────────────────────────────────────────────────────────────────────────────
def redraw_chunk(transformer, prompt_embeds: torch.Tensor, text_tags: torch.Tensor, video: torch.Tensor,
                 audio: torch.Tensor, *, sigma: float, seed: int = P.SEED, factor: int = 0,
                 guide: torch.Tensor | None = None, pin: torch.Tensor | None = None,
                 budget: int = P.TOKEN_BUDGET, device="cuda", audio_t_cond: float = AUDIO_COND_T,
                 anchor_t: float = P.ANCHOR_AUG, on_tile=None) -> torch.Tensor:
    """Un segmento: ogni tile fa UNA forward (tutti i tile INSIEME, poi la fusione), da `sigma` a 0.

    video  (1, 24, T, H, W) latente ingrandito (normalizzato), audio (1, 32, 2, Ta) congelato,
    guide  (1, 24, T, H/f, W/f) | None, pin (1, 24, 1, H, W) | None (frame 0 del segmento = ri-disegnato del
    segmento precedente). Ogni tile ritaglia la sua finestra del rumore (un unico campo al formato del segmento:
    due tile che coprono lo stesso punto partono dallo stesso rumore) e della guida/perno, e pesa il proprio
    risultato con la finestra a coseno rialzato. -> latente ri-disegnato (1, 24, T, H, W) sulla CPU."""
    _, ch, T, H, W = video.shape
    f = int(factor) if guide is not None else 0
    tiles, grid = P.tile_grid(H, W, T, f if f > 1 else 1, budget)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    field = torch.randn(video.shape, generator=gen, dtype=torch.float32)
    cgen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
    n_text = int(text_tags.shape[0])
    t0 = 1.0 - float(sigma)
    t_cond = max(t0, float(anchor_t))
    audio_rows = audio_to_rows(audio.to(torch.float32)).to(device)[None]
    embeds = prompt_embeds.to(device)
    acc = torch.zeros(video.shape, dtype=torch.float32)
    wsum = torch.zeros((1, 1, 1, H, W), dtype=torch.float32)
    layouts = {}
    _log(f"segmento {W * P.SPATIAL}x{H * P.SPATIAL}px · {T} passi latenti: {grid[0]}x{grid[1]} tile da "
         f"{grid[2]}x{grid[3]}px, sovrapposizione {grid[4]}/{grid[5]}px, "
         f"{P.guided_seq_len(grid[2], grid[3], T, f) if f > 1 else P.seq_len(grid[2], grid[3], T)} token per forward")
    for n_tile, (r0, c0, rows, cols, st, sl, top, left, bottom, right) in enumerate(tiles):
        key = (rows, cols)
        if key not in layouts:
            layouts[key] = build_tile_layout(text_tags, T, rows, cols, int(audio.shape[-1]),
                                             pin=pin is not None, guide_factor=f, guide_steps=T if f > 1 else None)
        lay = layouts[key]
        xt = video[:, :, :, r0:r0 + rows, c0:c0 + cols].to(torch.float32)
        nz = field[:, :, :, r0:r0 + rows, c0:c0 + cols]
        x_noised = t0 * xt + (1.0 - t0) * nz              # MiniMaxH3Scheduler.scale_noise(x, t0, noise)
        cond_rows = []
        if pin is not None:
            pc = pin[:, :, :, r0:r0 + rows, c0:c0 + cols].to(torch.float32)
            cn = torch.randn(pc.shape, generator=cgen, dtype=torch.float32)
            cond_rows.append(patchify(anchor_t * pc + (1.0 - anchor_t) * cn))
        if f > 1:
            gc = guide[:, :, :, r0 // f:(r0 + rows) // f, c0 // f:(c0 + cols) // f].to(torch.float32)
            cn = torch.randn(gc.shape, generator=cgen, dtype=torch.float32)
            cond_rows.append(patchify(anchor_t * gc + (1.0 - anchor_t) * cn))
        hid = torch.cat(cond_rows + [patchify(x_noised)]).to(device)
        if hid.shape[0] != len(lay["video_indices"]):
            raise RuntimeError(f"righe video {hid.shape[0]} ≠ indici del layout {len(lay['video_indices'])}")
        ts, ts_idx = row_timesteps(lay, n_text, t0, t_cond, audio_t_cond)
        with torch.inference_mode():
            vout, _aout = transformer(
                hidden_states=hid[None], audio_hidden_states=audio_rows, encoder_hidden_states=embeds,
                timestep=ts.to(device), timestep_indices=ts_idx.to(device),
                token_tags=lay["token_tags"].to(device), position_ids=lay["position_ids"].to(device),
                video_indices=lay["video_indices"].to(device), audio_indices=lay["audio_indices"].to(device),
                text_indices=lay["text_indices"].to(device), return_dict=False)
        v_rows = vout[0, lay["n_cond_rows"]:].float().cpu()
        x_rows = patchify(x_noised)
        x0 = unpatchify(x_rows + float(sigma) * v_rows, T, rows, cols)       # un solo passo fino a sigma 0
        m = P.feather(rows, cols, st, sl, top, left, bottom, right)[None, None, None]
        acc[:, :, :, r0:r0 + rows, c0:c0 + cols] += x0 * m
        wsum[:, :, :, r0:r0 + rows, c0:c0 + cols] += m[:, :, 0]
        del hid, vout
        if on_tile:
            on_tile(n_tile + 1, len(tiles))
    return (acc / wsum.clamp(min=1e-8)).to(video.dtype)


# ── VAE a finestre (memoria limitata) ─────────────────────────────────────────────────────────────────────────
def encode_frames(vae, frames_iter, n_frames: int, *, pixel_mean, pixel_std, latents_mean, latents_std,
                  device, clips_per_batch: int = 1, seed: int = 42):
    """Fotogrammi (iteratore di blocchi float 0..1 [n, H, W, 3]) → latente normalizzato (1, 24, T, H/16, W/16).

    Stessa ricetta del condizionamento di H3 (encode_vae_condition): pixel ImageNet-normalizzati, posteriore
    CAMPIONATO con generatore seminato 42 (indipendente dalla richiesta), arrotondato a fp16, poi normalizzato.
    La codifica è a clip da 17 fotogrammi (ognuna indipendente), l'ultima ripetuta fino a 17, e si scartano gli
    ultimi `token_drop` passi dell'insieme: identica a vae.encode ma senza tenere tutto il video in VRAM."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    clip = int(vae.config.clip_length)
    drop = int(vae.config.token_drop)
    pm = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
    ps = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)
    moments, buf, seen = [], [], 0

    def flush(frames):                                 # frames: [n, H, W, 3] float 0..1, n multiplo di `clip`
        x = frames.to(device, torch.float32).permute(3, 0, 1, 2)[None]
        x = (x - pm) / ps
        for i in range(x.shape[2] // clip):
            moments.append(vae._encode_clip(x[:, :, i * clip:(i + 1) * clip].to(next(vae.encoder.parameters()).dtype)).cpu())

    for blk in frames_iter:
        buf.append(blk)
        seen += int(blk.shape[0])
        have = sum(int(b.shape[0]) for b in buf)
        if have >= clip * clips_per_batch:
            allf = torch.cat(buf)
            use = (allf.shape[0] // clip) * clip
            flush(allf[:use])
            buf = [allf[use:]] if allf.shape[0] > use else []
    if seen != n_frames:
        raise RuntimeError(f"attesi {n_frames} fotogrammi, ricevuti {seen}")
    if buf:
        rest = torch.cat(buf)
        if rest.shape[0]:
            pad = rest[-1:].repeat((-rest.shape[0]) % clip, 1, 1, 1)
            flush(torch.cat([rest, pad]))
    mom = torch.cat(moments, dim=2)
    if drop > 0:
        mom = mom[:, :, :-drop]
    z = DiagonalGaussianDistribution(mom).sample(generator=torch.Generator().manual_seed(seed))
    z = z.to(torch.float16).float()
    return (z - torch.tensor(latents_mean).view(1, -1, 1, 1, 1)) / torch.tensor(latents_std).view(1, -1, 1, 1, 1)


def decode_stream(vae, z: torch.Tensor, emit, *, pixel_mean, pixel_std, latents_mean, latents_std, device):
    """Latente normalizzato (1, 24, T, h, w), T = 5g+2 → fotogrammi, UN pezzo da 17 alla volta tramite `emit(frames
    uint8 [n, H, W, 3])`. Replica AutoencoderKLMiniMaxH3._decode (decodifica per clip da 5+2 passi, i primi 5
    fotogrammi sovrapposti fra clip vicine sfumati linearmente) senza materializzare il video intero, che a 2 MP e
    15 s sono ~9 GB di float32 in VRAM. L'identità con vae.decode è verificata da test_h3_upsample_layout.py."""
    T = int(z.shape[2])
    chunk = int(vae.tokens_chunk_size)
    drop = int(vae.config.token_drop)
    if (T + drop) % chunk:
        raise ValueError(f"{T} passi latenti non sono 5g+2: il video va portato a 17g+5 fotogrammi prima")
    n_chunks = (T + drop) // chunk - 1
    nf = chunk * int(vae.temporal_compression_ratio)
    pre = int(vae.frame_pre_padding)
    ov = int(vae.token_overlap)
    mean = torch.tensor(latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(latents_std, device=device).view(1, -1, 1, 1, 1)
    pm = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
    ps = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)
    dt = next(vae.decoder.parameters()).dtype

    def to_u8(piece):
        v = (piece.float() * ps + pm).clamp(0, 1)
        return (v[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8).cpu().numpy()

    overlap = None
    for i in range(n_chunks):
        zi = (z[:, :, i * chunk:i * chunk + chunk + ov].to(device) * std + mean).to(dt)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                            enabled=torch.device(device).type == "cuda"):
            clip = vae._decode_clip(zi)
        head = clip[:, :, 0:nf][:, :, pre:]
        tail = clip[:, :, nf:2 * nf][:, :, pre:]
        if overlap is not None:
            head = vae._blend(overlap, head, vae.frame_overlap, dim=-3)
        emit(to_u8(head))
        overlap = tail
        del clip, head, zi
    if overlap is not None:
        emit(to_u8(overlap))
