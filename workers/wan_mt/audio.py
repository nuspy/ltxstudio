"""Embedding audio per MultiTalk: wav2vec2 cinese (TencentGameMate/chinese-wav2vec2-base) -> [T_frames, 12, 768].

Come `get_embedding` del repo MultiTalk (hidden state di tutti i 12 strati, interpolati linearmente a `fps` frame
video al secondo). Niente ComfyUI: config e feature extractor sono locali. Normalizzazione del volume a -23 LUFS
(pyloudnorm se c'e', altrimenti una implementazione ITU-R BS.1770 equivalente con scipy).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000


# ---------------------------------------------------------------- loudness (BS.1770 / pyloudnorm) ----------
def _biquad(kind, fs, f0, q, gain_db=0.0):
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw, alpha = np.cos(w0), np.sin(w0) / (2.0 * q)
    if kind == "high_shelf":
        sa = 2.0 * np.sqrt(a_) * alpha
        b = [a_ * ((a_ + 1) + (a_ - 1) * cw + sa), -2 * a_ * ((a_ - 1) + (a_ + 1) * cw), a_ * ((a_ + 1) + (a_ - 1) * cw - sa)]
        a = [(a_ + 1) - (a_ - 1) * cw + sa, 2 * ((a_ - 1) - (a_ + 1) * cw), (a_ + 1) - (a_ - 1) * cw - sa]
    else:                                                    # high_pass
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def integrated_loudness(x: np.ndarray, sr: int = SR) -> float:
    """LUFS integrati di un segnale mono (gating assoluto -70 LUFS e relativo -10 LU, blocchi 400 ms, passo 100 ms)."""
    from scipy.signal import lfilter
    y = lfilter(*_biquad("high_shelf", sr, 1681.9744509555319, 0.7071752369554196, 3.999843853973347), x.astype(np.float64))
    y = lfilter(*_biquad("high_pass", sr, 38.13547087613982, 0.5003270373253953), y)
    blk, step = int(0.4 * sr), int(0.1 * sr)
    if len(y) < blk:
        return float("-inf")
    n = (len(y) - blk) // step + 1
    z = np.array([np.mean(y[i * step:i * step + blk] ** 2) for i in range(n)])
    l = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    keep = l > -70.0
    if not keep.any():
        return float("-inf")
    thr = -0.691 + 10 * np.log10(np.mean(z[keep])) - 10.0
    keep = keep & (l > thr)
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10 * np.log10(np.mean(z[keep])))


def loudness_norm(x: np.ndarray, sr: int = SR, lufs: float = -23.0) -> np.ndarray:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        ld = meter.integrated_loudness(x)
        return x if abs(ld) > 100 else pyln.normalize.loudness(x, ld, lufs)
    except ImportError:
        ld = integrated_loudness(x, sr)
        if not np.isfinite(ld) or abs(ld) > 100:
            return x
        return (x * 10.0 ** ((lufs - ld) / 20.0)).astype(x.dtype)


# ---------------------------------------------------------------- caricamento audio ----------
def load_audio_16k(path: str) -> np.ndarray:
    """File audio/video -> mono float32 a 16 kHz (torchaudio, ripiego ffmpeg)."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.numpy().astype(np.float32)
    except Exception:                                        # noqa: BLE001 — formato non letto da torchaudio
        import subprocess
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()


# ---------------------------------------------------------------- modello ----------
def _linear_interpolation(features: torch.Tensor, seq_len: int) -> torch.Tensor:
    f = features.transpose(1, 2)
    return F.interpolate(f, size=seq_len, align_corners=True, mode="linear").transpose(1, 2)


def build_wav2vec(bin_path: str, device, dtype=torch.float16):
    """Wav2Vec2Model dalla config locale + pesi `pytorch_model.bin` (prefisso 'wav2vec2.', weight_norm g/v)."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    cfg = Wav2Vec2Config(**json.load(open(os.path.join(_HERE, "wav2vec2_config.json"), encoding="utf-8")))
    with torch.device("meta"):
        model = Wav2Vec2Model(cfg)
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        if not k.startswith("wav2vec2."):
            continue
        k = k[len("wav2vec2."):]
        k = k.replace("encoder.pos_conv_embed.conv.weight_g", "encoder.pos_conv_embed.conv.parametrizations.weight.original0")
        k = k.replace("encoder.pos_conv_embed.conv.weight_v", "encoder.pos_conv_embed.conv.parametrizations.weight.original1")
        out[k] = v
    missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
    missing = [m for m in missing if m != "masked_spec_embed"]        # usato solo in training (spec-augment)
    if missing:
        raise RuntimeError(f"wav2vec2: pesi mancanti {missing[:5]} ({len(missing)})")
    if "masked_spec_embed" in dict(model.named_parameters()):
        model.masked_spec_embed = torch.nn.Parameter(torch.zeros(cfg.hidden_size))
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


@torch.no_grad()
def embed_audio(model, wav: np.ndarray, fps: float, device, dtype=torch.float16, normalize: bool = True) -> torch.Tensor:
    """wav mono 16 kHz -> [int(durata*fps), 12, 768] float32 su CPU."""
    if normalize:
        wav = loudness_norm(wav)
    # do_normalize=False nel feature extractor di TencentGameMate: il segnale va dato com'e'
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None].to(device=device, dtype=dtype)
    seq_len = max(1, int(len(wav) / SR * fps))
    feats = model.feature_extractor(x).transpose(1, 2)
    feats = _linear_interpolation(feats, seq_len)
    hs, _ = model.feature_projection(feats)
    enc = model.encoder(hs, output_hidden_states=True, return_dict=True)
    emb = torch.stack(enc.hidden_states[1:], dim=1).squeeze(0)       # [12, T, 768]
    return emb.permute(1, 0, 2).float().cpu().contiguous()           # [T, 12, 768]
