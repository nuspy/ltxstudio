"""Loader delle LoRA di controllo MagicQuill V2 (formato EasyControl).

Derivato da MagicQuillV2 `src/lora_helper.py` / `src/layers_cache.py` (CC BY-NC 4.0, (c) Zichen Liu et al.; da
EasyControl). Vedi `NOTICE.md` (cartella del componente) per attribuzione e avvertenza di licenza (uso NON
commerciale).

Chiavi attese (per ogni file, rank 128, network_alpha = rank -> scala 1):
    transformer_blocks.{0..18}.attn.processor.{q,k,v,proj}_loras.0.{down,up}.weight    (blocchi doppi)
    single_transformer_blocks.{0..37}.attn.processor.{q,k,v}_loras.0.{down,up}.weight  (blocchi singoli)
Totale 19*4*2 + 38*3*2 = 380 tensori. Il loader PRETENDE questo schema: nessuna chiave orfana, nessuna mancante.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

_KEY_RE = re.compile(
    r"^(?P<kind>transformer_blocks|single_transformer_blocks)\.(?P<idx>\d+)\.attn\.processor\."
    r"(?P<proj>q|k|v|proj)_loras\.(?P<n>\d+)\.(?P<ud>down|up)\.weight$")

DOUBLE_PROJS = ("q", "k", "v", "proj")
SINGLE_PROJS = ("q", "k", "v")


class ControlLoRA:
    """Pesi LoRA di UNA modalità di controllo: {(kind, idx, proj): (down[rank,dim], up[dim,rank])}.

    kind = 'd' (blocco doppio) | 's' (blocco singolo). I tensori restano dove li ha messi il chiamante (CPU, bf16):
    `delta` li porta sul device/dtype dell'attivazione al momento dell'uso (solo nel passaggio di scrittura)."""

    def __init__(self, layers: dict, rank: int, dim: int, n_double: int, n_single: int, name: str = ""):
        self.layers = layers
        self.rank = rank
        self.dim = dim
        self.n_double = n_double
        self.n_single = n_single
        self.name = name

    @property
    def scale(self) -> float:
        # network_alpha == rank nel codice upstream (`network_alphas=ranks`) -> alpha/rank = 1
        return 1.0

    @classmethod
    def from_state_dict(cls, sd: dict, *, n_double: int = 19, n_single: int = 38, dim: int = 3072,
                        name: str = "") -> "ControlLoRA":
        layers: dict[tuple, list] = {}
        orphans: list[str] = []
        for k, v in sd.items():
            m = _KEY_RE.match(k)
            if not m:
                orphans.append(k)
                continue
            if int(m["n"]) != 0:
                raise ValueError(f"{name or 'LoRA'}: indice LoRA {m['n']} in '{k}': supportato solo 1 LoRA per file "
                                 f"(i file MagicQuill ne hanno una, indice 0)")
            kind = "d" if m["kind"] == "transformer_blocks" else "s"
            idx = int(m["idx"])
            lim = n_double if kind == "d" else n_single
            if idx >= lim or (m["proj"] == "proj" and kind == "s"):
                orphans.append(k)
                continue
            layers.setdefault((kind, idx, m["proj"]), [None, None])[0 if m["ud"] == "down" else 1] = v
        if orphans:
            raise ValueError(f"{name or 'LoRA'}: {len(orphans)} chiavi non riconosciute/orfane, es. {orphans[:3]}")
        expected = [("d", i, p) for i in range(n_double) for p in DOUBLE_PROJS] + \
                   [("s", i, p) for i in range(n_single) for p in SINGLE_PROJS]
        missing = [e for e in expected if e not in layers or layers[e][0] is None or layers[e][1] is None]
        if missing:
            raise ValueError(f"{name or 'LoRA'}: {len(missing)} coppie down/up mancanti, es. {missing[:3]}")
        rank = None
        out: dict[tuple, tuple] = {}
        for key, (down, up) in layers.items():
            r = down.shape[0]
            if tuple(down.shape) != (r, dim) or tuple(up.shape) != (dim, r):
                raise ValueError(f"{name or 'LoRA'}: shape errate per {key}: down {tuple(down.shape)} up "
                                 f"{tuple(up.shape)} (attese ({r},{dim}) e ({dim},{r}))")
            if rank is None:
                rank = r
            elif rank != r:
                raise ValueError(f"{name or 'LoRA'}: rank non uniforme ({rank} vs {r} in {key})")
            out[key] = (down, up)
        return cls(out, int(rank), dim, n_double, n_single, name)

    @classmethod
    def from_file(cls, path: str, **kw) -> "ControlLoRA":
        from safetensors.torch import load_file
        return cls.from_state_dict(load_file(path, device="cpu"), name=os.path.basename(path), **kw)

    def n_tensors(self) -> int:
        return 2 * len(self.layers)

    def delta(self, kind: str, idx: int, proj: str, x):
        """Contributo LoRA `up(down(x))` (scala 1) con i pesi portati su device/dtype di `x`."""
        import torch.nn.functional as F
        down, up = self.layers[(kind, idx, proj)]
        d = down.to(device=x.device, dtype=x.dtype)
        u = up.to(device=x.device, dtype=x.dtype)
        return F.linear(F.linear(x, d), u) * self.scale


@dataclass
class ControlSpec:
    """Una condizione di controllo: la sua LoRA, la immagine-condizione (PIL, già preprocessata), la forza (gamma =
    bias log() sull'attenzione immagine->condizione), il peso LoRA (config upstream: 1.0) e la dimensione `cond_size`."""
    lora: ControlLoRA
    image: Any                       # PIL.Image
    gamma: float = 1.0
    lora_weight: float = 1.0
    cond_size: int = 512

    @property
    def n_tokens(self) -> int:
        # upstream: w//8 * h//8 * 16 // 64  == (cond_size/16)^2  (512 -> 1024 token)
        return (self.cond_size // 16) ** 2
