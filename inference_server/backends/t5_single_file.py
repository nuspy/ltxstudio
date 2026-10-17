"""T5-XXL per FLUX.1 da SINGLE-FILE (fp8 e4m3 «t5xxl_enconly» o fp16 «t5xxl_fp16»), senza cartella HF con i pesi.

Perché: `T5EncoderModel.from_pretrained(_te/t5xxl)` vuole la cartella con model.safetensors fp16 (9.1 GB). I pesi
fp8 (4.9 GB, solo-encoder, nomi HF) bastano: qui si leggono TENSORE PER TENSORE (niente mmap, niente picco di RAM:
vedi [[server-mmap-uvicorn-crash]]), le matrici lineari restano in fp8 (≈4.4 GB residenti invece di 9.1) e si
promuovono a bf16 solo dentro `forward` (layerwise upcast), il resto (embedding, norme, bias relativo) va in bf16.

Trappola risolta: transformers (T5DenseGatedActDense) fa `hidden_states.to(self.wo.weight.dtype)`: con un
`wo.weight` fp8 le attivazioni verrebbero castate a fp8. `Fp8Linear` espone `weight = None` (non è un Tensor →
il controllo viene saltato) e tiene i pesi nel buffer `w8`.

Scelta della sorgente (`pick_weights`): 1) env NATIVE_T5_WEIGHTS (percorso esplicito, per i test), 2) la cartella
HF `_te/t5xxl/model.safetensors` (comportamento storico, INVARIATO se c'è), 3) solo se manca, il single-file in
TextEncoders (prima fp8 enconly, poi fp16). Nessuna scelta automatica fra due pesi presenti: vince sempre il 2).
Config e tokenizer restano quelli della cartella `_te/t5xxl` (pochi MB). La chiave dello STORE resta «t5xxl»
(la sorgente è deterministica per processo; NATIVE_T5_WEIGHTS è solo per i test).
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

TEXT_ENCODERS_DIR = r"E:\AI\Models\TextEncoders"
SINGLE_FILES = ("t5xxl_enconly.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors")      # ordine di ripiego: fp8 prima
_DT = {"F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F16": "float16", "BF16": "bfloat16",
       "F32": "float32"}
_FP8 = ("float8_e4m3fn", "float8_e5m2")
# nomi dei Linear di T5 da tenere in fp8 (matrici grandi); il resto → dtype di calcolo
_LINEAR_SUFFIX = (".SelfAttention.q.weight", ".SelfAttention.k.weight", ".SelfAttention.v.weight",
                  ".SelfAttention.o.weight", ".DenseReluDense.wi_0.weight", ".DenseReluDense.wi_1.weight",
                  ".DenseReluDense.wo.weight")


def pick_weights(te_dir: str, text_encoders_dir: str = TEXT_ENCODERS_DIR) -> tuple[str, str]:
    """-> (kind, path): kind 'folder' (from_pretrained storico) | 'single' (questo loader)."""
    env = os.environ.get("NATIVE_T5_WEIGHTS", "").strip()
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"NATIVE_T5_WEIGHTS non esiste: {env}")
        return "single", env
    folder = os.path.join(te_dir, "t5xxl", "model.safetensors")
    if os.path.isfile(folder):
        return "folder", os.path.join(te_dir, "t5xxl")
    for name in SINGLE_FILES:
        p = os.path.join(text_encoders_dir, name)
        if os.path.isfile(p):
            return "single", p
    return "folder", os.path.join(te_dir, "t5xxl")          # nessuno: from_pretrained darà l'errore storico


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _fp8_linear_cls():
    import torch
    import torch.nn.functional as F

    class Fp8Linear(torch.nn.Module):
        """Linear senza bias con pesi fp8 in buffer; upcast al dtype delle attivazioni dentro forward."""

        def __init__(self, w8):
            super().__init__()
            self.register_parameter("weight", None)          # vedi docstring del modulo
            self.register_buffer("w8", w8, persistent=False)
            self.out_features, self.in_features = w8.shape

        def forward(self, x):
            return F.linear(x, self.w8.to(x.dtype))

    return Fp8Linear


def load_t5_single_file(path: str, config_dir: str, dtype=None, keep_fp8: bool = True):
    """Costruisce T5EncoderModel da `config_dir/config.json` + pesi single-file. Ritorna il modello (CPU, eval)."""
    import torch
    from transformers import T5Config, T5EncoderModel
    dtype = dtype or torch.bfloat16
    header, base = read_header(path)
    header.pop("__metadata__", None)
    cfg = T5Config.from_pretrained(config_dir)
    with torch.device("meta"):
        model = T5EncoderModel(cfg)
    want = set(model.state_dict())
    sd, fp8_keys = {}, set()
    with open(path, "rb") as f:
        for k, info in header.items():
            if k not in want:
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            src = getattr(torch, _DT[info["dtype"]])
            t = torch.frombuffer(buf, dtype=torch.uint8).view(src).reshape(info["shape"]) if buf else \
                torch.empty(info["shape"], dtype=src)
            if keep_fp8 and src in (torch.float8_e4m3fn, torch.float8_e5m2) and k.endswith(_LINEAR_SUFFIX):
                sd[k] = t.clone()
                fp8_keys.add(k)
            else:
                sd[k] = t.to(dtype)
    missing = sorted(want - set(sd))
    if missing:
        raise ValueError(f"{os.path.basename(path)}: mancano {len(missing)} tensori T5 (es. {missing[:3]})")
    # i Linear fp8 NON passano da load_state_dict (dtype ≠ parametro): li si sostituisce dopo
    plain = {k: v for k, v in sd.items() if k not in fp8_keys}
    model.load_state_dict(plain, strict=False, assign=True)
    Fp8Linear = _fp8_linear_cls()
    for k in sorted(fp8_keys):
        mod_name = k[: -len(".weight")]
        parent_name, _, leaf = mod_name.rpartition(".")
        setattr(model.get_submodule(parent_name), leaf, Fp8Linear(sd[k]))
    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    if leftover:
        raise ValueError(f"parametri T5 rimasti su meta: {leftover[:3]}")
    return model.eval().requires_grad_(False)


def make_loader(te_dir: str, dtype=None):
    """Callable per `STORE.acquire("t5xxl", ...)`: dalla sorgente scelta da pick_weights."""
    import torch
    dtype = dtype or torch.bfloat16
    kind, path = pick_weights(te_dir)
    cfg_dir = os.path.join(te_dir, "t5xxl")
    if kind == "folder":
        from transformers import T5EncoderModel
        return lambda: T5EncoderModel.from_pretrained(path, torch_dtype=dtype)
    print(f"[t5] single-file {Path(path).name} (fp8 layerwise) — config/tokenizer da {cfg_dir}", flush=True)
    return lambda: load_t5_single_file(path, cfg_dir, dtype)

