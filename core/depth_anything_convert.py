"""Depth-Anything V2 «.pth» originale (repo DepthAnything) -> cartella formato HF (transformers).

Nativo, senza ComfyUI e senza scaricare nulla: rimappa le chiavi del checkpoint originale (`pretrained.*` DINOv2 +
`depth_head.*` DPT) ai nomi di `DepthAnythingForDepthEstimation` (stessa mappa di
transformers/models/depth_anything/convert_depth_anything_to_hf.py, verificata con load strict). L'ORIGINALE NON
SI CANCELLA. Serve a `core/controlnet_preprocess.depth()`, al ControlNet depth dei motori immagine e al lab DLSS.

Uso:  python -m core.depth_anything_convert [SRC.pth] [DST_DIR]
Config e preprocessor sono quelli di `depth-anything/Depth-Anything-V2-Large-hf` (Large: out_indices 5/12/18/24).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

DEFAULT_SRC = r"E:\AI\Models\Depth\Wan2.1\depth_anything_v2_vitl.pth"
DEFAULT_DST = r"E:\AI\Models\Segmentation\hf\depth-anything-v2-large"
EXPECTED_KEYS = 407                       # tensori del .pth Large V2 (verificato 17/10/2026)

LARGE_CONFIG = {
    "architectures": ["DepthAnythingForDepthEstimation"],
    "backbone_config": {
        "architectures": ["Dinov2Model"], "hidden_size": 1024, "image_size": 518, "model_type": "dinov2",
        "num_attention_heads": 16, "num_hidden_layers": 24,
        "out_features": ["stage5", "stage12", "stage18", "stage24"], "out_indices": [5, 12, 18, 24],
        "patch_size": 14, "reshape_hidden_states": False,
        "stage_names": ["stem"] + [f"stage{i}" for i in range(1, 25)],
    },
    "fusion_hidden_size": 256, "head_hidden_size": 32, "head_in_index": -1, "initializer_range": 0.02,
    "model_type": "depth_anything", "neck_hidden_sizes": [256, 512, 1024, 1024], "patch_size": 14,
    "reassemble_factors": [4, 2, 1, 0.5], "reassemble_hidden_size": 1024, "torch_dtype": "float32",
    "use_pretrained_backbone": False,
}
PREPROCESSOR = {
    "do_normalize": True, "do_pad": False, "do_rescale": True, "do_resize": True, "ensure_multiple_of": 14,
    "image_mean": [0.485, 0.456, 0.406], "image_processor_type": "DPTImageProcessor",
    "image_std": [0.229, 0.224, 0.225], "keep_aspect_ratio": True, "resample": 3,
    "rescale_factor": 0.00392156862745098, "size": {"height": 518, "width": 518}, "size_divisor": None,
}


def rename_key(k: str) -> list[tuple[str, tuple[int, int] | None]]:
    """Chiave originale -> [(chiave HF, (indice, di 3) se è un blocco qkv da spezzare)]."""
    # ── backbone DINOv2 ──
    simple = {"pretrained.cls_token": "backbone.embeddings.cls_token",
              "pretrained.mask_token": "backbone.embeddings.mask_token",
              "pretrained.pos_embed": "backbone.embeddings.position_embeddings"}
    if k in simple:
        return [(simple[k], None)]
    m = re.fullmatch(r"pretrained\.patch_embed\.proj\.(weight|bias)", k)
    if m:
        return [(f"backbone.embeddings.patch_embeddings.projection.{m[1]}", None)]
    m = re.fullmatch(r"pretrained\.norm\.(weight|bias)", k)
    if m:
        return [(f"backbone.layernorm.{m[1]}", None)]
    m = re.fullmatch(r"pretrained\.blocks\.(\d+)\.(.+)", k)
    if m:
        n, rest = m[1], m[2]
        p = f"backbone.encoder.layer.{n}."
        mm = re.fullmatch(r"attn\.qkv\.(weight|bias)", rest)
        if mm:
            return [(p + f"attention.attention.{name}.{mm[1]}", (i, 3)) for i, name in enumerate(("query", "key", "value"))]
        for a, b in (("attn.proj.", "attention.output.dense."), ("ls1.gamma", "layer_scale1.lambda1"),
                     ("ls2.gamma", "layer_scale2.lambda1"), ("norm1.", "norm1."), ("norm2.", "norm2."),
                     ("mlp.fc1.", "mlp.fc1."), ("mlp.fc2.", "mlp.fc2.")):
            if rest.startswith(a):
                return [(p + b + rest[len(a):], None)]
    # ── neck + head DPT ──
    m = re.fullmatch(r"depth_head\.projects\.(\d+)\.(weight|bias)", k)
    if m:
        return [(f"neck.reassemble_stage.layers.{m[1]}.projection.{m[2]}", None)]
    m = re.fullmatch(r"depth_head\.resize_layers\.(\d+)\.(weight|bias)", k)
    if m:
        return [(f"neck.reassemble_stage.layers.{m[1]}.resize.{m[2]}", None)]
    m = re.fullmatch(r"depth_head\.scratch\.layer(\d)_rn\.weight", k)
    if m:
        return [(f"neck.convs.{int(m[1]) - 1}.weight", None)]
    m = re.fullmatch(r"depth_head\.scratch\.refinenet(\d)\.(.+)", k)
    if m:
        li, rest = 4 - int(m[1]), m[2]                       # refinenet4 -> fusion layer 0 ... refinenet1 -> 3
        p = f"neck.fusion_stage.layers.{li}."
        if rest.startswith("out_conv."):
            return [(p + "projection." + rest.split(".", 1)[1], None)]
        mm = re.fullmatch(r"resConfUnit([12])\.conv([12])\.(weight|bias)", rest)
        if mm:
            return [(p + f"residual_layer{mm[1]}.convolution{mm[2]}.{mm[3]}", None)]
    m = re.fullmatch(r"depth_head\.scratch\.output_conv1\.(weight|bias)", k)
    if m:
        return [(f"head.conv1.{m[1]}", None)]
    m = re.fullmatch(r"depth_head\.scratch\.output_conv2\.([02])\.(weight|bias)", k)
    if m:
        return [(f"head.conv{2 if m[1] == '0' else 3}.{m[2]}", None)]
    raise KeyError(f"chiave Depth-Anything non riconosciuta: {k}")


def convert_state_dict(sd: dict) -> dict:
    import torch
    out = {}
    for k, v in sd.items():
        for nk, part in rename_key(k):
            if part is None:
                out[nk] = v.detach().clone().contiguous()
            else:
                i, n = part
                out[nk] = v.detach().chunk(n, dim=0)[i].clone().contiguous()
    return out


def is_hf_dir(d: str | os.PathLike) -> bool:
    d = Path(d)
    return (d / "config.json").is_file() and (d / "model.safetensors").is_file() and (d / "preprocessor_config.json").is_file()


def convert(src: str = DEFAULT_SRC, dst: str = DEFAULT_DST, log=print) -> Path:
    """Scrive `dst` (cartella HF). Non cancella `src`. Salta se `dst` è già completa."""
    import torch
    from safetensors.torch import save_file
    dst_p = Path(dst)
    if is_hf_dir(dst_p):
        log(f"[depth] {dst_p} già in formato HF: niente da fare")
        return dst_p
    if not Path(src).is_file():
        raise FileNotFoundError(src)
    from core import resources
    resources.require(commit_gb=4.0, what="conversione Depth-Anything V2")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if len(sd) != EXPECTED_KEYS:
        raise ValueError(f"{src}: {len(sd)} tensori, attesi {EXPECTED_KEYS} (non è Depth-Anything V2 Large?)")
    new = convert_state_dict(sd)
    tmp = dst_p.with_name(dst_p.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    save_file(new, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    (tmp / "config.json").write_text(json.dumps(LARGE_CONFIG, indent=2), encoding="utf-8")
    (tmp / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR, indent=2), encoding="utf-8")
    _verify(tmp, sd)
    if dst_p.exists():
        dst_p.rmdir()                                         # vuota (is_hf_dir era falso): sicura da togliere
    os.replace(tmp, dst_p)
    log(f"[depth] scritto {dst_p} ({len(new)} tensori); originale {src} NON toccato")
    return dst_p


def _verify(d: Path, orig_sd: dict) -> None:
    """Load STRICT nel modello HF + confronto tensore per tensore con l'originale."""
    import torch
    from safetensors.torch import load_file
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
    cfg = DepthAnythingConfig.from_pretrained(str(d))
    with torch.device("meta"):
        model = DepthAnythingForDepthEstimation(cfg)
    new = load_file(str(d / "model.safetensors"))
    want = model.state_dict()
    missing = sorted(set(want) - set(new))
    extra = sorted(set(new) - set(want))
    if missing or extra:
        raise ValueError(f"mappa chiavi incompleta: mancanti {missing[:5]} (+{len(missing)}), extra {extra[:5]} (+{len(extra)})")
    for k, t in want.items():
        if tuple(new[k].shape) != tuple(t.shape):
            raise ValueError(f"shape {k}: {tuple(new[k].shape)} != {tuple(t.shape)}")
    # ogni valore dell'originale ritrovato (qkv ricomposto) nei tensori nuovi
    for k, v in orig_sd.items():
        parts = rename_key(k)
        got = torch.cat([new[nk] for nk, _ in parts], dim=0) if parts[0][1] else new[parts[0][0]]
        if not torch.equal(got, v):
            raise ValueError(f"valori diversi dopo la conversione: {k}")


if __name__ == "__main__":
    a = sys.argv[1:]
    convert(a[0] if a else DEFAULT_SRC, a[1] if len(a) > 1 else DEFAULT_DST)
