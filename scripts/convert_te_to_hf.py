"""Converte i text encoder ComfyUI (file singoli) in cartelle HF caricabili da transformers,
SENZA riscaricare i pesi: usa i tuoi .safetensys locali e scarica solo config+tokenizer (pochi MB).

Le chiavi dei file ComfyUI sono già in formato HF (verificato), quindi basta affiancare il config
e il tokenizer dell'architettura standard e copiare i pesi locali come model.safetensors.

Output: <base>/diffusers/_te/{clip_l, t5xxl, qwen3_4b}/  (config.json + tokenizer + model.safetensors)
Uso: python scripts/convert_te_to_hf.py
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

BASE = Path(r"E:\AI\Models\diffusers\_te")
TE = {
    "clip_l": {
        "src": r"E:\AI\Models\TextEncoders\long_clip_l_hi_dream.safetensors",
        "repo": "openai/clip-vit-large-patch14",
        "patterns": ["*.json", "*.txt", "vocab*", "merges*", "tokenizer*", "special_tokens*"],
        "kind": "clip",
    },
    "t5xxl": {
        "src": r"E:\AI\Models\TextEncoders\t5xxl_fp16.safetensors",
        "repo": "google/t5-v1_1-xxl",
        "patterns": ["config.json", "*.model", "tokenizer*", "special_tokens*", "spiece*"],
        "kind": "t5",
    },
    "qwen3_4b": {
        "src": r"E:\AI\Models\TextEncoders\Qwen\qwen_3_4b.safetensors",
        "repo": "Qwen/Qwen3-4B",
        "patterns": ["config.json", "tokenizer*", "merges*", "vocab*", "generation_config*"],
        "kind": "qwen",
    },
}


def _pos_emb_len(src: str) -> int | None:
    try:
        with safe_open(src, "pt") as f:
            for k in f.keys():
                if k.endswith("position_embedding.weight"):
                    return f.get_slice(k).get_shape()[0]
    except Exception:                                   # noqa: BLE001
        pass
    return None


def convert(name: str, spec: dict) -> Path:
    dest = BASE / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[te] {name}: scarico config+tokenizer da {spec['repo']} (solo metadata)…", flush=True)
    snapshot_download(repo_id=spec["repo"], local_dir=str(dest), allow_patterns=spec["patterns"])
    # CLIP: estrai il text_config e adatta max_position_embeddings alla shape reale del peso
    if spec["kind"] == "clip":
        cfg = json.loads((dest / "config.json").read_text("utf-8"))
        tcfg = cfg.get("text_config", cfg)
        n = _pos_emb_len(spec["src"])
        if n:
            tcfg["max_position_embeddings"] = int(n)
        tcfg["model_type"] = "clip_text_model"
        tcfg["architectures"] = ["CLIPTextModel"]
        (dest / "config.json").write_text(json.dumps(tcfg, indent=2), "utf-8")
    print(f"[te] {name}: copio i pesi LOCALI ({Path(spec['src']).name})…", flush=True)
    out = dest / "model.safetensors"
    if spec["kind"] == "clip":
        # i file CLIP ComfyUI spesso non hanno il metadata 'format' → ri-salva per transformers
        from safetensors.torch import load_file, save_file
        save_file(load_file(spec["src"]), str(out), metadata={"format": "pt"})
    else:
        shutil.copy2(spec["src"], out)
    return dest


def main() -> int:
    for name, spec in TE.items():
        if not Path(spec["src"]).exists():
            print(f"[te] SALTO {name}: sorgente mancante {spec['src']}"); continue
        convert(name, spec)
    print("\n✔ Text encoder convertiti in", BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
