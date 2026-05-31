"""Ri-pacchettizza il gemma-3-12B abliterated (huihui-ai) nel formato encoder Comfy-Org per LTX.

Prende i 5 shard HF bf16 da _abliterated_staging/, li unisce in un singolo safetensors e ci
inietta il tensore `spiece_model` (tokenizer SentencePiece) estratto dal file Comfy esistente,
così la struttura combacia con `gemma_3_12B_it.safetensors` (model.* + vision_model.* +
multi_modal_projector.* + spiece_model). Output: gemma_3_12B_it_abliterated.safetensors (bf16).

Eseguire DOPO il completamento del download dei 5 shard.
"""
from pathlib import Path
import json, struct
import torch
from safetensors.torch import load_file, save_file

TE = Path(r"E:\AI\Models\TextEncoders")
STAGING = TE / "_abliterated_staging"
REF = TE / "gemma_3_12B_it.safetensors"            # file Comfy esistente (per spiece_model)
OUT = TE / "gemma_3_12B_it_abliterated.safetensors"


def read_spiece(path: Path):
    """Estrae il solo tensore spiece_model dal file di riferimento senza caricare tutto."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        info = hdr["spiece_model"]
        s, e = info["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
    # U8 1-D
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    shape = info["shape"]
    return t.reshape(shape)


def rename(k):
    """huihui (transformers gemma3) → formato Comfy:
    language_model.model.X → model.X ; vision_tower.vision_model.X → vision_model.X ;
    multi_modal_projector.* e spiece_model invariati."""
    if k.startswith("language_model."):
        return k[len("language_model."):]
    if k.startswith("vision_tower."):
        return k[len("vision_tower."):]
    return k


def main():
    shards = sorted(STAGING.glob("model-*-of-*.safetensors"))
    assert shards, f"Nessuno shard in {STAGING}"
    print(f"Shard trovati: {len(shards)}")
    merged = {}
    for sh in shards:
        d = load_file(str(sh))
        for k, v in d.items():
            merged[rename(k)] = v
        print(f"  {sh.name}: +{len(d)} tensori (tot {len(merged)})")
    # inietta il tokenizer SentencePiece dal file Comfy
    merged["spiece_model"] = read_spiece(REF)
    print(f"spiece_model iniettato: {merged['spiece_model'].shape} {merged['spiece_model'].dtype}")
    # diagnostica prefissi
    from collections import Counter
    c = Counter(k.split(".")[0] for k in merged)
    print("prefissi:", dict(c))
    save_file(merged, str(OUT), metadata={"format": "pt"})
    gb = OUT.stat().st_size / 1e9
    print(f"\nOK → {OUT}  ({gb:.2f} GB, {len(merged)} tensori)")


if __name__ == "__main__":
    main()
