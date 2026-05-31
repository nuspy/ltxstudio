"""Rimuove le chiavi '.comfy_quant' (metadati di quantizzazione inerti per il ComfyUI in uso)
dal modello LTX-2.3 distilled-1.1 fp8_scaled, eliminando il warning 'unet unexpected'.

I pesi fp8 (.weight) e le loro scale (.weight_scale) restano intatti → dequantizzazione identica,
ZERO impatto sulla qualità. Scrive su .tmp e sostituisce atomicamente l'originale.
"""
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import json, struct, os

SRC = Path(r"E:\AI\Models\diffusion_models\LTXVideo\v2\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors")
TMP = SRC.with_suffix(".clean.tmp")


def main():
    tensors = {}
    dropped = 0
    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            if k.endswith(".comfy_quant"):
                dropped += 1
                continue
            tensors[k] = f.get_tensor(k)
    print(f"tensori tenuti: {len(tensors)} | comfy_quant rimossi: {dropped}")
    save_file(tensors, str(TMP), metadata=meta)
    # verifica: 0 comfy_quant, weight_scale ancora presenti
    with open(TMP, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    cq = [k for k in hdr if k.endswith(".comfy_quant")]
    ws = [k for k in hdr if k.endswith(".weight_scale")]
    assert not cq, f"comfy_quant ancora presenti: {len(cq)}"
    assert ws, "ERRORE: weight_scale spariti! abort"
    print(f"verifica OK: 0 comfy_quant, {len(ws)} weight_scale intatti, {len(hdr)-1} tensori")
    os.replace(str(TMP), str(SRC))
    print(f"sostituito → {SRC} ({SRC.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
