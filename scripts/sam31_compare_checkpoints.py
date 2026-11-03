"""Confronto tensore per tensore fra il checkpoint di Meta (sam3.1_multiplex.pt) e la copia Comfy-Org (sam3.1_multiplex_fp16.safetensors). Senza codice di Meta:
il .pt si apre con torch.load(weights_only=True, mmap=True), il file Comfy con safetensors. Solo CPU. Stampa il report e (con --json) lo salva.

  E:/AI/Tools/diffusers_next_venv/Scripts/python.exe -I scripts/sam31_compare_checkpoints.py [--meta FILE] [--comfy FILE] [--json out.json]
"""
import argparse
import json

META = r"E:\AI\Models\Segmentation\sam3.1\sam3.1_multiplex.pt"
COMFY = r"E:\AI\Models\Segmentation\sam3.1_multiplex\sam3.1_multiplex_fp16.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=META)
    ap.add_argument("--comfy", default=COMFY)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import torch
    from safetensors import safe_open
    ck = torch.load(a.meta, map_location="cpu", weights_only=True, mmap=True)
    meta = ck["model"] if isinstance(ck.get("model"), dict) else ck
    mk = {k for k, v in meta.items() if hasattr(v, "shape")}
    with safe_open(a.comfy, framework="pt", device="cpu") as f:
        ck_keys = set(f.keys())
        only_meta = sorted(mk - ck_keys)
        only_comfy = sorted(ck_keys - mk)
        both = sorted(mk & ck_keys)
        shape_diff, dtype_meta, dtype_comfy = [], {}, {}
        worst = []                      # (max_abs, mean_abs, rel_max, key)
        sum_abs, n_el, global_max, n_exact = 0.0, 0, 0.0, 0
        for k in both:
            tm, tc = meta[k], f.get_tensor(k)
            dtype_meta[str(tm.dtype)] = dtype_meta.get(str(tm.dtype), 0) + 1
            dtype_comfy[str(tc.dtype)] = dtype_comfy.get(str(tc.dtype), 0) + 1
            if tuple(tm.shape) != tuple(tc.shape):
                shape_diff.append((k, tuple(tm.shape), tuple(tc.shape)))
                continue
            if tm.is_complex():
                continue
            d = (tm.float() - tc.float()).abs()
            mx = float(d.max()) if d.numel() else 0.0
            me = float(d.mean()) if d.numel() else 0.0
            scale = float(tm.float().abs().max()) or 1.0
            sum_abs += float(d.sum()); n_el += d.numel(); global_max = max(global_max, mx)
            n_exact += int(mx == 0.0)
            worst.append((mx, me, mx / scale, k))
    worst.sort(reverse=True)
    rep = {
        "chiavi_meta": len(mk), "chiavi_comfy": len(ck_keys), "in_comune": len(both),
        "solo_meta": only_meta, "solo_comfy": only_comfy, "forme_diverse": shape_diff,
        "dtype_meta": dtype_meta, "dtype_comfy": dtype_comfy,
        "errore_massimo_assoluto": global_max, "errore_medio_assoluto": (sum_abs / n_el) if n_el else 0.0,
        "tensori_identici_al_bit": n_exact, "tensori_confrontati": len(worst),
        "peggiori_10": [{"chiave": k, "max": mx, "medio": me, "max_relativo_al_massimo_del_tensore": rel} for mx, me, rel, k in worst[:10]],
        "errore_relativo_massimo": max((w[2] for w in worst), default=0.0),
    }
    print(json.dumps(rep, indent=1, ensure_ascii=False)[:6000])
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
