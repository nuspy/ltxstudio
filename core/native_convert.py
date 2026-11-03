"""Conversione UNA TANTUM dei pesi single-file in formato ComfyUI → file NATIVI (layout diffusers/transformers).

Regola del progetto: ComfyUI non esiste più nell'app. I single-file piccoli pubblicati da Comfy-Org & co.
(fp8/nvfp4, stanno in 32 GB) si possono scaricare, ma appena arrivati vengono convertiti qui:
  1. il file nativo si scrive accanto (`<nome>.part`) UN TENSORE ALLA VOLTA (niente mmap, commit minimo);
  2. si verifica (chiavi, forme, dtype e confronto bit-a-bit di un campione di tensori);
  3. si rinomina al nome definitivo e SOLO ALLORA si cancella il file ComfyUI.
Se il nativo esiste già non si riconverte (e il sorgente residuo si cancella).

Formato nativo («ltx-native-1», metadata `ltx_native`):
  · nomi dei moduli diffusers/transformers; proiezioni FUSE dove il file quantizzato le ha fuse
    (`attn.to_qkv`, `ff.gate_up`, `img_mlp.gate_up`): i worker le montano così (dividere i blocchi nvfp4
    richiederebbe di ri-impacchettare le scale swizzled);
  · quantizzati riconosciuti dal dtype, un solo schema per tutti:
      nvfp4  → .weight U8 [out,in/2] + .weight_scale F8 [out,in/16] + .weight_scale_2 F32 [] (+ .pre_quant_scale AWQ)
      fp8    → .weight F8_E4M3 [out,in] + .weight_scale F32 [out,1]   (la scala per-tensore diventa per-riga)
      int8   → .weight I8 + .weight_scale F32 [out,1] (+ .convrot_group I32: il worker ruota l'ingresso)
      embedding int8+ConvRot → .weight BF16 già de-ruotato
  · niente marcatori `comfy_quant` / `input_scale`.

Si usa dal downloader (voci di catalogo col campo `convert`), dal primo avvio per i file già su disco, e da riga
di comando:  python -m core.native_convert <converter> <sorgente> <destinazione>
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
from pathlib import Path

FORMAT = "ltx-native-1"
# marcatori solo-ComfyUI, niente pesi: schema di quantizzazione (letto prima), scala d'ingresso, e il suggerimento
# di kernel d'attenzione `comfy_attention.config` ({"attention": "comfy_kitchen_int8"}, es. Turbo INT8 di Comfy-Org)
_DROP_SUFFIX = (".comfy_quant", ".input_scale", ".comfy_attention.config")


def _torch():
    import torch
    return torch


def _dtypes():
    t = _torch()
    return {"F64": t.float64, "F32": t.float32, "F16": t.float16, "BF16": t.bfloat16,
            "F8_E4M3": t.float8_e4m3fn, "F8_E5M2": t.float8_e5m2, "I64": t.int64, "I32": t.int32,
            "I16": t.int16, "I8": t.int8, "U8": t.uint8, "BOOL": t.bool}


def read_header(path: str | os.PathLike) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def is_complete(path: str | os.PathLike) -> bool:
    """True se il safetensors è INTERO: la dimensione su disco copre l'ultimo tensore dichiarato
    nell'header. Un file troncato (download interrotto, fallback single-stream a metà) NON va convertito:
    la conversione esploderebbe a metà («shape … is invalid for input of size …»)."""
    try:
        h, base = read_header(path)
        end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
        return os.path.getsize(path) >= base + end
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def is_native(path: str | os.PathLike) -> bool:
    try:
        h, _ = read_header(path)
    except (OSError, ValueError, struct.error):
        return False
    return (h.get("__metadata__") or {}).get("ltx_native") == FORMAT


# ── ConvRot (Hadamard a gruppi, formato «int8_convrot») ────────────────────────────────────────
def hadamard(size: int, device=None, dtype=None):
    """Hadamard REGOLARE normalizzato (potenze di 4), simmetrico e ortogonale → auto-inverso.
    Identico a quello dei kernel convrot (verificato sul codice diffusers-modular MiniMaxH3ConvRotLinear)."""
    torch = _torch()
    seed = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                        dtype=dtype or torch.float32, device=device)
    m, w = seed, 4
    while w < size:
        m, w = torch.kron(m, seed), w * 4
    if w != size:
        raise ValueError(f"gruppo ConvRot non potenza di 4: {size}")
    return m / size ** 0.5


def rotate(t, group: int):
    """t @ blockdiag(H, …, H) sull'ultima dimensione."""
    shp = t.shape
    return (t.reshape(-1, shp[-1] // group, group) @ hadamard(group, t.device, t.dtype)).reshape(shp)


def _module_of(key: str) -> str:
    for suf in (".weight_scale_2", ".weight_scale", ".pre_quant_scale", ".weight", ".comfy_quant", ".input_scale",
                ".weight_s_rel", ".weight_s_channel", ".weight_codebook", ".weight_correction"):
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


W4A8_FORMATS = ("asym_w4a8_int8", "w6a8_int8")       # comfy-kitchen ≥ 0.2.37 (DiT H3 DaSiWa int4 / int6)


# ── regole comuni ──────────────────────────────────────────────────────────────────────────────
def _quant_common(key: str, info: dict, header: dict):
    """Regole condivise dai quantizzati ComfyUI. Ritorna None (nessuna regola) o [(chiave, fn)]."""
    torch = _torch()
    if key.endswith(_DROP_SUFFIX):
        return []
    mod = _module_of(key)
    conf = header.get("__qconf__", {}).get(mod, {})
    if conf.get("format") in W4A8_FORMATS:
        # W4A8/W6A8 comfy-kitchen: tutto invariato (kernel fuso nel worker) + marcatore [gruppo, gruppo ConvRot]
        if key == mod + ".weight":
            gs, cr = int(conf.get("group_size", 16)), int(conf.get("convrot_groupsize", 256))
            return [(key, lambda t: t),
                    (mod + ".w4a8", lambda t, _v=(gs, cr): torch.tensor(_v, dtype=torch.int32))]
        return [(key, lambda t: t)]
    if conf.get("convrot"):
        g = int(conf.get("convrot_groupsize", 256))
        wi = header.get(mod + ".weight", {})
        if "embed" in mod.rsplit(".", 1)[-1]:
            # embedding ruotato: lo porto in bf16 DE-RUOTATO (E = (q·s) @ H) e butto la scala
            if key == mod + ".weight_scale":
                return []
            if key == mod + ".weight":
                get = header["__ctx__"]["get"]

                def fn(t, _m=mod, _g=g):
                    s = get(_m + ".weight_scale")
                    if t.device.type == "meta":
                        return torch.empty(t.shape, dtype=torch.bfloat16, device="meta")
                    out = torch.empty(t.shape, dtype=torch.bfloat16)
                    for a in range(0, t.shape[0], 8192):      # a blocchi: RAM limitata
                        blk = t[a:a + 8192].float() * s[a:a + 8192].float()
                        out[a:a + 8192] = rotate(blk, _g).to(torch.bfloat16)
                    return out
                return [(key, fn)]
        elif key == mod + ".weight" and wi.get("dtype") == "I8":
            # Linear int8 ruotato: resta int8, il worker ruota l'ingresso (marcatore .convrot_group)
            return [(key, lambda t: t), (mod + ".convrot_group", lambda t, _g=g: torch.tensor(_g, dtype=torch.int32))]
    if key.endswith(".weight_scale") and info["shape"] in ([], [1]) and info["dtype"] == "F32":
        w = header.get(key[: -len("_scale")])            # <m>.weight
        if w and w["dtype"] in ("F8_E4M3", "F8_E5M2"):
            out = w["shape"][0]
            return [(key, lambda t, _o=out: t.float().reshape(1, 1).expand(_o, 1).contiguous())]
    if key.endswith(".weight_scale") and info["dtype"] == "F32" and len(info["shape"]) == 2:
        return [(key, lambda t: t.float())]
    return None


def _renamer(rename, *, extra=None, fp32=()):
    """Converter generico: regole comuni + rinomina chiave per chiave (+ regole extra, + cast a fp32)."""
    def plan(key, info, header):
        if extra is not None:
            r = extra(key, info, header)
            if r is not None:
                return r
        r = _quant_common(key, info, header)
        if r is not None:
            return [(rename(k), fn) for k, fn in r if rename(k) is not None]
        nk = rename(key)
        if nk is None:
            return []
        if any(p in nk for p in fp32):
            return [(nk, lambda t: t.float())]
        return [(nk, lambda t: t)]
    return plan


# ── converter registrati ───────────────────────────────────────────────────────────────────────
def _workers_path():
    wp = str(Path(__file__).resolve().parents[1] / "workers")
    if wp not in sys.path:
        sys.path.insert(0, wp)


def _mmh3_dit():
    _workers_path()
    import minimax_h3_convert as M
    return _renamer(lambda k: None if k in M.DIT_SKIP else M.dit_key(k), fp32=("folded_bias",))


def _mmh3_te():
    _workers_path()
    import minimax_h3_convert as M
    return _renamer(M.te_key)


def _mmh3_vae():
    _workers_path()
    import minimax_h3_convert as M
    torch = _torch()

    def plan(key, info, header):
        out = []
        for nk, sl in M.vae_video_key(key, info["shape"]):
            if isinstance(sl, tuple) and sl[0] == "head3":
                def fn(t, _i=sl[1]):
                    heads = t.shape[0] // (3 * M.VAE_HEAD_DIM)
                    return t.reshape(heads, 3, M.VAE_HEAD_DIM, *t.shape[1:])[:, _i].reshape(-1, *t.shape[1:]).contiguous()
            elif sl == "swap":
                def fn(t):
                    h = t.shape[0] // 2
                    return torch.cat([t[h:], t[:h]], 0)
            else:
                def fn(t):
                    return t
            out.append((nk, fn))
        return out
    return plan


def _mmh3_lora_rename():
    """Rinomina chiave per chiave delle LoRA H3 (ComfyUI, kohya, diffusers) → nomi del DiT nativo montato."""
    _workers_path()
    import minimax_h3_convert as M
    kohya = re.compile(r"^lora_unet_(token_refiner_)?blocks_(\d+)_(attn|mlp)_([a-z0-9_]+?)\.(.+)$")

    def rename(k):
        m = kohya.match(k)                               # kohya (lynaNSFW, AfterMidnight…): lora_unet_blocks_N_attn_qkv_proj
        if m:
            k = f"{'token_refiner.' if m.group(1) else ''}blocks.{m.group(2)}.{m.group(3)}.{m.group(4)}.{m.group(5)}"
        for pre in ("diffusion_model.", "model.diffusion_model.", "transformer."):
            if k.startswith(pre):
                k = k[len(pre):]
        return M.dit_key(k)
    return rename


def _mmh3_lora():
    """LoRA H3 in nomi ComfyUI (diffusion_model.blocks.N.attn.qkv_proj.lora_A…) o kohya
    (lora_unet_blocks_N_attn_qkv_proj.lora_down + alpha) → nomi nativi del modello montato
    (transformer_blocks.N.attn.to_qkv…, ff.gate_up/ff.out): stesse matrici, qkv e gate/up già fusi.

    Alcuni file portano PIÙ LoRA diverse sullo stesso modulo sotto convenzioni diverse (es. «Nipple Sucking»
    Civitai 2985148: tre LoRA scorrelate in kohya, diffusion_model. e nomi nudi). ComfyUI ne applica UNA per
    modulo, l'ultima della sua mappa chiavi (comfy/lora.py: lora_unet_ < diffusion_model. < nomi nudi per
    MiniMaxH3): stessa precedenza qui, così l'effetto è quello delle immagini d'esempio."""
    rename = _mmh3_lora_rename()

    def prio(k):
        if k.startswith("lora_unet_"):
            return 0
        return 1 if k.startswith(("diffusion_model.", "model.diffusion_model.", "transformer.")) else 2

    def extra(key, info, header):
        pick = header.get("__mmh3_pick__")
        if pick is None:
            pick = {}
            for k in header:
                if k.startswith("__"):
                    continue
                t = rename(k)
                if t is not None and (t not in pick or prio(k) > prio(pick[t])):
                    pick[t] = k
            header["__mmh3_pick__"] = pick
        t = rename(key)
        return [] if t is not None and pick.get(t) != key else None
    return _renamer(rename, extra=extra)


def _mmh3_lora_diffusers():
    """LoRA H3 in formato diffusers/peft (lightx2v «minimax-h3-diffusers»: Turbo ref2v 4 passi v1.0 / 8 passi v1.1 768p) → nomi nativi del DiT montato.

    Chiavi sorgente: `transformer_blocks.N.attn.{to_q,to_k,to_v,to_out.0}` e `ff.net.0.proj` / `ff.net.2` (+ `token_refiner.refiner_blocks.N.*`), con
    `lora_A.default.weight` / `lora_B.default.weight` e SENZA chiavi alpha (alpha nei metadati del file, rank dalle forme: scala = alpha/rank).
    Nativo: q, k, v FUSI in `to_qkv` (A = A_q;A_k;A_v impilate sul rank, B = blocco-diagonale di B_q,B_k,B_v: stesso delta di ogni proiezione),
    `ff.net.0.proj` → `ff.gate_up`, `ff.net.2` → `ff.out`. La scala alpha/rank e COTTA in B (il nativo senza chiavi alpha applica scala 1; con alpha 8 e rank
    128 e 1/16, una potenza di 2: nessuna perdita in bf16). Non ci sono coppie AdaLN nel file (la LoRA non le ha)."""
    torch = _torch()
    pat = re.compile(r"^((?:transformer_blocks|token_refiner\.refiner_blocks)\.\d+)\.(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))\.lora_([AB])\.default\.weight$")

    def plan(key, info, header):
        m = pat.match(key)
        if m is None:
            return []                                   # tutto cio che non e una coppia LoRA del DiT (non ce ne sono) non entra nel nativo
        blk, mod, which = m.group(1), m.group(2), m.group(5)
        meta = header.get("__srcmeta__") or {}
        ctx = header.get("__ctx__", {})
        alpha = float(meta.get("alpha") or 0)

        def scale_of(a_key):
            r = header[a_key]["shape"][0]
            return (alpha / r) if alpha else 1.0

        if mod in ("attn.to_k", "attn.to_v"):
            return []                                   # consumate insieme a quella di to_q
        if mod == "attn.to_q":
            ks = {n: f"{blk}.attn.{n}.lora_{which}.default.weight" for n in ("to_q", "to_k", "to_v")}
            a_keys = {n: f"{blk}.attn.{n}.lora_A.default.weight" for n in ("to_q", "to_k", "to_v")}
            if which == "A":
                def fuse_a(t, _ks=ks):
                    if t.device.type == "meta":
                        return torch.empty((3 * t.shape[0], t.shape[1]), dtype=t.dtype, device="meta")
                    return torch.cat([t, ctx["get"](_ks["to_k"]), ctx["get"](_ks["to_v"])], 0).contiguous()
                return [(f"{blk}.attn.to_qkv.lora_A.weight", fuse_a)]

            def fuse_b(t, _ks=ks, _ak=a_keys):
                if t.device.type == "meta":
                    return torch.empty((3 * t.shape[0], 3 * t.shape[1]), dtype=t.dtype, device="meta")
                parts = [t, ctx["get"](_ks["to_k"]), ctx["get"](_ks["to_v"])]
                out = torch.zeros((sum(p.shape[0] for p in parts), sum(p.shape[1] for p in parts)), dtype=t.dtype)
                r0 = c0 = 0
                for p in parts:
                    out[r0:r0 + p.shape[0], c0:c0 + p.shape[1]] = (p.float() * scale_of(_ak["to_q"])).to(t.dtype)
                    r0 += p.shape[0]; c0 += p.shape[1]
                return out
            return [(f"{blk}.attn.to_qkv.lora_B.weight", fuse_b)]
        target = {"attn.to_out.0": "attn.to_out.0", "ff.net.0.proj": "ff.gate_up", "ff.net.2": "ff.out"}[mod]
        nk = f"{blk}.{target}.lora_{which}.weight"
        if which == "A":
            return [(nk, lambda t: t)]
        a_key = key.replace("lora_B", "lora_A")
        sc = scale_of(a_key)
        return [(nk, lambda t, _s=sc: t if _s == 1.0 else (t.float() * _s).to(t.dtype))]
    return plan


def _mmh3_lora_guide():
    """LoRA H3 che porta nell'intestazione la geometria della guida (h3upscale: reference_downscale_factor,
    minimax_h3_guide_*). Stessa conversione di _mmh3_lora; in più il metadata del sorgente SOPRAVVIVE: senza,
    dopo la conversione (che cancella l'originale) la LoRA non si riconoscerebbe più come LoRA di guida."""
    plan = _mmh3_lora()
    plan.keep_meta = True
    return plan


def _krea2_dit():
    """Krea 2 single-file ComfyUI (es. Krea Noct V4 int8 ConvRot) → nomi Krea2Transformer2DModel; i Linear int8
    restano compressi (+ .convrot_group), `blocks.N.mod.lin` piatto → scale_shift_table [6, dim]."""
    _workers_path()
    import krea2_custom as K

    def extra(key, info, header):
        if K.is_block_modulation(key):
            return [(K.dit_key(key), lambda t: t.reshape(6, -1))]
        return None
    return _renamer(K.dit_key, extra=extra)


def _krea2_te():
    """TE Krea 2 (Qwen3VLModel 4B) fp8 «scaled» ComfyUI → nomi transformers del text_encoder dello snapshot: i Linear
    restano fp8 (scala per-tensore espansa per riga, _qlinear.RowFp8Linear), l'embedding torna bf16 (×scala)."""
    torch = _torch()
    emb = "model.language_model.embed_tokens"

    def rename(k):
        return k[len("model."):] if k.startswith("model.") else k

    def extra(key, info, header):
        if key == emb + ".weight_scale":
            return []
        if key == emb + ".weight" and info["dtype"].startswith("F8"):
            get = header["__ctx__"]["get"]

            def fn(t):
                if t.device.type == "meta":
                    return torch.empty(t.shape, dtype=torch.bfloat16, device="meta")
                sc = get(emb + ".weight_scale").float().reshape(())
                out = torch.empty(t.shape, dtype=torch.bfloat16)
                for a in range(0, t.shape[0], 16384):           # a blocchi: RAM limitata
                    out[a:a + 16384] = (t[a:a + 16384].float() * sc).to(torch.bfloat16)
                return out
            return [(rename(key), fn)]
        return None
    return _renamer(rename, extra=extra)


def _ltx25_te_assets() -> tuple[dict, dict]:
    """(metadata, {nome: tensore uint8}) che ltx-core esige in un TE Gemma 4 da file singolo: `gemma_config` e i
    tensori `hf_asset__*`. Dal TE ufficiale bf16 se è su disco, altrimenti solo quei pochi MB via Range."""
    torch = _torch()
    try:
        from core.model_paths import p as _mp
        off = _mp("ltx25-te-bf16", "")
    except Exception:                                        # noqa: BLE001
        off = ""
    if off and os.path.isfile(off) and is_complete(off):
        h, base = read_header(off)
        md = dict(h.get("__metadata__") or {})
        out = {}
        with open(off, "rb") as f:
            for k, info in h.items():
                if k.startswith("hf_asset__"):
                    a, b = info["data_offsets"]
                    f.seek(base + a)
                    out[k] = torch.frombuffer(bytearray(f.read(b - a)), dtype=torch.uint8).clone()
        return md, out
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import ltx25_te_fix_assets as A
    md, out = A.fetch_assets(A.SOURCE_URL)
    return dict(md), {k: v for k, v in out.items() if k.startswith("hf_asset__")}


def _ltx25_te():
    """TE LTX-2.5 (Gemma 4 + proiezioni) int8 per-tensore ComfyUI (es. uncensored v1.1) → stesso file con gli int8
    INTATTI (13 GB invece dei 24.5 del bf16) e l'innesto di `gemma_config` e dei tensori `hf_asset__*` che il loader
    Gemma di ltx-core esige in un TE single-file. Il fold int8 → bf16 lo fa ltx-core al caricamento
    (with_int8_fold sul text encoder, repo LTX-2.5)."""
    plan = _renamer(lambda k: k)
    plan.extras = _ltx25_te_assets                          # vedi convert(): tensori + metadata in coda
    return plan


def _ltx25_dit():
    """Transformer LTX-2.5 single-file int8_tensorwise SENZA prefisso (es. ChrisColeTech uncensored v1.1) → nomi
    `model.diffusion_model.*` che il loader di ltx-core si aspetta. Gli int8 restano tali (scala per-tensore): il
    fold a bf16→fp8 lo fa ltx-core al caricamento (fp8_cast). Il metadata (config del modello) si conserva."""
    pre = "model.diffusion_model."
    plan = _renamer(lambda k: k if k.startswith(pre) else pre + k)
    plan.keep_meta = True
    return plan


# ── dequant generico ComfyUI → bf16 standard (ex scripts/dequant_comfy_worker.py, che girava nel venv ComfyUI) ──
_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _from_blocked(stored, rows: int, cols: int):
    """Inversa del layout «to_blocked» di cublas per le scale nvfp4 (righe paddate a 128, colonne a 4)."""
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    x = stored.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3)
    return x.reshape(nrb * 128, ncb * 4)[:rows, :cols]


def dequant_nvfp4(packed, block_scale, scale2):
    """U8 [out, in/2] (primo valore nel nibble ALTO) + scale e4m3 swizzled [out, in/16] + scala globale → bf16."""
    torch = _torch()
    out_f, in_f = packed.shape[0], packed.shape[1] * 2
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32)
    idx = packed.to(torch.int32)
    w = torch.stack([lut[(idx >> 4) & 0xF], lut[idx & 0xF]], dim=-1).reshape(out_f, in_f)
    bs = _from_blocked(block_scale, out_f, in_f // 16).float() * scale2.float().reshape(())
    return (w.reshape(out_f, in_f // 16, 16) * bs.unsqueeze(-1)).reshape(out_f, in_f).to(torch.bfloat16)


def _remap_gemma(k: str) -> str:
    """Naming Gemma 3 ComfyUI → quello di ltx-core/transformers (come il vecchio worker --remap gemma)."""
    if k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return "model.model.language_model." + k[len("model."):]
    if k.startswith("vision_model."):
        return "model.model.vision_tower." + k
    return "model.model." + k


def _comfy_bf16(remap: str = "none"):
    """Single-file quantizzato ComfyUI (comfy_quant fp8 / nvfp4 / int8±ConvRot) → bf16 STANDARD con gli stessi nomi
    (o rimappati per Gemma): per i loader che vogliono pesi pieni (diffusers from_single_file, transformers).
    Formati non riconosciuti (es. mxfp8) → errore chiaro, mai pesi crudi scambiati per bf16."""
    torch = _torch()
    rn = _remap_gemma if remap == "gemma" else (lambda k: k)
    drop = (".weight_scale", ".weight_scale_2", ".pre_quant_scale") + _DROP_SUFFIX

    def plan(key, info, header):
        if key.endswith(drop):
            return []
        mod = _module_of(key)
        conf = header.get("__qconf__", {}).get(mod)
        get = header["__ctx__"]["get"]
        if conf is None or key != mod + ".weight":
            return [(rn(key), lambda t: t)]
        fmt = str(conf.get("format", ""))

        def fn(t, _m=mod, _fmt=fmt, _g=int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 0):
            if t.device.type == "meta":
                shp = (t.shape[0], t.shape[1] * 2) if _fmt == "nvfp4" else tuple(t.shape)
                return torch.empty(shp, dtype=torch.bfloat16, device="meta")
            if _fmt.startswith("float8"):
                sc = get(_m + ".weight_scale").float()
                return (t.float() * (sc.reshape(()) if sc.numel() == 1 else sc.reshape(-1, 1))).to(torch.bfloat16)
            if _fmt == "nvfp4":
                return dequant_nvfp4(t, get(_m + ".weight_scale"), get(_m + ".weight_scale_2"))
            if _fmt.startswith("int8"):
                sc = get(_m + ".weight_scale").float()
                w = t.float() * (sc.reshape(()) if sc.numel() == 1 else sc.reshape(-1, 1))
                return (rotate(w, _g) if _g else w).to(torch.bfloat16)
            raise ValueError(f"formato comfy_quant non supportato dal dequant nativo: {_fmt} ({_m})")
        return [(rn(key), fn)]
    plan.keep_meta = True
    return plan


def _mmh3_lora_released(partition: str):
    """LoRA H3 addestrata sul modello COMPLETO (adaLN larga 2688, es. Turbo v4 di larryvrh) → nomi nativi del DiT
    pruned. Le coppie adaLN si proiettano sulle coordinate pruned come in MiniMaxH3PrunedTransformer3DModel.
    project_adaln_lora, salvate come delta pieni: `.diff` = B·(A·baseᵀ) [out, 8] sul peso, `.diff_b` = B·(A·media)
    [out] sul folded_bias. Base e media del ramo (`fl2va`/`ref2va`) da adaln_affine.safetensors (verificato il
    14/10: tabelle temporali identiche ai nostri DiT nvfp4, diff 0.0)."""
    torch = _torch()
    rename = _mmh3_lora_rename()
    adaln = re.compile(r"(?:^|\.)((?:blocks\.\d+|final_layer)\.adaln_proj\.linear)\.lora_([AB])\.weight$")
    aff = {}

    def affine():
        if not aff:
            from core import model_paths as mp
            path = mp.p(f"mmh3-adaln-affine-{'ref' if partition == 'ref2va' else 'fl'}", "")
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"adaln_affine {partition} assente: serve per proiettare la LoRA adaLN")
            from safetensors.torch import load_file
            d = load_file(path)
            aff["basis"], aff["mean"] = d["adaln_basis"].double(), d["adaln_mean"].double()
        return aff["basis"], aff["mean"]

    def plan(key, info, header):
        m = adaln.search(key)
        if m is None:
            nk = rename(key)
            return [] if nk is None else [(nk, lambda t: t)]
        if m.group(2) == "B":
            return []                                   # consumata insieme alla sua lora_A
        mod, ctx = m.group(1), header.get("__ctx__", {})
        b_key = key[: -len("lora_A.weight")] + "lora_B.weight"
        base = rename(mod + ".lora_A.weight")[: -len(".lora_A.weight")]
        out_f = header[b_key]["shape"][0]

        def diff(t, _bk=b_key, _o=out_f):
            if t.device.type == "meta":
                return torch.empty((_o, 8), dtype=torch.float32, device="meta")
            basis, _mean = affine()
            return (ctx["get"](_bk).double() @ (t.double() @ basis.T)).float().contiguous()

        def diff_b(t, _bk=b_key, _o=out_f):
            if t.device.type == "meta":
                return torch.empty((_o,), dtype=torch.float32, device="meta")
            _basis, mean = affine()
            return (ctx["get"](_bk).double() @ (t.double() @ mean)).float().contiguous()
        return [(base + ".diff", diff), (base + ".diff_b", diff_b)]
    return plan


def _qwen21_te():
    def rename(k):
        if k == "lm_head.weight":                         # 1,2 GB mai usati (la pipeline legge gli hidden state)
            return None
        for a, b in (("model.layers.", "model.language_model.layers."),
                     ("model.embed_tokens.", "model.language_model.embed_tokens."),
                     ("model.norm.", "model.language_model.norm.")):
            if k.startswith(a):
                return b + k[len(a):]
        return k
    return _renamer(rename)


CONVERTERS = {
    "mmh3_dit": _mmh3_dit,            # MiniMax H3 DiT pruned (bf16/fp8/nvfp4) → MiniMaxH3PrunedTransformer3DModel
    "mmh3_te": _mmh3_te,              # MiniMax H3 TE Qwen3-VL-32B (nvfp4-AWQ / fp8) → Qwen3VLForConditionalGeneration
    "mmh3_vae": _mmh3_vae,            # MiniMax H3 VAE video fp16 → AutoencoderKLMiniMaxH3
    "mmh3_lora": _mmh3_lora,
    "mmh3_lora_diffusers": _mmh3_lora_diffusers,   # LoRA H3 in formato diffusers/peft (lightx2v turbo ref2v v1.0/v1.1 768p): q/k/v fusi, alpha/rank cotto in B
    "mmh3_lora_guide": _mmh3_lora_guide,   # LoRA H3 di GUIDA (h3upscale): come mmh3_lora ma conserva l'intestazione (fattore e geometria della guida)
    "mmh3_lora_fl": lambda: _mmh3_lora_released("fl2va"),    # LoRA H3 del modello completo → DiT pruned FL2VA
    "mmh3_lora_ref": lambda: _mmh3_lora_released("ref2va"),  # idem per il DiT pruned Ref2VA          # MiniMax H3 LoRA (turbo 4/8 passi ecc.) → nomi del modello nativo
    "krea2_dit": _krea2_dit,
    "krea2_te": _krea2_te,
    "ltx25_te": _ltx25_te,
    "ltx25_dit": _ltx25_dit,
    "comfy_bf16": _comfy_bf16,        # qualunque single-file comfy_quant → bf16 standard (core/dequant_comfy)
    "comfy_bf16_gemma": lambda: _comfy_bf16("gemma"),   # idem + naming Gemma 3 di ltx-core/transformers          # transformer LTX-2.5 int8 senza prefisso → model.diffusion_model.* (+ config)            # TE LTX-2.5 Gemma 4 int8 per-tensore ComfyUI (es. uncensored) → bf16 ufficiale            # TE Krea 2 Qwen3-VL-4B fp8 «scaled» ComfyUI (es. uncensored) → Qwen3VLModel          # Krea 2 single-file ComfyUI (int8 ConvRot / fp8 / bf16) → Krea2Transformer2DModel
    "qwen21_dit": lambda: _renamer(lambda k: k),   # Qwen-Image 2.1 nvfp4 «Prism»: nomi già diffusers
    "qwen21_te": _qwen21_te,          # Qwen-Image 2.1 TE Qwen3-VL-8B (nvfp4 ComfyUI) → transformers
    "qwen21_cn": lambda: _renamer(lambda k: k),    # Qwen-Image 2.1 ControlNet Union Fun (INT8 ConvRot): nomi già diffusers
    "qwen21_pe": lambda: _renamer(lambda k: k),    # arricchitori PE Qwen3.5-VL 9B (INT8 ConvRot): nomi già transformers
}


# ── motore di conversione ──────────────────────────────────────────────────────────────────────
def _meta_out(fn, info):
    torch = _torch()
    t = torch.empty(info["shape"], dtype=_dtypes()[info["dtype"]], device="meta")
    o = fn(t)
    return list(o.shape), o.dtype


def _to_bytes(t) -> bytes:
    t = t.contiguous().reshape(-1)
    return t.view(_torch().uint8).numpy().tobytes() if t.numel() else b""


def convert(converter: str, src: str | os.PathLike, dst: str | os.PathLike, *, delete_source: bool = True,
            log=print) -> Path:
    """Converte `src` (ComfyUI) → `dst` (nativo). Verifica e poi cancella `src`. Idempotente."""
    src, dst = Path(src), Path(dst)
    if dst.exists() and is_native(dst):
        if delete_source and src.exists() and src.resolve() != dst.resolve():
            log(f"[native] {dst.name} già nativo: cancello il residuo {src.name}")
            src.unlink()
        return dst
    if not src.exists():
        raise FileNotFoundError(f"sorgente da convertire assente: {src}")
    if converter not in CONVERTERS:
        raise KeyError(f"converter sconosciuto: {converter}")
    plan_fn = CONVERTERS[converter]()
    torch = _torch()
    rev = {v: k for k, v in _dtypes().items()}
    header, base = read_header(src)
    meta = header.pop("__metadata__", None) or {}
    order = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    dts = _dtypes()
    fsrc = open(src, "rb")

    def _read(k):
        info = header[k]
        a, b = info["data_offsets"]
        fsrc.seek(base + a)
        buf = bytearray(fsrc.read(b - a))
        return torch.frombuffer(buf, dtype=dts[info["dtype"]]).reshape(info["shape"]) if buf else             torch.empty(info["shape"], dtype=dts[info["dtype"]])
    ctx = {"meta": True}
    ctx["get"] = lambda k: (torch.empty(header[k]["shape"], dtype=dts[header[k]["dtype"]], device="meta")
                            if ctx["meta"] else _read(k))
    qconf = {}
    try:
        qconf.update(json.loads(meta.get("_quantization_metadata", "{}")).get("layers", {}))
    except ValueError:
        pass
    for k in [k for k in header if k.endswith(".comfy_quant")]:
        try:
            qconf[k[: -len(".comfy_quant")]] = json.loads(_read(k).numpy().tobytes())
        except ValueError:
            pass
    header = dict(header)
    header["__qconf__"], header["__ctx__"] = qconf, ctx
    header["__srcmeta__"] = meta                                # metadati del sorgente (es. `alpha` delle LoRA diffusers lightx2v)
    order = [(k, v) for k, v in order if not k.startswith("__")]
    plan, out_header, off = [], {}, 0
    for key, info in order:
        items = []
        for nk, fn in plan_fn(key, info, header):
            shape, dt = _meta_out(fn, info)
            n = 1
            for s in shape:
                n *= s
            size = n * torch.empty((), dtype=dt).element_size()
            if nk in out_header:
                raise ValueError(f"chiave duplicata dopo la conversione: {nk}")
            out_header[nk] = {"dtype": rev[dt], "shape": shape, "data_offsets": [off, off + size]}
            off += size
            items.append((nk, fn))
        if items:
            plan.append((key, info, items))
    # converter con `extras`: tensori aggiuntivi (scritti in coda) e metadata da innestare (es. TE LTX-2.5)
    extra_md, extra_t = plan_fn.extras() if hasattr(plan_fn, "extras") else ({}, {})
    added = [(nk, t) for nk, t in extra_t.items() if nk not in out_header]   # se il sorgente ce l'ha, vince lui
    for nk, t in added:
        size = t.numel() * t.element_size()
        out_header[nk] = {"dtype": rev[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + size]}
        off += size
    if getattr(plan_fn, "keep_meta", False):               # es. transformer LTX: la config del modello sta qui
        extra_md = {**{k: v for k, v in meta.items() if k != "_quantization_metadata"}, **extra_md}
    out_header["__metadata__"] = {**{str(k): str(v) for k, v in extra_md.items()},
                                  "ltx_native": FORMAT, "converter": converter, "source": src.name}
    hb = json.dumps(out_header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ctx["meta"] = False
    log(f"[native] {src.name} → {dst.name} ({converter}, {len(out_header) - 1} tensori)")
    with open(src, "rb") as fi, open(part, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for key, info, items in plan:
            a, b = info["data_offsets"]
            fi.seek(base + a)
            buf = bytearray(fi.read(b - a))
            t = torch.frombuffer(buf, dtype=dts[info["dtype"]]).reshape(info["shape"]) if buf else \
                torch.empty(info["shape"], dtype=dts[info["dtype"]])
            for nk, fn in items:
                data = _to_bytes(fn(t))
                exp = out_header[nk]["data_offsets"]
                if len(data) != exp[1] - exp[0]:
                    raise ValueError(f"{nk}: {len(data)} byte, attesi {exp[1] - exp[0]}")
                fo.write(data)
            del buf, t
        for _nk, t in added:                                  # stesso ordine degli offset assegnati sopra
            fo.write(_to_bytes(t))
    try:
        _verify(src, part, plan, out_header)
    finally:
        fsrc.close()
    os.replace(part, dst)
    log(f"[native] verificato: {dst.name} ({dst.stat().st_size / 2**30:.2f} GB)")
    if delete_source and src.resolve() != dst.resolve():
        src.unlink()
        log(f"[native] cancellato il file ComfyUI {src.name}")
    return dst


def _verify(src: Path, out: Path, plan, out_header) -> None:
    """Header identico al piano + confronto bit-a-bit di un campione (primi, ultimi e ogni 97° tensore)."""
    torch = _torch()
    h, base = read_header(out)
    exp = {k: v for k, v in out_header.items() if k != "__metadata__"}
    got = {k: v for k, v in h.items() if k != "__metadata__"}
    if exp != got:
        raise ValueError(f"header del file convertito diverso dal piano ({out.name})")
    size = out.stat().st_size
    last = max((v["data_offsets"][1] for v in got.values()), default=0)
    if size != base + last:
        raise ValueError(f"dimensione inattesa {size} ≠ {base + last} ({out.name})")
    dts = _dtypes()
    _, sbase = read_header(src)
    idx = set(range(min(3, len(plan)))) | set(range(max(0, len(plan) - 3), len(plan))) | set(range(0, len(plan), 97))
    with open(src, "rb") as fs, open(out, "rb") as fo:
        for i in sorted(idx):
            key, info, items = plan[i]
            a, b = info["data_offsets"]
            fs.seek(sbase + a)
            t = torch.frombuffer(bytearray(fs.read(b - a)), dtype=dts[info["dtype"]]).reshape(info["shape"])
            for nk, fn in items:
                ref = _to_bytes(fn(t))
                oa, ob = got[nk]["data_offsets"]
                fo.seek(base + oa)
                if fo.read(ob - oa) != ref:
                    raise ValueError(f"contenuto diverso dopo la conversione: {nk}")


# ── integrazione col catalogo ─────────────────────────────────────────────────────────────────
def source_path(entry: dict, dest_dir: Path) -> Path:
    """Dove il downloader mette il file ComfyUI prima della conversione (nome remoto, stessa cartella)."""
    return Path(dest_dir) / Path(entry.get("remote_filename") or entry["filename"]).name


def ensure_entry(entry: dict, dest_dir: Path, log=print) -> Path | None:
    """Voce col campo `convert`: se c'è il sorgente ComfyUI e non il nativo, converte (e cancella il sorgente)."""
    conv = entry.get("convert")
    if not conv or entry.get("kind") != "file":
        return None
    dst = Path(dest_dir) / entry["filename"]
    src = source_path(entry, dest_dir)
    if src.exists() and src.resolve() != dst.resolve():
        if not is_complete(src):
            log(f"[native] {src.name} è incompleto (download interrotto): niente conversione")
            return dst if dst.exists() else None
        return convert(conv, src, dst, log=log)
    return dst if dst.exists() else None


def migrate_all(base_dir: str | None = None, log=print) -> list[str]:
    """Converte i file ComfyUI già presenti su disco per tutte le voci `convert` del catalogo."""
    from core import weights_catalog as wc
    done = []
    for e in wc.load():
        if not e.get("convert") or e.get("kind") != "file":
            continue
        d = wc.dest_for(e, base_dir)
        src = source_path(e, d)
        if src.exists() and src.name != e["filename"] and is_complete(src):
            ensure_entry(e, d, log=log)
            done.append(e["id"])
    return done


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--migrate":
        print(migrate_all())
    elif len(sys.argv) == 4:
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
