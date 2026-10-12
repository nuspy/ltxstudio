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
    for suf in (".weight_scale_2", ".weight_scale", ".pre_quant_scale", ".weight", ".comfy_quant", ".input_scale"):
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


# ── regole comuni ──────────────────────────────────────────────────────────────────────────────
def _quant_common(key: str, info: dict, header: dict):
    """Regole condivise dai quantizzati ComfyUI. Ritorna None (nessuna regola) o [(chiave, fn)]."""
    torch = _torch()
    if key.endswith(_DROP_SUFFIX):
        return []
    mod = _module_of(key)
    conf = header.get("__qconf__", {}).get(mod, {})
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
    if key.endswith(".weight_scale") and info["shape"] == [] and info["dtype"] == "F32":
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


def _mmh3_lora():
    """LoRA H3 in nomi ComfyUI (diffusion_model.blocks.N.attn.qkv_proj.lora_A…) → nomi nativi del modello
    montato (transformer_blocks.N.attn.to_qkv…, ff.gate_up/ff.out): stesse matrici, qkv e gate/up già fusi."""
    _workers_path()
    import minimax_h3_convert as M

    def rename(k):
        for pre in ("diffusion_model.", "model.diffusion_model.", "transformer."):
            if k.startswith(pre):
                k = k[len(pre):]
        return M.dit_key(k)
    return _renamer(rename)


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
    "mmh3_lora": _mmh3_lora,          # MiniMax H3 LoRA (turbo 4/8 passi ecc.) → nomi del modello nativo
    "qwen21_dit": lambda: _renamer(lambda k: k),   # Qwen-Image 2.1 nvfp4 «Prism»: nomi già diffusers
    "qwen21_te": _qwen21_te,          # Qwen-Image 2.1 TE Qwen3-VL-8B (nvfp4 ComfyUI) → transformers
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
    out_header["__metadata__"] = {"ltx_native": FORMAT, "converter": converter, "source": src.name}
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
        if source_path(e, d).exists() and source_path(e, d).name != e["filename"]:
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
