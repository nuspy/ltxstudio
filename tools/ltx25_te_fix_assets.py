"""Ripara un text encoder LTX-2.5 derivato in locale che ha PERSO gli asset HF.

PROBLEMA (trovato live il 17/08): `ltx-core` carica un TE Gemma da **file singolo** solo se il
safetensors porta con sé (a) il metadata `gemma_config` e (b) i tensori `tokenizer_json` e
`hf_asset__*` (tokenizer_config.json, processor_config.json, …). I file UFFICIALI ce li hanno; una
quantizzazione fatta in casa (es. `gemma4-12b-with-proj-nvfp4-torchao.safetensors`) li perde,
perché i quantizzatori copiano solo i pesi. Risultato: LTX-2.5 non parte —
«No files matching pattern 'tokenizer.model'».

SOLUZIONE ECONOMICA: gli asset stanno all'INIZIO del file ufficiale e pesano pochi MB. Qui si
legge l'header remoto (~100 KB), si scaricano con richieste Range SOLO i tensori-asset (~7 MB) e
si riscrive il file locale aggiungendoli. Niente download da 24 GB.

Uso:
  python tools/ltx25_te_fix_assets.py                      # ripara il TE nvfp4 locale
  python tools/ltx25_te_fix_assets.py --check              # dice solo se il file è a posto
  python tools/ltx25_te_fix_assets.py --dest ALTRO.safetensors --source-url URL
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# mirror UNGATE (file byte-identici all'ufficiale, niente token) — vedi catalogo ltx25-te-bf16
SOURCE_URL = ("https://huggingface.co/dummy9996/LTX-2.5-22b-ungate/resolve/main/"
              "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
CONFIG_KEY = "gemma_config"
TOKENIZER_KEY = "tokenizer_json"
ASSET_PREFIX = "hf_asset__"


def _http(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    try:
        from core.downloader import resolve_token
        tok = resolve_token(None)
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
    except Exception:                                        # noqa: BLE001 — il mirror non è gated
        pass
    return urllib.request.urlopen(req, timeout=120).read()


def remote_header(url: str) -> tuple[dict, int]:
    """(header JSON, offset dei dati) leggendo solo i primi byte del file remoto."""
    head = _http(url, 0, 8)
    n = struct.unpack("<Q", head[:8])[0]
    hdr = json.loads(_http(url, 8, 8 + n - 1).decode("utf-8"))
    return hdr, 8 + n


def local_status(path: str) -> dict:
    """Cosa manca al file locale perché ltx-core lo accetti come TE da file singolo."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        md = f.metadata() or {}
    return {
        "config": CONFIG_KEY in md,
        "tokenizer": TOKENIZER_KEY in keys,
        "assets": sorted(k[len(ASSET_PREFIX):] for k in keys if k.startswith(ASSET_PREFIX)),
        "n_tensori": len(keys),
    }


def fetch_assets(url: str) -> tuple[dict, dict]:
    """Scarica dal file remoto SOLO metadata + tensori-asset. Ritorna (metadata, {nome: tensor})."""
    import torch
    hdr, data_off = remote_header(url)
    md = hdr.get("__metadata__") or {}
    voluti = [k for k in hdr
              if k != "__metadata__" and (k == TOKENIZER_KEY or k.startswith(ASSET_PREFIX))]
    if not voluti:
        raise SystemExit(f"Il file remoto non contiene asset ({TOKENIZER_KEY}/{ASSET_PREFIX}*): "
                         f"sorgente sbagliata?")
    out: dict = {}
    tot = 0
    for k in voluti:
        info = hdr[k]
        a, b = info["data_offsets"]
        raw = _http(url, data_off + a, data_off + b - 1)
        tot += len(raw)
        # gli asset sono byte grezzi salvati come tensore 1-D uint8
        out[k] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        print(f"  ↓ {k}: {len(raw) / 1024:.0f} KB", flush=True)
    print(f"  totale scaricato: {tot / 2**20:.1f} MB (invece dell'intero file)", flush=True)
    return md, out


def repair(dest: str, url: str) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    st = local_status(dest)
    if st["config"] and st["tokenizer"]:
        print(f"[=] {os.path.basename(dest)} è già completo (asset: {', '.join(st['assets']) or '-'})")
        return 0
    print(f"[!] mancano: " + ", ".join(
        ([] if st["config"] else ["metadata gemma_config"]) +
        ([] if st["tokenizer"] else ["tensore tokenizer_json"])))
    print(f"[+] recupero gli asset da {url.rsplit('/', 1)[-1]}", flush=True)
    if st["tokenizer"]:
        # I tensori-asset ci sono già: serve SOLO il metadata, che sta nell'header (~100 KB).
        md_remota, asset = remote_header(url)[0].get("__metadata__") or {}, {}
        print(f"  ↓ solo metadata dall'header: {len(md_remota.get(CONFIG_KEY, ''))} caratteri", flush=True)
    else:
        md_remota, asset = fetch_assets(url)

    print(f"[+] rileggo i {st['n_tensori']} tensori locali…", flush=True)
    tens: dict = {}
    with safe_open(dest, framework="pt") as f:
        md_locale = dict(f.metadata() or {})
        for k in f.keys():
            tens[k] = f.get_tensor(k)
    tens.update(asset)
    md_locale[CONFIG_KEY] = md_remota.get(CONFIG_KEY, md_locale.get(CONFIG_KEY, ""))
    for k, v in md_remota.items():                           # fallback usati da ltx-core
        md_locale.setdefault(k, v)
    if not md_locale.get(CONFIG_KEY):
        raise SystemExit("il file remoto non porta 'gemma_config': impossibile riparare")

    tmp = dest + ".fix"
    print(f"[+] riscrivo {os.path.basename(dest)} ({len(tens)} tensori)…", flush=True)
    save_file(tens, tmp, metadata={k: str(v) for k, v in md_locale.items()})
    bak = dest + ".senza-asset.bak"
    if not os.path.exists(bak):
        os.replace(dest, bak)
    else:
        os.remove(dest)
    os.replace(tmp, dest)
    st2 = local_status(dest)
    ok = st2["config"] and st2["tokenizer"]
    print(f"[{'OK' if ok else 'FAIL'}] config={st2['config']} tokenizer={st2['tokenizer']} "
          f"asset={', '.join(st2['assets']) or '-'}")
    print(f"      originale conservato in {os.path.basename(bak)}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="")
    ap.add_argument("--source-url", default=SOURCE_URL)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dest = a.dest
    if not dest:
        from core import ltx_release as lr
        dest = lr.get("2.5").text_encoder
    if not os.path.exists(dest):
        print(f"text encoder non trovato: {dest}"); return 2
    if a.check:
        print(json.dumps(local_status(dest), ensure_ascii=False, indent=1)); return 0
    return repair(dest, a.source_url)


if __name__ == "__main__":
    raise SystemExit(main())
