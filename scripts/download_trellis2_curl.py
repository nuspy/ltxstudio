"""Download TRELLIS.2-4B via curl a CHUNK CONCORRENTI (anti-throttle HuggingFace).

HF throttla ~0.3-0.5 MB/s per connessione e hf_hub si pianta sui file Xet. Workaround (memoria
[[hf-download-throttle]]): per ogni file si scaricano range da 16MB con N curl in parallelo e si
riassemblano. Resumabile: salta i file già completi e i chunk già scaricati.

Scarica in una CARTELLA LOCALE (struttura repo) → poi TRELLIS la usa con from_pretrained(<dir>).
Uso:  python scripts/download_trellis2_curl.py [--dir DIR] [--jobs 8] [--chunk-mb 16]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
from pathlib import Path

REPO = "microsoft/TRELLIS.2-4B"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"
DEFAULT_DIR = r"E:\AI\Tools\TRELLIS2\models\TRELLIS.2-4B"


def _curl(args: list[str]) -> int:
    return subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "3",
                           "--connect-timeout", "30", *args], capture_output=True).returncode


def _remote_size(url: str) -> int:
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    size = 0
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return size


def _download_file(relpath: str, dest_dir: Path, jobs: int, chunk: int) -> bool:
    url = f"{BASE_URL}/{relpath}"
    dest = dest_dir / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = _remote_size(url)
    if size and dest.is_file() and dest.stat().st_size == size:
        print(f"  ✓ già completo: {relpath}", flush=True)
        return True
    if not size:                                            # size ignota (file piccolo/json) → curl diretto resume
        rc = _curl(["-C", "-", "-o", str(dest), url])
        ok = rc == 0 and dest.is_file()
        print(f"  {'✓' if ok else '✗'} {relpath} (diretto)", flush=True)
        return ok
    # split in range da `chunk` byte, scaricati in parallelo come .partN, poi concatenati
    parts = [(i, s, min(s + chunk - 1, size - 1)) for i, s in enumerate(range(0, size, chunk))]
    tmp = dest.with_suffix(dest.suffix + ".dl")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    def grab(p):
        i, a, b = p
        pf = dest.parent / f"{dest.name}.part{i}"
        if pf.is_file() and pf.stat().st_size == (b - a + 1):
            return True
        rc = _curl(["-r", f"{a}-{b}", "-o", str(pf), url])
        return rc == 0 and pf.is_file() and pf.stat().st_size == (b - a + 1)

    print(f"  ⬇ {relpath}  ({size/1e9:.2f} GB, {len(parts)} chunk × {chunk//1024//1024}MB, {jobs} paralleli)", flush=True)
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(grab, parts))
    if not all(results):
        print(f"  ✗ chunk mancanti per {relpath} (rilancia per riprendere)", flush=True)
        return False
    with open(tmp, "wb") as out:                            # concatena i chunk in ordine
        for i, _, _ in parts:
            pf = dest.parent / f"{dest.name}.part{i}"
            with open(pf, "rb") as fh:
                out.write(fh.read())
    if tmp.stat().st_size != size:
        print(f"  ✗ size finale errata {relpath}", flush=True); return False
    os.replace(tmp, dest)
    for i, _, _ in parts:                                   # pulizia chunk
        try:
            (dest.parent / f"{dest.name}.part{i}").unlink()
        except OSError:
            pass
    print(f"  ✓ {relpath}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=16)
    a = ap.parse_args()
    dest_dir = Path(a.dir)
    chunk = a.chunk_mb * 1024 * 1024

    # 1) pipeline json (piccoli) → definiscono i modelli necessari
    metas = ["pipeline.json", "texturing_pipeline.json"]
    for m in metas:
        _download_file(m, dest_dir, a.jobs, chunk)
    import json
    needed: set[str] = set()
    for m in metas:
        p = dest_dir / m
        if not p.is_file():
            continue
        try:
            cfg = json.load(open(p, encoding="utf-8"))
            args = cfg.get("args", cfg)
            for v in (args.get("models", {}) or {}).values():
                if isinstance(v, str) and v.startswith("ckpts/"):
                    needed.add(v)
        except Exception as e:                              # noqa: BLE001
            print(f"[warn] parse {m}: {e}", flush=True)
    # encoder shape (non sempre in pipeline.json ma necessario) + relativi .json
    needed.add("ckpts/shape_enc_next_dc_f16c32_fp16")
    files = []
    for n in sorted(needed):
        files.append(n + ".safetensors")
        files.append(n + ".json")
    print(f"[TRELLIS2-curl] {len(files)} file da garantire in {dest_dir}", flush=True)

    ok = True
    for f in files:
        if not _download_file(f, dest_dir, a.jobs, chunk):
            ok = False
    print(f"[TRELLIS2-curl] {'✔ COMPLETO' if ok else '✗ INCOMPLETO (rilancia per riprendere)'}", flush=True)
    if ok:
        (dest_dir / ".ltx_ready").write_text("ok", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
