"""CLI download veloce (multi-connessione + resume) — usa core/fast_downloader.

Uso:
  python scripts/fast_download.py --repo ByteDance-Seed/SeedVR2-3B --dest E:\\AI\\Models\\SeedVR2\\SeedVR2-3B
  python scripts/fast_download.py --repo <repo> --file <path/nel/repo> --dest <file_locale>
  python scripts/fast_download.py --url <url> --dest <file_locale>
Opz: --connections 16  --allow "*.safetensors" "*.json"  (glob per --repo)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import fast_downloader as FD


def _bar():
    state = {"n": 0, "t0": time.time(), "last": 0.0}

    def cb(nbytes, name=None):
        state["n"] += nbytes
        now = time.time()
        if now - state["last"] > 0.5:
            mb = state["n"] / 1048576
            spd = mb / max(0.1, now - state["t0"])
            print(f"\r  {mb:8.0f} MB  ({spd:6.1f} MB/s){'  ' + name if name else ''}   ", end="", flush=True)
            state["last"] = now
    return cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--file"); ap.add_argument("--url")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--connections", type=int, default=16)
    ap.add_argument("--allow", nargs="*")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    a = ap.parse_args()
    cb = _bar()
    t0 = time.time()
    if a.url:
        FD.download_url(a.url, a.dest, connections=a.connections, token=a.token, progress=lambda n: cb(n))
    elif a.repo and a.file:
        FD.download_hf_file(a.repo, a.file, a.dest, token=a.token, connections=a.connections, progress=lambda n: cb(n))
    elif a.repo:
        FD.download_hf_repo(a.repo, a.dest, allow=a.allow, token=a.token, connections=a.connections, progress=cb)
    else:
        ap.error("specifica --url, oppure --repo [--file]")
    print(f"\n[fast_download] completato in {time.time()-t0:.0f}s → {a.dest}")


if __name__ == "__main__":
    main()
