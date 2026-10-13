"""Downloader veloce multi-connessione con resume — replica del sistema SwarmUI.

Perché: hf/curl single-stream + xet si pianta/rallenta su connessioni che invece reggono 150 Mbps.
Soluzione (come SwarmUI `utilities/HF_model_downloader.py`): per file > soglia, dividi in N chunk e
scaricali in PARALLELO con header HTTP Range (206 Partial Content), `.partN` per il resume, poi merge.
Fallback a single-stream (con resume) se il server non supporta i range. + retry exponential backoff +
session pooling keep-alive. Per HuggingFace usa /resolve/main + token opzionale + lista file via API.

API:
  download_url(url, dest, *, connections=16, token=None, progress=None) -> str
  download_hf_file(repo_id, filename, dest, *, token=None, progress=None) -> str
  download_hf_repo(repo_id, dest_dir, *, allow=None, token=None, progress=None) -> list[str]
"""
from __future__ import annotations

import concurrent.futures
import os
import time
import urllib.parse
from pathlib import Path

import requests

CFG = {"num_connections": 16, "chunk_size": 10 * 1024 * 1024, "max_retries": 5,
       "retry_delay": 2, "max_retry_delay": 30, "timeout": 300, "parallel_threshold": 10 * 1024 * 1024}
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ltxstudio-downloader"


def _session() -> requests.Session:
    s = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24, max_retries=3)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": _UA})
    return s


def _auth(url: str, token: str | None) -> dict:
    if token and urllib.parse.urlparse(url).hostname and "huggingface" in urllib.parse.urlparse(url).hostname:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _file_size_and_range(sess: requests.Session, url: str, token: str | None) -> tuple[int, bool]:
    """Ritorna (size, supports_range). Probe con Range bytes=0-0 (206 = range ok)."""
    h = _auth(url, token)
    try:
        r = sess.get(url, headers={**h, "Range": "bytes=0-0"}, timeout=60, allow_redirects=True, stream=True)
        cr = r.headers.get("content-range")
        r.close()
        if r.status_code == 206 and cr and "/" in cr:
            return int(cr.split("/")[-1]), True
        cl = r.headers.get("content-length")
        return (int(cl) if cl else 0), False
    except Exception:                                       # noqa: BLE001
        return 0, False


def _dl_chunk(sess, url, dest, cid, start, end, token, prog) -> bool:
    part = f"{dest}.part{cid}"
    want = end - start + 1
    resume = 0
    if os.path.exists(part):
        resume = os.path.getsize(part)
        if resume == want:
            if prog:
                prog(want)
            return True
        if resume > want:
            os.remove(part); resume = 0
    for attempt in range(CFG["max_retries"]):
        try:
            hdr = {**_auth(url, token), "Range": f"bytes={start + resume}-{end}"}
            r = sess.get(url, headers=hdr, timeout=CFG["timeout"], stream=True, allow_redirects=True)
            if r.status_code not in (206, 200):
                raise RuntimeError(f"HTTP {r.status_code}")
            with open(part, "ab" if resume else "wb") as fh:
                for data in r.iter_content(CFG["chunk_size"]):
                    if data:
                        fh.write(data)
                        if prog:
                            prog(len(data))
            if os.path.getsize(part) >= want:
                return True
            resume = os.path.getsize(part)
        except Exception:                                  # noqa: BLE001
            time.sleep(min(CFG["retry_delay"] * (2 ** attempt), CFG["max_retry_delay"]))
            resume = os.path.getsize(part) if os.path.exists(part) else 0
    return False


def _merge(dest, n) -> None:
    tmp = f"{dest}.tmp"
    buf = 64 * 1024 * 1024
    with open(tmp, "wb") as out:
        for i in range(n):
            part = f"{dest}.part{i}"
            with open(part, "rb") as inp:
                while True:
                    d = inp.read(buf)
                    if not d:
                        break
                    out.write(d)
            os.remove(part)
    os.replace(tmp, dest)


def _dl_single(sess, url, dest, size, token, prog) -> bool:
    resume = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size and resume >= size:
        return True
    hdr = {**_auth(url, token)}
    if resume:
        hdr["Range"] = f"bytes={resume}-"
    for attempt in range(CFG["max_retries"]):
        try:
            r = sess.get(url, headers=hdr, timeout=CFG["timeout"], stream=True, allow_redirects=True)
            with open(dest, "ab" if resume else "wb") as fh:
                for data in r.iter_content(CFG["chunk_size"]):
                    if data:
                        fh.write(data)
                        if prog:
                            prog(len(data))
            return (not size) or os.path.getsize(dest) >= size
        except Exception:                                  # noqa: BLE001
            time.sleep(min(CFG["retry_delay"] * (2 ** attempt), CFG["max_retry_delay"]))
            resume = os.path.getsize(dest) if os.path.exists(dest) else 0
            hdr["Range"] = f"bytes={resume}-"
    return False


def download_url(url: str, dest: str, *, connections: int = 16, token: str | None = None, progress=None) -> str:
    """Scarica un URL su `dest`: parallelo multi-chunk se grande e il server supporta i range; altrimenti
    single-stream con resume. `progress(nbytes)` chiamato a ogni blocco scritto."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    sess = _session()
    size, rng = _file_size_and_range(sess, url, token)
    if size and os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest                                        # già completo
    if rng and size > CFG["parallel_threshold"]:
        n = max(1, min(connections, size))
        base = size // n
        chunks = [(i, i * base, (size - 1 if i == n - 1 else (i + 1) * base - 1)) for i in range(n)]
        # Il parallelo si RIPROVA prima di arrendersi: le .partN complete vengono saltate (resume), si
        # riscarica solo il chunk fallito. Prima un solo chunk andato male (PinkCherry H3, 13/10: 15/16
        # parti complete) faceva ripartire da zero in single-stream, lasciando ~20GB di parti orfane.
        for _try in range(3):
            ok = True
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(_dl_chunk, sess, url, dest, cid, s, e, token, progress) for cid, s, e in chunks]
                for f in concurrent.futures.as_completed(futs):
                    ok = ok and f.result()
            if ok:
                _merge(dest, n)
                return dest
            time.sleep(10 * (_try + 1))
        # fallback single-stream: le parti del parallelo non servono più (il single scrive in dest)
        for cid, _s, _e in chunks:
            try:
                os.remove(f"{dest}.part{cid}")
            except OSError:
                pass
    if not _dl_single(sess, url, dest, size, token, progress):
        raise RuntimeError(f"download fallito: {url}")
    return dest


# ─────────── HuggingFace ───────────
def _hf_url(repo_id: str, filename: str) -> str:
    from urllib.parse import quote                      # nomi con spazi (es. LoRA «NSFW Qwen by … V2»)
    return f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='/')}"


def download_hf_file(repo_id: str, filename: str, dest: str, *, token: str | None = None,
                     connections: int = 16, progress=None) -> str:
    return download_url(_hf_url(repo_id, filename), dest, connections=connections, token=token, progress=progress)


def download_hf_repo(repo_id: str, dest_dir: str, *, allow: list | None = None, token: str | None = None,
                     connections: int = 16, progress=None) -> list[str]:
    """Scarica tutti i file di un repo HF (o solo quelli che matchano i glob `allow`), in serie ma ogni
    file con N connessioni parallele. Ritorna i path scaricati."""
    from huggingface_hub import list_repo_files
    import fnmatch
    files = list_repo_files(repo_id=repo_id, token=token)
    if allow:
        files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in allow)]
    out = []
    for f in files:
        dest = os.path.join(dest_dir, f.replace("/", os.sep))
        if progress:
            progress(0, f)
        download_hf_file(repo_id, f, dest, token=token, connections=connections,
                         progress=(lambda n, _f=f: progress(n, _f)) if progress else None)
        out.append(dest)
    return out
