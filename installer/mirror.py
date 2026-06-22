"""Fallback mirror tadamoo.com — SOLO per custom nodes + comfy_compat (mai modelli).

Se l'origine (GitHub) non è raggiungibile, scarica un tarball versionato dal mirror
(<mirror_base>/<id>.tar.gz) e lo estrae in <ComfyUI>/custom_nodes/. L'integrità è
garantita da SHA256SUMS sul mirror (NON dal TLS): per questo, se la verifica del
certificato fallisce (es. cert scaduto), si ricade su una connessione non verificata
MA si rifiuta il file se l'hash SHA256 non corrisponde. Il comfy_compat 'bundled'
viene invece copiato dal repo via core.comfy_compat.
"""
from __future__ import annotations

import hashlib
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def _http_get(url: str) -> bytes:
    """GET con verifica TLS; in caso di errore TLS (cert scaduto/self-signed) ricade
    su connessione non verificata. L'integrità è poi garantita da SHA256."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except ssl.SSLError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
            return r.read()
    except urllib.error.URLError as e:                      # alcuni errori cert arrivano qui
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
                return r.read()
        raise


def _expected_sha(base: str, filename: str) -> str | None:
    try:
        sums = _http_get(f"{base.rstrip('/')}/SHA256SUMS").decode("utf-8", "replace")
        for line in sums.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == filename:
                return parts[0].lower()
    except Exception:
        pass
    return None


def fetch_node(spec: dict, node_id: str, comfy_root: str | None) -> bool:
    if node_id == "comfy_compat":
        try:
            from core import comfy_compat
            comfy_compat.install()
            return True
        except Exception:
            pass  # prosegue col tarball del mirror
    base = spec.get("mirror_base")
    if not base or not comfy_root:
        print(f"  [mirror] indisponibile per {node_id} (base/comfy_root mancanti)")
        return False
    filename = f"{node_id}.tar.gz"
    url = f"{base.rstrip('/')}/{filename}"
    try:
        print(f"  [mirror] scarico {url}")
        data = _http_get(url)
        # verifica integrità via SHA256SUMS del mirror
        want = _expected_sha(base, filename)
        if want:
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                print(f"  [mirror] SHA256 NON corrisponde per {node_id} → rifiuto")
                return False
        else:
            print(f"  [mirror] (nessun SHA256SUMS: estraggo senza verifica hash)")
        dest = Path(comfy_root) / "custom_nodes"
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(data); tmp.flush()
            with tarfile.open(tmp.name) as tf:
                tf.extractall(dest)
        print(f"  [mirror] {node_id} estratto in {dest}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [mirror] fallito per {node_id}: {e}")
        return False
