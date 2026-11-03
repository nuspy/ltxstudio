"""Downloader UNICO dei pesi (FASE B weights-unification) — un motore, tante porte.

Prima c'erano TRE stack HTTP (hf_hub nel wizard, urllib per Civitai/facerestore, requests
multi-chunk in fast_downloader MAI usato dalla GUI) e quattro manifesti. Da qui in poi ogni
download di pesi passa da questo modulo, che monta:

  - `core/fast_downloader` (16 connessioni con Range+resume — l'anti-throttle che era orfano)
    per QUALSIASI sorgente: file HF via /resolve/main, url diretti (GitHub), snapshot HF via
    lista file (con `allow_patterns` del catalogo v2);
  - PREFLIGHT GATED: i repo gated vengono verificati PRIMA di partire (HTTP sulla API HF) →
    `GatedRepoError` con l'url della licenza, così la GUI può mostrare il dialog token invece
    del generico "✗ accesso negato" a download già fallito;
  - VERIFICA D'INTEGRITÀ post-download: sha256 se a catalogo, altrimenti controllo strutturale
    (header safetensors parsabile, magic GGUF, size>0) — il "video verde da GGUF corrotto" si
    intercetta QUI, non a generazione partita;
  - nomi remoti ≠ locali (`remote_filename`) e voci `no_remote` rifiutate con errore CHIARO;
  - `DownloadQueue`: coda SERIALE con pausa/riprendi/annulla (connessione metered: un peso alla
    volta, mai fetch concorrenti di modelli).

Il chiamante GUI resta `scripts/download_weights.py` in QProcess (kill = cancel duro);
API/installer usano `download_entry`/`DownloadQueue` direttamente.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path

from core import weights_catalog as wc


class DownloadUnavailable(RuntimeError):
    """La voce non ha una fonte remota (no_remote): scan/link/import, niente download."""


class GatedRepoError(RuntimeError):
    """Repo gated senza token/licenza. `license_url` = dove accettare; la GUI mostra il dialog."""

    def __init__(self, repo_id: str, msg: str = ""):
        self.repo_id = repo_id
        self.license_url = f"https://huggingface.co/{repo_id}"
        super().__init__(msg or f"repo gated: {repo_id} — accetta la licenza su "
                                f"{self.license_url} e imposta un HF token")


class IntegrityError(RuntimeError):
    """File scaricato ma corrotto/incompleto (sha o struttura non validi)."""


# ─────────────────────────── token ───────────────────────────
def resolve_token(explicit: str | None = None) -> str | None:
    """Token HF: esplicito → env → user_prefs('hf_token') → quello salvato da `hf auth login` (lo usa già
    hf_hub_download da sé; senza, il preflight dei repo gated rifiutava un accesso in realtà valido).
    Mai stampato nei log."""
    if explicit:
        return explicit
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    try:
        from core import user_prefs
        v = str(user_prefs.get("hf_token", "") or "").strip()
        if v:
            return v
    except Exception:                                    # noqa: BLE001
        pass
    try:
        from huggingface_hub import get_token
        return get_token() or None
    except Exception:                                    # noqa: BLE001
        return None


# ─────────────────────────── preflight gated ───────────────────────────
def check_gated(repo_id: str, token: str | None) -> None:
    """Solleva GatedRepoError se il repo è gated e il token non basta. Best-effort: problemi di
    RETE non bloccano (il download vero darà l'errore); un 401/403 esplicito sì."""
    import requests
    try:
        h = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(f"https://huggingface.co/api/models/{repo_id}", headers=h, timeout=15)
    except Exception:                                    # noqa: BLE001 — rete: non è un verdetto
        return
    if r.status_code in (401, 403):
        raise GatedRepoError(repo_id)
    if r.status_code == 200:
        try:
            info = r.json()
            if info.get("gated") and not token:
                raise GatedRepoError(repo_id)
        except GatedRepoError:
            raise
        except Exception:                                # noqa: BLE001
            pass


# ─────────────────────────── verifica integrità ───────────────────────────
def verify_file(path: str | Path, *, sha256: str | None = None) -> None:
    """Controllo post-download SENZA torch: sha256 se noto, altrimenti struttura.
    safetensors → header JSON parsabile (8 byte little-endian di lunghezza + JSON);
    GGUF → magic b'GGUF'. Un file troncato/corrotto fallisce QUI, non al load."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise IntegrityError(f"file assente o vuoto: {p}")
    if sha256:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
                h.update(blk)
        if h.hexdigest().lower() != sha256.lower():
            raise IntegrityError(f"sha256 diverso dall'atteso: {p.name}")
        return
    suf = p.suffix.lower()
    try:
        if suf == ".safetensors":
            with open(p, "rb") as fh:
                (hlen,) = struct.unpack("<Q", fh.read(8))
                if not (0 < hlen < 200 * 1024 * 1024):
                    raise IntegrityError(f"header safetensors implausibile: {p.name}")
                json.loads(fh.read(hlen).decode("utf-8"))
        elif suf == ".gguf":
            with open(p, "rb") as fh:
                if fh.read(4) != b"GGUF":
                    raise IntegrityError(f"magic GGUF assente: {p.name}")
    except IntegrityError:
        raise
    except Exception as ex:                              # noqa: BLE001
        raise IntegrityError(f"struttura non valida ({p.name}): {ex}") from ex


# ─────────────────────────── download di UNA voce ───────────────────────────
def download_entry(entry: dict, *, base_dir: str | None = None, token: str | None = None,
                   progress=None, log=print) -> Path:
    """Scarica una voce del catalogo v2 nella sua destinazione canonica. Ritorna il path finale.
    `progress(nbytes_delta, filename)` per il conteggio byte del chiamante."""
    if entry.get("no_remote"):
        raise DownloadUnavailable(
            f"'{entry.get('id')}' non ha una fonte remota (no_remote): "
            f"{entry.get('note') or 'usa scan/link o l’import manuale.'}")
    token = resolve_token(token)
    repo = entry.get("repo_id", "")
    if not repo and "civitai.com" in str(entry.get("url") or ""):
        from core import fast_downloader as _fd
        if not _fd.civitai_key():
            raise DownloadUnavailable(
                f"'{entry.get('id')}' si scarica da Civitai, che chiede il login: inserisci la chiave API "
                f"Civitai nel tab «Pesi modelli» (civitai.com → Impostazioni account → API Keys).")
    if repo and entry.get("gated"):
        check_gated(repo, token)

    from core import fast_downloader as fd
    dest_dir = wc.dest_for(entry, base_dir)
    if entry.get("kind") == "file":
        fname = entry.get("filename", "")
        if not fname:
            raise DownloadUnavailable(f"'{entry.get('id')}': kind=file senza filename")
        out = Path(dest_dir) / fname
        remote = entry.get("remote_filename") or fname
        if entry.get("convert"):
            # il file ComfyUI è GIÀ su disco (copiato a mano, collegato dall'installer, download interrotto dopo
            # la fine dello scarico): solo conversione nativa in locale, niente rete
            from core import native_convert as nc
            _src = nc.source_path(entry, Path(dest_dir))
            if _src.exists() and nc.is_complete(_src):
                done = nc.ensure_entry(entry, Path(dest_dir), log=log)
                if done is not None and done.exists():
                    verify_file(done)
                    return done
        # voce `convert`: si scarica il file ComfyUI col suo nome, poi diventa il file NATIVO `filename`
        # (core/native_convert: verifica e cancella il sorgente)
        target = out
        if entry.get("convert"):
            from core import native_convert as nc
            target = nc.source_path(entry, Path(dest_dir))
        prog = (lambda n, _f=fname: progress(n, _f)) if progress else None
        if repo:
            fd.download_hf_file(repo, remote, str(target), token=token, progress=prog)
        elif entry.get("url"):
            fd.download_url(entry["url"], str(target), token=token, progress=prog)
        else:                                            # downloadable() lo esclude, ma difendiamoci
            raise DownloadUnavailable(f"'{entry.get('id')}': nessuna fonte remota")
        verify_file(target, sha256=entry.get("sha256"))
        if entry.get("convert"):
            log(f"[native] conversione in formato nativo: {target.name}")
            nc.convert(entry["convert"], target, out, delete_source=not entry.get("keep_source"), log=log)
            verify_file(out)
        return out
    # snapshot
    if not repo:
        raise DownloadUnavailable(f"'{entry.get('id')}': snapshot senza repo_id")
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    files = fd.download_hf_repo(repo, str(dest_dir), allow=entry.get("allow_patterns"),
                                token=token,
                                progress=(lambda n, f: progress(n, f)) if progress else None)
    if not files:
        raise IntegrityError(f"snapshot vuoto: {repo} (allow_patterns troppo stretti?)")
    for f in files:                                      # struttura dei pesi scaricati
        if str(f).lower().endswith((".safetensors", ".gguf")):
            verify_file(f)
    return Path(dest_dir)


def plan_ids(ids: list[str], *, enforce_matrix: bool = True) -> list[dict]:
    """Espande gli id con le dipendenze ricorsive (deps PRIMA), dedup, in ordine di download.
    Le dipendenze rispettano la matrice dell'edizione: per «a|b» si sceglie l'alternativa consentita; se
    nessuna lo e' → `wc.DepNotAllowedError` (mai un download silenzioso di cio' che l'edizione non include).
    `enforce_matrix=False` per gli strumenti di sviluppo."""
    by_id = {e["id"]: e for e in wc.load()}
    out: list[dict] = []
    seen: set[str] = set()

    def visit(eid: str) -> None:
        if eid in seen or eid not in by_id:
            return
        seen.add(eid)
        for d in by_id[eid].get("deps", []):
            visit(wc.resolve_dep(d, by_id, strict=enforce_matrix, owner=eid))   # «a|b»: presente/consentita
        out.append(by_id[eid])

    for i in ids:
        visit(i)
    return out


# ─────────────────────────── coda SERIALE con pausa/riprendi ───────────────────────────
class DownloadQueue:
    """Un download alla volta (connessione metered), pausa/riprendi/annulla TRA i file.
    Callbacks (tutti opzionali, chiamati dal thread della coda):
      on_start(entry) · on_bytes(entry_id, nbytes_delta, filename) · on_done(entry, path)
      on_error(entry, exc) · on_idle()"""

    def __init__(self, *, base_dir: str | None = None, token: str | None = None,
                 on_start=None, on_bytes=None, on_done=None, on_error=None, on_idle=None):
        self._items: list[dict] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self.base_dir, self.token = base_dir, token
        self.on_start, self.on_bytes = on_start, on_bytes
        self.on_done, self.on_error, self.on_idle = on_done, on_error, on_idle
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, ids: list[str]) -> list[str]:
        entries = [e for e in plan_ids(ids) if wc.downloadable(e) and not wc.is_present(e, self.base_dir)]
        with self._lock:
            have = {e["id"] for e in self._items}
            fresh = [e for e in entries if e["id"] not in have]
            self._items.extend(fresh)
        self._wake.set()
        return [e["id"] for e in fresh]

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()
        self._wake.set()

    def cancel_pending(self) -> None:
        """Svuota la coda (il file in corso finisce: il taglio duro è il kill del processo)."""
        with self._lock:
            self._items.clear()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def pending(self) -> list[str]:
        with self._lock:
            return [e["id"] for e in self._items]

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.5)
                continue
            with self._lock:
                e = self._items.pop(0) if self._items else None
            if e is None:
                if self.on_idle:
                    try:
                        self.on_idle()
                    except Exception:                    # noqa: BLE001
                        pass
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            try:
                if self.on_start:
                    self.on_start(e)
                path = download_entry(
                    e, base_dir=self.base_dir, token=self.token,
                    progress=(lambda n, f, _id=e["id"]: self.on_bytes and self.on_bytes(_id, n, f)))
                if self.on_done:
                    self.on_done(e, path)
            except Exception as ex:                      # noqa: BLE001
                if self.on_error:
                    try:
                        self.on_error(e, ex)
                    except Exception:                    # noqa: BLE001
                        pass
