"""Bus preview/progresso per-step tra il backend (image_flux) e l'endpoint /preview.

I job dell'InferenceServer girano SERIALI (un solo job GPU alla volta), quindi un unico stato globale
basta. Due canali:
- PROGRESSO (step/total): aggiornato ad OGNI step (leggerissimo) → barra credibile.
- PREVIEW (png): aggiornato ogni ~1/8 degli step (decode latent2rgb) → immagine dell'anteprima.
La GUI polla GET /preview: legge sempre step/total dagli header; ridisegna il PNG solo quando
`png_seq` cambia. Best-effort: non deve MAI rompere la generazione.

ISOLAMENTO IN SUBPROCESS: il job gira in un PROCESSO figlio (job_runner) separato dal server HTTP —
torch e uvicorn non convivono mai (anti-crash). Il figlio e il server NON condividono lo stato globale:
se `LTX2_PREVIEW_DIR` è impostato (nel figlio), preview_bus scrive lo stato su FILE là dentro
(preview.png + preview.json), e l'endpoint /preview del server li LEGGE. Nel server (senza quell'env)
`get_preview` legge dal file; nel figlio scrive. In-process (test) funziona ancora via lo stato globale.
"""
from __future__ import annotations

import json
import os
import threading

_lock = threading.Lock()
_latest = {"step": 0, "total": 0, "png_seq": 0, "png": b"", "msg": ""}

_PREVIEW_DIR = os.environ.get("LTX2_PREVIEW_DIR") or ""     # set solo nel processo figlio (job_runner)
_PNG_PATH = os.path.join(_PREVIEW_DIR, "preview.png") if _PREVIEW_DIR else ""
_META_PATH = os.path.join(_PREVIEW_DIR, "preview.json") if _PREVIEW_DIR else ""


def _write_meta() -> None:
    try:
        tmp = _META_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: _latest[k] for k in ("step", "total", "png_seq", "msg")}, f)
        os.replace(tmp, _META_PATH)
    except Exception:                                       # noqa: BLE001
        pass


def set_message(msg: str) -> None:
    """Fase corrente (es. 'Caricamento componenti…', 'Generazione…', 'Decodifica…')."""
    with _lock:
        _latest["msg"] = str(msg or "")
        if _META_PATH:
            _write_meta()


def set_progress(step: int, total: int) -> None:
    """Aggiorna solo il progresso (ogni step, senza decodificare l'anteprima)."""
    with _lock:
        _latest["step"] = int(step)
        _latest["total"] = int(total)
        if _META_PATH:
            _write_meta()


def set_preview(png: bytes, step: int, total: int) -> None:
    """Aggiorna l'anteprima (PNG) + il progresso. png_seq cresce → la GUI ridisegna."""
    with _lock:
        _latest["step"] = int(step)
        _latest["total"] = int(total)
        if png:
            _latest["png_seq"] += 1
            _latest["png"] = png
            if _PNG_PATH:
                try:
                    tmp = _PNG_PATH + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(png)
                    os.replace(tmp, _PNG_PATH)
                except Exception:                           # noqa: BLE001
                    pass
        if _META_PATH:
            _write_meta()


def get_preview() -> dict:
    """Ritorna {step, total, png_seq, png}. Nel SERVER (LTX2_PREVIEW_DIR non impostato ma i file esistono)
    legge lo stato scritto dal processo figlio; altrimenti (in-process) ritorna lo stato globale."""
    srv_dir = os.environ.get("LTX2_SERVER_PREVIEW_DIR") or ""
    if srv_dir:
        meta_p = os.path.join(srv_dir, "preview.json")
        png_p = os.path.join(srv_dir, "preview.png")
        try:
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            png = b""
            if meta.get("png_seq"):
                try:
                    with open(png_p, "rb") as f:
                        png = f.read()
                except Exception:                           # noqa: BLE001
                    png = b""
            return {"step": int(meta.get("step", 0)), "total": int(meta.get("total", 0)),
                    "png_seq": int(meta.get("png_seq", 0)), "png": png, "msg": meta.get("msg", "")}
        except Exception:                                   # noqa: BLE001 — nessun job attivo / file assente
            return {"step": 0, "total": 0, "png_seq": 0, "png": b"", "msg": ""}
    with _lock:
        return dict(_latest)


def reset() -> None:
    """Azzera all'inizio di un nuovo job (niente anteprima/progresso del job precedente)."""
    with _lock:
        _latest.update(step=0, total=0, png_seq=0, png=b"", msg="")
        if _META_PATH:
            _write_meta()
    # nel server: azzera anche i file del figlio precedente
    srv_dir = os.environ.get("LTX2_SERVER_PREVIEW_DIR") or ""
    if srv_dir:
        try:
            with open(os.path.join(srv_dir, "preview.json"), "w", encoding="utf-8") as f:
                json.dump({"step": 0, "total": 0, "png_seq": 0, "msg": ""}, f)
        except Exception:                                   # noqa: BLE001
            pass
