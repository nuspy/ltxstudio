"""LongCat-Video-Avatar 1.5: foto (o solo testo) + voce → video di una persona che parla.

Motore STANDALONE nel venv `E:\\AI\\Tools\\LongCatAvatar\\venv` (codice ufficiale meituan-longcat, MIT),
pilotato via subprocess con workers/longcat_avatar_infer.py. Stesso schema di core/trellis_service:
is_installed / preflight / make_job / kill_orphans, ambiente isolato dal venv dell'app.
La durata del video la decide l'audio (core.avatar.timing).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from core.music_service import Job          # job subprocess generico (lock GPU inter-processo incluso)

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
ROOT = TOOLS / "LongCatAvatar"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "workers" / "longcat_avatar_infer.py"
OUT_DIR = REPO / "output" / "avatar"

PRECISIONS = ("int8", "fp8", "bf16")
PRECISION_IDS = {"int8": "lcav15-int8", "fp8": "lcav15-fp8", "bf16": "lcav15-bf16"}
RESOLUTIONS = ("480p", "720p")
ASPECTS = ("16:9", "9:16", "1:1", "4:3", "3:4")
AUDIO_TYPES = ("add", "para")


def kill_orphans() -> int:
    """Killa subprocess longcat_avatar_infer orfani (crash precedente dell'app)."""
    from core.proc_utils import kill_orphans as _ko
    return _ko("longcat_avatar_infer")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "avatar").strip()) or "avatar"


def is_installed() -> bool:
    """Marker scritto dall'installer solo dopo l'import riuscito del codice ufficiale."""
    return VENV_PY.exists() and (ROOT / ".ltx_ready").exists()


def weight_ids(precision: str) -> list[str]:
    """Voci di catalogo che servono per questa precisione (la voce DiT tira le dipendenze)."""
    return [PRECISION_IDS.get(precision, "lcav15-int8")]


def missing_weights(precision: str) -> list[str]:
    """ID di catalogo mancanti (DiT scelto + dipendenze), in ordine di download."""
    try:
        from core import weights_catalog as wc
        from core import model_paths as mp
        e = mp.entry(PRECISION_IDS.get(precision, ""))
        return wc.missing_ids_with_deps(e, str(mp.models_root())) if e else [PRECISION_IDS.get(precision, "?")]
    except Exception:                                                # noqa: BLE001
        return []


def installed_precisions() -> list[str]:
    """Precisioni con TUTTI i pesi presenti (per popolare la combo della GUI e /avatar/options)."""
    return [p for p in PRECISIONS if not missing_weights(p)]


def output_path(name: str = "") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{_safe(name or 'avatar')}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


def preflight(params: dict) -> list[str]:
    errs: list[str] = []
    if not is_installed():
        errs.append("Motore LongCat Avatar non installato: python scripts/install_longcat_avatar.py")
    prec = params.get("precision", "int8")
    if prec not in PRECISIONS:
        errs.append(f"Precisione sconosciuta: {prec}")
    else:
        miss = missing_weights(prec)
        if miss:
            errs.append("Pesi mancanti: " + ", ".join(miss) + " — scaricali dal tab «Pesi modelli».")
    mode = params.get("mode", "ai2v")
    if mode == "ai2v":
        ip = params.get("image_path") or ""
        if not ip or not Path(ip).is_file():
            errs.append(f"Immagine iniziale inesistente: {ip or '(nessuna)'}")
    elif mode != "at2v":
        errs.append(f"Modalità sconosciuta: {mode}")
    ap = params.get("audio_path") or ""
    if not ap or not Path(ap).is_file():
        errs.append(f"Audio inesistente: {ap or '(nessuno)'}")
    if params.get("audio2_path"):
        if not Path(params["audio2_path"]).is_file():
            errs.append(f"Audio persona 2 inesistente: {params['audio2_path']}")
        if mode != "ai2v":
            errs.append("La modalità due persone richiede un'immagine iniziale.")
    return errs


def make_job(params: dict, **cbs) -> Job:
    """Job subprocess (non avviato). params: vedi workers/longcat_avatar_infer.DEFAULTS."""
    kill_orphans()
    p = dict(params)
    p.setdefault("output", str(output_path(p.get("name", ""))))
    p["parent_pid"] = os.getpid()                    # watchdog: se l'app muore, muore anche il worker
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pj = OUT_DIR / f"_params_{int(time.time() * 1000)}.json"
    pj.write_text(json.dumps(p, ensure_ascii=False, indent=1), "utf-8")
    clean_path = os.pathsep.join(x for x in os.environ.get("PATH", "").split(os.pathsep)
                                 if x and "comfy" not in x.lower())
    env: dict = {"PYTHONPATH": None, "PYTHONHOME": None, "PYTHONSTARTUP": None, "PATH": clean_path,
                 "LONGCAT_CODE_DIR": str(ROOT / "LongCat-Video"), "PYTHONIOENCODING": "utf-8"}
    job = Job([VENV_PY, "-u", str(WORKER), str(pj)], ROOT, env=env, **cbs)
    job.output = p["output"]
    job.params_file = str(pj)
    return job
