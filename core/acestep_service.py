r"""ACE-Step XL 1.5 — generazione musica/canzoni STANDALONE, niente ComfyUI (alternativa a HeartMuLa).

venv isolato `E:\AI\Tools\ACE-Step\venv` (repo ace-step/ACE-Step) via subprocess. Apache-2.0.
"""
from __future__ import annotations
import re
from pathlib import Path
from core.music_service import Job

TOOLS = Path(r"E:\AI\Tools")
AS_DIR = TOOLS / "ACE-Step"
AS_VENV_PY = AS_DIR / "venv" / "Scripts" / "python.exe"
GEN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acestep_generate.py"
OUT_DIR = Path(r"E:\AI\ltxstudio\output\audio")
# cache HF dei pesi ACE-Step (dove li scarichiamo). Serve a verificare che il download sia COMPLETO.
ACE_CACHE = Path.home() / ".cache" / "ace-step" / "checkpoints"
ACE_REPO_DIR = ACE_CACHE / "models--ACE-Step--ACE-Step-v1-3.5B"


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "song").strip()) or "song"


def _weights_ready() -> bool:
    """True se i pesi ACE-Step sono COMPLETI (nessun .incomplete + transformer presente).
    Se la cache non è nel percorso noto (modello gestito altrove dal runtime) non blocca → True."""
    if not ACE_REPO_DIR.exists():
        return True                                          # scaricato altrove → non bloccare
    if any(ACE_REPO_DIR.rglob("*.incomplete")):
        return False                                         # download interrotto
    blobs = (ACE_REPO_DIR / "blobs")
    if blobs.exists() and any(b.is_file() and b.stat().st_size > 5_000_000_000 for b in blobs.iterdir()):
        return True                                          # transformer ~6.6GB presente
    return bool(list(ACE_REPO_DIR.rglob("*.safetensors")))


def _tool_ready() -> bool:
    return AS_VENV_PY.exists() and (AS_DIR / ".ltx_ready").exists()


def is_installed() -> bool:
    return _tool_ready() and _weights_ready()


def make_job(tags: str, lyrics: str = "", out_name: str = "song", *, duration: float = 60.0,
             steps: int = 27, guidance: float = 15.0, lazy: bool = False, **cbs) -> Job:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{_safe(out_name)}.wav"
    cmd = [AS_VENV_PY, "-u", str(GEN_SCRIPT), "--tags", tags, "--lyrics", lyrics or "",
           "--duration", float(duration), "--steps", int(steps), "--guidance", float(guidance),
           "--output", str(out)]
    if lazy:                                              # low-VRAM: CPU offload dei componenti
        cmd.append("--lazy")
    return Job(cmd, AS_DIR, **cbs)


def preflight(tags: str) -> list[str]:
    errs = []
    if not _tool_ready():
        errs.append("ACE-Step non installato: usa l'installer (scripts/install_acestep.py).")
    elif not _weights_ready():
        errs.append("ACE-Step: pesi del modello INCOMPLETI (download interrotto). Ri-scarica con "
                    "`hf download ACE-Step/ACE-Step-v1-3.5B --cache-dir ~/.cache/ace-step/checkpoints`.")
    if not (tags or "").strip():
        errs.append("Tag/stile vuoti.")
    return errs
