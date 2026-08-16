"""Restauro volti per-fotogramma su un video (CodeFormer / GFPGAN via facerestore_cf).

A differenza di FaceDetailer (che RI-GENERA il volto con un modello di diffusione, rischiando
di cambiare ancora l'identità), il restauro CodeFormer/GFPGAN RICOSTRUISCE il volto preservando
l'identità e correggendo l'aspetto "plastico"/sfocato — ideale come passaggio finale su un
video LTX. `codeformer_fidelity`: 0 = massima ricostruzione (più nitido, può scostarsi),
1 = massima fedeltà all'input (preserva di più). ~0.7 è un buon compromesso.

Richiede il custom node `facerestore_cf` (nodi FaceRestoreModelLoader + FaceRestoreCFWithModel).
Se assente, `node_installed()` è False e l'UI offre l'installazione.
"""
from __future__ import annotations

from pathlib import Path

# repo del custom node (CodeFormer/GFPGAN per ComfyUI)
FACERESTORE_REPO = {
    "url":  "https://github.com/mav-rik/facerestore_cf.git",
    "name": "facerestore_cf",
    "note": "Restauro volti CodeFormer/GFPGAN. Scarica i modelli (~350MB) al primo uso.",
}

# modelli di restauro selezionabili. NB: il nodo NON li scarica da solo — vanno messi in
# ComfyUI/models/facerestore_models (altrimenti la lista del nodo è vuota → validation fail).
# `ensure_model()` li scarica on-demand dalle release ufficiali col nome atteso qui sotto.
FACERESTORE_MODELS = ["codeformer-v0.1.0.pth", "GFPGANv1.4.pth", "GFPGANv1.3.pth"]
FACE_DETECTORS = ["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"]

FACERESTORE_MODEL_URLS = {
    "codeformer-v0.1.0.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}


def models_dir() -> "Path | None":
    """Cartella ComfyUI/models/facerestore_models (da cui il nodo legge la lista)."""
    from core.tts_engine import custom_nodes_root
    root = custom_nodes_root()      # …/ComfyUI/custom_nodes
    if root is None:
        return None
    return root.parent / "models" / "facerestore_models"


def available_models() -> list[str]:
    d = models_dir()
    if d and d.exists():
        return sorted(f.name for f in d.glob("*.pth"))
    return []


def ensure_model(name: str, *, log=print, confirm=None) -> tuple[bool, str]:
    """Garantisce che il modello `name` sia presente in models/facerestore_models; se manca e
    ne conosciamo l'URL, lo scarica COL CONSENSO. Ritorna (ok, messaggio).

    weights-unification C: prima scaricava ~350MB da GitHub senza chiedere (connessione
    metered). `confirm(nome, mb)` -> bool: la GUI passa il dialog; None (headless) = consenso
    implicito del chiamante. Il download passa dal motore unico (resume + retry) invece di
    urlretrieve (zero resume)."""
    d = models_dir()
    if d is None:
        return False, "Cartella ComfyUI/models non trovata."
    dest = d / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return True, f"{name} già presente."
    url = FACERESTORE_MODEL_URLS.get(name)
    if not url:
        return False, (f"Modello «{name}» non presente e senza URL noto. Scaricalo a mano in "
                       f"{d}.")
    if confirm is not None and not confirm(name, 350):
        return False, f"Download di {name} annullato dall'utente."
    try:
        d.mkdir(parents=True, exist_ok=True)
        log(f"⬇ Scarico {name} (~350MB)…")
        from core import fast_downloader
        fast_downloader.download_url(url, str(dest))
        return True, f"✓ {name} scaricato."
    except Exception as e:
        return False, f"❌ Download di {name} fallito: {e}"


def _comfy_node_present(candidates: tuple[str, ...]) -> bool:
    """True se uno dei nomi nodo è registrato in ComfyUI (scan custom_nodes)."""
    from core.tts_engine import _comfy_node_present as _scan   # riusa lo scanner + cache
    return _scan(candidates)


def node_installed() -> bool:
    return _comfy_node_present(("FaceRestoreCFWithModel", "FaceRestoreModelLoader"))


def install_node(*, log=print) -> tuple[bool, str]:
    """Clona facerestore_cf in custom_nodes + pip install requirements. Riusa il meccanismo
    generico di core.tts_engine (git clone + venv pip + invalidate cache)."""
    import shutil
    import subprocess
    from core.tts_engine import custom_nodes_root, venv_python, invalidate_engine_cache
    if shutil.which("git") is None:
        return False, "❌ Git non trovato nel PATH. Installa Git da https://git-scm.com/."
    root = custom_nodes_root()
    if root is None:
        return False, "❌ Cartella ComfyUI/custom_nodes non trovata."
    target = root / FACERESTORE_REPO["name"]
    log_lines = [f"📦 Installazione facerestore_cf → {target}"]
    try:
        if target.exists():
            res = subprocess.run(["git", "pull"], cwd=str(target),
                                 capture_output=True, text=True, timeout=120)
            log_lines.append(res.stdout.strip() or res.stderr.strip() or "(git pull ok)")
        else:
            res = subprocess.run(["git", "clone", FACERESTORE_REPO["url"], str(target)],
                                 capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                return False, "\n".join(log_lines + [f"❌ git clone fallito:\n{res.stderr.strip()}"])
            log_lines.append("✓ clonato")
        req = target / "requirements.txt"
        py = venv_python()
        if req.exists() and py:
            res = subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)],
                                 capture_output=True, text=True, timeout=900)
            log_lines.extend((res.stdout + res.stderr).strip().splitlines()[-6:])
            if res.returncode != 0:
                return False, "\n".join(log_lines + ["❌ pip install fallito (vedi sopra)."])
        invalidate_engine_cache()
        log_lines.append("\n✅ Installato. Riavvia ComfyUI per caricare il nodo.")
        return True, "\n".join(log_lines)
    except subprocess.TimeoutExpired:
        return False, "\n".join(log_lines + ["❌ Timeout."])
    except Exception as e:
        return False, "\n".join(log_lines + [f"❌ Errore: {e}"])


class FaceRestorePipeline:
    def build(self, video_name: str, *, fps: int = 24, fidelity: float = 0.7,
              model: str = "codeformer-v0.1.0.pth", detection: str = "retinaface_resnet50",
              out_format: str = "video/h264-mp4") -> dict:
        """Carica il video, restaura i volti frame-by-frame, ri-salva mantenendo l'audio."""
        fidelity = max(0.0, min(1.0, float(fidelity)))
        p: dict = {}
        p["1"] = {"class_type": "VHS_LoadVideo",
                  "inputs": {"video": video_name, "force_rate": 0, "custom_width": 0,
                             "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
                             "select_every_nth": 1, "format": "AnimateDiff"}}
        p["2"] = {"class_type": "FaceRestoreModelLoader", "inputs": {"model_name": model}}
        p["3"] = {"class_type": "FaceRestoreCFWithModel",
                  "inputs": {"facerestore_model": ["2", 0], "image": ["1", 0],
                             "facedetection": detection, "codeformer_fidelity": fidelity}}
        # ricombina; tieni l'audio originale del video (VHS_LoadVideo slot 2 = AUDIO)
        p["4"] = {"class_type": "VHS_VideoCombine",
                  "inputs": {"images": ["3", 0], "audio": ["1", 2], "frame_rate": int(fps),
                             "loop_count": 0, "filename_prefix": "ltxstudio_facerestore",
                             "format": out_format, "pingpong": False, "save_output": True}}
        return p
