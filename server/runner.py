"""Runner per le REST API: invia un grafo a ComfyUI, attende il completamento, raccoglie gli
output. Gestisce gli input come FILE caricato (multipart) OPPURE come PATH locale sul server.

Riusa core.comfy_driver. Tutti gli output finiscono in OUTPUT_DIR e sono scaricabili via /download.
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from core.comfy_driver import ComfyDriver

OUTPUT_DIR = Path(r"E:\AI\ltxstudio\output")
UPLOAD_DIR = Path(r"E:\AI\ltxstudio\server\_uploads")
COMFY_INPUT = Path(r"E:\AI\Comfy_Nexus\ComfyUI\input")
VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov", ".gif")

_driver = ComfyDriver()


def driver() -> ComfyDriver:
    return _driver


def _save_upload(data: bytes, suffix: str = "") -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    p.write_bytes(data)
    return str(p)


def resolve_local(upload_bytes: bytes | None, filename: str | None, path: str | None) -> str | None:
    """Ritorna un path LOCALE sul server: dal file caricato (salvato) o dal path indicato.
    Solleva se il path indicato non esiste."""
    if upload_bytes is not None:
        suffix = Path(filename or "").suffix
        return _save_upload(upload_bytes, suffix)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path non trovato sul server: {path}")
        return str(p)
    return None


def to_comfy_image(local_path: str) -> str:
    """Carica un'immagine nella input dir di ComfyUI e ritorna il nome (subfolder/name)."""
    r = _driver.upload_image(local_path)
    name = r.get("name", ""); sub = r.get("subfolder", "")
    return f"{sub}/{name}" if sub else name


def to_comfy_video(local_path: str) -> str:
    """Copia un video nella input dir di ComfyUI (VHS_LoadVideo carica per nome)."""
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT / Path(local_path).name
    if str(dest) != str(local_path):
        shutil.copy2(local_path, dest)
    return Path(local_path).name


def upload_comfy_mask(mask_local: str, ref_comfy_name: str) -> None:
    """Carica una maschera associata a un'immagine già caricata in ComfyUI."""
    name = ref_comfy_name.split("/")[-1]
    sub = ref_comfy_name.rsplit("/", 1)[0] if "/" in ref_comfy_name else ""
    _driver.upload_mask(mask_local, name, sub, "input")


def run_graph(prompt: dict, *, timeout: int = 1200) -> dict:
    """Invia il grafo, attende il completamento, scarica gli output in OUTPUT_DIR.
    Ritorna {prompt_id, files:[paths], videos:[paths], images:[paths], node_count}."""
    if not _driver.is_up():
        raise RuntimeError("ComfyUI non è in esecuzione (http://127.0.0.1:8188).")
    pid = _driver.submit(prompt)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = _driver.history(pid) or {}
        entry = h.get(pid)
        if entry:
            st = entry.get("status", {})
            if st.get("status_str") == "error" or st.get("completed") is False and st.get("messages"):
                # prosegui: lo stato finale arriva con completed True/False
                pass
            if st.get("completed") or st.get("status_str") == "success":
                files, videos, images = [], [], []
                for out in entry.get("outputs", {}).values():
                    for key in ("gifs", "videos", "images", "audio"):
                        for item in out.get(key, []) or []:
                            fn = item.get("filename")
                            if not fn:
                                continue
                            data = _driver.download_view(fn, item.get("subfolder", ""), item.get("type", "output"))
                            dest = OUTPUT_DIR / fn; dest.write_bytes(data)
                            files.append(str(dest))
                            if fn.lower().endswith(VIDEO_EXT):
                                videos.append(str(dest))
                            elif fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                                images.append(str(dest))
                return {"prompt_id": pid, "node_count": len(prompt),
                        "files": files, "videos": videos, "images": images}
        time.sleep(1.5)
    raise TimeoutError(f"Timeout ({timeout}s) in attesa del job {pid}.")
