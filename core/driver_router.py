"""Router unico di generazione: instrada al backend NATIVO o a ComfyUI secondo `backend.is_native`.

Obiettivo migrazione (#28): un solo punto che decide il backend per modulo, così i chiamanti (GUI,
Film Studio gen_service) non duplicano la logica. In modalità nativa gli handler leggono path LOCALI
(niente upload/download via ComfyUI).

API:
  is_native(module) -> bool
  generate(module, native_params, *, comfy=None) -> dict   # {status, output:[paths]}
  prepare_input(path) -> str        # path locale per il nativo (no upload); upload se serve comfy
"""
from __future__ import annotations

from core import backend, native_driver

# moduli serviti dal server nativo (inference_server/server.py HANDLERS)
NATIVE_MODULES = {"image", "video", "upscale", "segment", "video_upscale", "video_face",
                  "video_fps", "video_edit", "video_long"}


def is_native(module: str) -> bool:
    return backend.is_native(module)


def native_up() -> bool:
    try:
        return native_driver.is_up()
    except Exception:                                       # noqa: BLE001
        return False


def generate(module: str, native_params: dict, *, timeout: float = 3600, comfy=None) -> dict:
    """Esegue una generazione sul backend giusto. Se il modulo è nativo (config) usa l'InferenceServer
    e ritorna {status, output:[paths]}. Altrimenti, se `comfy` (callable) è fornito, lo invoca."""
    if is_native(module):
        return native_driver.run_and_wait(module, native_params, timeout=timeout)
    if comfy is not None:
        return comfy()
    raise RuntimeError(f"modulo '{module}' non nativo e nessun fallback comfy fornito")


def prepare_input(path: str, *, comfy_upload=None) -> str:
    """Path d'ingresso per la generazione: in nativo (o comfy giù) ritorna il path LOCALE; altrimenti
    usa `comfy_upload(path)` per caricarlo su ComfyUI."""
    if comfy_upload is None or native_up():                 # nativo disponibile → path locale
        return path
    try:
        return comfy_upload(path)
    except Exception:                                       # noqa: BLE001
        return path
