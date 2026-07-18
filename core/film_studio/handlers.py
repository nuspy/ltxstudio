"""Handler del Job Scheduler mappati sul Generation Service nativo (immagine/video).

Estratti dalla finestra UI in CORE così che possano essere registrati anche in modalità headless
(Lucky Director) senza aprire la finestra Film Studio. Idempotente.
"""
from __future__ import annotations

from core.film_studio import gen_service, scheduler


def _img(p: dict):
    kw = {}
    if p.get("width"):
        kw["width"] = int(p["width"])
    if p.get("height"):
        kw["height"] = int(p["height"])
    return gen_service.gen_image(p["prompt"], p["out"], face_ref=p.get("face_ref", ""),
                                 ref_images=p.get("ref_images"), loras=p.get("loras"),
                                 arch=p.get("arch", "flux"), model=p.get("model", ""),
                                 negative=p.get("negative", ""), seed=int(p.get("seed", 7)),
                                 coherence_engine=p.get("coherence_engine", "flux"),
                                 grounding_px=int(p.get("grounding_px", 1024)),
                                 krea2_variant=p.get("krea2_variant", "turbo"), **kw)


def _vid(p: dict):
    keyframes = p.get("keyframes")
    cf = p.get("chain_from")                                  # raccordo: ultimo frame del video precedente
    if cf and not keyframes:
        from core.film_studio import reference_chain
        link_png = str(p["out"]).rsplit(".", 1)[0] + "_link.png"
        keyframes = reference_chain.prev_shot_link(cf, link_png, strength=0.35)
    kw = {}
    if p.get("width"):
        kw["width"] = int(p["width"])
    if p.get("height"):
        kw["height"] = int(p["height"])
    return gen_service.gen_shot_video(p["prompt"], p["out"], duration_s=p.get("duration_s", 4),
                                      keyframes=keyframes, loras=p.get("loras"),
                                      audio_path=p.get("audio_path", "") or "",
                                      model_gguf=p.get("model_gguf", ""),
                                      negative=p.get("negative", ""), seed=int(p.get("seed", 7)),
                                      video_arch=p.get("video_arch", "ltx"),
                                      lingbot_variant=p.get("lingbot_variant", "dense"),
                                      run_refiner=bool(p.get("run_refiner")), **kw)


def register_default_handlers() -> None:
    """Registra gli handler image/video sul scheduler (idempotente: register sovrascrive)."""
    scheduler.register("image", _img)
    scheduler.register("video", _vid)
