"""Gating feature-matrix per la REST API — STESSE attivazioni/disattivazioni della GUI desktop.

Il client mobile deve rispettare il feature matrix identico al desktop: `GET /features` espone lo stato
(abilitato + upsell) di ogni capability, e gli endpoint applicano `require(fid)` (403 con info upsell se
disabilitato). In dev (manifest assente) è tutto fail-open, come nella GUI.
"""
from __future__ import annotations

from fastapi import HTTPException

# capability note all'app mobile (schermata → capability), stesse del feature matrix desktop.
KNOWN_CAPS = [
    "cap.gen_images", "cap.inpaint", "cap.krea2_edit", "cap.sam", "cap.upscale_image", "cap.restore",
    "cap.video", "cap.video_lingbot", "cap.controlnet", "cap.extend_retake",
    "cap.audio_tts_clone", "cap.music_song", "cap.music_sonify", "cap.music_video",
    "cap.lora_image_train", "cap.lora_video_train",
    "cap.adult_mode", "cap.stego", "cap.threed", "cap.filmstudio", "cap.lucky_director",
    "cap.import_models", "cap.civitai_import",
    "cap.avatar_longcat", "cap.avatar_h3", "cap.image_qwen21", "cap.video_minimax_h3",
    "cap.qwen21_controlnet", "cap.qwen21_prompt_enhancer",
    "cap.video_restore_iclora", "cap.video_outpaint", "cap.video_restyle", "cap.video_camera_ic",
    "cap.video_lipdub", "cap.video_hdr_ic", "cap.video_motion_track", "cap.video_h3_upsampler",
    "cap.video_inpaint_r2v",
    "cap.image_controlnet", "cap.image_redux", "cap.image_cn_inpaint", "cap.image_magicquill",
]


def is_on(fid: str) -> bool:
    try:
        from core import features
        return bool(features.is_enabled(fid))
    except Exception:                                        # noqa: BLE001
        return True                                         # fail-open come la GUI in dev


def _info(fid: str) -> dict:
    try:
        from core import features
        return features.manifest().get("features", {}).get(fid, {}) or {}
    except Exception:                                        # noqa: BLE001
        return {}


def require(fid: str) -> None:
    """Solleva 403 (con info upsell) se la capability è disabilitata nell'edizione. No-op in dev."""
    if is_on(fid):
        return
    info = _info(fid)
    raise HTTPException(403, {
        "error": "feature_disabled", "capability": fid,
        "i18n_key": info.get("i18n_key", ""), "upsell": info.get("upsell"),
    })


def snapshot() -> dict:
    """Stato completo del feature matrix per l'app (edizione + enabled/upsell per capability)."""
    try:
        from core import features
        edition = features.edition()
        allows_adult = bool(features.allows_adult())
    except Exception:                                        # noqa: BLE001
        edition, allows_adult = "dev", True
    feats = {}
    for c in KNOWN_CAPS:
        info = _info(c)
        feats[c] = {"enabled": is_on(c), "i18n_key": info.get("i18n_key", ""),
                    "upsell": info.get("upsell")}
    return {"edition": edition, "allows_adult": allows_adult, "features": feats}
