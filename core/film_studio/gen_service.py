"""Generation Service dell'AI Film Studio — facciata sul backend NATIVO (zero ComfyUI).

Espone le primitive che servono allo storyboard/scenografia, risolvendo i due percorsi di coerenza:
- Rapido: immagini di riferimento dell'asset + PuLID (volto) + multi-ref.
- LoRA: carica le LoRA degli @Nome (trigger) con peso; keyframe come conditioning aggiuntivo.

Immagini → core/native_driver "image"; video → worker ltx2 (riusa _NativeLtx2Worker tramite subprocess
del worker, qui via native? no: il video nativo gira col worker ltx2_infer come negli altri moduli).
@Reference: (project, "@Nome#Variante") → AssetVariation → {ref images | Lora+trigger}.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

from core import native_driver
from core.film_studio import db

LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
LTX2_WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "workers", "ltx2_infer.py")
GGUF = r"E:\AI\Models\diffusion_models\LTXVideo\LTX-2\quantstack\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
NEG = "blurry, low quality, deformed, distorted face, extra limbs, watermark, text"

_REF_RE = re.compile(r"@([A-Za-z0-9_]+)(?:#([A-Za-z0-9_]+))?")


# ─────────────── @Reference resolver ───────────────
def resolve_refs(text: str, path_mode: str = "fast", flavor: str = "image") -> dict:
    """Estrae gli @Nome[#Variante] dal testo e li risolve in {ref_images, loras, trigger_map, clean_text}.
    path_mode: 'fast' (immagini+PuLID) | 'lora' (LoRA+trigger) | 'mixed' (per-asset).
    flavor: 'image' (keyframe) | 'video' (shot) → sceglie la LoRA del flavor giusto (B2: prima si caricava
    sempre flavor='image' anche per gli shot video → LoRA video mai usate).
    #Variante ora è ONORATA (B1): immagini+LoRA della VARIAZIONE (es. @Marco#insanguinato), con fallback
    all'asset se la variante non ha propri asset."""
    refs, ref_images, loras, trigger_map = [], [], [], {}
    for m in _REF_RE.finditer(text or ""):
        name, variant = m.group(1), m.group(2)
        assets = db.find_assets(name_like=name)
        asset = next((a for a in assets if a["name"].lower() == name.lower()), assets[0] if assets else None)
        if not asset:
            continue
        var = None                                           # risolvi la variante citata (#Variante)
        if variant:
            vrows = db.query("asset_variation", "asset_id=? AND name=?", (asset["id"], variant))
            var = vrows[0] if vrows else None
        refs.append((asset, variant))
        mode = path_mode
        if path_mode == "mixed":
            mode = (json.loads(asset.get("fields") or "{}").get("path_mode")) or "fast"
        if mode == "lora":
            lr = []
            if var:                                          # LoRA della VARIAZIONE (flavor giusto), poi fallback asset
                lr = db.query("lora", "variation_id=? AND flavor=?", (var["id"], flavor), order="version")
            if not lr:
                lr = db.query("lora", "asset_id=? AND flavor=?", (asset["id"], flavor), order="version")
            if lr:
                _w = float(lr[-1].get("weight") or 0.9)      # peso dalla LoRA (col opz.), default 0.9
                loras.append({"file": lr[-1]["file"], "trigger": lr[-1]["trigger"], "weight": _w})
                trigger_map[name] = lr[-1]["trigger"]
                continue
        # Rapido (o LoRA assente): immagini di riferimento della VARIAZIONE se citata, altrimenti dell'asset
        imgs = []
        if var:
            imgs = db.query("image", "owner_kind='variation' AND owner_id=?", (var["id"],))
        if not imgs:
            imgs = db.query("image", "owner_kind='asset' AND owner_id=?", (asset["id"],))
        ref_images += [i["file"] for i in imgs[:4] if i.get("file")]
    clean = _REF_RE.sub(lambda m: trigger_map.get(m.group(1), m.group(1)), text or "")
    return {"refs": refs, "ref_images": ref_images, "loras": loras,
            "trigger_map": trigger_map, "clean_text": clean}


# ─────────────── immagini (keyframe / asset sheet) ───────────────
def gen_image(prompt: str, out_png: str, *, width: int = 1024, height: int = 1024, seed: int = 7,
              face_ref: str = "", ref_image: str = "", ref_images: list | None = None,
              loras: list | None = None, arch: str = "flux", model: str = "", negative: str = "") -> str:
    """Genera UNA immagine col motore nativo. face_ref→PuLID; ref_image→Kontext; loras opzionali.
    arch/model: architettura+checkpoint scelti nel progetto (B5: prima hardcoded flux). negative: estende il
    negative di default (continuity elementi vietati)."""
    neg = NEG + (", " + negative if negative else "")
    params = {"arch": arch, "prompt": prompt, "negative": neg, "width": width, "height": height,
              "steps": 22, "guidance": 3.5, "seed": seed, "output": out_png}
    if model:                                                # checkpoint esplicito (altrimenti default del server)
        params["model"] = model
    if face_ref:
        params["pulid_face"] = face_ref; params["pulid_strength"] = 0.85
    # Kontext/multi-ref SOLO se non c'è un volto PuLID (attori→PuLID; props/luoghi/architetture→Kontext ref).
    _refs = [r for r in ([ref_image] + list(ref_images or [])) if r]
    if _refs and not face_ref:
        params["arch"] = "kontext"; params["image_path"] = _refs[0]
        if len(_refs) > 1:
            params["ref_images"] = _refs                     # multi-reference (composizione senza incollaggio)
    if loras:
        params["loras"] = loras
    res = native_driver.run_and_wait("image", params, timeout=900)
    p = (res.get("output") or [out_png])
    p = p[0] if isinstance(p, list) else p
    if p and os.path.exists(p) and p != out_png:
        import shutil
        shutil.copyfile(p, out_png)
    if res.get("status") != "success" and not os.path.exists(out_png):
        raise RuntimeError(res.get("error", "gen_image fallita"))
    return out_png


# ─────────────── video (shot) ───────────────
def gen_shot_video(prompt: str, out_mp4: str, *, width: int = 768, height: int = 512, fps: int = 25,
                   duration_s: float = 4.0, seed: int = 7, keyframes: list | None = None,
                   loras: list | None = None, audio_path: str = "", model_gguf: str = "",
                   negative: str = "") -> str:
    """Genera il video di uno shot col worker nativo LTX-2.3 GGUF. keyframes=[{path,frame,strength}].
    model_gguf: transformer scelto nel progetto (B5), altrimenti il distilled Q4 di default."""
    nf = max(9, int(round(duration_s * fps)) // 8 * 8 + 1)
    neg = NEG + (", " + negative if negative else "")
    params = {"gguf_transformer": model_gguf or GGUF, "prompt": prompt, "negative": neg, "width": width,
              "height": height, "num_frames": nf, "fps": fps, "steps": 8, "guidance": 3.0,
              "seed": seed, "offload": "cpu", "output": out_mp4}
    if keyframes:
        params["keyframes"] = keyframes
    if loras:
        params["loras"] = loras
    if audio_path:
        params["audio_path"] = audio_path
    if nf > 150:
        params["chunk_frames"] = 121
    jf = out_mp4 + ".json"
    with open(jf, "w") as _f:                        # fix: handle non chiuso (leak FD nei loop)
        json.dump(params, _f)
    env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run([LTX2_PY, "-u", LTX2_WORKER, jf], env=env, capture_output=True, text=True)
    if not os.path.exists(out_mp4):
        raise RuntimeError("gen_shot_video fallita: " + (r.stdout[-400:] + r.stderr[-400:]))
    return out_mp4
