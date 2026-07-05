"""LoRA immagini (FLUX.2) — generazione automatica dataset (immagini) + training via ai-toolkit.

Stessa logica del LoRA video ma per IMMAGINI:
- Copertura best-practice per tipo (core/lora_coverage) + situazioni custom illimitate + materiale reale.
- Volti: FLUX.1 + PuLID (foto dell'angolo giusto) → forte somiglianza. Oggetti/luoghi: FLUX.1 Kontext
  (foto di riferimento → nuova vista/luce). Concetti: t2i.
- FOCUS VOLTO: varia abiti/sfondo (in caption, non nel trigger) + auto-crop sul viso → la LoRA impara
  il volto, riusabile con abiti diversi.
- Training: trainer FLUX.2 ESISTENTE (core/training_service, ai-toolkit) su cartella immagini+caption.

Le immagini si generano col server immagini nativo (:8189). Zero ComfyUI.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from core import lora_coverage
from core.lora_video_service import Subject, _subj_desc, plan_clips   # riuso modello + pianificazione

NEG = "blurry, low quality, deformed, distorted face, extra limbs, watermark, text"


def _gen_image(subject_type: str, ref: str, prompt: str, out_png: str, width: int, height: int,
               use_face: bool) -> bool:
    """Genera UNA immagine col motore nativo: volto→FLUX.1+PuLID, oggetto/luogo con foto→Kontext, altrimenti t2i."""
    from core import native_driver
    if not native_driver.is_up():
        return False
    if use_face and ref:                          # personaggio: identità forte via PuLID
        params = {"arch": "flux", "prompt": prompt, "negative": NEG, "pulid_face": ref,
                  "pulid_strength": 0.85, "width": width, "height": height, "steps": 20,
                  "guidance": 3.5, "seed": 7, "output": out_png}
    elif ref:                                     # oggetto/luogo: foto di riferimento via Kontext
        params = {"arch": "kontext", "prompt": prompt, "negative": NEG, "image_path": ref,
                  "width": width, "height": height, "steps": 20, "guidance": 3.0, "seed": 7,
                  "output": out_png}
    else:                                         # concetto / senza foto: t2i
        params = {"arch": "flux", "prompt": prompt, "negative": NEG, "width": width, "height": height,
                  "steps": 20, "guidance": 3.5, "seed": 7, "output": out_png}
    res = native_driver.run_and_wait("image", params, timeout=600)
    p = res.get("output") or out_png
    if p and os.path.exists(p) and p != out_png:
        shutil.copyfile(p, out_png)
    return os.path.exists(out_png)


def _face_crop_image(png: str, margin: float = 2.4) -> None:
    """Ritaglia l'immagine attorno al volto (insightface) → la LoRA si concentra sul viso."""
    try:
        import cv2
        from core.lora_video_service import _face_app
    except Exception:                             # noqa: BLE001
        return
    try:
        img = cv2.imread(png)
        H, W = img.shape[:2]
        app = _face_app()
        faces = app.get(img) if app else []
        if not faces:
            return
        x1, y1, x2, y2 = faces[0].bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = min(max(x2 - x1, y2 - y1) * margin, W, H)
        left = int(max(0, min(W - side, cx - side / 2)))
        top = int(max(0, min(H - side, cy - side / 2)))
        side = int(side)
        cv2.imwrite(png, img[top:top + side, left:left + side])
    except Exception:                             # noqa: BLE001
        pass


def generate_dataset(subjects: list[Subject], dataset_dir: str, *, width: int = 1024, height: int = 1024,
                     caption_engine: str = "procedural", caption_mode: str = "descriptive",
                     progress=None) -> dict:
    """Genera le IMMAGINI (+ caption .txt) in un'unica cartella (formato ai-toolkit).
    caption_engine: 'procedural' (trigger+tag) | 'joycaption' | 'florence2' — se VLM, ricaptiona con quello."""
    def log(m):
        if progress:
            progress(m)
    os.makedirs(dataset_dir, exist_ok=True)
    n = 0
    total = sum(len([c for c in plan_clips(s) if not c.get("keyframes_seq")]) + len(s.real_media)
                for s in subjects)
    log(f"Dataset immagini: {total} da generare/includere")
    for s in subjects:
        for clip in plan_clips(s):
            if clip.get("keyframes_seq"):          # la rotazione è solo-video → salta nelle immagini
                continue
            stem = f"{s.trig()}__{clip['id']}".replace(" ", "_")
            out_png = os.path.join(dataset_dir, stem + ".png")
            if not os.path.exists(out_png):
                ref = clip.get("ref") or ""
                if not _gen_image(s.type, ref, clip["prompt"], out_png, width, height, clip["use_face"]):
                    log(f"  [skip] '{stem}': server immagini non attivo o nessuna foto")
                    continue
                if clip.get("use_face") and s.face_focus:
                    _face_crop_image(out_png)
            open(os.path.splitext(out_png)[0] + ".txt", "w", encoding="utf-8").write(clip["caption"])
            n += 1
            log(f"Immagine {n}/{total}: {stem}")
        for rm in s.real_media:                    # materiale reale (foto): copiato; video: estrae 1 frame
            if not os.path.exists(rm):
                continue
            stem = f"{s.trig()}__real_{os.path.splitext(os.path.basename(rm))[0]}".replace(" ", "_")
            dst = os.path.join(dataset_dir, stem + ".png")
            ext = os.path.splitext(rm)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp"):
                if not os.path.exists(dst):
                    shutil.copyfile(rm, dst)
            else:                                  # video reale → primo frame come immagine
                if not os.path.exists(dst):
                    subprocess.run(["ffmpeg", "-v", "error", "-i", rm, "-vframes", "1", dst, "-y"])
            if os.path.exists(dst):
                open(os.path.splitext(dst)[0] + ".txt", "w", encoding="utf-8").write(f"{s.trig()}. {_subj_desc(s)}.")
                n += 1
                log(f"Reale {n}/{total}: {stem}")
    # Ricaptioning VLM opzionale: sostituisce le caption procedurali con caption reali (JoyCaption/Florence-2)
    if caption_engine and caption_engine != "procedural" and n > 0:
        from core import caption_service as cs
        trig = subjects[0].trig() if len(subjects) == 1 else ""
        log(f"Ricaptioning con {caption_engine} ({caption_mode})…")
        rc = cs.run_blocking(dataset_dir, engine=caption_engine, mode=caption_mode,
                             trigger=trig, overwrite=True, on_log=log)
        log(f"Ricaptioning {caption_engine}: {'OK' if rc == 0 else 'fallito/saltato (uso caption procedurali)'}")
    return {"dir": dataset_dir, "images": n}


def train(dataset_dir: str, name: str, out_dir: str, *, base: str = "flux2-dev", trigger: str = "",
          rank: int = 16, steps: int = 2000, progress=None) -> str:
    """Allena la LoRA FLUX.2 col trainer esistente (ai-toolkit) sulla cartella immagini+caption."""
    def log(m):
        if progress:
            progress(m)
    from core import training_service as ts
    p = ts.TrainParams(name=name, base_model=base, dataset_path=dataset_dir, rank=rank, alpha=rank,
                       steps=steps, trigger_word=trigger, low_vram=(base == "flux2-dev"))
    errs = ts.preflight(p)
    if errs:
        raise RuntimeError("; ".join(errs))
    cfg = ts.write_config(p)
    log(f"Training FLUX.2 LoRA (ai-toolkit, rank {rank}, {steps} step)…")
    cmd = [str(ts.VENV_PYTHON), str(ts.RUN_PY), str(cfg)]
    proc = subprocess.Popen(cmd, cwd=str(ts.AI_TOOLKIT_DIR), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if line and ("step" in line.lower() or "loss" in line.lower() or "error" in line.lower()):
            log(line[-160:])
    proc.wait()
    safet = list((p.project_dir()).rglob("*.safetensors"))
    if not safet:
        raise RuntimeError("training fallito (LoRA non prodotta) — vedi log")
    final = max(safet, key=os.path.getmtime)
    return str(final)


def make_lora(subjects: list[Subject], name: str, out_dir: str, *, base: str = "flux2-dev",
              rank: int = 16, steps: int = 2000, width: int = 1024, height: int = 1024,
              caption_engine: str = "procedural", caption_mode: str = "descriptive", progress=None) -> dict:
    """Pipeline COMPLETA immagini: dataset → train (ai-toolkit FLUX.2). Ritorna {dataset, lora}."""
    ds = os.path.join(out_dir, name, "dataset")
    generate_dataset(subjects, ds, width=width, height=height,
                     caption_engine=caption_engine, caption_mode=caption_mode, progress=progress)
    trig = ", ".join(sorted({s.trig() for s in subjects}))
    lora = train(ds, name, out_dir, base=base, trigger=trig, rank=rank, steps=steps, progress=progress)
    return {"dataset": ds, "lora": lora}
