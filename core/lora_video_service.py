"""LoRA video (LTX-2.3) — generazione automatica dataset + preprocess + training.

Best practice automatizzate:
- Copertura per tipo (core/lora_coverage): personaggio = viste/espressioni/emozioni/gesti; luogo =
  orari-luci/viste/movimenti; oggetto = 360°/luci/dettaglio. + situazioni custom illimitate.
- IDENTITÀ personaggi: keyframe con PuLID (volto dalla foto) → i2v in clip → somiglianza forte e
  costante anche in espressioni estreme. Oggetti/luoghi: i2v dalla foto di riferimento.
- Caption automatica = TRIGGER + tag del caso (+ descrizione), EDITABILE.
- Dataset MISTO: oltre ai clip generati si possono includere video/foto REALI dell'utente.
- Preprocess + training con il trainer UFFICIALE LTX-2 (packages/ltx-trainer), config low-VRAM.

Zero ComfyUI. I clip video si generano col motore nativo (worker ltx2_infer, GGUF); i keyframe
PuLID col server immagini nativo (:8189). Pensato per un worker GUI (callback progress) o CLI.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core import lora_coverage

REPO = Path(__file__).resolve().parents[1]
LTX2_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
LTX2_WORKER = str(REPO / "workers" / "ltx2_infer.py")
TRAIN_RUN = str(REPO / "workers" / "ltx_train_run.py")
GGUF = r"E:\AI\Models\diffusion_models\LTXVideo\LTX-2\quantstack\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
TRAINER_DIR = r"E:\AI\Tools\LTX-2\packages\ltx-trainer"
LTX_CKPT = r"E:\AI\Models\diffusion_models\LTXVideo\LTX-2\ltx-2.3-22b-dev-fp8.safetensors"
GEMMA = r"E:\AI\Models\TextEncoders\gemma-3-ltx"
NEG = "blurry, low quality, deformed, distorted face, extra limbs, watermark, text"

TYPE_SUBJ = {       # come descrivere il soggetto nel prompt di generazione
    "person": "a person named {name}",
    "place": "the location known as {name}",
    "object": "the object known as {name}",
    "concept": "{name}",
}


@dataclass
class Subject:
    name: str                                   # nome = trigger/tag
    type: str = "person"                        # person|place|object|concept
    photos: list = field(default_factory=list)  # [{"path":..,"angle":"front|three_quarter|profile|back|any"}]
    case_ids: list[str] = field(default_factory=list)   # casi di copertura attivi
    customs: list[str] = field(default_factory=list)    # situazioni descritte a mano (illimitate)
    real_media: list[str] = field(default_factory=list) # video/foto reali da includere
    trigger: str = ""                           # se vuoto → derivato dal nome
    face_focus: bool = True                      # personaggi: impara il VOLTO, non abiti/sfondo
    photo: str = ""                             # back-compat: foto singola

    def trig(self) -> str:
        return (self.trigger or self.name or "subject").strip()

    def all_photos(self) -> list:
        ph = list(self.photos or [])
        if self.photo and not any(p.get("path") == self.photo for p in ph):
            ph.append({"path": self.photo, "angle": "front"})
        return [p for p in ph if p.get("path")]

    def photo_for(self, angle: str) -> str:
        """Foto dell'angolo richiesto (fallback: 'any'/'front'/prima disponibile)."""
        ph = self.all_photos()
        if not ph:
            return ""
        for want in (angle, "any", "front"):
            for p in ph:
                if p.get("angle") == want:
                    return p["path"]
        return ph[0]["path"]


def trainer_installed() -> bool:
    try:
        r = subprocess.run([LTX2_PY, "-c", "import ltx_trainer"], capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:                            # noqa: BLE001
        return False


def _subj_desc(s: Subject) -> str:
    return TYPE_SUBJ.get(s.type, "{name}").format(name=s.trig())


def plan_clips(s: Subject) -> list[dict]:
    """Espande i casi attivi + le situazioni custom in clip da generare.
    Ogni voce: {id, prompt, caption, use_face, ref, keyframes?}.
    FOCUS VOLTO (personaggi): abiti+sfondo VARIATI per clip (e messi in caption, non nel trigger),
    così la LoRA impara solo il viso. Multi-foto: la foto dell'angolo giusto fa da riferimento.
    Se ≥2 foto con angoli diversi → aggiunge una clip di ROTAZIONE multi-keyframe."""
    subj = _subj_desc(s)
    trig = s.trig()
    is_person = s.type == "person"
    face_focus = is_person and s.face_focus
    out: list[dict] = []
    by_id = {cid: (tmpl, tags) for cid, tmpl, tags in lora_coverage.cases_for(s.type)}
    angle_of = lora_coverage.PERSON_CASE_ANGLE if is_person else {}

    def _var(idx: int):                          # abito+sfondo deterministici per indice (no random → riproducibile)
        o = lora_coverage.OUTFITS[idx % len(lora_coverage.OUTFITS)]
        b = lora_coverage.BACKDROPS[(idx * 5 + 3) % len(lora_coverage.BACKDROPS)]
        return o, b

    k = 0
    for cid in s.case_ids:
        if cid not in by_id:
            continue
        tmpl, tags = by_id[cid]
        prompt = tmpl.format(subj=subj)
        cap_tags = list(tags)
        if face_focus:                           # varia ciò che NON deve entrare nel trigger e mettilo in caption
            o, b = _var(k)
            prompt += f", wearing {o}, in {b}"
            cap_tags += [f"wearing {o}", b]
        caption = trig + ", " + ", ".join(cap_tags) + (f". {prompt}." if prompt else "")
        out.append({"id": cid, "prompt": prompt, "caption": caption, "use_face": is_person,
                    "ref": s.photo_for(angle_of.get(cid, "any")) or (s.all_photos()[0]["path"] if s.all_photos() else "")})
        k += 1
    # clip ROTAZIONE multi-keyframe se ci sono più angoli
    photos = s.all_photos()
    if is_person and len({p.get("angle") for p in photos}) >= 2:
        order = ["front", "three_quarter", "profile", "back"]
        seq = [p for a in order for p in photos if p.get("angle") == a] or photos
        out.append({"id": "rotation", "prompt": lora_coverage.ROTATION_CASE["prompt"].format(subj=subj),
                    "caption": trig + ", " + ", ".join(lora_coverage.ROTATION_CASE["tags"]) + ".",
                    "use_face": False, "ref": seq[0]["path"], "keyframes_seq": [p["path"] for p in seq]})
    for i, situation in enumerate(s.customs):
        situation = (situation or "").strip()
        if not situation:
            continue
        prompt = f"{subj}, {situation}"
        cap = f"{trig}, {situation}."
        if face_focus:
            o, b = _var(k); prompt += f", wearing {o}, in {b}"; cap += f" wearing {o}, {b}."
            k += 1
        out.append({"id": f"custom_{i}", "prompt": prompt, "caption": cap, "use_face": is_person,
                    "ref": s.photo_for("any")})
    return out


def _gen_keyframe_pulid(photo: str, prompt: str, out_png: str, width: int, height: int,
                        strength: float = 0.85) -> bool:
    """Keyframe con PuLID (volto dalla foto) via server immagini nativo (:8189)."""
    from core import native_driver
    if not native_driver.is_up():
        return False
    res = native_driver.run_and_wait("image", {
        "arch": "flux", "prompt": prompt, "negative": NEG,
        "pulid_face": photo, "pulid_strength": strength,
        "width": width, "height": height, "steps": 18, "guidance": 3.5, "seed": 7,
        "output": out_png,
    }, timeout=600)
    p = res.get("output") or out_png
    if p and os.path.exists(p) and p != out_png:
        shutil.copyfile(p, out_png)
    return os.path.exists(out_png)


def _gen_clip(cond_image: str, prompt: str, out_mp4: str, width: int, height: int,
              num_frames: int, fps: int) -> None:
    params = {
        "image_path": cond_image, "gguf_transformer": GGUF, "prompt": prompt, "negative": NEG,
        "width": width, "height": height, "num_frames": num_frames, "fps": fps,
        "steps": 8, "guidance": 3.0, "seed": 7, "offload": "cpu", "ref_strength": 1.0, "output": out_mp4,
    }
    jf = out_mp4 + ".json"
    json.dump(params, open(jf, "w"))
    env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run([LTX2_PY, "-u", LTX2_WORKER, jf], env=env, capture_output=True, text=True)
    if not os.path.exists(out_mp4):
        raise RuntimeError("clip fallita: " + (r.stdout[-400:] + r.stderr[-400:]))


def _gen_clip_kf(keyframes: list[dict], prompt: str, out_mp4: str, width: int, height: int,
                 num_frames: int, fps: int) -> None:
    """Clip con keyframe multipli (foto/angoli a frame diversi) = rotazione/angoli coerenti."""
    params = {
        "keyframes": keyframes, "gguf_transformer": GGUF, "prompt": prompt, "negative": NEG,
        "width": width, "height": height, "num_frames": num_frames, "fps": fps,
        "steps": 8, "guidance": 3.0, "seed": 7, "offload": "cpu", "output": out_mp4,
    }
    jf = out_mp4 + ".json"
    json.dump(params, open(jf, "w"))
    env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run([LTX2_PY, "-u", LTX2_WORKER, jf], env=env, capture_output=True, text=True)
    if not os.path.exists(out_mp4):
        raise RuntimeError("clip multi-keyframe fallita: " + (r.stdout[-400:] + r.stderr[-400:]))


def _face_crop(mp4: str, target_w: int, target_h: int, margin: float = 2.4) -> None:
    """Ritaglia il clip attorno al VOLTO (rileva sul frame centrale con insightface) e riscala.
    Toglie peso ad abiti/sfondo → la LoRA si concentra sul viso. No-op se il volto non si rileva."""
    try:
        import tempfile
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except Exception:                            # noqa: BLE001
        return
    tmp = mp4 + ".frame.png"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.3", "-i", mp4, "-vframes", "1", tmp, "-y"])
    if not os.path.exists(tmp):
        return
    try:
        img = cv2.imread(tmp)
        H, W = img.shape[:2]
        app = _face_app()
        faces = app.get(img) if app else []
        if not faces:
            return
        x1, y1, x2, y2 = faces[0].bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * margin
        side = min(side, W, H)
        left = int(max(0, min(W - side, cx - side / 2)))
        top = int(max(0, min(H - side, cy - side / 2)))
        side = int(side)
        crop = mp4 + ".crop.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", mp4,
                        "-vf", f"crop={side}:{side}:{left}:{top},scale={target_w}:{target_h}",
                        "-c:a", "copy", crop, "-y"])
        if os.path.exists(crop):
            os.replace(crop, mp4)
    except Exception:                            # noqa: BLE001
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


_FACE_APP = {}


def _face_app():
    if "app" not in _FACE_APP:
        try:
            from insightface.app import FaceAnalysis
            a = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=-1, det_size=(640, 640))
            _FACE_APP["app"] = a
        except Exception:                        # noqa: BLE001
            _FACE_APP["app"] = None
    return _FACE_APP["app"]


def generate_dataset(subjects: list[Subject], dataset_dir: str, *, width: int = 768, height: int = 512,
                     clip_seconds: float = 3.0, fps: int = 25, use_pulid: bool = True, progress=None) -> dict:
    """Genera i clip (con caption .txt) per tutti i soggetti + include il materiale reale.
    Ritorna {clips, captions, dir}."""
    def log(m):
        if progress:
            progress(m)
    os.makedirs(dataset_dir, exist_ok=True)
    nf = int(round(clip_seconds * fps) // 8 * 8 + 1)
    n_done = 0
    total = sum(len(plan_clips(s)) + len(s.real_media) for s in subjects)
    log(f"Dataset: {total} clip da generare/includere")
    for s in subjects:
        kf_dir = os.path.join(dataset_dir, "_keyframes", s.trig().replace(" ", "_"))
        os.makedirs(kf_dir, exist_ok=True)
        for clip in plan_clips(s):
            stem = f"{s.trig()}__{clip['id']}".replace(" ", "_")
            out_mp4 = os.path.join(dataset_dir, stem + ".mp4")
            if not os.path.exists(out_mp4):
                seq = clip.get("keyframes_seq") or []
                if len(seq) >= 2:                            # ROTAZIONE multi-keyframe (angoli coerenti)
                    kfs = [{"path": ph, "frame": int(j * (nf - 1) / (len(seq) - 1)), "strength": 1.0}
                           for j, ph in enumerate(seq)]
                    _gen_clip_kf(kfs, clip["prompt"], out_mp4, width, height, nf, fps)
                else:
                    cond = clip.get("ref") or s.photo
                    if clip["use_face"] and use_pulid and cond:   # keyframe PuLID = somiglianza forte
                        kf = os.path.join(kf_dir, clip["id"] + ".png")
                        if _gen_keyframe_pulid(cond, clip["prompt"], kf, width, height):
                            cond = kf
                        else:
                            log("  (PuLID/server immagini non attivo → i2v dalla foto)")
                    if not cond:
                        log(f"  [skip] '{stem}': nessuna foto (serve almeno una foto del soggetto)")
                        continue
                    _gen_clip(cond, clip["prompt"], out_mp4, width, height, nf, fps)
                if clip.get("use_face") and s.face_focus and os.path.exists(out_mp4):
                    _face_crop(out_mp4, width, height)          # ritaglia sul volto → LoRA del viso
            open(os.path.splitext(out_mp4)[0] + ".txt", "w", encoding="utf-8").write(clip["caption"])
            n_done += 1
            log(f"Clip {n_done}/{total}: {stem}")
        # materiale reale
        for rm in s.real_media:
            if not os.path.exists(rm):
                continue
            ext = os.path.splitext(rm)[1].lower()
            stem = f"{s.trig()}__real_{os.path.splitext(os.path.basename(rm))[0]}".replace(" ", "_")
            cap = f"{s.trig()}. {_subj_desc(s)}."
            if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
                dst = os.path.join(dataset_dir, stem + ".mp4")
                if not os.path.exists(dst):
                    shutil.copyfile(rm, dst)
            else:   # foto reale → breve clip i2v (il trainer vuole video)
                dst = os.path.join(dataset_dir, stem + ".mp4")
                if not os.path.exists(dst):
                    _gen_clip(rm, _subj_desc(s) + ", subtle natural motion", dst, width, height, nf, fps)
            open(os.path.splitext(dst)[0] + ".txt", "w", encoding="utf-8").write(cap)
            n_done += 1
            log(f"Reale {n_done}/{total}: {stem}")
    return {"dir": dataset_dir, "clips": n_done}


def build_dataset_json(dataset_dir: str) -> str:
    """Crea dataset.json (colonne video+caption) dai clip+caption .txt presenti."""
    rows = []
    for f in sorted(os.listdir(dataset_dir)):
        if not f.endswith(".mp4"):
            continue
        cap = os.path.join(dataset_dir, os.path.splitext(f)[0] + ".txt")
        rows.append({"video": os.path.join(dataset_dir, f),
                     "caption": open(cap, encoding="utf-8").read().strip() if os.path.exists(cap) else ""})
    dj = os.path.join(dataset_dir, "dataset.json")
    json.dump(rows, open(dj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return dj


def preprocess(dataset_dir: str, work_dir: str, *, width: int = 768, height: int = 512,
               num_frames: int = 49, trigger: str = "", progress=None) -> str:
    """Preprocess col trainer ufficiale: video+caption → latents+conditions (preprocessed_data_root)."""
    def log(m):
        if progress:
            progress(m)
    dj = build_dataset_json(dataset_dir)
    out = os.path.join(work_dir, "preprocessed")
    os.makedirs(out, exist_ok=True)
    log("Preprocess dataset (latents + text)…")
    cmd = [LTX2_PY, "-u", os.path.join(TRAINER_DIR, "scripts", "process_dataset.py"), dj,
           "--resolution-buckets", f"{width}x{height}x{num_frames}",
           "--model-path", LTX_CKPT, "--text-encoder-path", GEMMA, "--output-dir", out]
    if trigger:
        cmd += ["--lora-trigger", trigger]
    env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"
    r = subprocess.run(cmd, cwd=TRAINER_DIR, env=env, capture_output=True, text=True)
    if not os.path.isdir(os.path.join(out, "latents")):
        raise RuntimeError("preprocess fallito: " + (r.stdout[-600:] + r.stderr[-600:]))
    return out


def train(preprocessed_dir: str, out_lora: str, *, trigger: str = "", rank: int = 16, steps: int = 2000,
          learning_rate: float = 1e-4, low_vram: bool = True, mode: str = "t2v", progress=None) -> str:
    """Lancia il training LoRA col trainer ufficiale (wrapper in ltx2_venv per gestire lo YAML)."""
    def log(m):
        if progress:
            progress(m)
    template = os.path.join(TRAINER_DIR, "configs",
                            "t2v_lora_low_vram.yaml" if low_vram else f"{mode}_lora.yaml")
    overrides = {                                  # chiavi reali del config LtxTrainerConfig
        "model.model_path": LTX_CKPT, "model.text_encoder_path": GEMMA, "model.training_mode": "lora",
        "lora.rank": rank, "lora.alpha": rank,
        "optimization.steps": steps, "optimization.learning_rate": learning_rate,
        "data.preprocessed_data_root": preprocessed_dir,
        "output_dir": os.path.dirname(out_lora) or ".",
        "validation.skip_initial_validation": True,
    }
    of = out_lora + ".overrides.json"
    json.dump(overrides, open(of, "w"))
    os.makedirs(os.path.dirname(out_lora) or ".", exist_ok=True)
    log(f"Training LoRA (rank {rank}, {steps} step)…")
    cmd = [LTX2_PY, "-u", TRAIN_RUN, "--template", template, "--overrides", of,
           "--out_lora", out_lora]
    env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.Popen(cmd, cwd=TRAINER_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:                     # progress live
        line = line.rstrip()
        if line and ("step" in line.lower() or "loss" in line.lower() or "error" in line.lower()):
            log(line[-160:])
    proc.wait()
    if not os.path.exists(out_lora):
        raise RuntimeError("training fallito (LoRA non prodotta) — vedi log")
    return out_lora


import re as _re
_FFMPEG = r"E:\AI\Tools\ffmpeg-shared\bin\ffmpeg.exe"


def _safe_stem(s: str) -> str:
    s = _re.sub(r"[^0-9A-Za-z_.-]+", "_", s).strip("_.")
    return s or "clip"


def build_clip_dataset(clips: list[dict], dataset_dir: str, *, global_caption: str = "",
                       progress=None) -> dict:
    """Prepara il dataset da CLIP ESISTENTI. clips: [{path, caption?, start?, end?}].
    Ogni clip → <stem>.mp4 (trim+normalizzato h264) + <stem>.txt (caption: per-clip · sibling _*.txt · globale)."""
    def log(m):
        if progress:
            progress(m)
    os.makedirs(dataset_dir, exist_ok=True)
    n = 0
    for i, c in enumerate(clips):
        src = c.get("path", "")
        if not src or not os.path.isfile(src):
            continue
        stem = _safe_stem(os.path.splitext(os.path.basename(src))[0]) or f"clip{i:03d}"
        dst = os.path.join(dataset_dir, f"{stem}.mp4")
        start = float(c.get("start") or 0.0); end = float(c.get("end") or 0.0)
        cmd = [_FFMPEG, "-y"]
        if start > 0:
            cmd += ["-ss", str(start)]
        cmd += ["-i", src]
        if end > 0 and end > start:
            cmd += ["-t", str(end - start)]
        cmd += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
        log(f"Clip {i + 1}/{len(clips)}: {stem}")
        subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(dst):
            log(f"⚠️ ffmpeg non ha prodotto {stem}.mp4 (clip saltato)"); continue
        # caption: per-clip → sibling <name>_prompt.txt/.txt → globale
        cap = (c.get("caption") or "").strip()
        if not cap:
            for sib in (os.path.splitext(src)[0] + ".txt", os.path.splitext(src)[0] + "_prompt.txt"):
                if os.path.exists(sib):
                    cap = open(sib, encoding="utf-8").read().strip(); break
        cap = cap or global_caption or ""
        open(os.path.join(dataset_dir, f"{stem}.txt"), "w", encoding="utf-8").write(cap)
        n += 1
    return {"dir": dataset_dir, "clips": n}


def make_lora_from_clips(clips: list[dict], name: str, out_dir: str, *, global_caption: str = "",
                         trigger: str = "", rank: int = 16, steps: int = 2000,
                         width: int = 768, height: int = 512, clip_seconds: float = 3.0,
                         low_vram: bool = True, progress=None) -> dict:
    """Pipeline da CLIP esistenti: build dataset → preprocess → train. Ritorna {dataset, lora, clips}."""
    nf = int(round(clip_seconds * 25) // 8 * 8 + 1)
    work = os.path.join(out_dir, name)
    ds = os.path.join(work, "dataset")
    info = build_clip_dataset(clips, ds, global_caption=global_caption, progress=progress)
    if info["clips"] < 1:
        raise RuntimeError("Nessun clip valido nel dataset (controlla i file audio/video).")
    pre = preprocess(ds, work, width=width, height=height, num_frames=nf, trigger=trigger, progress=progress)
    lora = os.path.join(work, name + ".safetensors")
    train(pre, lora, trigger=trigger, rank=rank, steps=steps, low_vram=low_vram, progress=progress)
    return {"dataset": ds, "lora": lora, "clips": info["clips"]}


def make_lora(subjects: list[Subject], name: str, out_dir: str, *, rank: int = 16, steps: int = 2000,
              clip_seconds: float = 3.0, use_pulid: bool = True, low_vram: bool = True, progress=None) -> dict:
    """Pipeline COMPLETA: dataset → preprocess → train. Ritorna {dataset, lora}."""
    W, H, FPS = 768, 512, 25
    nf = int(round(clip_seconds * FPS) // 8 * 8 + 1)
    work = os.path.join(out_dir, name)
    ds = os.path.join(work, "dataset")
    generate_dataset(subjects, ds, width=W, height=H, clip_seconds=clip_seconds,
                     use_pulid=use_pulid, progress=progress)
    trig = ", ".join(sorted({s.trig() for s in subjects}))
    pre = preprocess(ds, work, width=W, height=H, num_frames=nf, trigger=trig, progress=progress)
    lora = os.path.join(work, name + ".safetensors")
    train(pre, lora, trigger=trig, rank=rank, steps=steps, low_vram=low_vram, progress=progress)
    return {"dataset": ds, "lora": lora}
