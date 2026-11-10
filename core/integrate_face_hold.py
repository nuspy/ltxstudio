"""Integra: il volto del soggetto incollato NON si rigenera (04/11, controllo automatico dei turni 31 e 35).

Nei test dal vivo il volto incollato (33 px, somiglianza al riferimento 0,82) tornava dall'armonizzazione Qwen (strength 0,45-0,65) e da Klein ref-inpaint con somiglianza
0,05-0,34: la maschera (sagoma + contesto) copre anche la testa, che viene rumorata e ridisegnata senza alcun condizionamento d'identita. Qui si toglie il volto dalla zona
generativa (riquadro insightface + 30 %, bordo sfumato) e, per Klein ref-inpaint, si ricava il ritaglio del volto del soggetto da dare come secondo riferimento.

insightface sta nell'inference_server_venv, non nel venv della GUI: i riquadri si chiedono a `workers/face_boxes_job.py` in un processo figlio su CPU.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

INFERENCE_PY = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"
JOB = str(Path(__file__).resolve().parents[1] / "workers" / "face_boxes_job.py")
HOLD_GROW = 0.30                 # il riquadro del volto cresce del 30 % per lato (fronte, capelli, mento)
HOLD_FEATHER = 0.35              # bordo sfumato: frazione del semiasse tenuta come discesa a zero
CROP_MARGIN = 0.60               # ritaglio del volto del soggetto: riquadro + 60 % per lato (capelli e spalle superiori)
RESULT_MARKER = "@@JSON@@"


def detect_face_boxes(image_paths: list[str], runner=None) -> list[list[dict]]:
    """Per ogni immagine la lista dei volti [{"bbox": [x0, y0, x1, y1], "score": s}], i piu grandi per primi. `runner(argv, stdin) -> stdout` iniettabile nei test."""
    stdin = json.dumps({"images": list(image_paths)})
    argv = [INFERENCE_PY, "-I", JOB]
    if runner is not None:
        stdout, error_tail = runner(argv, stdin), ""
    else:
        if not os.path.isfile(INFERENCE_PY):
            raise RuntimeError(f"i riquadri dei volti servono insightface dell'inference_server_venv: manca {INFERENCE_PY}")
        env = dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, env=env, timeout=300, encoding="utf-8", errors="replace")
        stdout, error_tail = done.stdout, (done.stderr or done.stdout)[-400:]
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(RESULT_MARKER)), "")
    if not line:
        raise RuntimeError(f"face_boxes_job senza risposta: {error_tail or stdout[-400:]}")
    return json.loads(line[len(RESULT_MARKER):])["res"]


def boxes_inside_mask(faces: list[dict], mask: np.ndarray, image_size: tuple[int, int]) -> list[list[float]]:
    """Riquadri (in pixel dell'immagine) il cui centro cade nella sagoma `mask` (True = soggetto incollato): gli altri volti della scena non si toccano."""
    height, width = mask.shape
    scale_x, scale_y = width / image_size[0], height / image_size[1]
    inside = []
    for face in faces:
        x0, y0, x1, y1 = face["bbox"]
        cx, cy = int(round((x0 + x1) / 2 * scale_x)), int(round((y0 + y1) / 2 * scale_y))
        if 0 <= cx < width and 0 <= cy < height and mask[cy, cx]:
            inside.append([x0, y0, x1, y1])
    return inside


def hold_map(shape: tuple[int, int], boxes: list[list[float]], box_scale: tuple[float, float] = (1.0, 1.0), grow: float = HOLD_GROW, feather: float = HOLD_FEATHER) -> np.ndarray:
    """Mappa float [0, 1] (1 = volto da tenere): ellisse del riquadro cresciuto di `grow`, poi un bordo che scende a 0 in `feather` semiassi. Solo numpy."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    hold = np.zeros(shape, np.float32)
    for x0, y0, x1, y1 in boxes:
        x0, x1, y0, y1 = x0 * box_scale[0], x1 * box_scale[0], y0 * box_scale[1], y1 * box_scale[1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = max(1.0, (x1 - x0) / 2 * (1 + 2 * grow)), max(1.0, (y1 - y0) / 2 * (1 + 2 * grow))
        radius = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
        ramp = np.clip((1.0 + feather - radius) / feather, 0.0, 1.0)
        hold = np.maximum(hold, ramp * ramp * (3 - 2 * ramp))                       # smoothstep: niente spigolo tra volto tenuto e zona generativa
    return hold


def hold_faces_in_mask(mask_path: str, boxes: list[list[float]], image_size: tuple[int, int], out_path: str) -> str:
    """Scrive in `out_path` la maschera generativa (bianco = rigenera) con i volti tenuti fuori. Senza riquadri la maschera resta quella di prima."""
    mask = Image.open(mask_path).convert("L")
    if not boxes:
        mask.save(out_path)
        return out_path
    scale = (mask.width / image_size[0], mask.height / image_size[1])
    hold = hold_map((mask.height, mask.width), boxes, scale)
    kept = np.asarray(mask, np.float32) * (1.0 - hold)
    Image.fromarray(np.clip(kept, 0, 255).astype(np.uint8), "L").save(out_path)
    return out_path


def face_crop(reference_path: str, out_path: str, runner=None, margin: float = CROP_MARGIN) -> str:
    """Ritaglio del volto piu grande della foto del soggetto (riquadro + `margin` per lato), da dare a Klein come secondo riferimento. Ritorna "" se non trova volti."""
    faces = detect_face_boxes([reference_path], runner)[0]
    if not faces:
        return ""
    image = Image.open(reference_path).convert("RGB")
    x0, y0, x1, y1 = faces[0]["bbox"]
    w, h = x1 - x0, y1 - y0
    box = (max(0, int(x0 - margin * w)), max(0, int(y0 - margin * h)), min(image.width, int(x1 + margin * w)), min(image.height, int(y1 + margin * h)))
    image.crop(box).save(out_path)
    return out_path


def hold_face_of_subject(export: dict, out_dir: str, log, runner=None) -> dict:
    """Copia di `export` (risultato di export_integrate) con la maschera `mask` priva del volto del soggetto incollato e, se serve a Klein, `face_crop` del suo riferimento.
    Se il volto non si trova lo dice nel log e lascia l'export com'era (mai in silenzio)."""
    image_path, position_path = export["image"], export["position_mask"]
    faces = detect_face_boxes([image_path], runner)[0]
    size = Image.open(image_path).size
    subject = np.asarray(Image.open(position_path).convert("L").resize(size)) > 127
    boxes = boxes_inside_mask(faces, subject, size)
    if not boxes:
        log(f"[integra] volto da tenere: nessun volto trovato dentro la sagoma del soggetto ({len(faces)} volti nella scena): la maschera resta com'e")
        return dict(export)
    held = hold_faces_in_mask(export["mask"], boxes, size, os.path.join(out_dir, "integrate_mask_volto_tenuto.png"))
    log(f"[integra] volto da tenere: {len(boxes)} volto/i fuori dalla zona generativa (riquadro +{int(HOLD_GROW * 100)} %)")
    result = dict(export, mask=held)
    refs = export.get("layer_refs") or []
    if refs:
        crop = face_crop(refs[0], os.path.join(out_dir, "integrate_volto_soggetto.png"), runner)
        if crop:
            result["face_crop"] = crop
    return result
