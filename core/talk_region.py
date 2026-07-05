"""Talk 'region': anima SOLO bocca/volto tenendo fermo il resto (reimagine=off).

Il pannello Talk offre regione (mouth/face), offset maschera e "reimagine". Quando reimagine è
OFF, il video a2v generato viene compositato SOLO nella regione scelta sopra la foto/frame
sorgente statico → si muove solo la bocca (o il volto), sfondo e corpo restano fermi.

Rilevamento volto con cv2 Haar cascade (nessuna dipendenza/pesi extra). Fail-safe: se non
rileva volti o cv2 manca, ritorna il video generato invariato (la generazione non si blocca).
"""
from __future__ import annotations

import os
import subprocess


def composite_talk_region(generated: str, source_image: str, out: str, *,
                          region: str = "mouth", mask_offset: int = 0, on_log=None) -> str:
    """Composita `generated` (a2v) nella regione volto/bocca sopra `source_image` statico.
    Ritorna `out` (con audio del generato) o `generated` se qualcosa fallisce."""
    try:
        import cv2
        import numpy as np
    except Exception:                                        # noqa: BLE001
        return generated
    if not (generated and os.path.isfile(generated) and source_image and os.path.isfile(source_image)):
        return generated
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(generated)
        if not cap.isOpened():
            return generated
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bg = cv2.imread(source_image)
        if bg is None:
            cap.release(); return generated
        bg = cv2.resize(bg, (w, h))
        tmp = out + ".noaudio.mp4"
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        last_box = None
        matched = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
            box = max(faces, key=lambda b: b[2] * b[3]) if len(faces) else last_box
            last_box = box
            mask = np.zeros((h, w), np.float32)
            if box is not None:
                matched += 1
                x, y, fw, fh = [int(v) for v in box]
                if region == "mouth":                        # terzo inferiore del volto
                    mx, my, mw, mh = x + int(fw * 0.15), y + int(fh * 0.58), int(fw * 0.70), int(fh * 0.42)
                else:                                        # face: tutto il riquadro volto
                    mx, my, mw, mh = x, y, fw, fh
                o = int(mask_offset)
                x0, y0 = max(0, mx - o), max(0, my - o)
                x1, y1 = min(w, mx + mw + o), min(h, my + mh + o)
                mask[y0:y1, x0:x1] = 1.0
                k = max(5, ((min(fw, fh) // 8) | 1))         # feather dispari proporzionale al volto
                mask = cv2.GaussianBlur(mask, (k, k), 0)
            m3 = mask[..., None]
            comp = (frame.astype(np.float32) * m3 + bg.astype(np.float32) * (1.0 - m3)).astype("uint8")
            vw.write(comp)
        cap.release(); vw.release()
        if matched == 0:                                     # nessun volto → il composite sarebbe tutto sfondo
            os.remove(tmp) if os.path.exists(tmp) else None
            return generated
        # riporta l'audio del video generato sul composito
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", generated,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", out], check=False)
        try:
            os.remove(tmp)
        except OSError:
            pass
        # #106 watermark obbligatorio: il composite ricostruisce lo sfondo dalla FOTO sorgente (non
        # marchiata) → la banda del video a2v generato andrebbe persa; si marchia QUI l'output
        # finale consegnato (GUI Talk + REST /video/lipsync). Se forced e non applicabile,
        # apply_to_file rimuove `out` e solleva → il fail-safe esterno ritorna `generated`
        # (già marchiato da ltx2_infer): mai un output non marchiato.
        if os.path.exists(out):
            from core import watermark
            out = watermark.apply_to_file(out)
        return out if os.path.exists(out) else generated
    except Exception:                                        # noqa: BLE001 — fail-safe totale
        return generated
