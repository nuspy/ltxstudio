"""Integra con Krea 2 Edit: la posizione del soggetto come RIQUADRO disegnato sulla scena, oltre al testo («sulla sinistra / al centro / sulla destra»).

Krea 2 Edit non ha una maschera: nei test del 03-04/11 il soggetto tornava piu piccolo (volto 16 px contro 36 nel composito) e in un punto qualsiasi. Il riquadro (rettangolo rosso attorno
alla sagoma di posizione, con la sua altezza) da` al modello dove e quanto grande; il prompt dice che e solo una guida da non disegnare.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

BOX_COLOR = (255, 0, 0)
BOX_LINE_FRACTION = 0.004          # spessore del contorno: frazione del lato lungo, almeno 3 px


def silhouette_bbox(position_mask_path: str, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Riquadro (x0, y0, x1, y1) della sagoma di posizione, portato alla dimensione `size` (larghezza, altezza) della scena. None se la maschera e vuota."""
    mask = np.asarray(Image.open(position_mask_path).convert("L").resize(size)) > 127
    columns, rows = np.where(mask.any(0))[0], np.where(mask.any(1))[0]
    if len(columns) == 0 or len(rows) == 0:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]), int(rows[-1])


def draw_position_box(scene_path: str, position_mask_path: str, out_path: str) -> tuple[str, tuple[int, int, int, int]] | None:
    """Scrive in `out_path` la scena con il riquadro rosso attorno alla sagoma; ritorna (percorso, riquadro) o None se la maschera e vuota (la scena non si tocca)."""
    scene = Image.open(scene_path).convert("RGB")
    box = silhouette_bbox(position_mask_path, scene.size)
    if box is None:
        return None
    draw = ImageDraw.Draw(scene)
    draw.rectangle(box, outline=BOX_COLOR, width=max(3, int(round(max(scene.size) * BOX_LINE_FRACTION))))
    scene.save(out_path)
    return out_path, box


def box_prompt(where: str, extra: str = "") -> str:
    """Istruzione Krea con il riquadro: il soggetto della seconda immagine va DENTRO il rettangolo rosso della prima, grande quanto il rettangolo; il rettangolo non deve comparire."""
    return ("Add the person from the second image to the first image, standing inside the red rectangle (" + where + " of the scene), as tall as the rectangle, as a real part of "
            "the scene: match the perspective, lighting and colour grading, add contact shadows, let the foreground objects overlap her where they should. Keep her face, "
            "hair and clothes identical to the second image. The red rectangle is only a placement guide: do not draw it. Do not change anything else in the scene and do not "
            "add anyone else. " + extra).strip()
