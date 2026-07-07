"""Reference-chaining visivo: l'ULTIMO frame dello shot N diventa il conditioning (frame 0) dello shot N+1,
così i tagli fra riprese consecutive della STESSA scena risultano raccordati (niente stacco brusco di
posa/luce/sfondo). Estrae il frame con ffmpeg; il conditioning riusa il meccanismo keyframe già esistente
(keyframes=[{path, frame, strength}]). Opzionale e per-scena: non è un'automazione sul contenuto, è un raccordo
richiesto esplicitamente dall'utente."""
from __future__ import annotations

import os
import subprocess


def last_frame(video_path: str, out_png: str) -> str | None:
    """Estrae l'ultimo frame di un video in un PNG (per il raccordo con lo shot successivo). None se fallisce."""
    if not (video_path and os.path.exists(video_path)):
        return None
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    # -sseof -0.1 posiziona a fine file; -update 1 tiene solo l'ultimo frame decodificato
    r = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.2", "-i", video_path,
                        "-update", "1", "-frames:v", "1", out_png, "-y"], check=False)
    return out_png if (r.returncode == 0 and os.path.exists(out_png)) else None


def prev_shot_link(prev_video: str, out_png: str, strength: float = 0.35) -> list[dict] | None:
    """Costruisce il keyframe di raccordo dallo shot precedente: [{path, frame:0, strength}] o None.
    strength BASSA (default 0.35): àncora la continuità ma lascia evolvere il movimento (non congela)."""
    fp = last_frame(prev_video, out_png)
    if not fp:
        return None
    return [{"path": fp, "frame": 0, "strength": float(strength)}]
