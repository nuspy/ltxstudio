"""Durata ↔ segmenti di LongCat-Video-Avatar 1.5 (stdlib-only: lo importa anche il worker nel suo venv).

Il modello genera clip da 93 fotogrammi a 25 fps; ogni continuazione riusa gli ultimi 13 come
condizione e ne aggiunge 80. La durata del video la decide quindi l'AUDIO: si calcolano i segmenti
che lo coprono e il video finale viene tagliato alla lunghezza esatta della voce.
"""
from __future__ import annotations

import math

FPS = 25
NUM_FRAMES = 93          # fotogrammi del primo segmento
NUM_COND_FRAMES = 13     # fotogrammi di condizione riusati da ogni continuazione
FIRST_S = NUM_FRAMES / FPS                        # 3.72 s
STEP_S = (NUM_FRAMES - NUM_COND_FRAMES) / FPS     # 3.20 s


def segments_for(duration_s: float) -> int:
    """Segmenti necessari a coprire `duration_s` secondi di audio (minimo 1)."""
    d = max(0.0, float(duration_s or 0.0))
    if d <= FIRST_S:
        return 1
    return 1 + math.ceil((d - FIRST_S) / STEP_S - 1e-9)


def covered_seconds(num_segments: int) -> float:
    """Secondi generati da `num_segments` segmenti (prima del taglio sulla durata audio)."""
    n = max(1, int(num_segments))
    return (NUM_FRAMES + (n - 1) * (NUM_FRAMES - NUM_COND_FRAMES)) / FPS


def effective_duration(audio_s: float, *, trim_start: float = 0.0, trim_end: float = 0.0,
                       max_s: float = 0.0) -> float:
    """Durata usata davvero: regione selezionata (trim) e limite opzionale `max_s` (0 = nessuno)."""
    a = max(0.0, float(audio_s or 0.0))
    s = min(max(0.0, float(trim_start or 0.0)), a)
    e = float(trim_end or 0.0)
    e = a if e <= s or e > a else e
    d = e - s
    if max_s and max_s > 0:
        d = min(d, float(max_s))
    return max(0.0, d)
