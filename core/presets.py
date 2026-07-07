"""Preset impostazioni predefinite per LTX Studio — allineati alla pipeline verificata LTX-2.3.

Campionamento UNIFORME e verificato: sampler `euler_ancestral_cfg_pp` + `LTXVScheduler (shift)` +
CFG 1.0 (la guida la dà il NAG). I sampler RES e i CFG alti (2.5-4.5) dei vecchi preset davano
rumore/over-cooking → rimossi. I preset ora variano solo per RISOLUZIONE / FPS / DURATA / STEP
(il loro vero scopo); il campionamento resta quello buono per tutti.
"""
from __future__ import annotations

_SAMPLER = "euler_ancestral_cfg_pp"
_SCHED = "LTXVScheduler (shift)"

VIDEO_PRESETS: dict[str, dict] = {
    # ── Modalità ESPLICITE Fast/Pro/HQ (review 07/07): tre livelli chiari, stessa risoluzione di
    # lavoro 1280×704 (cambia solo il compromesso velocità/qualità). I preset SCRIVONO nei controlli
    # GUI (regola: l'ottimizzazione imposta i controlli, non il pipeline direttamente).
    # teacache (audit acceleratori 07/07): FAST usa la cache aggressiva (bozza, +2× velocità), PRO
    # moderata (drift impercettibile a 24 step), HQ 0.0 (purezza assoluta, nessuna scorciatoia).
    "⚡ FAST (bozza ~1 min)": {
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": _SCHED,
        "width": 704, "height": 384, "fps": 16, "length_s": 4.0, "teacache": 0.05,
        "description": "Modalità FAST: 8 step, risoluzione ridotta, TeaCache — bozza per validare l'idea",
    },
    "🎯 PRO (equilibrato)": {
        "steps": 24, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 1280, "height": 704, "fps": 24, "length_s": 6.0, "teacache": 0.03,
        "description": "Modalità PRO: 24 step full-res, TeaCache moderata — miglior compromesso",
    },
    "💎 HQ (massima qualità)": {
        "steps": 44, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 1280, "height": 704, "fps": 24, "length_s": 8.0, "teacache": 0.0,
        "description": "Modalità HQ: 44 step full-res, nessuna cache — massima qualità, il più lento",
    },
    "🎯 Migliore qualità": {
        "steps": 40, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "description": "Qualità massima, 40 step (lento)",
    },
    "⚡ Preview rapida": {
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": _SCHED,
        "width": 480, "height": 832, "fps": 16, "length_s": 4.0,
        "description": "Bozza veloce per testare l'idea (~1 min)",
    },
    "📺 Broadcast 16:9": {
        "steps": 30, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 1280, "height": 704, "fps": 25, "length_s": 6.0,
        "description": "Formato TV/YouTube, 25fps PAL",
    },
    "📱 Social 9:16": {
        "steps": 30, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 704, "height": 1280, "fps": 30, "length_s": 6.0,
        "description": "Reels/TikTok/Shorts verticale 30fps",
    },
    "🎬 Cinemascope": {
        "steps": 35, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 1280, "height": 544, "fps": 24, "length_s": 10.0,
        "description": "2.35:1 cinematografico, 24fps film",
    },
    "🎥 Anamorphic 2.39:1": {
        "steps": 35, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 1280, "height": 544, "fps": 24, "length_s": 8.0,
        "description": "Formato anamorphic cinema",
    },
    "🚀 GGUF fast (RAM bassa)": {
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": _SCHED,
        "width": 704, "height": 1280, "fps": 24, "length_s": 5.0,
        "description": "Ottimizzato per GGUF quantizzato, meno VRAM",
    },
    "🌟 Balanced": {
        "steps": 24, "cfg": 1.0, "sampler": _SAMPLER, "scheduler": _SCHED,
        "width": 704, "height": 1280, "fps": 24, "length_s": 8.0,
        "description": "Bilanciamento qualità/velocità per uso quotidiano",
    },
}
