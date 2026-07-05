"""Generazione file LRC (liriche sincronizzate / karaoke) da un audio, via Whisper (transformers).

Usa un modello Whisper GIÀ in cache (default openai/whisper-tiny) su CPU: nessun download, nessun
conflitto con la GPU. Trascrive l'audio con timestamp per-segmento e scrive un .lrc `[mm:ss.xx]testo`.
Se le liriche originali sono note NON servono qui: questo è ASR puro (utile anche per strumentale/voce).
"""
from __future__ import annotations

from pathlib import Path


def _fmt(ts: float) -> str:
    ts = max(0.0, float(ts or 0.0))
    m = int(ts // 60)
    return f"[{m:02d}:{ts - m * 60:05.2f}]"


def transcribe_to_lrc(audio_path: str, out_lrc: str | None = None,
                      model: str = "openai/whisper-tiny") -> str:
    """Trascrive l'audio e scrive un .lrc con timestamp per segmento. Ritorna il path del .lrc."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model, device="cpu",
                   chunk_length_s=30, return_timestamps=True)
    res = asr(str(audio_path), return_timestamps=True)
    chunks = res.get("chunks") or []
    lines = []
    for ch in chunks:
        start = (ch.get("timestamp") or (0.0, 0.0))[0]
        text = (ch.get("text") or "").strip()
        if text:
            lines.append(f"{_fmt(start)}{text}")
    if not lines and res.get("text"):
        lines.append(f"{_fmt(0.0)}{res['text'].strip()}")
    out = out_lrc or str(Path(audio_path).with_suffix(".lrc"))
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
