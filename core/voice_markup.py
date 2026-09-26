"""Convenzioni espressive della voce, UGUALI per tutti i motori (TTS dell'app e MiniMax H3).

Nel testo di una battuta si possono mettere questi marcatori (maiuscole/minuscole indifferenti):

  pause      [pause:800ms]  [pause:1.5s]  [pause]  (= 500 ms)   [breath] (= respiro, 350 ms)
  emozione   [emotion:happy] … [/emotion]    (anche la forma compatibile <emotion type="happy">…</emotion>)
             valori: neutral happy sad angry whisper calm excited fearful surprised tender serious
  velocità   [speed:1.2] … [/speed]          0.5–2.0 (1 = normale)
  accelerando/rallentando  [accel:0.9-1.3] … [/accel]   la velocità passa gradualmente da 0.9 a 1.3
  tono       [pitch:+2] … [/pitch]           semitoni −12…+12
  volume     [volume:-6] … [/volume]         dB −30…+12   ([loud] … [/loud] = +5 dB, [soft] … [/soft] = −6 dB)
  sussurro   [whisper] … [/whisper]          (= emozione whisper)
  enfasi     [emph] … [/emph]                (+3 dB, +1 semitono)
  compatibile SSML: <prosody rate="+20%" pitch="+2st" volume="-3dB"> … </prosody>

I marcatori si possono annidare; i valori si combinano con quelli della voce (velocità ×, tono +, volume +).
Il TTS sintetizza segmento per segmento e applica velocità/tono/volume in post-produzione quando il motore non
li supporta; MiniMax H3 genera la voce da sé: lì gli stessi marcatori diventano indicazioni di recitazione
("whispers", "quickly", "loudly"…) fuori dal tag <d>, e il testo dentro <d> resta pulito (guida ufficiale H3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

EMOTIONS = {        # emozione → (velocità ×, tono semitoni, volume dB) applicati in post-produzione
    "neutral": (1.0, 0.0, 0.0), "happy": (1.05, 1.5, 0.0), "sad": (0.88, -1.5, -3.0),
    "angry": (1.10, 0.5, 4.0), "whisper": (0.92, -1.0, -12.0), "calm": (0.95, -0.5, -1.0),
    "excited": (1.15, 2.0, 3.0), "fearful": (1.08, 1.5, -2.0), "surprised": (1.05, 2.5, 2.0),
    "tender": (0.92, 0.5, -4.0), "serious": (0.95, -1.0, 0.0),
}
DELIVERY_EN = {     # per MiniMax H3 (indicazioni di recitazione in inglese, come vuole la guida dei prompt)
    "happy": "cheerfully", "sad": "in a sad, subdued voice", "angry": "angrily", "whisper": "in a whisper",
    "calm": "calmly", "excited": "excitedly", "fearful": "in a frightened, shaky voice",
    "surprised": "in surprise", "tender": "tenderly", "serious": "in a serious tone",
}
PAUSE_DEFAULT_MS, BREATH_MS = 500, 350

_TOKEN = re.compile(
    r"\[(?P<close>/)?(?P<tag>pause|breath|emotion|speed|accel|pitch|volume|loud|soft|whisper|emph)"
    r"(?::(?P<arg>[^\]]*))?\]"
    r"|<\s*emotion\s+type\s*=\s*['\"]?(?P<emo>[a-z]+)['\"]?\s*>|<\s*/\s*emotion\s*>"
    r"|<\s*prosody(?P<pattrs>[^>]*)>|<\s*/\s*prosody\s*>", re.I)


@dataclass
class Style:
    speed: float = 1.0
    pitch: float = 0.0
    volume_db: float = 0.0
    emotion: str = ""
    accel: tuple | None = None          # (da, a) della velocità, graduale lungo il segmento

    def combined(self, speed: float = 1.0, pitch: float = 0.0, volume_db: float = 0.0) -> "Style":
        return replace(self, speed=self.speed * speed, pitch=self.pitch + pitch, volume_db=self.volume_db + volume_db)


@dataclass
class Segment:
    kind: str                            # "text" | "pause"
    text: str = ""
    ms: int = 0
    style: Style = field(default_factory=Style)


def _f(s: str, default: float) -> float:
    try:
        return float(str(s).strip().replace(",", "."))
    except (TypeError, ValueError):
        return default


def _pause_ms(arg: str | None) -> int:
    if not arg:
        return PAUSE_DEFAULT_MS
    a = arg.strip().lower()
    if a.endswith("ms"):
        return int(_f(a[:-2], PAUSE_DEFAULT_MS))
    if a.endswith("s"):
        return int(_f(a[:-1], 0.5) * 1000)
    return int(_f(a, PAUSE_DEFAULT_MS))


def _prosody(attrs: str) -> dict:
    out = {}
    for k, v in re.findall(r"(rate|pitch|volume)\s*=\s*['\"]([^'\"]+)['\"]", attrs or "", re.I):
        v = v.strip().lower()
        if k.lower() == "rate":
            out["speed"] = 1.0 + _f(v.rstrip("%"), 0) / 100.0 if v.endswith("%") else _f(v, 1.0)
        elif k.lower() == "pitch":
            out["pitch"] = _f(v.replace("st", "").rstrip("%"), 0.0)
        else:
            out["volume_db"] = _f(v.replace("db", ""), 0.0)
    return out


def parse(text: str, base: Style | None = None) -> list[Segment]:
    """Testo con marcatori → segmenti (testo con il suo stile | pausa). Marcatori sconosciuti restano testo."""
    stack = [base or Style()]
    segs: list[Segment] = []
    pos = 0

    def emit(chunk: str):
        if chunk.strip():
            segs.append(Segment("text", re.sub(r"\s+", " ", chunk).strip(), style=stack[-1]))

    for m in _TOKEN.finditer(text or ""):
        emit(text[pos:m.start()])
        pos = m.end()
        tok = m.group(0).lower()
        cur = stack[-1]
        if m.group("emo"):
            e = m.group("emo").lower()
            stack.append(replace(cur, emotion=e if e in EMOTIONS else cur.emotion))
            continue
        if tok.startswith("</") or m.group("close"):
            if len(stack) > 1:
                stack.pop()
            continue
        if m.group("pattrs") is not None:
            stack.append(cur.combined(**_prosody(m.group("pattrs"))))
            continue
        tag, arg = (m.group("tag") or "").lower(), m.group("arg")
        if tag == "pause":
            segs.append(Segment("pause", ms=max(0, min(10000, _pause_ms(arg)))))
        elif tag == "breath":
            segs.append(Segment("pause", ms=BREATH_MS))
        elif tag == "emotion":
            e = (arg or "").strip().lower()
            stack.append(replace(cur, emotion=e if e in EMOTIONS else cur.emotion))
        elif tag == "whisper":
            stack.append(replace(cur, emotion="whisper"))
        elif tag == "speed":
            stack.append(cur.combined(speed=max(0.5, min(2.0, _f(arg, 1.0)))))
        elif tag == "accel":
            a, _, b = (arg or "1-1").partition("-")
            stack.append(replace(cur, accel=(max(0.5, min(2.0, _f(a, 1.0))), max(0.5, min(2.0, _f(b, 1.0))))))
        elif tag == "pitch":
            stack.append(cur.combined(pitch=max(-12.0, min(12.0, _f(arg, 0.0)))))
        elif tag == "volume":
            stack.append(cur.combined(volume_db=max(-30.0, min(12.0, _f(arg, 0.0)))))
        elif tag == "loud":
            stack.append(cur.combined(volume_db=5.0))
        elif tag == "soft":
            stack.append(cur.combined(volume_db=-6.0))
        elif tag == "emph":
            stack.append(cur.combined(pitch=1.0, volume_db=3.0))
    emit((text or "")[pos:])
    return segs


def has_markup(text: str) -> bool:
    return bool(_TOKEN.search(text or ""))


def plain(text: str) -> str:
    """Solo le parole (per sottotitoli, <d> di H3, anteprime)."""
    out = " ".join(s.text for s in parse(text) if s.kind == "text").strip()
    return re.sub(r"\s+([,.;:!?…])", lambda m: m.group(1), out)


def effective(style: Style) -> tuple[float, float, float]:
    """(velocità, tono, volume) finali dello stile, emozione compresa."""
    es, ep, ev = EMOTIONS.get(style.emotion or "neutral", (1.0, 0.0, 0.0))
    return style.speed * es, style.pitch + ep, style.volume_db + ev


def accel_chunks(seg: Segment) -> list[Segment]:
    """Accelerando/rallentando: divide il segmento per frasi/virgole e interpola la velocità."""
    if not seg.style.accel:
        return [seg]
    parts = [p.strip() for p in re.split(r"(?<=[,.;:!?])\s+", seg.text) if p.strip()]
    if len(parts) < 2:
        words = seg.text.split()
        k = max(2, min(4, len(words) // 3))
        step = max(1, len(words) // k)
        parts = [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
    a, b = seg.style.accel
    n = len(parts)
    return [Segment("text", t, style=replace(seg.style, accel=None,
                                             speed=seg.style.speed * (a + (b - a) * (i / max(1, n - 1)))))
            for i, t in enumerate(parts)]


def delivery_en(text: str) -> str:
    """Indicazione di recitazione in inglese (per MiniMax H3) dallo stile PREVALENTE della battuta."""
    segs = [s for s in parse(text) if s.kind == "text"]
    if not segs:
        return ""
    main = max(segs, key=lambda s: len(s.text)).style
    bits = []
    if main.emotion and main.emotion in DELIVERY_EN:
        bits.append(DELIVERY_EN[main.emotion])
    speed = main.speed if not main.accel else (main.accel[0] + main.accel[1]) / 2
    if main.accel and main.accel[1] > main.accel[0] * 1.1:
        bits.append("speeding up as the line goes on")
    elif main.accel and main.accel[1] < main.accel[0] * 0.9:
        bits.append("slowing down as the line goes on")
    elif speed >= 1.15:
        bits.append("quickly")
    elif speed <= 0.85:
        bits.append("slowly")
    if main.volume_db >= 4:
        bits.append("loudly")
    elif main.volume_db <= -5 and main.emotion != "whisper":
        bits.append("softly")
    if main.pitch >= 2:
        bits.append("in a higher-pitched voice")
    elif main.pitch <= -2:
        bits.append("in a deeper voice")
    if any(s.kind == "pause" for s in parse(text)):
        bits.append("with deliberate pauses")
    return ", ".join(bits)
