"""Catena audio dell'avatar parlante: voce (TTS / clone / da descrizione) → dialogo a 1–2 persone →
musica di sottofondo ed effetti → mix → wav pronto per LongCat (o riferimento timbro per H3).

Usata IDENTICA dal pannello «Genera voce» del tab Avatar e dall'endpoint /avatar/generate.
Motori voce: quelli di core/tts_engine (kokoro, edge, handy_clone, cosyvoice2, f5tts, xtts2, fish,
vibevoice). Velocità/tono/volume sono applicati in post-produzione quando il motore non li supporta,
così ogni motore accetta gli stessi parametri.

Dialogo a due: LongCat vuole UNA traccia per persona, lunghe uguale, con silenzio quando parla l'altra
(`audio_type="para"`): costruiamo esattamente quello dal copione «A: … / B: …», più il mix per il video.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SR = 44100
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "output" / "avatar"

ENGINES_NATIVE_SPEED = {"kokoro", "edge"}   # gli altri: velocità via time-stretch in post
SCRIPT_LINE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*(.+)$")


@dataclass
class Speaker:
    """Voce di una persona. Uno fra voice_id (preset della libreria o id Edge/Kokoro),
    voice_description (l'LLM sceglie la voce) o clone_sample (+ trascrizione)."""
    name: str = "A"
    engine: str = "edge"
    voice_id: str = ""
    voice_description: str = ""
    clone_sample: str = ""
    clone_transcript: str = ""
    language: str = ""            # lingua del TESTO (it-IT, en-US…); '' = quella della voce della libreria
    allow_accent: bool = False    # True = testo in una lingua diversa da quella della voce, accento voluto
    speed: float = 1.0            # 0.5–2.0
    pitch: float = 0.0            # semitoni (−12…+12)
    volume_db: float = 0.0        # guadagno
    emotion: str = ""             # neutral|happy|sad|angry|whisper|calm (Edge nativo; marker per gli altri)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Speaker":
        d = dict(d or {})
        clone = d.pop("clone", None) or {}
        s = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if clone:
            s.clone_sample = clone.get("sample_path", s.clone_sample) or ""
            s.clone_transcript = clone.get("transcript", s.clone_transcript) or ""
        return s


@dataclass
class AudioResult:
    path: str                                    # mix finale (voce + musica + effetti)
    tracks: dict[str, str] = field(default_factory=dict)   # person1/person2 → traccia voce (per LongCat)
    audio_type: str = "para"
    duration_s: float = 0.0
    warnings: list = field(default_factory=list)            # es. voce italiana su testo inglese (accento)
    languages: dict = field(default_factory=dict)            # speaker → lingue del testo usate


# ── util audio (numpy/librosa/soundfile: presenti nel venv dell'app) ─────────────────────────────
def _uid(prefix: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"


def load(path: str):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def save(y, path: Path) -> str:
    import numpy as np
    import soundfile as sf
    sf.write(str(path), np.clip(y, -1.0, 1.0).astype("float32"), SR)
    return str(path)


def duration(path: str) -> float:
    import librosa
    return float(librosa.get_duration(path=path))


def silence(seconds: float):
    import numpy as np
    return np.zeros(max(0, int(round(seconds * SR))), dtype="float32")


def _db(y, gain_db: float):
    return y * (10.0 ** (float(gain_db) / 20.0)) if gain_db else y


def postprocess_voice(y, *, speed: float = 1.0, pitch: float = 0.0, volume_db: float = 0.0,
                      native_speed: bool = False):
    import librosa
    if not native_speed and speed and abs(float(speed) - 1.0) > 1e-3:
        y = librosa.effects.time_stretch(y, rate=float(speed))
    if pitch and abs(float(pitch)) > 1e-3:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=float(pitch))
    return _db(y, volume_db)


def loudnorm(path: str, target_lufs: float = -16.0) -> str:
    """Normalizzazione EBU R128 con ffmpeg; se ffmpeg manca, normalizzazione di picco."""
    ff = shutil.which("ffmpeg")
    if ff:
        out = str(Path(path).with_name(Path(path).stem + "_ln.wav"))
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", path, "-af",
                            f"loudnorm=I={float(target_lufs)}:TP=-1.5:LRA=11", "-ar", str(SR), out],
                           capture_output=True, text=True)
        if r.returncode == 0 and Path(out).is_file():
            return out
    import numpy as np
    y = load(path)
    peak = float(np.max(np.abs(y))) or 1.0
    return save(y * (0.89 / peak), Path(path).with_name(Path(path).stem + "_pk.wav"))


# ── voce ──────────────────────────────────────────────────────────────────────────────────────
def _voice_preset(sp: Speaker):
    """(VoicePreset, motore) per tts_engine: clone da campione, voce salvata in libreria (il suo motore),
    id diretto Edge/Kokoro, voce da descrizione (Edge)."""
    from core import tts_engine as T
    if sp.clone_sample:
        return (T.VoicePreset(name=f"avatar_{sp.name}", engine=sp.engine, voice_id=sp.voice_id,
                              language=sp.language, sample_path=sp.clone_sample,
                              sample_transcript=sp.clone_transcript), sp.engine)
    if sp.voice_id:
        for v in T.list_user_voices():                      # id = slug o nome di una voce salvata
            if sp.voice_id in (v.slug(), v.name):
                return v, (v.engine or sp.engine)
        return (T.VoicePreset(name=f"avatar_{sp.name}", engine=sp.engine, voice_id=sp.voice_id,
                              language=sp.language), sp.engine)
    if sp.voice_description:
        vid = voice_from_description(sp.voice_description, sp.language)
        return T.VoicePreset(name=f"avatar_{sp.name}", engine="edge", voice_id=vid,
                             language=sp.language), "edge"
    # nessuna voce indicata: quella predefinita DELLA LINGUA DEL TESTO (non sempre italiana)
    from core import lang_codes as LC
    lang = LC.base(sp.language) or "it"
    vid = (DEFAULT_VOICES.get(sp.engine) or {}).get(lang, "")
    return (T.VoicePreset(name=f"avatar_{sp.name}", engine=sp.engine, voice_id=vid,
                          language=sp.language), sp.engine)


# voce predefinita per motore e lingua del testo (usata SOLO se lo speaker non indica nessuna voce)
DEFAULT_VOICES = {
    "edge": {"it": "it-IT-ElsaNeural", "en": "en-US-AriaNeural", "es": "es-ES-ElviraNeural",
             "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural", "pt": "pt-BR-FranciscaNeural",
             "zh": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural",
             "ru": "ru-RU-SvetlanaNeural", "ar": "ar-EG-SalmaNeural", "hi": "hi-IN-SwaraNeural",
             "uk": "uk-UA-PolinaNeural", "el": "el-GR-AthinaNeural", "hu": "hu-HU-NoemiNeural",
             "he": "he-IL-HilaNeural", "nl": "nl-NL-ColetteNeural", "pl": "pl-PL-ZofiaNeural",
             "tr": "tr-TR-EmelNeural", "cs": "cs-CZ-VlastaNeural"},
    "kokoro": {"it": "if_sara", "en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "pt": "pf_dora",
               "hi": "hf_alpha", "ja": "jf_alpha", "zh": "zf_xiaobei"},
}


def voice_from_description(description: str, language: str = "it-IT") -> str:
    """Voce Edge scelta dall'LLM a partire da una descrizione («donna anziana, calda»).
    Niente scritture su DB/libreria (a differenza di film_studio.voice_studio)."""
    from core import tts_engine as T
    gender = "female"
    try:
        from core import llm_service as L
        attrs = L.complete_json(
            'Map a voice description to JSON: {"gender":"male|female","age":"child|young|adult|old"}. '
            "Output ONLY the JSON object.", description, max_tokens=80)
        gender = str(attrs.get("gender", "female")).lower()
    except Exception:                                        # noqa: BLE001 — euristica di riserva
        low = description.lower()
        if any(w in low for w in ("uomo", "man", "male", "maschi", "ragazzo", "signore", "voce grave")):
            gender = "male"
    presets = T.edge_list_voices((language or "it")[:2]) if T.edge_available() else []
    for p in presets:
        if ("female" in (p.name or "").lower()) == (gender == "female"):
            return p.voice_id
    if presets:
        return presets[0].voice_id
    return "it-IT-DiegoNeural" if gender == "male" else "it-IT-ElsaNeural"


def text_language(sp: Speaker, voice=None) -> str:
    """Lingua del TESTO dello speaker: quella dichiarata, altrimenti quella della voce della libreria /
    dell'id Edge o Kokoro. Se non si ricava, errore chiaro: niente lingua indovinata."""
    from core import lang_codes as LC
    if LC.base(sp.language):
        return LC.bcp47(sp.language)
    if voice is not None and LC.base(getattr(voice, "language", "")) and getattr(voice, "custom", True):
        return LC.bcp47(voice.language)
    for engine in (sp.engine,):
        langs = LC.voice_langs(engine, sp.voice_id)
        if langs and len(langs) == 1:
            return LC.bcp47(next(iter(langs)))
    raise ValueError(f"speaker «{sp.name}»: indica la lingua del testo (language, es. it-IT / en-US): "
                     "con questa voce non si ricava da sola")


def synth_line(sp: Speaker, text: str, notes: list | None = None) -> str:
    """Una battuta → wav 44.1 kHz. Accetta i marcatori espressivi di core/voice_markup (pause, emozioni,
    velocità/accelerando, tono, volume, enfasi): si sintetizza SEGMENTO PER SEGMENTO e velocità/tono/volume
    si applicano in post quando il motore non li gestisce, così ogni motore capisce gli stessi marcatori."""
    import numpy as np
    from core import tts_engine as T
    from core import voice_markup as VM
    from core import lang_codes as LC
    if (sp.voice_description or not (sp.voice_id or sp.clone_sample)) and not LC.base(sp.language):
        raise ValueError(f"speaker «{sp.name}»: indica la lingua del testo (language, es. it-IT / en-US): "
                         "serve per scegliere la voce")
    v, engine = _voice_preset(sp)
    native = engine in ENGINES_NATIVE_SPEED
    line_lang = text_language(sp, v)
    base = VM.Style(speed=sp.speed, pitch=sp.pitch, volume_db=sp.volume_db,
                    emotion=sp.emotion if sp.emotion in VM.EMOTIONS else "", lang=LC.base(line_lang))
    fade = int(SR * 0.008)
    parts = []
    for seg in VM.parse(text, base):
        if seg.kind == "pause":
            parts.append(silence(seg.ms / 1000.0))
            continue
        seg_lang = LC.bcp47(seg.style.lang) if seg.style.lang else line_lang
        warn = LC.check(engine, seg_lang, voice_id=v.voice_id, voice_language=v.language,
                        cloned=bool(sp.clone_sample), allow_accent=sp.allow_accent)
        if warn and notes is not None and f"{sp.name}: {warn}" not in notes:
            notes.append(f"{sp.name}: {warn}")
        for c in VM.accel_chunks(seg):
            speed, pitch, vol = VM.effective(c.style)
            raw = T.synthesize(engine, v, c.text, speed=(speed if native else 1.0), language=seg_lang)
            y = postprocess_voice(load(str(raw)), speed=speed, pitch=pitch, volume_db=vol, native_speed=native)
            if len(y) > 2 * fade:                            # micro-dissolvenze: niente click fra i segmenti
                y = y.copy()
                y[:fade] *= np.linspace(0, 1, fade)
                y[-fade:] *= np.linspace(1, 0, fade)
            parts.append(y)
    if not parts:
        raise ValueError("battuta vuota")
    return save(np.concatenate(parts), _uid(f"voice_{sp.name}"))


def parse_script(text: str, speaker_names: list[str]) -> list[tuple[int, str]]:
    """Copione «A: …» / «Nome: …» → [(indice_speaker, battuta)]. Righe senza prefisso = speaker 0."""
    keys = [n.strip().lower() for n in speaker_names]
    alias = {"a": 0, "b": 1, "1": 0, "2": 1, "person1": 0, "person2": 1}
    out: list[tuple[int, str]] = []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        m = SCRIPT_LINE.match(raw)
        if m:
            who = m.group(1).strip().lower()
            idx = keys.index(who) if who in keys else alias.get(who)
            if idx is not None and idx < max(2, len(keys)):
                out.append((idx, m.group(2).strip()))
                continue
        if out:                                              # continuazione della battuta precedente
            i, prev = out[-1]
            out[-1] = (i, prev + " " + raw.strip())
        else:
            out.append((0, raw.strip()))
    return out


def build_speech(text: str, speakers: list[Speaker], *, gap_ms: int = 350,
                 pad_start_ms: int = 0, pad_end_ms: int = 0) -> AudioResult:
    """Voce da testo. 1 speaker → una traccia; 2 speaker → tracce allineate (silenzio mentre parla l'altro)."""
    import numpy as np
    if not speakers:
        speakers = [Speaker()]
    lines = parse_script(text, [s.name for s in speakers]) if len(speakers) > 1 else [(0, text.strip())]
    if not lines or not any(t for _, t in lines):
        raise ValueError("Testo della voce vuoto.")
    n = min(2, len(speakers))
    tracks = [silence(pad_start_ms / 1000.0) for _ in range(n)]
    gap = silence(gap_ms / 1000.0)
    notes: list[str] = []
    for i, (who, line) in enumerate(lines):
        who = min(who, n - 1)
        y = load(synth_line(speakers[who], line, notes))
        for k in range(n):
            tracks[k] = np.concatenate([tracks[k], y if k == who else np.zeros_like(y)])
            if i < len(lines) - 1:
                tracks[k] = np.concatenate([tracks[k], gap])
    tail = silence(pad_end_ms / 1000.0)
    tracks = [np.concatenate([t, tail]) for t in tracks]
    paths = {f"person{k + 1}": save(t, _uid(f"track{k + 1}")) for k, t in enumerate(tracks)}
    mix = tracks[0] if n == 1 else tracks[0] + tracks[1]
    mix_path = save(mix, _uid("speech"))
    from core import voice_markup as VM
    langs = {}
    for who, line in lines:
        sp = speakers[min(who, n - 1)]
        try:
            main = text_language(sp, _voice_preset(sp)[0])
        except ValueError:
            main = sp.language
        found = {main} | {s.style.lang for s in VM.parse(line) if s.kind == "text" and s.style.lang}
        langs.setdefault(sp.name, set()).update(x for x in found if x)
    return AudioResult(path=mix_path, tracks=paths, audio_type="para", duration_s=len(mix) / SR,
                       warnings=notes, languages={k: sorted(v) for k, v in langs.items()})


def trim_file(path: str, start: float = 0.0, end: float = 0.0) -> str:
    """Regione [start, end] di un file (0 = fino alla fine)."""
    y = load(path)
    s = max(0, int(start * SR))
    e = int(end * SR) if end and end > start else len(y)
    return save(y[s:e], _uid("trim"))


def fit_duration(path: str, target_s: float, mode: str = "pad") -> str:
    """Adatta la voce a una durata: pad (silenzio in coda), speed (accelera/rallenta ≤25%), error."""
    import numpy as np
    y = load(path)
    cur = len(y) / SR
    if not target_s or abs(cur - target_s) < 0.05:
        return path
    if mode == "error":
        raise ValueError(f"La voce dura {cur:.1f}s, richiesti {target_s:.1f}s.")
    if mode == "speed" and 0.75 <= cur / target_s <= 1.25:
        import librosa
        y = librosa.effects.time_stretch(y, rate=cur / target_s)
    elif cur < target_s:
        y = np.concatenate([y, silence(target_s - cur)])
    else:
        y = y[: int(target_s * SR)]
    return save(y, _uid("fit"))


def mix_bed(voice_path: str, bed_path: str, *, bed_db: float = -18.0, ducking: bool = True) -> str:
    """Mixa musica/effetti sotto la voce; con ducking la base scende di altri 8 dB mentre si parla."""
    import numpy as np
    v = load(voice_path)
    b = load(bed_path)
    if len(b) < len(v):                                      # base più corta: loop
        b = np.tile(b, int(np.ceil(len(v) / max(1, len(b)))))
    b = _db(b[: len(v)], bed_db)
    if ducking:
        win = int(0.05 * SR) or 1
        env = np.convolve(np.abs(v), np.ones(win) / win, mode="same")
        talk = env > (0.02 * (np.max(env) or 1.0))
        duck = np.where(talk, 10 ** (-8 / 20), 1.0).astype("float32")
        k = int(0.2 * SR) or 1
        duck = np.convolve(duck, np.ones(k) / k, mode="same")
        b = b * duck
    return save(v + b, _uid("mix"))


def generate_music(prompt: str, seconds: float, engine: str = "acestep", on_log=None) -> str:
    """Base musicale strumentale con ACE-Step (o HeartMuLa). Solleva se il motore non è installato."""
    t0 = time.time()
    name = f"avatar_bed_{uuid.uuid4().hex[:6]}"
    if engine == "heartmula":
        from core import music_service as ms
        p = ms.SongParams(name=name, tags=prompt, duration_s=int(max(10, seconds + 2)), instrumental=True)
        errs = ms.preflight_song(p)
        if errs:
            raise RuntimeError("; ".join(errs))
        ms.make_song_job(p, on_log=on_log).run()
        out = p.out_mp3()
    else:
        from core import acestep_service as ac
        errs = ac.preflight(prompt)
        if errs:
            raise RuntimeError("; ".join(errs))
        ac.make_job(prompt + ", instrumental", "[instrumental]", out_name=name,
                    duration=float(max(10, seconds + 2)), on_log=on_log).run()
        out = next(iter(sorted(ac.OUT_DIR.glob(f"{name}.*"))), None)
    if not out or not Path(out).is_file() or Path(out).stat().st_mtime < t0:
        raise RuntimeError(f"Generazione musica ({engine}) senza output.")
    return str(out)


def generate_sfx(prompt: str, seconds: float, video_path: str = "", on_log=None) -> str:
    """Effetti/ambiente con MMAudio: da testo, oppure sonorizzando il video finale (video_path)."""
    from core import music_service as ms
    t0 = time.time()
    p = ms.SonifyParams(name=f"avatar_sfx_{uuid.uuid4().hex[:6]}", video_path=video_path or "",
                        prompt=prompt or "", duration_s=float(max(1.0, seconds)))
    errs = ms.preflight_sonify(p)
    if errs:
        raise RuntimeError("; ".join(errs))
    ms.make_sonify_job(p, on_log=on_log).run()
    out = ms.find_latest_audio(after=t0)
    if not out:
        raise RuntimeError("MMAudio senza output.")
    return str(out)
