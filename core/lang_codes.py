"""Lingue: la lingua del TESTO e quella della VOCE sono due cose diverse.

  · lingua del testo  = come leggere le parole (fonemi, numeri, sigle). Si dichiara per speaker / battuta e
                        anche dentro una frase col marcatore [lang:en]…[/lang] (core/voice_markup).
  · lingua della voce = proprietà della voce scelta. Edge e Kokoro sono MONOLINGUA (it-IT-ElsaNeural, if_sara:
                        italiane → un testo inglese esce con accento italiano); le voci Edge «Multilingual» e le
                        voci CLONATE (XTTS, F5, Fish, CosyVoice, VibeVoice, Handy clone) sono cross-lingua: timbro
                        dal campione, lingua dal testo.
Nessuna sostituzione automatica della voce: se testo e voce non combaciano si segnala (avviso), a meno che
l'accento sia voluto (allow_accent).
"""
from __future__ import annotations

# codice base → (BCP-47 predefinito, nome inglese usato da MiniMax H3, nomi/alias accettati)
LANGS = {
    "it": ("it-IT", "Italian", ("italiano", "italian", "ita")),
    "en": ("en-US", "English", ("inglese", "english", "eng")),
    "es": ("es-ES", "Spanish", ("spagnolo", "spanish", "español", "espanol", "spa")),
    "fr": ("fr-FR", "French", ("francese", "french", "français", "francais", "fra")),
    "de": ("de-DE", "German", ("tedesco", "german", "deutsch", "deu")),
    "pt": ("pt-BR", "Portuguese", ("portoghese", "portuguese", "português", "portugues", "por")),
    "zh": ("zh-CN", "Chinese", ("cinese", "chinese", "mandarin", "中文", "zho")),
    "ja": ("ja-JP", "Japanese", ("giapponese", "japanese", "日本語", "jpn")),
    "ko": ("ko-KR", "Korean", ("coreano", "korean", "한국어", "kor")),
    "ru": ("ru-RU", "Russian", ("russo", "russian", "русский", "rus")),
    "ar": ("ar-EG", "Arabic", ("arabo", "arabic", "العربية", "ara")),
    "hi": ("hi-IN", "Hindi", ("hindi", "hin")),
    "uk": ("uk-UA", "Ukrainian", ("ucraino", "ukrainian", "українська", "ukr")),
    "el": ("el-GR", "Greek", ("greco", "greek", "ελληνικά", "ell")),
    "hu": ("hu-HU", "Hungarian", ("ungherese", "hungarian", "magyar", "hun")),
    "he": ("he-IL", "Hebrew", ("ebraico", "hebrew", "עברית", "heb")),
    "nl": ("nl-NL", "Dutch", ("olandese", "dutch", "nederlands", "nld")),
    "pl": ("pl-PL", "Polish", ("polacco", "polish", "polski", "pol")),
    "tr": ("tr-TR", "Turkish", ("turco", "turkish", "türkçe", "tur")),
    "cs": ("cs-CZ", "Czech", ("ceco", "czech", "čeština", "ces")),
}

H3_LANGS = ("it", "en", "zh", "es", "fr", "de", "pt", "ja", "ko", "ru", "ar")
# lingue che il motore sa LEGGERE (fonemizzazione / normalizzazione del testo)
ENGINE_TEXT_LANGS = {
    "kokoro": ("it", "en", "es", "fr", "pt", "hi", "ja", "zh"),
    "xtts2": ("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"),
    "f5tts": ("en", "zh", "it", "es", "fr", "de", "ja"),
    "fish": ("en", "zh", "ja", "ko", "fr", "de", "ar", "es", "it", "pt", "ru", "nl", "pl"),
    "cosyvoice2": ("zh", "en", "ja", "ko", "it", "es", "fr", "de", "ru"),
    "vibevoice": ("en", "zh"),
    "handy_clone": ("en", "it", "es", "fr", "de", "pt", "zh", "ja", "ko", "ru", "ar", "hi", "nl", "pl", "tr", "he",
                    "el", "hu", "cs", "uk"),
    "edge": tuple(LANGS),                               # dipende dalla voce: vedi voice_langs()
}
CLONE_ENGINES = {"xtts2", "f5tts", "fish", "cosyvoice2", "vibevoice", "handy_clone"}
# Kokoro: la prima lettera dell'id voce è la lingua (af_/am_ = inglese USA, bf_/bm_ = inglese UK, if_/im_ = …)
_KOKORO_PREFIX = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh"}


def base(code: str | None) -> str:
    """'it-IT' / 'Italian' / 'italiano' / 'it' → 'it'. '' se sconosciuta o vuota."""
    c = (code or "").strip()
    if not c:
        return ""
    low = c.lower().replace("_", "-")
    head = low.split("-")[0]
    if head in LANGS:
        return head
    for k, (_bcp, name, aliases) in LANGS.items():
        if low == name.lower() or low in aliases:
            return k
    return ""


def bcp47(code: str | None) -> str:
    b = base(code)
    if not b:
        return ""
    c = (code or "").strip().replace("_", "-")
    if "-" in c and c.split("-")[0].lower() == b and len(c.split("-")[1]) == 2:
        return f"{b}-{c.split('-')[1].upper()}"                 # rispetta la variante data (en-GB, pt-PT…)
    return LANGS[b][0]


def h3_name(code: str | None) -> str:
    """Nome del tag di battuta di MiniMax H3 ('Italian'…); '' se H3 non supporta la lingua."""
    b = base(code)
    return LANGS[b][1] if b in H3_LANGS else ""


def voice_langs(engine: str, voice_id: str = "", voice_language: str = "", cloned: bool = False) -> set | None:
    """Lingue che la VOCE parla senza accento. None = cross-lingua (clone / Edge multilingue): parla tutte le
    lingue che il motore sa leggere col timbro del campione."""
    if cloned or engine in CLONE_ENGINES:
        return None
    vid = (voice_id or "").strip()
    if engine == "edge":
        if "multilingual" in vid.lower():
            return None
        b = base(vid[:5]) if len(vid) >= 5 and vid[2] == "-" else ""
        return {b} if b else ({base(voice_language)} if base(voice_language) else set())
    if engine == "kokoro":
        b = _KOKORO_PREFIX.get(vid[:1].lower(), "") if len(vid) > 2 and vid[1] in "fm" and vid[2] == "_" else ""
        return {b} if b else ({base(voice_language)} if base(voice_language) else set())
    return {base(voice_language)} if base(voice_language) else set()


def check(engine: str, text_lang: str, *, voice_id: str = "", voice_language: str = "", cloned: bool = False,
          allow_accent: bool = False) -> str:
    """'' se tutto a posto, altrimenti l'avviso da mostrare (testo in una lingua che la voce/motore non copre)."""
    b = base(text_lang)
    if not b:
        return f"lingua del testo non riconosciuta: {text_lang!r}"
    readable = ENGINE_TEXT_LANGS.get(engine)
    if readable and b not in readable:
        return (f"il motore {engine} non sa leggere la lingua {b} (supporta: {', '.join(readable)}): "
                "scegli un altro motore voce")
    langs = voice_langs(engine, voice_id, voice_language, cloned)
    if langs is None or not langs or b in langs or allow_accent:
        return ""
    return (f"la voce {voice_id or engine} è {'/'.join(sorted(langs))} ma il testo è {b}: verrà letto con "
            f"l'accento della voce. Usa una voce {b}, una voce multilingue o clonata, oppure allow_accent=true "
            "se l'accento è voluto")
