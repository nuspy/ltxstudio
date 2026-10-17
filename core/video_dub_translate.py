"""«Estrai e traduci» del Doppiaggio da testo: STT locale dal video → traduzione con l'LLM locale → testo della battuta.

Tutto LOCALE (nessun servizio esterno): STT = faster-whisper se installato (da' anche lingua e tempi reali), altrimenti
Handy-API (`core/handy_api.py`, Whisper, :8720, solo testo); traduzione = `core/llm_service.py` (LM Studio o Gemma locale).
Nessun fallback silenzioso verso altro: se non c'e' nessun STT o l'LLM non e' attivo, `DubTranslateError` con il messaggio
su cosa avviare. STT e LLM usano la GPU: la GUI li lancia in un thread e a fine lavoro l'LLM viene scaricato.

La traduzione deve STARE NEL TEMPO del video (il labiale e' generato per la durata del sorgente): all'LLM si chiede una
lunghezza parlata simile all'originale (parole attese = secondi di parlato × ~2,5; caratteri × ~4,5 per CJK); se il testo
tradotto e' comunque molto piu' lungo dei secondi disponibili, un SECONDO giro chiede di accorciarlo e poi si avvisa.

Modulo senza Qt e senza torch (i backend sono importati al momento dell'uso: i test li sostituiscono con mock).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

from core import video_edit_ic as vic
from core import video_ic_prep as prep


class DubTranslateError(Exception):
    """Errore per l'utente: `key` = chiave i18n del messaggio, `str(e)` = testo italiano di riserva."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


def spoken_langs() -> list[tuple[str, str]]:
    """[(codice, nome inglese)] delle SOLE lingue parlabili (vedi video_edit_ic.LIPDUB_SPOKEN_LANGS: proposta)."""
    return [(c, n) for c, n, _e in vic.LIPDUB_SPOKEN_LANGS]


def _lang_name(code: str) -> str:
    return dict(spoken_langs()).get((code or "").strip().lower(), "")


def _extract_wav(video: str, out_wav: str) -> str:
    r = subprocess.run([prep.ffmpeg_path(), "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        out_wav], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(out_wav):
        raise DubTranslateError("vedit.dub.err_no_speech",
                                "Non riesco a leggere l'audio del video: " + (r.stderr or "")[-200:])
    return out_wav


def extract_speech(video: str, log=None) -> dict:
    """STT del parlato del video → {"text", "language" ('' se ignota), "speech_seconds" (0 se ignoti), "backend"}."""
    log = log or (lambda s: None)
    with tempfile.TemporaryDirectory(prefix="dubstt_") as td:
        wav = _extract_wav(video, os.path.join(td, "speech.wav"))
        try:
            import faster_whisper  # noqa: F401
            have_fw = True
        except Exception:                                              # noqa: BLE001
            have_fw = False
        if have_fw:
            from faster_whisper import WhisperModel
            import torch
            cuda = torch.cuda.is_available()
            log("[dub] STT: faster-whisper locale…")
            model = WhisperModel("small", device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
            segs, info = model.transcribe(wav, word_timestamps=False)
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            if not text:
                raise DubTranslateError("vedit.dub.err_no_speech", "Nessun parlato riconosciuto nel video.")
            return {"text": text, "language": (info.language or "").lower(),
                    "speech_seconds": float(sum(max(0.0, s.end - s.start) for s in segs)), "backend": "faster-whisper"}
        from core import handy_api
        if not (handy_api.available() or handy_api.ensure()):
            raise DubTranslateError("vedit.dub.err_stt",
                                    "STT locale non disponibile: avvia Handy-API (C:\\Projects\\Handy-API, porta 8720) "
                                    "oppure installa faster-whisper.")
        log("[dub] STT: Handy-API (Whisper)…")
        text = (handy_api.transcribe(wav) or "").strip()
        if not text:
            raise DubTranslateError("vedit.dub.err_no_speech", "Nessun parlato riconosciuto nel video.")
        return {"text": text, "language": "", "speech_seconds": 0.0, "backend": "handy"}


def _ask_llm(text: str, target_name: str, n_words: int, seconds: float, *, hard: bool, cjk: bool) -> dict:
    from core import llm_service as llm
    unit = "characters" if cjk else "words"
    system = ("You are a professional dubbing translator. Translate spoken dialogue for LIP-SYNC dubbing: the translation "
              "must be natural spoken language and take about the same time to say as the original. Reply with ONLY a "
              'JSON object: {"source_language": "<ISO 639-1 code of the original>", "translation": "<translated text>"}. '
              "Write the translation in the native script of the target language, no quotes, no notes, no stage directions.")
    limit = (f"HARD LIMIT: at most {n_words} {unit}. The previous attempt was too long; shorten it by dropping filler."
             if hard else f"Aim for about {n_words} {unit} (the original lasts about {seconds:.1f} seconds when spoken).")
    user = f"Target language: {target_name}\n{limit}\n\nOriginal text:\n{text}"
    out = llm.complete_json(system, user, retries=2, max_tokens=900, temperature=0.3)
    if not isinstance(out, dict) or not str(out.get("translation", "")).strip():
        raise DubTranslateError("vedit.dub.err_translate", "L'LLM non ha restituito una traduzione valida.")
    return out


def translate(text: str, target: str, video_seconds: float, *, speech_seconds: float = 0.0, log=None) -> dict:
    """Traduce `text` nella lingua `target` (codice di LIPDUB_SPOKEN_LANGS) → {"text", "source_language", "fit", "est_s",
    "warning"}. `warning` ('' | 'long' | 'short') segnala una traduzione che non sta nel tempo del video."""
    log = log or (lambda s: None)
    name = _lang_name(target)
    if not name:
        raise DubTranslateError("vedit.dub.err_no_target", f"Lingua di destinazione non valida: {target!r}.")
    from core import llm_service as llm
    if not llm.available():
        cfg = llm.get_config()
        if cfg.get("provider") == "lmstudio" and llm.ensure_lmstudio():
            pass
        else:
            raise DubTranslateError("vedit.dub.err_llm",
                                    f"LLM locale non attivo (provider «{cfg.get('provider')}»): avvia LM Studio "
                                    f"(porta {cfg.get('lmstudio_port')}) o scegli il provider Gemma nelle impostazioni LLM.")
    src_est = vic.estimate_speech_seconds(text)
    secs = speech_seconds if speech_seconds > 0 else (min(video_seconds, src_est) if video_seconds > 0 else src_est)
    secs = max(secs, 0.5)
    cjk = target in ("zh", "ja", "ko")
    n = max(1, int(round(secs * (4.5 if cjk else 2.5))))
    log(f"[dub] traduzione in {name} (~{n} {'caratteri' if cjk else 'parole'}) con l'LLM locale…")
    try:
        out = _ask_llm(text, name, n, secs, hard=False, cjk=cjk)
        fit, est = vic.lipdub_text_fit(str(out["translation"]), video_seconds)
        if fit == "long":
            log("[dub] traduzione troppo lunga per il video: secondo giro per accorciarla…")
            out = _ask_llm(text, name, max(1, int(n * 0.85)), secs, hard=True, cjk=cjk)
            fit, est = vic.lipdub_text_fit(str(out["translation"]), video_seconds)
    finally:
        try:
            llm.unload()                                               # libera la VRAM prima della generazione video
        except Exception:                                              # noqa: BLE001
            pass
    return {"text": " ".join(str(out["translation"]).split()),
            "source_language": str(out.get("source_language") or "").strip().lower()[:5],
            "fit": fit, "est_s": est, "warning": fit if fit in ("long", "short") else ""}


def run(video: str, target: str, *, source_text: str = "", video_seconds: float = 0.0, log=None) -> dict:
    """Flusso completo: STT (salvo `source_text` dato) → traduzione. Ritorna `translate()` + "original_text",
    "original_language" (whisper se c'e', altrimenti quella dichiarata dall'LLM), "backend"."""
    log = log or (lambda s: None)
    if video_seconds <= 0:
        try:
            video_seconds = float(prep.probe(video).duration)
        except Exception:                                              # noqa: BLE001
            video_seconds = 0.0
    if source_text.strip():
        sp = {"text": source_text.strip(), "language": "", "speech_seconds": 0.0, "backend": "testo dato"}
    else:
        sp = extract_speech(video, log=log)
    tr = translate(sp["text"], target, video_seconds, speech_seconds=sp["speech_seconds"], log=log)
    tr.update(original_text=sp["text"], original_language=sp["language"] or tr["source_language"], backend=sp["backend"])
    return tr


def fill_cfg(cfg: dict, log=None) -> dict:
    """Per l'API (e chi vuole lo stesso flusso): se `translate_to` e' presente e battuta/prompt no, traduce e riempie
    `lipdub_line` + `lipdub_lang`. Ritorna la risposta di run() (o {})."""
    target = (cfg.get("translate_to") or "").strip().lower()
    if not target or (cfg.get("lipdub_line") or "").strip() or (cfg.get("prompt") or "").strip():
        return {}
    res = run(cfg["video"], target, source_text=cfg.get("source_text", ""), log=log)
    cfg["lipdub_line"] = res["text"]
    cfg["lipdub_lang"] = target
    return res
