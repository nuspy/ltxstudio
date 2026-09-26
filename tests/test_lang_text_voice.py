"""Lingua del TESTO ≠ lingua della VOCE: normalizzazione, avvisi di incompatibilità, [lang:…] nei marcatori,
battute H3 per lingua, validazione dello skill. Solo Python puro: niente modelli, GPU né Qt."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "skills", "ltx-avatar-film", "scripts"))

import pytest  # noqa: E402

from core import lang_codes as LC  # noqa: E402
from core import voice_markup as VM  # noqa: E402


def test_normalizzazione():
    assert LC.base("Italian") == LC.base("it-IT") == LC.base("italiano") == LC.base("it") == "it"
    assert LC.bcp47("en-gb") == "en-GB" and LC.bcp47("english") == "en-US"
    assert LC.h3_name("es-ES") == "Spanish" and LC.h3_name("hu") == ""


def test_voce_monolingua_vs_testo():
    assert "accento" in LC.check("edge", "en-US", voice_id="it-IT-ElsaNeural")
    assert LC.check("edge", "en-US", voice_id="it-IT-ElsaNeural", allow_accent=True) == ""
    assert LC.check("edge", "en-US", voice_id="it-IT-GiuseppeMultilingualNeural") == ""
    assert LC.check("kokoro", "it", voice_id="im_nicola") == ""
    assert "accento" in LC.check("kokoro", "en", voice_id="im_nicola")
    assert "non sa leggere" in LC.check("kokoro", "de", voice_id="im_nicola")
    assert LC.check("xtts2", "en", cloned=True) == ""                  # clone = cross-lingua


def test_marcatore_lang():
    t = "Mi ha detto: [lang:en][emotion:happy]see you later[/emotion][/lang] e se n'è andato."
    segs = [s for s in VM.parse(t) if s.kind == "text"]
    assert [s.style.lang for s in segs] == ["", "en", ""] and segs[1].style.emotion == "happy"
    assert VM.split_by_lang(t, "it") == [("it", "Mi ha detto:"), ("en", "see you later"),
                                         ("it", "e se n'è andato.")]
    assert VM.plain(t) == "Mi ha detto: see you later e se n'è andato."


def test_lingua_speaker_esplicita():
    from core.avatar import audio_chain as A
    assert A.text_language(A.Speaker(language="english")) == "en-US"
    assert A.text_language(A.Speaker(engine="edge", voice_id="fr-FR-DeniseNeural")) == "fr-FR"
    assert A.text_language(A.Speaker(engine="kokoro", voice_id="im_nicola")) == "it-IT"
    with pytest.raises(ValueError):                                     # clone senza lingua: niente default
        A.text_language(A.Speaker(engine="xtts2", clone_sample="x.wav"))
    v, _ = A._voice_preset(A.Speaker(engine="kokoro", language="es-ES"))
    assert v.voice_id == "ef_dora", "senza voce: la predefinita della LINGUA DEL TESTO, non sempre italiana"


def test_battute_h3_per_lingua():
    from core import h3_prompt as P
    s = P._line_sentence({"speaker": "Nonna", "language": "it-IT",
                          "text": "Mio nipote dice: [lang:en]no problem, grandma[/lang] e sparisce"}, 1)
    assert s == ("Nonna (S1) says <d>[Italian] Mio nipote dice:</d> <d>[English] no problem, grandma</d> "
                 "<d>[Italian] e sparisce.</d>")
    with pytest.raises(ValueError):
        P._line_sentence({"text": "ciao"}, 1)                           # lingua obbligatoria
    with pytest.raises(ValueError):
        P._line_sentence({"language": "Hungarian", "text": "szia"}, 1)  # non supportata da H3


def test_skill_valida_le_lingue():
    import avatar_film as F
    assert F.check_markup("[lang:xx]boh[/lang]")
    e, w = F.check_languages({"engine": "longcat", "audio": {"mode": "tts", "text": "hi", "speakers": [
        {"name": "Tom", "engine": "edge", "voice_id": "it-IT-ElsaNeural", "language": "en-US"},
        {"name": "X", "voice_description": "old man"}]}})
    assert any("«X»" in x for x in e) and any("«Tom»" in x and "accento" in x for x in w)
    e, _ = F.check_languages({"engine": "minimax_h3", "video": {"h3": {"dialogue_lines": [{"text": "hola"}]}}})
    assert e and "language" in e[0]
