"""Traduzione + arricchimento del testo dell'utente nell'edit H3 (core/h3_edit_text.py): una chiamata per job, soggetto fedele, inglese saltato, cache, avviso senza blocco."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import h3_edit_text as T  # noqa: E402


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_TEXT_CACHE_DIR", str(tmp_path))


def test_is_english():
    assert T.is_english("") and T.is_english("a mountain refuge at night, snowy") and T.is_english("person")
    assert not T.is_english("un rifugio di montagna di notte") and not T.is_english("persona") and not T.is_english("café")
    assert not T.is_english("la ragazza in rosso")


class Fake:
    def __init__(self, reply=None):
        self.calls, self.reply = [], reply

    def enhance(self, payload, kind="image", nsfw=False, max_new_tokens=220):
        self.calls.append((payload, kind))
        if self.reply is not None:
            return self.reply
        d = json.loads(payload)
        out = {}
        for k, v in d.items():
            out[k] = "the girl in red" if k == "subject" else "A snowy mountain refuge at night, warm firelight on timber walls (" + v + ")"
        return json.dumps(out)


def cfg(**kw):
    base = {"h3_translate": True, "object": "la ragazza in rosso", "prompt": "un rifugio di montagna di notte", "picture_desc": "", "shots": ""}
    base.update(kw)
    return base


def test_one_call_translates_subject_faithfully_and_enriches_the_description():
    f, logs, c = Fake(), [], cfg()
    info = T.apply_to_cfg(c, log=logs.append, enhance=f.enhance, available=lambda: True)
    assert len(f.calls) == 1 and f.calls[0][1] == "h3_fields"                       # UNA sola chiamata per job, kind dedicato
    payload = json.loads(f.calls[0][0])
    assert set(payload) == {"subject", "setting"}                                   # i campi vuoti non partono
    assert c["object"] == "the girl in red" and c["prompt"].startswith("A snowy mountain refuge")
    assert info["status"] == "ok" and info["original"]["object"] == "la ragazza in rosso" and info["sent"]["object"] == "the girl in red"
    assert any("la ragazza in rosso" in m and "the girl in red" in m for m in logs)    # originale e inviato nel log
    assert T.summary_line(info).startswith("[h3text] ") and "prompt" in T.summary_line(info)


def test_known_subject_label_needs_no_llm_and_english_text_is_skipped():
    f = Fake()
    c = cfg(object="persona", prompt="a snowy refuge")
    info = T.apply_to_cfg(c, log=lambda m: None, enhance=f.enhance, available=lambda: True)
    assert f.calls == [] and c["object"] == "person" and c["prompt"] == "a snowy refuge" and info["status"] == "dict"
    c2 = cfg(object="person", prompt="a snowy refuge")
    info2 = T.apply_to_cfg(c2, log=lambda m: None, enhance=f.enhance, available=lambda: True)
    assert f.calls == [] and info2["status"] == "english" and c2["prompt"] == "a snowy refuge"           # niente da fare: i params restano quelli di sempre


def test_flag_off_changes_nothing():
    f, c = Fake(), cfg(h3_translate=False)
    info = T.apply_to_cfg(c, log=lambda m: None, enhance=f.enhance, available=lambda: True)
    assert info["status"] == "off" and f.calls == [] and c["object"] == "la ragazza in rosso"


def test_cache_by_text_hash():
    f = Fake()
    T.apply_to_cfg(cfg(), log=lambda m: None, enhance=f.enhance, available=lambda: True)
    f2, c = Fake(), cfg()
    info = T.apply_to_cfg(c, log=lambda m: None, enhance=f2.enhance, available=lambda: True)
    assert f2.calls == [] and info["status"] == "cache" and c["object"] == "the girl in red"


def test_unavailable_or_failing_enhancer_warns_and_sends_the_original():
    logs, c = [], cfg()
    info = T.apply_to_cfg(c, log=logs.append, enhance=Fake().enhance, available=lambda: False)
    assert info["status"] == "unavailable" and c["object"] == "la ragazza in rosso" and any("AVVISO" in m and "ORIGINALE" in m for m in logs)
    for bad in ("", "non e json", json.dumps({"subject": "x"})):                    # risposta vuota/non valida/chiavi mancanti
        c = cfg()
        info = T.apply_to_cfg(c, log=lambda m: None, enhance=Fake(reply=bad).enhance, available=lambda: True)
        assert info["status"] == "failed" and c["prompt"] == "un rifugio di montagna di notte"
    c = cfg()
    def boom(*a, **k): raise RuntimeError("OOM")
    assert T.apply_to_cfg(c, log=lambda m: None, enhance=boom, available=lambda: True)["status"] == "failed"       # non solleva mai


def test_overlong_subject_translation_is_rejected_label_stays_original():
    long_subject = json.dumps({"subject": "a beautiful young woman wearing an elegant red evening dress on stage", "setting": "A cabin."})
    c, logs = cfg(), []
    T.apply_to_cfg(c, log=logs.append, enhance=Fake(reply=long_subject).enhance, available=lambda: True)
    assert c["object"] == "la ragazza in rosso" and c["prompt"] == "A cabin." and any("troppo lunga" in m for m in logs)    # l'etichetta non si arricchisce


def test_enhancer_system_prompt_for_h3_fields_is_deterministic_and_single_line():
    from core import prompt_enhance as PE
    s = PE._sys_for("h3_fields")
    assert "FAITHFULLY" in s and "never add details" in s and "single line" in s
    src = (ROOT / "core" / "prompt_enhance.py").read_text(encoding="utf-8")
    assert 'if kind == "h3_fields":' in src and "do_sample=False" in src


def test_wiring_main_window_build_params_and_run_edit():
    mw = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "h3_edit_text as _h3t" in mw and "apply_to_cfg(cfg" in mw and "on_progress=self._edit_ai_progress" in mw
    assert mw.index("apply_to_cfg(cfg") < mw.index("_vh3.build_params(cfg, out)")                     # prima dei params, quindi prima di SAM e dei clip
    ev = (ROOT / "core" / "video_edit_h3.py").read_text(encoding="utf-8")
    assert 'e["text_info"]' in ev and 'not in (None, "off", "english")' in ev
    rr = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert "text_job.json" in rr and 'e.get("text_info")' in rr
    meta_line = [ln for ln in rr.splitlines() if ln.strip().startswith('meta = {"ci": ci')][0]
    assert "text_info" not in meta_line                                                               # la chiave dei clip dipende solo dal testo INVIATO (prompt_text), non dall'originale
