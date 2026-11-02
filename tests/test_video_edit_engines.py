"""Tabella azione -> motori (core/video_edit_engines.py): filtro per azione, default = motore di oggi, mai download/cambio automatico."""
import pytest

from core import video_edit_engines as E
from core import video_edit_modes as vem


def test_every_action_has_at_least_one_engine_and_default_is_first():
    for a in E.ACTIONS:
        items = E.engines_for(a)
        assert items, a
        assert items[0].id == E.default_engine(a)
        assert all(e.supports(a) for e in items)


def test_dropdown_filters_engines_that_do_not_do_the_action():
    ids = {a: [e.id for e in E.engines_for(a)] for a in E.ACTIONS}
    assert ids[E.GLOBAL_EDIT] == ["ltx_edit_anything"]                   # H3 e inpaint non fanno l'edit globale
    assert "ltx_edit_anything" not in ids[E.REPLACE_CHARACTER] and "ltx_inpaint" not in ids[E.REPLACE_CHARACTER]
    assert ids[E.REPLACE_CHARACTER] == ["ltx_union_canny", "h3_edit"]    # storico per primo, H3 in piu'
    assert ids[E.CHANGE_BACKGROUND] == ["h3_edit"]
    assert ids[E.OBJECT_EDIT] == ["ltx_inpaint"]


def test_defaults_are_todays_engines():
    assert E.DEFAULT_ENGINE[E.GLOBAL_EDIT] == "ltx_edit_anything"
    assert E.DEFAULT_ENGINE[E.OBJECT_EDIT] == "ltx_inpaint"
    assert E.DEFAULT_ENGINE[E.REPLACE_CHARACTER] == "ltx_union_canny"


def test_modes_map_to_actions_and_historical_modes_have_the_same_lora():
    assert E.action_of_mode("edit") == E.GLOBAL_EDIT and E.action_of_mode("inpaint") == E.OBJECT_EDIT
    assert E.action_of_mode("replace_char") == E.REPLACE_CHARACTER and E.action_of_mode("nope") is None
    # il peso del motore storico = quello della MODE_SPEC (la tabella non puo' divergere dal backend di oggi)
    assert vem.MODE_SPECS["edit"].edit_lora.replace("\\", "/") in E.ENGINES["ltx_edit_anything"].weights[0][0]
    assert vem.MODE_SPECS["inpaint"].edit_lora.replace("\\", "/") in E.ENGINES["ltx_inpaint"].weights[0][0]


def test_not_installed_engine_stays_visible_and_is_refused_with_clear_message():
    absent = lambda item: False                                          # noqa: E731
    items = E.dropdown_items(E.CHANGE_BACKGROUND, present=absent)
    assert [i["id"] for i in items] == ["h3_edit"] and items[0]["installed"] is False
    msg = E.missing_message("h3_edit", present=absent)
    assert "non e' installato" in msg and "Pesi modelli" in msg and "nessun download automatico" in msg


def test_weight_group_is_any_of():
    only_fp8 = lambda item: item in {"mmh3-ref2va-fp8", "mmh3-te-fp8", "mmh3-video-vae", "mmh3-audio-vae", "mmh3-configs"}  # noqa: E731
    assert E.is_installed("h3_edit", present=only_fp8)
    assert E.missing_groups("h3_edit", present=lambda i: i != "mmh3-video-vae" and only_fp8(i)) == [("mmh3-video-vae",)]


def test_engine_params_carry_the_chosen_engine_and_refuse_wrong_action():
    p = E.engine_params(E.CHANGE_BACKGROUND, "h3_edit")
    assert p["engine"] == "h3_edit" and p["engine_kind"] == E.KIND_H3_EDIT and p["engine_action"] == E.CHANGE_BACKGROUND
    assert p["engine_overlap_frames"] == 39 and p["engine_clip_s"] == 6.0
    with pytest.raises(ValueError):
        E.engine_params(E.GLOBAL_EDIT, "h3_edit")


def test_remember_and_fallback(monkeypatch):
    store = {}
    from core import user_prefs
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(user_prefs, "set", lambda k, v: store.__setitem__(k, v))
    assert E.remembered(E.REPLACE_CHARACTER) == "ltx_union_canny"        # niente scelta: default (motore di oggi)
    E.remember(E.REPLACE_CHARACTER, "h3_edit")
    assert E.remembered(E.REPLACE_CHARACTER) == "h3_edit"
    E.remember(E.REPLACE_CHARACTER, "ltx_edit_anything")                 # motore che non fa l'azione: ignorato
    assert E.remembered(E.REPLACE_CHARACTER) == "h3_edit"
    store["vedit_engine.replace_character"] = "inesistente"              # preferenza non valida: ripiego sul default, non errore
    assert E.remembered(E.REPLACE_CHARACTER) == "ltx_union_canny"
