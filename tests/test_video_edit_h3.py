"""core/video_edit_h3.py — params del worker H3 per Cambia sfondo / Sostituisci personaggio (controlli, nessuna scelta automatica)."""
import os

import pytest

from core import video_edit_engines as E
from core import video_edit_h3 as V

ALL = lambda item: True                       # noqa: E731
NONE = lambda item: False                     # noqa: E731


@pytest.fixture()
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    p.write_bytes(b"x")
    pic = tmp_path / "p.png"
    pic.write_bytes(b"x")
    return str(p), str(pic)


def _cfg(clip, **kw):
    c = {"video": clip[0], "mode": "change_background", "engine": "h3_edit", "object": "person", "prompt": "a rainy neon street",
         "h3_profile": "int8"}
    c.update(kw)
    return c


def test_background_params_carry_engine_and_all_defaults(clip):
    p = V.build_params(_cfg(clip), "out.mp4", present=ALL)
    assert p["arch"] == "minimax_h3" and p["engine"] == "h3_edit" and p["engine_action"] == "change_background"
    e = p["h3_edit"]
    assert e["action"] == "change_background" and e["mask_dir"].endswith(os.path.join("c_track", "masks"))
    assert (e["clip_s"], e["overlap"], e["grow_px"], e["headroom"], e["hold_frames"]) == (6.0, 39, 16, 0.10, 2)
    assert p["h3_profile"] == "int8" and p["h3_turbo"] == "v4" and p["h3_video_shift"] == 12.0
    assert e["picture"] == "" and e["prompt_text"] == "a rainy neon street"


def test_replace_needs_photo_and_person(clip):
    with pytest.raises(V.H3EditError, match="foto del nuovo personaggio"):
        V.build_params(_cfg(clip, mode="replace_char", ref_image=""), "o.mp4", present=ALL)
    p = V.build_params(_cfg(clip, mode="replace_char", ref_image=clip[1], prompt=""), "o.mp4", present=ALL)
    assert p["h3_edit"]["action"] == "replace_character" and p["h3_edit"]["picture"] == clip[1]
    with pytest.raises(V.H3EditError, match="personaggio da sostituire"):
        V.build_params(_cfg(clip, mode="replace_char", ref_image=clip[1], object=""), "o.mp4", present=ALL)


def test_background_needs_place_in_words_or_photo(clip):
    with pytest.raises(V.H3EditError, match="nuovo luogo"):
        V.build_params(_cfg(clip, prompt="", ref_image=""), "o.mp4", present=ALL)
    assert V.build_params(_cfg(clip, prompt="", ref_image=clip[1]), "o.mp4", present=ALL)["h3_edit"]["picture"] == clip[1]


def test_missing_weights_refuse_with_clear_message_never_download(clip):
    with pytest.raises(V.H3EditError, match="non e' installato"):
        V.build_params(_cfg(clip), "o.mp4", present=NONE)


def test_profile_must_be_chosen_and_installed(clip):
    only_int8 = lambda i: i != "mmh3-ref2va-fp8" and i != "mmh3-ref2va-nvfp4" and i != "mmh3-ref2va-int4"   # noqa: E731
    assert V.installed_profiles(only_int8) == ["int8"]
    with pytest.raises(V.H3EditError, match="profilo H3"):
        V.build_params(_cfg(clip, h3_profile=""), "o.mp4", present=ALL)          # nessun profilo scelto: nessuna scelta automatica
    with pytest.raises(V.H3EditError, match="profilo H3"):
        V.build_params(_cfg(clip, h3_profile="fp8"), "o.mp4", present=only_int8)


def test_other_actions_are_refused(clip):
    with pytest.raises(V.H3EditError):
        V.build_params(_cfg(clip, mode="edit"), "o.mp4", present=ALL)
