"""Test puri di core/video_edit_modes.py (nessuna GPU, nessun Qt)."""
import json
import re
from pathlib import Path

import pytest

from core import video_edit_modes as M

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_modes_identical_to_old_tab():
    assert M.legacy_modes() == [
        ("edit", "vedit.mode.edit", "LTX-2/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"),
        ("inpaint", "vedit.mode.inpaint", "LTX-2/ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors"),
        ("anime2real", "vedit.mode.anime2real", "LTX-2/ltx23_anime2real_rank64_v1_4500.safetensors"),
        ("real2anime", "vedit.mode.real2anime", "LTX-2/ltx23_real2anime_rank64_v1_5000.safetensors"),
        ("replace_char", "vedit.mode.replace_char", "LTX-2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
    ]


def test_tab_modes_match_if_tab_not_yet_derived():
    """Se il tab ha ancora il vecchio MODES letterale, deve coincidere con legacy_modes()."""
    src = (ROOT / "ui" / "tabs" / "video_edit_ai_tab.py").read_text(encoding="utf-8")
    if "legacy_modes" in src:
        return
    for key, i18n, lora in M.legacy_modes():
        assert f'("{key}", "{i18n}", "{lora}")' in src


def test_all_phase_1a_modes_present():
    for k in ("edit", "inpaint", "replace_char", "anime2real", "real2anime", "refocus", "uncompress",
              "deinterlace", "motiondeblur", "detailer", "outpaint", "restyle", "cameraman", "lipdub", "hdr", "motion_track"):
        assert k in M.MODE_SPECS


def test_key_matches_dict_and_fields_valid():
    for k, s in M.MODE_SPECS.items():
        assert s.key == k
        assert s.family in M.FAMILIES
        assert s.i18n_key == f"vedit.mode.{k}"
        assert s.needs_prompt in (M.NEEDS_REQUIRED, M.NEEDS_OPTIONAL, M.NEEDS_NONE)
        assert s.distilled in (M.DISTILLED_REQUIRED, M.DISTILLED_RECOMMENDED)
        assert s.preprocess in (M.PRE_NONE, M.PRE_OUTPAINT, M.PRE_RESTYLE_FRAME0)


def test_ceilings_valid():
    for s in M.MODE_SPECS.values():
        assert s.max_long_edge % 64 == 0 and s.max_long_edge >= 512, s.key
        assert s.max_frames_per_segment % 8 == 1 and s.max_frames_per_segment >= 9, s.key
        if s.min_long_edge:
            assert s.min_long_edge <= s.max_long_edge


def test_hdr_spec():
    """hdr attivo dalla fase 1c: pipeline dedicata, nessun prompt, il worker segmenta da solo, niente arrotondamento a 64."""
    s = M.MODE_SPECS["hdr"]
    assert (s.phase, s.enabled, s.distilled, s.kind) == ("1c", True, M.DISTILLED_RECOMMENDED, "hdr")
    assert (s.needs_prompt, s.cap, s.catalog_id) == (M.NEEDS_NONE, "cap.video_hdr_ic", "ltx-iclora-hdr")
    assert s.own_segments and s.native_res and not s.chunkable and s.keep_audio
    assert all(s.enabled for s in M.MODE_SPECS.values())


def test_own_segments_counted_but_not_chunked_by_worker():
    # hdr: i segmenti li conta la GUI (info), li esegue il worker da se' (own_segments); lipdub/inpaint restano a 1
    assert M.segments_needed(97, "hdr") == 1
    assert M.segments_needed(200, "hdr") == 3
    assert M.segments_needed(1000, "lipdub") == 1


def test_lipdub_spec():
    """Lip dub (fase 1b): abilitato, non spezzabile, voce nuova (niente mux dell'audio originale), cap dedicata."""
    s = M.MODE_SPECS["lipdub"]
    assert (s.phase, s.enabled, s.kind, s.cap) == ("1b", True, "lipdub", "cap.video_lipdub")
    assert s.needs_audio_in_video and not s.keep_audio and not s.chunkable and s.needs_prompt == M.NEEDS_REQUIRED
    assert s.max_frames_per_segment == 249 and M.segments_needed(1000, "lipdub") == 1


def test_lipdub_audio_spec():
    """Variante sperimentale da audio: stessa LoRA/cap, serve un file audio (non l'audio nel video), prompt opzionale."""
    a, t = M.MODE_SPECS["lipdub_audio"], M.MODE_SPECS["lipdub"]
    assert (a.kind, a.catalog_id, a.cap, a.family) == (t.kind, t.catalog_id, t.cap, t.family)
    assert a.needs_audio_file and not a.needs_audio_in_video and a.needs_prompt == M.NEEDS_OPTIONAL
    assert a.experimental_key == "vedit.note_lipdub_audio" and not a.keep_audio and not a.chunkable
    assert not t.needs_audio_file


def test_modes_for_family():
    assert [s.key for s in M.modes_for_family("edit")] == ["edit", "inpaint", "replace_char"]
    assert [s.key for s in M.modes_for_family("style")] == ["anime2real", "real2anime", "restyle"]
    assert [s.key for s in M.modes_for_family("repair")] == ["refocus", "uncompress", "deinterlace",
                                                             "motiondeblur", "detailer", "hdr"]
    assert [s.key for s in M.modes_for_family("outpaint")] == ["outpaint"]
    assert [s.key for s in M.modes_for_family("lipdub")] == ["lipdub", "lipdub_audio"]
    assert [s.key for s in M.modes_for_family("camera")] == ["cameraman", "motion_track"]
    assert "lipdub" in M.families() and "hdr" in [s.key for f in M.families() for s in M.modes_for_family(f)]


def test_every_mode_in_exactly_one_family():
    seen = [s.key for f in M.FAMILIES for s in M.modes_for_family(f, include_disabled=True)]
    assert sorted(seen) == sorted(M.MODE_SPECS)


def test_kinds_resolve_to_worker_registry_ids():
    """I kind IC-LoRA dei MODE_SPECS esistono in workers/ltx2_infer.py (_IC_IDS) con lo stesso id di catalogo."""
    src = (ROOT / "workers" / "ltx2_infer.py").read_text(encoding="utf-8")
    block = src[src.index("_IC_IDS = {"):src.index("_IC_FALLBACK")]
    ids = dict(re.findall(r'"([\w-]+)":\s*"(ltx-iclora-[\w-]+)"', block))
    for s in M.MODE_SPECS.values():
        if s.kind:
            assert ids.get(s.kind) == s.catalog_id, (s.key, s.kind, s.catalog_id)


def test_catalog_ids_exist():
    cat = json.loads((ROOT / "data" / "weights_catalog.json").read_text(encoding="utf-8"))
    ids = {e["id"] for e in cat["weights"]}
    for s in M.MODE_SPECS.values():
        if s.catalog_id:
            assert s.catalog_id in ids, (s.key, s.catalog_id)


def test_i18n_key_shapes():
    """Forma delle chiavi (le traduzioni in 15 lingue le aggiunge l'agente GUI)."""
    for s in M.MODE_SPECS.values():
        assert re.fullmatch(r"vedit\.mode\.\w+", s.i18n_key)
        assert not s.note_key or re.fullmatch(r"vedit\.note_\w+", s.note_key)


def test_spec_for_kind_and_get_spec():
    assert M.spec_for_kind("outpaint").key == "outpaint"
    assert M.spec_for_kind("union").key == "replace_char"
    assert M.spec_for_kind("lipdub").key == "lipdub"
    assert M.spec_for_kind("hdr").key == "hdr"
    assert M.spec_for_kind("nope") is None
    with pytest.raises(KeyError):
        M.get_spec("nope")


def test_clamp_resolution():
    assert M.clamp_resolution(3840, 2160, "refocus") == (1280, 704, True)
    assert M.clamp_resolution(1280, 704, "refocus") == (1280, 704, False)
    assert M.clamp_resolution(500, 300, "refocus") == (512, 320, False)    # mai ingrandire
    w, h, _ = M.clamp_resolution(2160, 3840, "detailer")
    assert max(w, h) <= 1024 and w % 64 == 0 and h % 64 == 0
    for spec, sp in M.MODE_SPECS.items():
        w, h, _ = M.clamp_resolution(5000, 3000, spec)
        assert max(w, h) <= sp.max_long_edge


def test_snap_and_segments_needed():
    assert [M.snap_frames(n) for n in (1, 9, 10, 16, 17, 100, 121)] == [9, 9, 9, 9, 17, 97, 121]
    assert M.segments_needed(50, "refocus") == 1
    assert M.segments_needed(97, "refocus") == 1
    assert M.segments_needed(98, "refocus") == 2
    assert M.segments_needed(1000, "inpaint") == 1          # non spezzabile: la GUI deve rifiutare


def test_ic_kinds_list():
    assert "cameraman" in M.ICLORA_VIDEO_KINDS and "motion-track" in M.ICLORA_VIDEO_KINDS
    assert "lipdub" in M.ICLORA_VIDEO_KINDS and "hdr" in M.ICLORA_VIDEO_KINDS
    assert "canny" not in M.ICLORA_VIDEO_KINDS


def test_motion_track_spec():
    sp = M.get_spec("motion_track")
    assert sp.kind == "motion-track" and sp.catalog_id == "ltx-iclora-motion-track"
    assert sp.needs_source_image and sp.needs_tracks and not sp.needs_video and not sp.chunkable
    assert sp.cap == "cap.video_motion_track" and sp.max_frames_per_segment == 257 and (sp.max_frames_per_segment - 1) % 8 == 0
    assert M.spec_for_kind("motion-track").key == "motion_track"
    assert M.segments_needed(1000, "motion_track") == 1      # non spezzabile: la GUI/API devono rifiutare
