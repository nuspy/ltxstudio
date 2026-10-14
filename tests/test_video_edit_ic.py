"""core/video_edit_ic.py: build_job/finalize/checkpoint_info su video sintetici (nessuna GPU, nessun Qt)."""
import os
import shutil
import subprocess

import pytest

from core import video_edit_ic as vic
from core import video_edit_modes as vem
from core import video_ic_prep as P

_HAVE_FFMPEG = bool(shutil.which(P.ffmpeg_path()) or os.path.exists(P.ffmpeg_path()))
pytestmark = pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg non disponibile")


def _gen(out, n, size="320x192", audio=False):
    cmd = [P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-shortest"]
    cmd += ["-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


@pytest.fixture(autouse=True)
def _weights_present(monkeypatch):
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "")


def test_checkpoint_info_states():
    assert vic.checkpoint_info(None)["state"] == "unknown"
    assert vic.checkpoint_info({"model_kind": "auto", "model": "auto"})["state"] == "auto"
    d = vic.checkpoint_info({"model_kind": "diffusion", "model": r"E:\m\ltx-2.3-22b-distilled-1.1.safetensors"})
    assert d["state"] == "distilled" and d["params"]["model"].endswith("distilled-1.1.safetensors")
    e = vic.checkpoint_info({"model_kind": "diffusion", "model": r"E:\m\ltx-2.3-22b-dev-fp8.safetensors"})
    assert e["state"] == "dev" and not e["distilled"]
    g = vic.checkpoint_info({"model_kind": "gguf", "model_gguf": r"E:\m\ltx-2.3-distilled-Q4_K_M.gguf"})
    assert g["state"] == "distilled" and "gguf_transformer" in g["params"] and "model" not in g["params"]
    assert vic.checkpoint_info({"model_kind": "minimax_h3", "model": ""})["state"] == "other_arch"


def test_restore_job_no_copy_and_audio(tmp_path):
    src = _gen(tmp_path / "s.mp4", 50, audio=True)
    ij = vic.build_job({"mode": "refocus", "video": src, "prompt": ""}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert p["control_video"] == src, "il restauro usa il sorgente senza copiarlo"
    assert p["control_type"] == "refocus" and p["edit_lora"] == "" and p["_strict_ic"] is True
    assert p["num_frames"] == 49 and (p["width"], p["height"]) == (320, 192)
    assert "chunk_frames" not in p and ij.mux_audio_from == src


def test_checkpoint_passed_as_chosen(tmp_path):
    src = _gen(tmp_path / "s.mp4", 17)
    ij = vic.build_job({"mode": "uncompress", "video": src, "model": "X/ltx-2.3-22b-dev-fp8.safetensors",
                        "ltx_release": "2.3"}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    assert ij.params["model"] == "X/ltx-2.3-22b-dev-fp8.safetensors" and ij.params["ltx_release"] == "2.3"
    ij2 = vic.build_job({"mode": "uncompress", "video": src}, str(tmp_path), 2, str(tmp_path / "o2.mp4"))
    assert "model" not in ij2.params                      # nessuna scelta automatica


def test_long_clip_segments_and_cap(tmp_path):
    src = _gen(tmp_path / "l.mp4", 200, size="1920x1088")
    ij = vic.build_job({"mode": "refocus", "video": src}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert max(p["width"], p["height"]) <= 1280 and p["width"] % 64 == 0 and p["height"] % 64 == 0
    assert p["chunk_frames"] == 97 and p["overlap_frames"] >= 8 and ij.segments >= 2
    assert any("ridotto" in n for n in ij.notes)


def test_outpaint_and_finalize(tmp_path):
    src = _gen(tmp_path / "v.mp4", 25, size="192x320")           # verticale
    out = str(tmp_path / "o.mp4")
    ij = vic.build_job({"mode": "outpaint", "video": src, "out_format": "16:9", "out_dark": True,
                        "out_paste": True, "keep_audio": True}, str(tmp_path), 1, out)
    lay = ij.layout
    assert (lay.canvas_w, lay.canvas_h) == (ij.params["width"], ij.params["height"])
    assert lay.canvas_w % 64 == 0 and lay.canvas_w > lay.canvas_h and lay.new_area > 0
    guide = ij.params["control_video"]
    gi = P.probe(guide, count_frames=True)
    assert (gi.width, gi.height, gi.nb_frames) == (lay.canvas_w, lay.canvas_h, 25)
    assert ij.gamma == 2.0 and ij.paste_original
    # il «risultato del worker» simulato = la guida stessa: finalize applica gamma inverso + originale incollato
    shutil_out = str(tmp_path / "gen.mp4")
    shutil.copy(guide, shutil_out)
    final = vic.finalize(ij, shutil_out, cleanup=False)
    fi = P.probe(final, count_frames=True)
    assert (fi.width, fi.height) == (lay.canvas_w, lay.canvas_h) and fi.nb_frames >= 24
    # custom margini senza area -> errore chiaro
    with pytest.raises(vic.IcJobError):
        vic.build_job({"mode": "outpaint", "video": src, "out_format": "custom", "out_margins": (0, 0, 0, 0)},
                      str(tmp_path), 3, out)


def test_restyle_short_vs_long(tmp_path):
    from PIL import Image
    img = str(tmp_path / "style.png")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(img)
    src = _gen(tmp_path / "s.mp4", 33)
    ij = vic.build_job({"mode": "restyle", "video": src, "prompt": "Make it Disney 2D Animation style.",
                        "style_image": img}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    assert ij.params["control_video"].endswith("guide_f0.mp4") and "ic_frame0_image" not in ij.params
    fr = list(P._read_frames(ij.params["control_video"], 320, 192))
    assert fr[0].mean(axis=(0, 1))[0] > 150                        # frame 0 = stile (rosso)
    long_src = _gen(tmp_path / "l.mp4", 200)
    ij2 = vic.build_job({"mode": "restyle", "video": long_src, "prompt": "x", "style_image": img},
                        str(tmp_path), 2, str(tmp_path / "o2.mp4"))
    assert ij2.params["ic_frame0_image"] == img and ij2.params["control_video"] == long_src
    with pytest.raises(vic.IcJobError):
        vic.build_job({"mode": "restyle", "video": src, "prompt": "x"}, str(tmp_path), 3, str(tmp_path / "o3.mp4"))


def test_cameraman_job(tmp_path):
    ref = _gen(tmp_path / "r.mp4", 49)
    img = str(tmp_path / "i.png")
    from PIL import Image
    Image.new("RGB", (64, 64)).save(img)
    ij = vic.build_job({"mode": "cameraman", "cam_ref": ref, "cam_img": img, "cam_img_strength": 0.6,
                        "width": 960, "height": 576, "cam_seconds": 1.0, "prompt": ""},
                       str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert p["control_video"] == ref and p["image_path"] == img and p["ref_strength"] == 0.6
    assert (p["width"], p["height"]) == (960, 576) and p["num_frames"] == 25 and ij.mux_audio_from == ""
    with pytest.raises(vic.IcJobError):
        vic.build_job({"mode": "cameraman"}, str(tmp_path), 2, str(tmp_path / "o2.mp4"))


def test_disabled_and_unknown_modes_rejected(tmp_path):
    src = _gen(tmp_path / "s.mp4", 17)
    for m in ("lipdub", "hdr", "nope", "edit"):
        with pytest.raises(vic.IcJobError):
            vic.build_job({"mode": m, "video": src}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
