"""Video di movimento (<Video 1>) a ~512 px sul lato lungo come l'originale Openjutsu (04/11): tela, piano VRAM, params, GUI."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workers"))
from core import h3_vram_plan as plan  # noqa: E402
from core import minimax_h3_spec as spec  # noqa: E402

VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"

CANVASES = [(736, 1344), (480, 864), (576, 1024), (1344, 736), (736, 736), (480, 1280), (544, 960)]


def test_canvas_is_512_on_the_long_side_with_multiples_of_32():
    for height, width in CANVASES:
        canvas = spec.reference_video_canvas(height, width)
        assert max(canvas["height"], canvas["width"]) in (512, 544)                       # 512, o 544 se l'arrotondamento a 32 sale
        assert canvas["height"] % 32 == 0 and canvas["width"] % 32 == 0
    landscape = spec.reference_video_canvas(736, 1344)
    assert (landscape["height"], landscape["width"]) == (288, 512)
    portrait = spec.reference_video_canvas(1344, 736)
    assert (portrait["height"], portrait["width"]) == (512, 288)


def test_zero_or_small_canvas_keeps_the_target_canvas():
    assert spec.reference_video_canvas(736, 1344, 0) == {"short_edge": 736, "max_pixels": 736 * 1344, "height": 736, "width": 1344}
    small = spec.reference_video_canvas(256, 480)                                         # gia sotto i 512: invariata
    assert (small["height"], small["width"]) == (256, 480)


def test_vram_plan_counts_the_reference_rows_at_the_smaller_canvas():
    references = [{"kind": "video", "frames": 141}]
    full = plan.ref_rows_estimate(references, 480, 736, 1344)
    reduced = plan.ref_rows_estimate(references, 480, 736, 1344, (288, 512))
    assert reduced * 5 < full < reduced * 8                                              # ~6,7 volte meno righe


def test_params_and_gui_option():
    from core import video_edit_h3 as vh3
    clip = Path(os.environ.get("TEMP", ".")) / "ref_canvas_probe.mp4"
    clip.write_bytes(b"x")
    try:
        cfg = {"engine": "h3_edit", "mode": "change_background", "video": str(clip), "object": "person", "prompt": "a refuge", "h3_profile": "nvfp4"}
        assert "h3_ref_max_side" not in vh3.build_params(cfg, "o.mp4", present=lambda item: True)           # default: ~512 px (lo decide il worker)
        assert vh3.build_params({**cfg, "h3_ref_full_res": True}, "o.mp4", present=lambda item: True)["h3_ref_max_side"] == 0
    finally:
        clip.unlink()


def test_worker_and_runkey_wiring():
    worker = (ROOT / "workers" / "minimax_h3_infer.py").read_text(encoding="utf-8")
    assert "reference_video_canvas_for(p, _th, _tw)" in worker and 'canvas_short_edge=int(video_canvas["short_edge"])' in worker
    assert "reference_video_canvas_for(p, h, w)" in worker                                # il piano conta le righe alla tela vera del riferimento
    edit = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert '("refmaxside", int(p.get("h3_ref_max_side", REF_VIDEO_MAX_SIDE)))' in edit


@pytest.mark.skipif(not os.path.isfile(VENV_PY), reason="serve il venv di H3")
def test_emulated_canvas_equals_the_pipelines_resolve_canvas_size():
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size\n"
        "from core import minimax_h3_spec as spec\n"
        "for h, w in %r:\n"
        "    c = spec.reference_video_canvas(h, w)\n"
        "    got = resolve_canvas_size(w, h, 32, c['short_edge'], c['max_pixels'])\n"
        "    assert got == (c['height'], c['width']), (h, w, got, c)\n"
        "print('CANVAS OK')\n" % (ROOT, CANVASES))
    result = subprocess.run([VENV_PY, "-c", script], capture_output=True, text=True, timeout=300)
    assert "CANVAS OK" in result.stdout, result.stdout[-500:] + result.stderr[-800:]
