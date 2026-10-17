"""Test di core/video_ic_prep.py su video sintetici piccoli generati con ffmpeg (nessuna GPU, nessun Qt)."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from core import video_ic_prep as P

_HAVE_FFMPEG = bool(shutil.which(P.ffmpeg_path()) or os.path.exists(P.ffmpeg_path()))
pytestmark = pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg non disponibile")


def _ffgen(out, src_expr, n, size="128x72", rate=25, audio=False):
    cmd = [P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", f"{src_expr}{':' if '=' in src_expr else '='}size={size}:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-shortest"]
    cmd += ["-frames:v", str(n), "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac"]
    cmd += [str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


def _frames(path):
    info = P.probe(path)
    return list(P._read_frames(path, info.width, info.height))


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("icprep")
    return _ffgen(d / "clip.mp4", "testsrc2", 50)


# ── probe / audio ─────────────────────────────────────────────────────────────────────────────────
def test_probe_and_has_audio(tmp_path):
    a = _ffgen(tmp_path / "a.mp4", "testsrc2", 25, audio=True)
    v = _ffgen(tmp_path / "v.mp4", "testsrc2", 25)
    ia = P.probe(a, count_frames=True)
    assert (ia.width, ia.height, ia.nb_frames) == (128, 72, 25)
    assert abs(ia.fps - 25.0) < 0.01 and ia.has_audio
    assert P.has_audio(a) is True and P.has_audio(v) is False
    assert not P.probe(v).has_audio


def test_snap_frames():
    assert [P.snap_frames(n) for n in (9, 16, 17, 97, 100)] == [9, 9, 17, 97, 97]


# ── outpaint: layout ──────────────────────────────────────────────────────────────────────────────
def test_layout_aspect_916_to_169_contains_source_and_multiple_of_64():
    lo = P.compute_outpaint_layout(720, 1280, aspect="16:9", max_long_edge=1280)
    assert lo.canvas_w % 64 == 0 and lo.canvas_h % 64 == 0
    assert max(lo.canvas) <= 1280
    assert lo.src_w <= lo.canvas_w and lo.src_h <= lo.canvas_h
    assert lo.x + lo.src_w <= lo.canvas_w and lo.y + lo.src_h <= lo.canvas_h
    assert lo.new_area > 0.4 and lo.clamped
    assert abs(lo.x - (lo.canvas_w - lo.src_w) / 2) <= 2                   # centrato
    assert abs(lo.canvas_w / lo.canvas_h - 16 / 9) < 0.15                 # arrotondamento in su: aspetto approssimato


def test_layout_same_aspect_no_new_area():
    lo = P.compute_outpaint_layout(1280, 704, aspect=(1280, 704), max_long_edge=1280)
    assert lo.canvas == (1280, 704) and lo.new_area < 0.01 and not lo.too_much_new_area


def test_layout_anchor_left_right_top_bottom():
    l = P.compute_outpaint_layout(512, 512, aspect="16:9", anchor="left")
    r = P.compute_outpaint_layout(512, 512, aspect="16:9", anchor="right")
    assert l.x == 0 and r.x + r.src_w == r.canvas_w
    t = P.compute_outpaint_layout(512, 512, aspect="9:16", anchor="top")
    b = P.compute_outpaint_layout(512, 512, aspect="9:16", anchor="bottom")
    assert t.y == 0 and b.y + b.src_h == b.canvas_h
    with pytest.raises(ValueError):
        P.compute_outpaint_layout(512, 512, aspect="16:9", anchor="diag")


def test_layout_custom_margins():
    lo = P.compute_outpaint_layout(640, 360, margins=(0, 320, 0, 0), max_long_edge=1280)   # +320 a destra
    assert lo.canvas_w % 64 == 0 and lo.canvas_h % 64 == 0
    assert lo.x == 0 and lo.y == 0 or (lo.x <= 2 and lo.y <= 2)
    assert abs(lo.src_w / lo.canvas_w - 640 / 960) < 0.05
    lo2 = P.compute_outpaint_layout(640, 360, margins=(64, 0, 0, 128), max_long_edge=2000)
    assert lo2.x > 0 and lo2.y > 0 and lo2.src_w <= lo2.canvas_w - lo2.x + 2
    assert 0 < lo2.new_area < 1


def test_layout_needs_exactly_one_mode():
    with pytest.raises(ValueError):
        P.compute_outpaint_layout(100, 100)
    with pytest.raises(ValueError):
        P.compute_outpaint_layout(100, 100, aspect="1:1", margins=(1, 1, 1, 1))


def test_layout_rounds_up_and_never_shrinks_source():
    """Live 14/10: 512x288 -> 21:9 rimpiccioliva il sorgente (456x256). Ora canvas arrotondato IN SU, scala 1.0."""
    for kw in (dict(aspect=(21, 9)), dict(aspect="9:16"), dict(margins=(0, 192, 0, 192)), dict(margins=(30, 0, 50, 0))):
        lo = P.compute_outpaint_layout(512, 288, **kw)
        assert (lo.src_w, lo.src_h) == (512, 288) and lo.scale == 1.0 and not lo.clamped, (kw, lo)
        assert lo.canvas_w % 64 == 0 and lo.canvas_h % 64 == 0
        assert lo.canvas_w >= 512 and lo.canvas_h >= 288
        assert lo.x + lo.src_w <= lo.canvas_w and lo.y + lo.src_h <= lo.canvas_h
    assert P.compute_outpaint_layout(512, 288, aspect=(21, 9)).canvas == (704, 320)
    assert P.compute_outpaint_layout(512, 288, margins=(0, 192, 0, 192)).canvas == (896, 320)
    lo = P.compute_outpaint_layout(500, 281, aspect="16:9")                    # dispari: mai sotto 1.0
    assert lo.scale >= 1.0 and lo.src_w >= 500 and lo.src_h >= 281


def test_layout_over_cap_reduces_canvas_then_scales():
    lo = P.compute_outpaint_layout(1280, 720, aspect="9:16", max_long_edge=1280)    # canvas ideale 1280x2276 > tetto
    assert lo.clamped and max(lo.canvas) <= 1280 and lo.scale < 1.0
    assert lo.src_w <= lo.canvas_w and lo.src_h <= lo.canvas_h
    lo2 = P.compute_outpaint_layout(1000, 500, margins=(0, 100, 0, 0), max_long_edge=1280)   # 1100 -> 1152 entro il tetto
    assert not lo2.clamped and lo2.scale == 1.0 and lo2.canvas_w == 1152


def test_narrow_bar_detection():
    assert P.compute_outpaint_layout(512, 288, aspect=(21, 9)).narrow_bar            # barre 96 su 704 = 13.6%
    assert P.compute_outpaint_layout(512, 288, margins=(0, 60, 0, 60)).narrow_bar
    assert not P.compute_outpaint_layout(512, 288, margins=(0, 192, 0, 192)).narrow_bar
    assert not P.compute_outpaint_layout(512, 288, aspect="9:16").narrow_bar
    lo = P.compute_outpaint_layout(1280, 704, aspect=(1280, 704))
    assert lo.narrow_bar_fraction == 1.0 and not lo.narrow_bar                       # nessuna barra
    # lo slop di arrotondamento (asse con <=64 px totali) non conta come barra
    assert P.compute_outpaint_layout(512, 288, margins=(0, 192, 0, 192)).narrow_bar_fraction > 0.2


def test_too_much_new_area_flag():
    assert P.compute_outpaint_layout(256, 256, aspect=(3, 1)).too_much_new_area


# ── outpaint: canvas con barre NERE pure ──────────────────────────────────────────────────────────
def test_build_outpaint_canvas_black_bars(tmp_path):
    src = _ffgen(tmp_path / "s.mp4", "testsrc2", 12, size="72x128")           # verticale
    lo = P.compute_outpaint_layout(72, 128, aspect="16:9", max_long_edge=256)
    out = P.build_outpaint_canvas(src, str(tmp_path / "c.mp4"), lo)
    fr = _frames(out)
    assert len(fr) == 12
    assert fr[0].shape[:2] == (lo.canvas_h - lo.canvas_h % 2, lo.canvas_w - lo.canvas_w % 2)
    f = fr[3].astype(int)
    left = f[:, : max(0, lo.x - 8)]                                           # barra sinistra (8 px di margine: chroma 4:2:0 sanguina al bordo)
    assert left.size > 0 and left.max() <= 2, f"barra non nera pura: max={left.max()}"
    inner = f[lo.y + 4: lo.y + lo.src_h - 4, lo.x + 4: lo.x + lo.src_w - 4]
    assert inner.mean() > 40                                                  # il contenuto c'e'


def test_gamma_roundtrip_lut_and_ffmpeg(tmp_path):
    lut_f = P.gamma_lut(1 / 2.0)
    lut_i = P.gamma_lut(2.0)
    rt = lut_i[lut_f]
    err = np.abs(rt.astype(int) - np.arange(256))
    assert err.max() <= 3                                                     # quantizzazione 8 bit, no deriva
    # via ffmpeg su una rampa: schiarisci con 1/2.0 poi riscurisci con 2.0
    ramp = tmp_path / "ramp.mp4"
    subprocess.run([P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "gradients=size=256x16:rate=25:duration=0.4:speed=0:type=linear:c0=black:c1=white:x0=0:y0=0:x1=255:y1=0",
                    "-c:v", "libx264", "-crf", "5", "-pix_fmt", "yuv420p", str(ramp)], check=True, capture_output=True)
    bright = P.apply_gamma(str(ramp), str(tmp_path / "b.mp4"), 0.5)
    back = P.inverse_gamma(bright, str(tmp_path / "r.mp4"), 2.0)
    a, b, c = (_frames(x)[2][8].astype(int).mean(axis=1) for x in (str(ramp), bright, back))
    assert b[32:200].mean() > a[32:200].mean() + 15                           # schiarito
    assert np.abs(c[16:240] - a[16:240]).mean() < 6                           # round-trip ~ originale


def test_outpaint_canvas_with_gamma_brightens_source_only(tmp_path):
    src = _ffgen(tmp_path / "d.mp4", "color=c=0x202020", 10, size="64x64")
    src = str(src)
    lo = P.compute_outpaint_layout(64, 64, aspect="16:9", max_long_edge=256)
    plain = _frames(P.build_outpaint_canvas(src, str(tmp_path / "p.mp4"), lo))[2]
    bright = _frames(P.build_outpaint_canvas(src, str(tmp_path / "g.mp4"), lo, gamma=2.0))[2]
    cx, cy = lo.x + lo.src_w // 2, lo.y + lo.src_h // 2
    assert bright[cy, cx].mean() > plain[cy, cx].mean() + 20
    assert bright[2, 2].max() <= 2 and plain[2, 2].max() <= 2                  # barre nere in entrambi


def test_looks_dark(tmp_path):
    dark = _ffgen(tmp_path / "k.mp4", "color=c=black", 30, size="64x64")
    light = _ffgen(tmp_path / "w.mp4", "color=c=white", 30, size="64x64")
    assert P.looks_dark(dark) and not P.looks_dark(light)
    assert P.mean_luma(light) > 0.8 > 0.2 > P.mean_luma(dark)


def test_paste_original_restores_center(tmp_path):
    src = _ffgen(tmp_path / "s.mp4", "testsrc2", 12, size="72x128")
    lo = P.compute_outpaint_layout(72, 128, aspect="16:9", max_long_edge=256)
    gen = _ffgen(tmp_path / "g.mp4", "color=c=red", 12, size=f"{lo.canvas_w}x{lo.canvas_h}")   # "generato" rosso
    out = P.paste_original(gen, src, str(tmp_path / "o.mp4"), lo, border=2)
    f = _frames(out)[4].astype(int)
    assert f[2, 2, 0] > 150 and f[2, 2, 1] < 80                               # fuori: rosso generato
    cy, cx = lo.y + lo.src_h // 2, lo.x + lo.src_w // 2
    orig = _frames(src)[4].astype(int)
    assert not (f[cy, cx, 0] > 150 and f[cy, cx, 1] < 80)                     # dentro: non piu' rosso uniforme
    assert np.abs(f[lo.y + 6: lo.y + lo.src_h - 6, lo.x + 6: lo.x + lo.src_w - 6].mean()
                  - orig[6:-6, 6:-6].mean()) < 25


# ── outpaint: tutti i formati, feather, min_bar ───────────────────────────────────────────────────
@pytest.mark.parametrize("fmt", ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"])
@pytest.mark.parametrize("anchor", ["center", "left", "right", "top", "bottom"])
@pytest.mark.parametrize("src", [(1280, 720), (720, 1280), (960, 540)])
def test_layout_all_formats_and_anchors_are_valid(fmt, anchor, src):
    lo = P.compute_outpaint_layout(src[0], src[1], aspect=fmt, anchor=anchor)
    assert lo.canvas_w % 64 == 0 and lo.canvas_h % 64 == 0 and max(lo.canvas) <= 1280
    assert lo.x >= 0 and lo.y >= 0 and lo.x + lo.src_w <= lo.canvas_w and lo.y + lo.src_h <= lo.canvas_h
    assert lo.x % 2 == 0 and lo.y % 2 == 0 and lo.src_w % 2 == 0 and lo.src_h % 2 == 0
    assert 0.0 <= lo.new_area < 1.0


def test_layout_vertical_only_extension():
    """16:9 -> 4:3 con sorgente in basso: tutta l'area nuova sta SOPRA (estensione verticale, una sola barra)."""
    lo = P.compute_outpaint_layout(1280, 720, aspect="4:3", anchor="bottom")
    assert lo.x == 0 and lo.y > 0 and lo.y + lo.src_h == lo.canvas_h
    up = P.compute_outpaint_layout(1280, 720, aspect="4:3", anchor="top")
    assert up.y == 0 and up.canvas_h - up.src_h > 0
    m = P.compute_outpaint_layout(1280, 720, margins=(256, 0, 0, 0))             # solo sopra, margini
    assert m.y > 0 and m.x == 0 and m.canvas_h > m.src_h and m.canvas_w == 1280


def test_min_bar_off_by_default_and_only_when_narrow():
    base = P.compute_outpaint_layout(1280, 720, aspect="4:3")
    assert base.narrow_bar and base.scale == 1.0
    mb = P.compute_outpaint_layout(1280, 720, aspect="4:3", min_bar=0.15)
    assert not mb.narrow_bar and mb.scale < 1.0 and mb.new_area > base.new_area
    wide = P.compute_outpaint_layout(1280, 720, aspect="9:16")                     # barre larghe: nessun effetto
    assert P.compute_outpaint_layout(1280, 720, aspect="9:16", min_bar=0.15) == wide
    one = P.compute_outpaint_layout(1280, 720, aspect="4:3", anchor="bottom", min_bar=0.15)
    assert one.scale == 1.0                                                        # barra unica del 25 %: gia' larga


def test_feather_mask_shape_and_sides():
    lo = P.compute_outpaint_layout(512, 288, aspect="4:3")                         # barre sopra/sotto, lati a filo
    m = P.feather_mask(lo, 24)
    assert m.shape == (lo.canvas_h, lo.canvas_w) and m.dtype == np.float32
    assert m.min() == 0.0 and m.max() == 1.0
    mid = lo.x + lo.src_w // 2
    col = m[lo.y: lo.y + 40, mid]
    assert col[0] == 0.0 and np.all(np.diff(col) >= -1e-6) and col[-1] == 1.0       # rampa monotona 0 -> 1
    assert m[lo.y + lo.src_h // 2, 0] == 1.0 and m[lo.y + lo.src_h // 2, lo.canvas_w - 1] == 1.0   # lati a filo
    assert m[: lo.y, :].max() == 0.0 and m[lo.y + lo.src_h:, :].max() == 0.0        # fuori dal sorgente
    assert P.feather_mask(lo, 0)[lo.y + lo.src_h // 2, mid] == 1.0
    sharp = P.feather_mask(lo, 0)
    assert sharp[lo.y + 2, mid] == 1.0 and sharp[lo.y + 1, mid] == 0.0              # taglio netto con bordo 2 px


def test_feather_mask_clamps_to_small_source():
    lo = P.compute_outpaint_layout(128, 72, aspect="9:16", max_long_edge=256)
    m = P.feather_mask(lo, 9999)
    assert m.shape == (lo.canvas_h, lo.canvas_w) and 0.0 <= m.min() and m.max() <= 1.0


def test_paste_feathered_blends_seam_and_is_static(tmp_path):
    src = _ffgen(tmp_path / "s.mp4", "color=c=white", 9, size="128x72")
    lo = P.compute_outpaint_layout(128, 72, aspect="9:16", max_long_edge=256)
    gen = _ffgen(tmp_path / "g.mp4", "color=c=black", 9, size=f"{lo.canvas_w}x{lo.canvas_h}")
    hard = _frames(P.paste_original(gen, src, str(tmp_path / "h.mp4"), lo, border=2))
    soft = _frames(P.paste_original(gen, src, str(tmp_path / "f.mp4"), lo, border=2, feather=16))
    mid = lo.x + lo.src_w // 2
    col_h = hard[3][lo.y: lo.y + 20, mid, 0].astype(int)
    col_f = soft[3][lo.y: lo.y + 20, mid, 0].astype(int)
    assert np.abs(np.diff(col_h)).max() > 150                                        # taglio netto: salto enorme
    assert np.abs(np.diff(col_f)).max() < 60                                         # sfumato: nessun salto
    assert col_f[0] < 40 and col_f[-1] > 215 and np.all(np.diff(col_f) >= -3)        # 0 -> 255 progressivo
    assert max(float(np.abs(a.astype(int) - soft[0].astype(int)).max()) for a in soft) <= 3   # stessa maschera, nessun flicker
    assert len(soft) == 9 and soft[0].shape == (lo.canvas_h, lo.canvas_w, 3)


def test_paste_feather_zero_equals_legacy_path(tmp_path):
    src = _ffgen(tmp_path / "s.mp4", "testsrc2", 9, size="72x128")
    lo = P.compute_outpaint_layout(72, 128, aspect="16:9", max_long_edge=256)
    gen = _ffgen(tmp_path / "g.mp4", "color=c=red", 9, size=f"{lo.canvas_w}x{lo.canvas_h}")
    a = _frames(P.paste_original(gen, src, str(tmp_path / "a.mp4"), lo))
    b = _frames(P.paste_original(gen, src, str(tmp_path / "b.mp4"), lo, feather=0))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# ── restyle: frame 0 ──────────────────────────────────────────────────────────────────────────────
def test_replace_first_frame(tmp_path, clip):
    from PIL import Image
    img = tmp_path / "style.png"
    Image.new("RGB", (200, 100), (250, 10, 10)).save(img)                     # aspetto diverso: cover+crop
    out = P.replace_first_frame(clip, str(img), str(tmp_path / "r.mp4"))
    fr, orig = _frames(out), _frames(clip)
    assert len(fr) == len(orig) == 50
    f0 = fr[0].astype(int)
    assert f0[..., 0].mean() > 200 and f0[..., 1].mean() < 60                 # frame 0 ~ immagine di stile
    assert np.abs(fr[1].astype(int) - orig[1].astype(int)).mean() < 6         # gli altri ~ sorgente


# ── segmentazione ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("total,seg,ov", [(50, 25, 9), (97, 97, 9), (200, 97, 9), (121, 49, 9), (30, 97, 9),
                                          (1000, 97, 9), (98, 97, 9), (60, 25, 0)])
def test_plan_segments_invariants(total, seg, ov):
    segs = P.plan_segments(total, seg, ov)
    s_ = P.snap_frames(seg)
    assert segs[0].start == 0 and segs[-1].end == (total if total > s_ else P.snap_frames(total))
    for a in segs:
        assert a.count % 8 == 1 and a.start >= 0
    for a, b in zip(segs, segs[1:]):
        assert b.start > a.start and b.start <= a.end - min(ov, a.count - 1)       # overlap >= richiesto, niente buchi
    assert all(o >= min(ov, s_ - 1) for o in P.overlaps_of(segs))
    if total <= s_:
        assert len(segs) == 1


def test_plan_segments_rejects_tiny():
    with pytest.raises(ValueError):
        P.plan_segments(5, 25)


def test_split_and_concat_roundtrip(tmp_path, clip):
    parts = P.split_video(clip, str(tmp_path / "segs"), 25, overlap=9)
    segs = [p.segment for p in parts]
    assert len(parts) >= 3 and all(P.probe(p.path, count_frames=True).nb_frames == p.segment.count for p in parts)
    out = P.concat_segments([p.path for p in parts], str(tmp_path / "joined.mp4"), P.overlaps_of(segs))
    assert P.probe(out, count_frames=True).nb_frames == 50                      # nessun frame perso/duplicato
    a, b = _frames(clip), _frames(out)
    assert np.mean([np.abs(x.astype(int) - y.astype(int)).mean() for x, y in zip(a, b)]) < 6   # ~ originale
    assert not P.has_audio(out)


def test_concat_crossfade_blends_overlap(tmp_path):
    red = _ffgen(tmp_path / "r.mp4", "color=c=red", 9, size="32x32")
    blue = _ffgen(tmp_path / "b.mp4", "color=c=blue", 9, size="32x32")
    out = P.concat_segments([red, blue], str(tmp_path / "x.mp4"), [4])
    fr = _frames(out)
    assert len(fr) == 9 + 9 - 4
    r = [int(f[..., 0].mean()) for f in fr]
    bl = [int(f[..., 2].mean()) for f in fr]
    assert r[0] > 200 and bl[-1] > 200 and r[4] > 200                           # fuori overlap: colori puri
    assert r[5] > r[7] > r[8] - 1 or r[5] >= r[8]                               # dentro overlap: il rosso cala...
    assert bl[5] < bl[8]                                                        # ...e il blu sale


def test_concat_rejects_size_mismatch_and_bad_overlaps(tmp_path):
    a = _ffgen(tmp_path / "a.mp4", "color=c=red", 9, size="32x32")
    b = _ffgen(tmp_path / "b.mp4", "color=c=red", 9, size="64x64")
    with pytest.raises(ValueError):
        P.concat_segments([a, b], str(tmp_path / "o.mp4"), [0])
    with pytest.raises(ValueError):
        P.concat_segments([a, a], str(tmp_path / "o.mp4"), [1, 2])


def test_fill_area_shrinks_only_symmetric_bars():
    """fill_area (17/10, live): barre opposte con area nuova < 60% -> sorgente rimpicciolito al centro; una barra sola no."""
    for asp, w, h in (("21:9", 704, 384), ("4:3", 704, 384), ("1:1", 704, 384)):
        off = P.compute_outpaint_layout(w, h, aspect=asp)
        on = P.compute_outpaint_layout(w, h, aspect=asp, fill_area=P.FILL_AREA)
        assert off.scale == 1.0 and off.new_area < P.FILL_AREA
        assert on.scale < 1.0 and abs(on.new_area - P.FILL_AREA) < 0.02
        assert (on.canvas_w, on.canvas_h) == (off.canvas_w, off.canvas_h)                 # stesso canvas
        assert abs((on.x + on.src_w / 2) - (off.x + off.src_w / 2)) <= 2 and abs((on.y + on.src_h / 2) - (off.y + off.src_h / 2)) <= 2
        assert on.x >= 0 and on.y >= 0 and on.x + on.src_w <= on.canvas_w and on.y + on.src_h <= on.canvas_h
    one = P.compute_outpaint_layout(704, 384, aspect="4:3", anchor="bottom", fill_area=P.FILL_AREA)
    assert one == P.compute_outpaint_layout(704, 384, aspect="4:3", anchor="bottom")       # barra singola: invariato
    big = P.compute_outpaint_layout(704, 384, aspect="9:16", fill_area=P.FILL_AREA)
    assert big == P.compute_outpaint_layout(704, 384, aspect="9:16")                       # area gia' >= 60%: invariato
    m = P.compute_outpaint_layout(704, 384, margins=(160, 0, 160, 0), fill_area=P.FILL_AREA)
    assert m.scale < 1.0 and abs(m.new_area - P.FILL_AREA) < 0.02                          # vale anche con i margini
