"""Test di core/motion_tracks.py: fedelta' al nodo ufficiale LTXVDrawTracks / LTXVSparseTrackEditor (Lightricks).

Tre livelli di confronto numerico, nessuna GPU:
1. reimplementazione di RIFERIMENTO scritta dal sorgente (algoritmo diverso: mappa di priorita' per pixel, non timbri);
2. ORACOLO: il codice puro del nodo (`sparse_tracks.py`) eseguito su torch-CPU con stub di `comfy_api` (nessun ComfyUI), se
   il file e' su disco (saltato altrimenti);
3. proprieta' (colori, raggi, scia, BGR, coordinate, conteggio frame).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

from core import motion_tracks as mt

NODE_SRC = r"E:\AI\workflows\LTX-2.3-curated\_Lightricks-official\sparse_tracks.py"
WORKFLOW = (r"E:\AI\workflows\LTX-2.3-curated\_Lightricks-official\example_workflows\2.3"
            r"\LTX-2.3_ICLoRA_Motion_Track_Distilled.json")


# ── riferimento indipendente ──────────────────────────────────────────────────────────────────────
def _ref_color(ratio: float):
    """Gradiente blu -> verde -> giallo -> rosso scritto a mano (stesso significato del nodo, float64)."""
    if ratio <= 1 / 3:
        t = ratio * 3
        return (0.0, t, 1 - t)
    if ratio <= 2 / 3:
        t = (ratio - 1 / 3) * 3
        return (t, 1.0, 0.0)
    t = (ratio - 2 / 3) * 3
    return (1.0, 1 - t, 0.0)


def _ref_bilinear(img: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """Bilinear align_corners=False per pixel (cicli), indipendente da resize_bilinear."""
    h, w, _ = img.shape
    out = np.zeros((oh, ow, 3), np.float64)
    for y in range(oh):
        sy = max((y + 0.5) * h / oh - 0.5, 0.0)
        y0 = min(int(np.floor(sy)), h - 1); y1 = min(y0 + 1, h - 1); fy = sy - y0
        for x in range(ow):
            sx = max((x + 0.5) * w / ow - 0.5, 0.0)
            x0 = min(int(np.floor(sx)), w - 1); x1 = min(x0 + 1, w - 1); fx = sx - x0
            top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
            bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
            out[y, x] = top * (1 - fy) + bot * fy
    return out


def reference_render(tracks, width, height):
    """Riferimento: per ogni frame, per ogni cerchio visibile (eta' decrescente) una mappa di PRIORITA' sull'intera tela
    (np.maximum), poi colore del cerchio con priorita' massima; riduzione bilinear a cicli; BGR."""
    n = max(len(t) for t in tracks)
    if height <= width:
        rw, rh = int(width * 1080 / height), 1080
    else:
        rw, rh = 1080, int(height * 1080 / width)
    sx, sy = rw / width, rh / height
    yy, xx = np.mgrid[0:rh, 0:rw]
    frames = []
    for t in range(n):
        lo = max(0, t - 50)
        circles = []                                    # (eta', indice traccia, raggio, colore, cx, cy)
        for ti, trk in enumerate(tracks):
            for k in range(lo, t + 1):
                if k >= len(trk):
                    continue
                age = t - k
                ratio = 1.0 - age / 50.0
                r = 2 + 6 * ratio
                circles.append((age, ti, r, _ref_color(ratio), round(trk[k]["x"] * sx), round(trk[k]["y"] * sy)))
        circles.sort(key=lambda c: (-c[0], c[1]))      # piu' vecchi prima; a pari eta' l'indice di traccia maggiore vince
        prio = np.full((rh, rw), -1, np.int64)
        for j, (_a, _ti, r, _c, cx, cy) in enumerate(circles):
            inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            prio[inside] = j                            # j cresce: l'ultimo vince
        canvas = np.zeros((rh, rw, 3), np.float64)
        for j, c in enumerate(circles):
            canvas[prio == j] = c[3]
        small = _ref_bilinear(canvas, height, width)
        frames.append(small[..., ::-1])
    return np.stack(frames)


def _tracks_two():
    ctrl1 = [{"x": 8, "y": 10}, {"x": 40, "y": 30}, {"x": 70, "y": 12}]
    ctrl2 = [{"x": 20, "y": 50}, {"x": 60, "y": 52}]
    return [mt.interpolate_spline(ctrl1, 6), mt.interpolate_spline(ctrl2, 6)]


# ── interpolazione ────────────────────────────────────────────────────────────────────────────────
def test_interpolate_cases():
    assert mt.interpolate_spline([], 5) == []
    one = mt.interpolate_spline([{"x": 3.4, "y": 7.6}], 4)
    assert one == [{"x": 3, "y": 8}] * 4
    two = mt.interpolate_spline([{"x": 0, "y": 0}, {"x": 10, "y": 20}], 5)
    assert two[0] == {"x": 0, "y": 0} and two[-1] == {"x": 10, "y": 20} and two[2] == {"x": 5, "y": 10}
    cps = [{"x": 0, "y": 0}, {"x": 100, "y": 50}, {"x": 200, "y": 0}, {"x": 300, "y": 80}]
    sp = mt.interpolate_spline(cps, 121)
    assert len(sp) == 121 and sp[0] == {"x": 0, "y": 0} and sp[-1] == {"x": 300, "y": 80}   # passa per gli estremi
    assert {"x": 100, "y": 50} in sp and {"x": 200, "y": 0} in sp                           # e per i punti di controllo
    assert mt.interpolate_spline([[1, 2], [3, 4]], 3)[1] == {"x": 2, "y": 3}                 # coppie accettate


def test_interpolate_matches_node_source():
    node = _node_module()
    rng = np.random.default_rng(3)
    for npts in (1, 2, 3, 5, 9):
        cps = [{"x": float(rng.uniform(0, 960)), "y": float(rng.uniform(0, 540))} for _ in range(npts)]
        for ns in (2, 17, 121):
            assert mt.interpolate_spline(cps, ns) == node._interpolate_spline(cps, ns)


# ── colori, raggi, finestra ───────────────────────────────────────────────────────────────────────
def test_age_colors_gradient():
    c = mt.age_colors(np.array([0.0, 1 / 3, 0.5, 2 / 3, 1.0], np.float32))
    assert np.allclose(c[0], (0, 0, 1)) and np.allclose(c[1], (0, 1, 0), atol=1e-6)
    assert np.allclose(c[2], (0.5, 1, 0), atol=1e-6) and np.allclose(c[3], (1, 1, 0), atol=1e-5)
    assert np.allclose(c[4], (1, 0, 0), atol=1e-6)


def test_render_resolution():
    assert mt.render_resolution(960, 544)[:2] == (1905, 1080)
    assert mt.render_resolution(544, 960)[:2] == (1080, 1905)
    rw, rh, sx, sy = mt.render_resolution(1024, 576)
    assert (rw, rh) == (1920, 1080) and sx == 1.875 and sy == 1.875


def test_single_static_dot_is_bgr_and_sized():
    """Un punto fermo: il cerchio piu' recente (eta' 0) ha raggio 8 a 1080p, ROSSO in RGB -> nel BGR del nodo il canale 2 e' 1."""
    tr = [[{"x": 480, "y": 272}] * 3]
    a = mt.render_array(tr, 960, 544)
    assert a.shape == (3, 544, 960, 3)
    px = a[0, 272, 480]
    assert px[0] < 0.05 and px[1] < 0.05 and px[2] > 0.9      # (B,G,R) scambiati: il rosso e' nel canale 2
    lit = (a[0].sum(-1) > 0.5).sum()
    assert 25 < lit < 120                                      # raggio 8/1080 ridotto a 544: ~4 px -> ~50 pixel
    assert a[0, 0, 0].sum() == 0                               # fuori dal cerchio e' nero
    # a frame 2 il punto fermo ha anche i cerchi piu' vecchi (eta' 1 e 2, raggio minore) sotto: il centro resta il piu' recente
    assert a[2, 272, 480][2] > 0.9


def test_trail_grows_and_ages():
    tr = [[{"x": 40 + i * 12, "y": 272} for i in range(60)]]
    a = mt.render_array(tr, 960, 544)
    lit_counts = [(a[t].sum(-1) > 0.1).sum() for t in (0, 10, 40, 59)]
    assert lit_counts[0] < lit_counts[1] < lit_counts[2]      # la scia si allunga fino a MAX_TRAIL
    assert lit_counts[3] >= lit_counts[2] * 0.9               # poi resta di lunghezza costante (finestra di 51)
    # colore per eta' (BGR del nodo): la testa (eta' 0) e' rossa -> canale 2; la coda (eta' 50) e' blu -> canale 0
    head = a[59, 270:275, 40 + 59 * 12 - 2:40 + 59 * 12 + 3]
    tail = a[59, 270:275, 40 + 9 * 12 - 2:40 + 9 * 12 + 3]
    assert head[..., 2].max() > 0.9 and head[..., 0].max() < 0.1
    assert tail[..., 0].max() > 0.5 and tail[..., 2].max() < 0.2


def test_frame_count_and_shorter_track_invisible():
    tr = [[{"x": 5, "y": 5}] * 10, [{"x": 30, "y": 30}] * 4]
    a = mt.render_array(tr, 64, 64)
    assert a.shape[0] == 10
    # la traccia 2 (4 frame) non esiste piu' dopo il frame 4: nessun pixel li' dopo la scia
    assert (a[8, 24:36, 24:36].sum()) == 0 or a[8, 24:36, 24:36].sum() < a[2, 24:36, 24:36].sum()
    assert mt.render_array([], 32, 32).shape == (1, 32, 32, 3)           # senza tracce: un frame nero (come il nodo)
    assert mt.render_array(tr, 64, 64, num_frames=12).shape[0] == 12


# ── riferimento indipendente ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("w,h", [(96, 64), (64, 96)])
def test_render_matches_reference(w, h):
    tr = _tracks_two()
    tr = [[{"x": p["x"] * w / 96, "y": p["y"] * h / 64} for p in t] for t in tr]
    got = mt.render_array(tr, w, h)
    ref = reference_render(tr, w, h)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 2e-5


# ── oracolo: codice del nodo su torch-CPU ─────────────────────────────────────────────────────────
_NODE = {}


def _node_module():
    if "m" in _NODE:
        return _NODE["m"]
    if not os.path.exists(NODE_SRC):
        pytest.skip("sorgente del nodo non presente")
    torch = pytest.importorskip("torch")

    class _Any:
        def __init__(self, *a, **k):
            self.a, self.k = a, k

        def __getattr__(self, n):
            return _Any

        def __call__(self, *a, **k):
            return _Any()

    class _Out:
        def __init__(self, *a, **k):
            self.args = a

    io = types.SimpleNamespace(ComfyNode=object, NodeOutput=_Out, Schema=_Any, Image=_Any, String=_Any, Int=_Any)
    pkg = types.ModuleType("ltxv_oracle"); pkg.__path__ = []
    reg = types.ModuleType("ltxv_oracle.nodes_registry")
    reg.comfy_node = lambda name=None, description=None: (lambda cls: cls)
    api = types.ModuleType("comfy_api"); latest = types.ModuleType("comfy_api.latest"); latest.io = io
    saved = {k: sys.modules.get(k) for k in ("comfy_api", "comfy_api.latest")}
    sys.modules.update({"ltxv_oracle": pkg, "ltxv_oracle.nodes_registry": reg, "comfy_api": api, "comfy_api.latest": latest})
    try:
        spec = importlib.util.spec_from_file_location("ltxv_oracle.sparse_tracks", NODE_SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ltxv_oracle.sparse_tracks"] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    _NODE["m"] = mod
    _NODE["torch"] = torch
    return mod


def _node_render(tracks, w, h, monkeypatch):
    node = _node_module()
    torch = _NODE["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)          # mai la GPU nei test
    out = node.LTXVDrawTracks.execute(json.dumps(tracks), w, h)
    return out.args[0].float().numpy()


@pytest.mark.parametrize("w,h", [(96, 64), (64, 96), (160, 96)])
def test_render_matches_node_oracle(w, h, monkeypatch):
    ctrl = [[{"x": 8, "y": 10}, {"x": 0.8 * w, "y": 0.5 * h}, {"x": 0.9 * w, "y": 6}],
            [{"x": 0.2 * w, "y": 0.8 * h}, {"x": 0.7 * w, "y": 0.85 * h}],
            [{"x": 0.5 * w, "y": 0.4 * h}]]
    tr = [mt.interpolate_spline(c, 24) for c in ctrl]
    got = mt.render_array(tr, w, h)
    exp = _node_render(tr, w, h, monkeypatch)
    assert got.shape == exp.shape
    # il nodo restituisce half: tolleranza = 1 ulp di float16 sui valori in [0,1]
    assert np.abs(got - exp).max() < 1.5e-3


def test_render_matches_node_oracle_long_trail(monkeypatch):
    """Piu' di 50 frame: finestra della scia a regime, tracce che escono dal quadro."""
    w, h = 96, 64
    ctrl = [[{"x": -10, "y": 20}, {"x": 50, "y": 30}, {"x": 120, "y": 40}], [{"x": 90, "y": 60}, {"x": 5, "y": 5}]]
    tr = [mt.interpolate_spline(c, 70) for c in ctrl]
    got = mt.render_array(tr, w, h)
    exp = _node_render(tr, w, h, monkeypatch)
    assert np.abs(got - exp).max() < 1.5e-3


# ── parsing ───────────────────────────────────────────────────────────────────────────────────────
def test_parse_tracks_formats():
    tr = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    assert mt.parse_tracks(json.dumps([tr])) == [tr]
    assert mt.parse_tracks([json.dumps(tr)]) == [tr]
    assert len(mt.parse_tracks([[tr, tr], [tr]])) == 3


def test_parse_track_input_variants():
    ctrl = [[{"x": 1, "y": 2}, {"x": 30, "y": 4}, {"x": 50, "y": 9}]]
    i = mt.parse_track_input(json.dumps(ctrl), 121)
    assert i.kind == "control" and len(i.tracks[0]) == 3
    dense = mt.interpolate_all(ctrl, 25)
    assert mt.parse_track_input(dense, 25).kind == "dense"                      # lunghezza == num_frames
    assert mt.parse_track_input(dense, 121).kind == "control"                   # altrimenti punti di controllo
    wrapped = {"points_store": ctrl, "coordinates": dense, "size": [960, 544]}
    i = mt.parse_track_input(json.dumps(wrapped), 25)
    assert i.kind == "control" and i.size == (960, 544)
    nodestyle = {"points_store": json.dumps(ctrl), "coordinates": json.dumps(dense)}
    assert mt.parse_track_input(nodestyle).kind == "control"
    only_dense = mt.parse_track_input({"coordinates": dense}, 25)
    assert only_dense.kind == "dense"
    with pytest.raises(ValueError):
        mt.parse_track_input("[]")


def test_dense_tracks_rescale_and_length():
    inp = mt.TrackInput("control", [[{"x": 0, "y": 0}, {"x": 960, "y": 544}]], (960, 544))
    d = mt.dense_tracks(inp, 9, 1920, 1088)
    assert len(d[0]) == 9 and d[0][-1] == {"x": 1920, "y": 1088}
    dense = mt.TrackInput("dense", mt.interpolate_all([[{"x": 0, "y": 0}, {"x": 10, "y": 10}]], 9))
    assert mt.dense_tracks(dense, 9, 64, 64)[0][4] == {"x": 5, "y": 5}
    assert len(mt.dense_tracks(dense, 17, 64, 64)[0]) == 17                     # ricampionata


@pytest.mark.skipif(not os.path.exists(WORKFLOW), reason="workflow ufficiale non presente")
def test_load_official_workflow_json():
    """I JSON del workflow ufficiale si riusano: le tracce dell'esempio si caricano e si rendono."""
    raw = open(WORKFLOW, encoding="utf-8").read()
    inp = mt.parse_track_input(raw, 121)
    assert inp.kind == "control" and len(inp.tracks) == 2
    d = mt.dense_tracks(inp, 121, 960, 544)
    assert len(d) == 2 and all(len(t) == 121 for t in d)
    node = _node_module() if os.path.exists(NODE_SRC) else None
    if node is not None:                                                           # stesso risultato del nodo
        wf = json.loads(raw)
        ps = next(n for n in wf["nodes"] if n["type"] == "LTXVSparseTrackEditor")["widgets_values"][0]
        exp = [node._interpolate_spline(sp, 121) for sp in json.loads(ps)]
        assert d == exp


# ── modello dell'editor ───────────────────────────────────────────────────────────────────────────
def test_editor_model_flow():
    m = mt.TrackEditorModel(960, 544)
    assert m.is_empty() and not m.undo_last_point()
    m.add_point(100, 100); m.add_point(300, 200); m.add_point(500, 150)
    assert len(m) == 1 and m.n_points == 3 and m.active == 0
    m.new_track()
    assert len(m) == 2 and m.new_track() == 1 and len(m) == 2                    # non crea tracce vuote in piu'
    m.add_point(50, 400)
    assert m.splines[1] == [[50.0, 400.0]]
    assert m.hit_test(102, 99, 6) == (0, 0) and m.hit_test(700, 500, 6) is None
    m.move_point(0, 1, 2000, -5)                                                 # clamp nello spazio dell'immagine
    assert m.splines[0][1] == [960.0, 0.0]
    m.nudge_point(0, 0, 10, 0)
    assert m.splines[0][0] == [110.0, 100.0]
    m.select(1)
    assert m.undo_last_point() and len(m) == 1                                   # traccia 1 aveva un punto: sparisce
    assert m.undo_last_point() and m.n_points == 2
    m.delete_point(0, 0)
    assert m.n_points == 1
    m.clear()
    assert m.is_empty() and m.active == -1


def test_editor_model_json_roundtrip_and_resize():
    m = mt.TrackEditorModel(960, 544)
    for p in ((100, 100), (300, 200), (500, 150)):
        m.add_point(*p)
    m.new_track(); m.add_point(50, 400)
    js = m.to_json(121)
    d = json.loads(js)
    assert set(d) >= {"points_store", "coordinates", "size"} and len(d["coordinates"][0]) == 121
    m2 = mt.TrackEditorModel(1920, 1088)                                         # altra risoluzione: riscala
    assert m2.load(js, 121) == 2
    assert m2.splines[0][0] == [200.0, 200.0] and m2.splines[1] == [[100.0, 800.0]]
    m3 = mt.TrackEditorModel(960, 544)
    assert m3.load(json.dumps({"coordinates": m.dense(25)}), 25) == 2            # dense -> punti di controllo modificabili
    assert len(m3.splines[0]) <= 12 and len(m3.splines[1]) == 1                  # la traccia ferma resta un punto
    m.set_size(480, 272)
    assert m.splines[0][0] == [50.0, 50.0]


def test_default_size_and_frames():
    assert mt.default_size(1920, 1080) == (1024, 576)
    assert mt.default_size(1080, 1920) == (576, 1024)
    assert mt.default_size(1000, 1000) == (576, 576)
    assert mt.default_size(4000, 500, max_long=1280)[0] <= 1280
    assert mt.frames_for_seconds(10, 24) == 241 and mt.frames_for_seconds(5, 24) == 121
    assert mt.frames_for_seconds(0.1, 24) == 9 and (mt.frames_for_seconds(7, 25) - 1) % 8 == 0


# ── video ─────────────────────────────────────────────────────────────────────────────────────────
def test_write_video_frames_and_colors(tmp_path):
    from core import video_ic_prep as prep
    if not os.path.exists(prep.ffmpeg_path()) and prep.ffmpeg_path() == "ffmpeg":
        pytest.skip("ffmpeg assente")
    tr = mt.interpolate_all([[{"x": 10, "y": 10}, {"x": 90, "y": 50}], [{"x": 48, "y": 60}]], 17)
    out = str(tmp_path / "tracks.mp4")
    mt.write_video(tr, 96, 64, 24, out)
    info = prep.probe(out, count_frames=True)
    assert (info.width, info.height, info.nb_frames) == (96, 64, 17)
    frames = list(prep._read_frames(out, 96, 64))
    assert len(frames) == 17
    ref = mt.to_uint8(mt.render_array(tr, 96, 64))
    # 4:4:4 lossless: i colori sopravvivono (tolleranza per la conversione RGB<->YUV)
    assert np.abs(frames[8].astype(int) - ref[8].astype(int)).max() <= 6
    with pytest.raises(ValueError):
        mt.write_video(tr, 95, 64, 24, out)
