"""core/hdr_export.py + modalita' HDR di video_edit_ic: PQ, tonemap, scrittore HEVC/EXR/anteprima, cucitura, build_job.
Nessuna GPU, nessun Qt: solo numpy + ffmpeg (tiene pochi MB)."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from core import hdr_export as H
from core import video_edit_ic as vic
from core import video_ic_prep as P

_HAVE_FFMPEG = bool(shutil.which(P.ffmpeg_path()) or os.path.exists(P.ffmpeg_path()))
needs_ff = pytest.mark.skipif(not _HAVE_FFMPEG or not H.ffmpeg_ok(), reason="ffmpeg con libx265 non disponibile")


def _ramp(w=128, h=72, peak=8.0):
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    return np.repeat(np.repeat(x, h, 0), 3, 2) * peak


def test_pq_known_values():
    # 203 nit (bianco diffuso) = codice PQ ~0.58; 10000 nit = 1.0; 0 = 0
    assert abs(float(H.pq_oetf(np.array([203 / 10000.0]))[0]) - 0.5807) < 2e-3
    assert abs(float(H.pq_oetf(np.array([1.0]))[0]) - 1.0) < 1e-6
    assert float(H.pq_oetf(np.array([0.0]))[0]) < 1e-5
    y = np.array([0.0, 0.0001, 0.02, 0.1, 0.5, 1.0], np.float32)
    assert np.allclose(H.pq_eotf(H.pq_oetf(y)), y, atol=2e-4)


def test_to_pq_u16_grey_and_primaries():
    g = np.full((2, 2, 3), 1.0, np.float32)                     # grigio neutro: R=G=B anche in BT.2020
    u = H.to_pq_u16(g)
    assert u.dtype == np.uint16 and abs(int(u[0, 0, 0]) - int(u[0, 0, 1])) <= 2 and abs(int(u[0, 0, 1]) - int(u[0, 0, 2])) <= 2
    assert abs(int(u[0, 0, 0]) / 65535.0 - 0.5807) < 3e-3, "lineare 1.0 = 203 nit"
    red = np.zeros((1, 1, 3), np.float32); red[..., 0] = 1.0     # rosso Rec.709 = meno saturo in BT.2020 (G,B > 0)
    r = H.to_pq_u16(red)[0, 0]
    assert r[0] > r[1] > 0 and r[2] > 0
    over = H.to_pq_u16(np.full((1, 1, 3), 1.0e4, np.float32))   # oltre i 10000 nit: clip a 65535, mai overflow
    assert int(over.max()) == 65535


def test_tonemap_sdr_matches_official_node_formula():
    f = np.array([[[0.0, 0.18, 1.0], [4.0, 16.0, 1.0e6]]], np.float32)
    got = H.tonemap_sdr_u8(f, 0.0)
    x = np.clip(f, 0, 1e4); y = x / (1 + x)
    ref = np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)
    assert np.abs(got.astype(float) - np.round(ref * 255)).max() <= 1
    assert H.tonemap_sdr_u8(f, 1.0).mean() > got.mean() > H.tonemap_sdr_u8(f, -1.0).mean()
    assert H.clamp_exposure(99) == H.MAX_EV and H.clamp_exposure(-99) == H.MIN_EV


def test_stitcher_identity_on_consistent_segments():
    for total, seg in ((200, 97), (41, 17), (49, 97)):          # (41,17,9): overlap su ENTRAMBI i lati dei segmenti
        segs = P.plan_segments(total, seg, 9)
        out = []
        st = H.Stitcher(out.append, [(sg.start, sg.count) for sg in segs])
        for sg in segs:
            st.add(np.stack([np.full((2, 2, 3), float(sg.start + k), np.float32) for k in range(sg.count)]))
        st.finish()
        assert len(out) == total == st.written, (total, len(out))
        assert max(abs(float(f[0, 0, 0]) - i) for i, f in enumerate(out)) < 1e-3
    with pytest.raises(ValueError):
        H.Stitcher(print, [])


def test_stitcher_blends_in_linear_light():
    a = np.stack([np.full((1, 1, 3), 0.0, np.float32)] * 9)     # segmento A tutto 0, B tutto 8: nel mezzo si mescolano
    b = np.stack([np.full((1, 1, 3), 8.0, np.float32)] * 9)
    out = []
    st = H.Stitcher(out.append, [(0, 9), (5, 9)])               # overlap di 4 frame
    st.add(a); st.add(b); st.finish()
    v = [float(f[0, 0, 0]) for f in out]
    assert len(v) == 14 and v[:5] == [0.0] * 5 and v[-5:] == [8.0] * 5
    assert v[5:9] == pytest.approx([8 * w / 5 for w in (1, 2, 3, 4)])


@needs_ff
def test_writer_outputs_tags_and_values(tmp_path):
    w, h, n = 128, 72, 9
    prev = str(tmp_path / "o.mp4")
    pq, exr = H.default_paths(prev)
    assert pq.endswith("_HDR_PQ.mp4") and exr.endswith("_EXR")
    with H.HdrWriter(prev, pq=pq, exr_dir=exr, width=w, height=h, fps=24, crf_hdr=6) as wr:
        for i in range(n):
            wr.write(_ramp(w, h, 8.0 + i))
    res = wr.result()
    assert res["frames"] == n and res["peak_linear"] == pytest.approx(16.0) and len(os.listdir(exr)) == n
    c = H.probe_color(pq)
    assert (c["codec"], c["pix_fmt"], c["color_transfer"], c["color_primaries"], c["color_space"]) == \
           ("hevc", "yuv420p10le", "smpte2084", "bt2020", "bt2020nc")
    assert "Mastering display metadata" in c["side_data"]
    s = H.probe_color(prev)
    assert (s["codec"], s["pix_fmt"], s["color_space"]) == ("h264", "yuv420p", "bt709")
    assert int(c["nb_frames"]) == n == int(s["nb_frames"])
    # il video HDR decodificato torna ai codici PQ attesi (errore massimo < 0.5%)
    r = subprocess.run([P.ffmpeg_path(), "-v", "error", "-i", pq, "-vf",
                        "scale=in_color_matrix=bt2020nc:in_range=tv:out_range=pc,format=rgb48le",
                        "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True)
    got = np.frombuffer(r.stdout, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535
    ref = H.to_pq_u16(_ramp(w, h, 8.0)).astype(np.float32) / 65535
    assert np.abs(got - ref).max() < 5e-3
    # l'EXR e' float lineare: il primo frame rilegge uguale (half: tolleranza)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    e = cv2.imread(os.path.join(exr, "frame_00000.exr"), cv2.IMREAD_UNCHANGED)
    assert e is not None and e.dtype == np.float32 and abs(float(e.max()) - 8.0) < 0.05


@needs_ff
def test_writer_rejects_bad_frames_and_sidecar(tmp_path):
    prev = str(tmp_path / "o.mp4")
    wr = H.HdrWriter(prev, width=64, height=64, fps=24)
    with pytest.raises(ValueError):
        wr.write(np.zeros((64, 64), np.float32))
    wr.write(np.zeros((64, 64, 3), np.float32))               # NaN/inf ripuliti, nessuna eccezione
    wr.write(np.full((64, 64, 3), np.nan, np.float32))
    res = wr.close()
    assert res["frames"] == 2 and res["pq"] == "" and res["exr_dir"] == ""
    H.write_sidecar(prev, {**res, "pq": str(tmp_path / "x.mp4"), "exr_dir": str(tmp_path)})
    assert H.extra_outputs(prev) == [str(tmp_path)], "solo i path esistenti"


@needs_ff
def test_mux_audio_keeps_video_tags(tmp_path):
    w, h = 64, 64
    pq = str(tmp_path / "a_HDR_PQ.mp4")
    with H.HdrWriter(str(tmp_path / "a.mp4"), pq=pq, width=w, height=h, fps=25) as wr:
        for _ in range(25):
            wr.write(_ramp(w, h, 4.0))
    src = str(tmp_path / "src.mp4")
    subprocess.run([P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-shortest", "-frames:v", "25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", src], check=True, capture_output=True)
    assert not H.mux_audio(pq, str(tmp_path / "nessuno.mp4"))
    assert H.mux_audio(pq, src) and P.has_audio(pq)
    assert H.probe_color(pq)["color_transfer"] == "smpte2084"


# ── modalita' hdr in build_job ─────────────────────────────────────────────────────────────────────
def _gen(out, n, size="320x190", audio=False, pix="yuv420p"):
    cmd = [P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-shortest"]
    cmd += ["-frames:v", str(n), "-c:v", "libx264" if pix == "yuv420p" else "libx265", "-pix_fmt", pix]
    cmd += (["-c:a", "aac"] if audio else []) + [str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


@needs_ff
def test_build_job_hdr_params(tmp_path, monkeypatch):
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "")
    src = _gen(tmp_path / "s.mp4", 50, size="1920x1080", audio=True)
    ij = vic.build_job({"mode": "hdr", "video": src, "keep_audio": True, "hdr_exposure_ev": 99,
                        "hdr_format": "both", "hdr_hq": True, "seed": 7}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert p["control_type"] == "hdr" and p["_strict_ic"] is True and p["native_res"] is True
    assert (p["width"], p["height"]) == (1280, 720), "tetto 1280 sul lato lungo, aspetto esatto, niente multipli di 64"
    assert p["num_frames"] == 49 and p["control_strength"] == 1.0 and p["prompt"] == ""
    assert p["hdr_format"] == "both" and p["hdr_hq"] is True and p["hdr_exposure_ev"] == H.MAX_EV
    assert p["hdr_audio_from"] == src and ij.mux_audio_from == "", "l'audio lo rimette il worker su anteprima e HEVC"
    assert "chunk_frames" not in p and p["hdr_seg_frames"] == 97 and p["hdr_overlap"] == 9
    assert any("EXR" in n for n in ij.notes)


@needs_ff
def test_build_job_hdr_long_clip_and_odd_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "")
    src = _gen(tmp_path / "s.mp4", 220, size="642x362")
    ij = vic.build_job({"mode": "hdr", "video": src}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert (p["width"], p["height"]) == (642, 362) and p["num_frames"] == 217 and ij.segments == 3
    assert "chunk_frames" not in p and p["hdr_format"] == "pq" and "hdr_audio_from" not in p
    with pytest.raises(vic.IcJobError, match="formato HDR"):
        vic.build_job({"mode": "hdr", "video": src, "hdr_format": "dpx"}, str(tmp_path), 2, str(tmp_path / "o2.mp4"))


@needs_ff
def test_build_job_hdr_warns_on_10bit_source(tmp_path, monkeypatch):
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "")
    src = _gen(tmp_path / "s10.mp4", 17, size="320x192", pix="yuv420p10le")
    ij = vic.build_job({"mode": "hdr", "video": src}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    assert any("yuv420p10le" in n for n in ij.notes)


def test_missing_weight_hdr_needs_scene_emb(tmp_path, monkeypatch):
    spec = vic.vem.get_spec("hdr")
    lora = tmp_path / "hdr.safetensors"; lora.write_bytes(b"x")
    monkeypatch.setattr(vic, "weight_path", lambda s: str(lora))
    monkeypatch.setattr(vic, "hdr_scene_emb_path", lambda: str(tmp_path / "nessuno.safetensors"))
    msg = vic.missing_weight(spec)
    assert "scene-emb" in msg and "ltx-iclora-hdr-scene-emb" in msg and "Peso «hdr»" not in msg
    (tmp_path / "nessuno.safetensors").write_bytes(b"x")
    assert vic.missing_weight(spec) == ""
    monkeypatch.setattr(vic, "weight_path", lambda s: str(tmp_path / "manca.safetensors"))
    assert "Peso «hdr» mancante" in vic.missing_weight(spec)


def test_job_outputs_reads_sidecar(tmp_path):
    out = str(tmp_path / "o.mp4")
    open(out, "wb").write(b"x")
    pq, exr = H.default_paths(out)
    open(pq, "wb").write(b"x"); os.makedirs(exr)
    H.write_sidecar(out, {"pq": pq, "exr_dir": exr})
    spec = vic.vem.get_spec("hdr")
    ij = vic.IcJob(params={"output": out}, spec=spec, work_dir=str(tmp_path))
    assert ij.outputs(out) == [out, pq, exr]
    ij2 = vic.IcJob(params={"output": out}, spec=vic.vem.get_spec("refocus"), work_dir=str(tmp_path))
    assert ij2.outputs(out) == [out]
