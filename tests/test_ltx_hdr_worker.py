"""workers/ltx_hdr.py con una pipeline FINTA (nessuna GPU, nessun peso): segmentazione, cucitura, scrittura di
anteprima SDR + HEVC PQ + EXR + audio, sidecar, errori. Verifica il cablaggio, NON la qualita' dell'HDR (live)."""
import os
import shutil
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core import hdr_export as H  # noqa: E402
from core import video_ic_prep as P  # noqa: E402
from workers import ltx_hdr  # noqa: E402

pytestmark = pytest.mark.skipif(not (shutil.which(P.ffmpeg_path()) or os.path.exists(P.ffmpeg_path())) or not H.ffmpeg_ok(),
                                reason="ffmpeg con libx265 non disponibile")


class FakePipe:
    """Come HDRICLoraPipeline.__call__: ritorna [f,h,w,3] float lineare; ogni frame vale (indice globale del
    segmento + k) cosi' la cucitura corretta dà una rampa continua."""

    def __init__(self, bad=None, starts=(0,)):
        self.calls = []
        self.bad = bad
        self.starts = list(starts)

    def __call__(self, seed, height, width, num_frames, frame_rate, video_conditioning, tiling_config=None,
                 high_quality_hdr=False):
        self.calls.append(dict(seed=seed, h=height, w=width, n=num_frames, fps=frame_rate, cond=video_conditioning,
                               hq=high_quality_hdr))
        start = self.starts[len(self.calls) - 1]
        # un frame vale (start + k) / 10: il picco cresce con l'indice (si controlla anche il picco del sidecar)
        t = torch.stack([torch.full((height, width, 3), (start + k) / 10.0) for k in range(num_frames)])
        if self.bad == "nan":
            t[0, 0, 0, 0] = float("nan")
        return t


def _src(tmp_path, n, audio=True):
    out = str(tmp_path / "src.mp4")
    cmd = [P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-shortest"]
    cmd += ["-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _params(tmp_path, src, n, **kw):
    p = {"control_video": src, "output": str(tmp_path / "o.mp4"), "fps": 25, "width": 96, "height": 64,
         "num_frames": n, "seed": 3, "control_strength": 1.0, "hdr_format": "both", "hdr_audio_from": src}
    p.update(kw)
    return p


def test_single_segment_all_outputs(tmp_path):
    src = _src(tmp_path, 25)
    pipe = FakePipe()
    logs = []
    info = ltx_hdr.run(pipe, _params(tmp_path, src, 25), None, log=logs.append)
    assert len(pipe.calls) == 1 and pipe.calls[0]["cond"] == [(src, 1.0)] and pipe.calls[0]["seed"] == 3
    assert (pipe.calls[0]["h"], pipe.calls[0]["w"], pipe.calls[0]["hq"]) == (64, 96, False)
    out = str(tmp_path / "o.mp4")
    pq, exr = H.default_paths(out)
    assert info["frames"] == 25 and info["segments"] == 1 and os.path.exists(out) and os.path.exists(pq)
    assert len(os.listdir(exr)) == 25 and P.has_audio(out) and P.has_audio(pq)
    assert info["peak_linear"] == pytest.approx(2.4, abs=1e-3)
    c = H.probe_color(pq)
    assert c["color_transfer"] == "smpte2084" and c["pix_fmt"] == "yuv420p10le"
    assert H.read_sidecar(out)["frames"] == 25 and H.extra_outputs(out) == [pq, exr]
    assert any("HEVC" in m for m in logs)


def test_long_clip_segments_and_continuity(tmp_path):
    src = _src(tmp_path, 41, audio=False)
    pipe = FakePipe(starts=[sg.start for sg in ltx_hdr.plan(41, 17, 9)[0]])
    info = ltx_hdr.run(pipe, _params(tmp_path, src, 41, hdr_seg_frames=17, hdr_overlap=9, hdr_format="exr",
                                     hdr_audio_from=""), None, log=lambda m: None)
    assert info["segments"] == len(pipe.calls) >= 3 and info["frames"] == 41
    assert all(c["n"] == 17 for c in pipe.calls)
    assert all("hdrseg_" in c["cond"][0][0] for c in pipe.calls), "ogni segmento ha il suo guide tagliato"
    exr = H.default_paths(str(tmp_path / "o.mp4"))[1]
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    vals = [float(cv2.imread(os.path.join(exr, f"frame_{i:05d}.exr"), cv2.IMREAD_UNCHANGED).mean()) for i in range(41)]
    assert all(abs(vals[i] - i / 10.0) < 0.06 for i in range(41)), vals
    assert not os.path.exists(H.default_paths(str(tmp_path / "o.mp4"))[0]), "formato exr: nessun HEVC"


def test_hq_and_bad_format_and_nan(tmp_path):
    src = _src(tmp_path, 17, audio=False)
    pipe = FakePipe()
    ltx_hdr.run(pipe, _params(tmp_path, src, 17, hdr_hq=True, hdr_format="pq", hdr_audio_from=""), None, log=lambda m: None)
    assert pipe.calls[0]["hq"] is True
    with pytest.raises(SystemExit, match="formato"):
        ltx_hdr.run(FakePipe(), _params(tmp_path, src, 17, hdr_format="dpx"), None, log=lambda m: None)
    with pytest.raises(RuntimeError, match="NaN"):
        ltx_hdr.run(FakePipe(bad="nan"), _params(tmp_path, src, 17, hdr_format="pq", output=str(tmp_path / "n.mp4")),
                    None, log=lambda m: None)


def test_plan_matches_core_planner():
    segs, ovs = ltx_hdr.plan(200, 97, 9)
    assert [s.start for s in segs] == [0, 88, 103] and ovs == [9, 82]
    segs, ovs = ltx_hdr.plan(49, 97, 9)
    assert len(segs) == 1 and ovs == []
