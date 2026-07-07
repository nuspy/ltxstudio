"""Backend test: reference-chaining visivo (estrazione ultimo frame + keyframe di raccordo)."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.film_studio import reference_chain as rc


def _mk_clip(path, color, dur=1.0):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c={color}:size=64x64:rate=25:duration={dur}", "-pix_fmt", "yuv420p", path, "-y"],
                   check=False)


def test_last_frame_e_link(tmp_path):
    clip = str(tmp_path / "shotN.mp4")
    _mk_clip(clip, "red")
    out = str(tmp_path / "link.png")
    kf = rc.prev_shot_link(clip, out, strength=0.4)
    assert kf and len(kf) == 1
    assert kf[0]["frame"] == 0 and abs(kf[0]["strength"] - 0.4) < 1e-6
    assert os.path.exists(kf[0]["path"])                     # il frame è stato estratto


def test_none_su_video_assente(tmp_path):
    assert rc.last_frame(str(tmp_path / "nope.mp4"), str(tmp_path / "x.png")) is None
    assert rc.prev_shot_link("", str(tmp_path / "x.png")) is None
