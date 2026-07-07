"""Backend test: storia dei take non-distruttiva (core/film_studio/takes.py)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.film_studio import db, takes


def _canon(tmp, name):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        f.write("v")
    return p


def test_record_e_restore(tmp_path):
    pid = db.insert("project", name="taketest", kind="short", config="{}", updated=0)
    sid = db.insert("scene", project_id=pid, ord=1, description="x", continuity="{}", assets="")
    sh = db.insert("shot", scene_id=sid, ord=1, description="y", dialogues="[]", keyframes="[]",
                   refs="", continuity="{}", camera="")
    canon = _canon(str(tmp_path), "shot_kf.png")

    # tre generazioni successive → tre take versionati, il canonico resta uno solo
    for content in ("A", "B", "C"):
        Path(canon).write_text(content)
        t = takes.record_take("shot", sh, "keyframe", canon)
        assert t and os.path.exists(t)
    tk = takes.list_takes("shot", sh, "keyframe")
    assert [x["version"] for x in tk] == [1, 2, 3]
    assert Path(tk[0]["file"]).read_text() == "A" and Path(tk[2]["file"]).read_text() == "C"

    # ora il canonico ha "C"; ripristino il take 1 ("A") come corrente
    assert Path(canon).read_text() == "C"
    takes.restore_take("shot", sh, "keyframe", 1, canon)
    assert Path(canon).read_text() == "A"                    # ripristino non-distruttivo: i take restano
    assert len(takes.list_takes("shot", sh, "keyframe")) == 3

    db.delete_project(pid)


def test_record_none_se_file_assente():
    assert takes.record_take("shot", 99999, "keyframe", "/non/esiste.png") is None
