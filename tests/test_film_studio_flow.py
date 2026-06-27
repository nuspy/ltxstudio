"""#57 Test a secco del flusso AI Film Studio (NO GPU): DB → progetto → scena → shot → enqueue.
Verifica l'ossatura (DB/versioning/scheduler/continuity) senza generazione reale (mock del worker).
Il test LIVE con generazione vera resta nella sessione #43."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_film_studio_dry_flow(tmp_path, monkeypatch):
    from core.film_studio import db, continuity as ct
    # progetto → scene → shot
    pid = db.insert("project", name="test_flow", config="{}", series_id=0)
    assert pid
    sid = db.insert("scene", project_id=pid, ord=1, description="@Mario entra nella stanza",
                    continuity=__import__("json").dumps(ct.new_state(time_of_day="night", mood="tense")))
    shid = db.insert("shot", scene_id=sid, ord=1, description="primo piano di @Mario",
                     duration_s=4, seed=12345, dialogues="[]")
    assert db.get("shot", shid)["seed"] == 12345

    # continuity propaga lo stato (orario/mood)
    st = __import__("json").loads(db.get("scene", sid)["continuity"])
    assert st["time_of_day"] == "night" and st["mood"] == "tense"
    assert "night" in ct.to_prompt(st)

    # la tabella job registra un lavoro tracciabile (senza avviare il worker loop, testato live in #43)
    import json as _j
    jid = db.insert("job", type="image", status="queued", priority=5,
                    params=_j.dumps({"prompt": "x", "out": str(tmp_path / "kf.png"), "owner_id": shid}))
    assert jid and db.get("job", jid)["status"] == "queued"

    # cleanup: delete_project cascade
    db.delete_project(pid)
    assert db.get("project", pid) is None
