"""Backend test: copertura dialoghi shot-reverse-shot (core/film_studio/coverage.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.film_studio import db, coverage


def _mk_scene():
    pid = db.insert("project", name="covtest", kind="short", duration_s=30, config="{}", updated=0)
    sid = db.insert("scene", project_id=pid, ord=1, description="bar, notte", continuity="{}", assets="")
    return pid, sid


def test_plan_coverage_alterna_angoli():
    dl = [{"actor": "@Marco", "text": "Dove sei stato?"},
          {"actor": "@Lena", "text": "Non sono affari tuoi."},
          {"actor": "@Marco", "text": "Invece sì."}]
    units = coverage.plan_coverage(dl, 6.0)
    assert len(units) == 3
    assert units[0]["speaker"] == "Marco" and units[1]["speaker"] == "Lena"
    # Marco e Lena hanno angoli diversi (campo/controcampo); Marco (unità 0 e 2) stesso angolo
    assert units[0]["camera"] != units[1]["camera"]
    assert units[0]["camera"] == units[2]["camera"]
    assert abs(sum(u["dur"] for u in units) - 6.0) < 0.1


def test_expand_crea_shot_e_sposta_i_successivi():
    pid, sid = _mk_scene()
    dl = [{"actor": "@Marco", "text": "A"}, {"actor": "@Lena", "text": "B"}, {"actor": "@Marco", "text": "C"}]
    sh = db.insert("shot", scene_id=sid, ord=1, description="due al bancone",
                   dialogues=json.dumps(dl, ensure_ascii=False), duration_s=6, keyframes="[]", refs="",
                   continuity="{}", camera="")
    tail = db.insert("shot", scene_id=sid, ord=2, description="uscita", dialogues="[]", duration_s=3,
                     keyframes="[]", refs="", continuity="{}", camera="")
    ids = coverage.expand_shot_to_coverage(sh)
    assert len(ids) == 3
    shots = db.query("shot", "scene_id=?", (sid,), order="ord")
    # 3 di copertura + la coda spostata a ord=4
    assert [s["ord"] for s in shots] == [1, 2, 3, 4]
    cov = [s for s in shots if s["ord"] in (1, 2, 3)]
    for s in cov:                                            # ogni copertura ha ESATTAMENTE 1 battuta
        assert len(json.loads(s["dialogues"])) == 1 and s["camera"]
    assert db.get("shot", tail)["ord"] == 4                  # la coda è stata spostata
    # cleanup
    db.delete_project(pid)


def test_wardrobe_prompts():
    ps = coverage.wardrobe_prompts("Marco, 30s detective", "trench coat insanguinato", n=6)
    assert len(ps) == 6
    for angle_id, prompt in ps:
        assert "trench coat insanguinato" in prompt and "Marco" in prompt
    assert ps[0][0] != ps[1][0]                              # angoli distinti
    assert coverage.wardrobe_prompts("x", "y", n=3) and len(coverage.wardrobe_prompts("x", "y", n=3)) == 3


def test_turnaround_prompt_mantiene_angolo():
    p = coverage.turnaround_prompt("Marco detective", "left")
    assert "Marco detective" in p and "left profile" in p
    assert coverage.turnaround_prompt("x", "back").count("back view") == 1


def test_accept_immagine_db():
    pid = db.insert("project", name="acc", kind="short", config="{}", updated=0)
    aid = db.insert("asset", type="actor", name="A", description="x", fields="{}", tags="", scope="project", series_id=0)
    i1 = db.insert("image", owner_kind="asset", owner_id=aid, angle="front", repr="front", file="a.png", seed=1, prompt="", version=1)
    db.insert("image", owner_kind="asset", owner_id=aid, angle="back", repr="back", file="b.png", seed=1, prompt="", version=1)
    db.update("image", i1, accepted=1)                        # curatela: accetta la 1ª
    accepted = [r for r in db.query("image", "owner_kind='asset' AND owner_id=?", (aid,)) if r.get("accepted")]
    assert len(accepted) == 1 and accepted[0]["angle"] == "front"
    db.delete("asset", aid); db.delete_project(pid)


def test_expand_noop_su_shot_singola_battuta():
    pid, sid = _mk_scene()
    sh = db.insert("shot", scene_id=sid, ord=1, description="x",
                   dialogues=json.dumps([{"actor": "@Marco", "text": "solo"}]), duration_s=4,
                   keyframes="[]", refs="", continuity="{}", camera="")
    assert coverage.expand_shot_to_coverage(sh) == []       # 1 battuta → niente copertura
    db.delete_project(pid)
