"""Test dei controlli automatici dei giri live (tests/app_live/autocheck.py) e dei casi a scenari (make_scenari.py).

I test «leggeri» girano sempre (immagini sintetiche, nessun modello). I test «sui file veri» usano le uscite dei giri del 03/11 in
orch/app_live e i modelli CLIP / volti gia sul disco (CPU): si saltano se i file mancano, e girano con AC_HEAVY=1 (ogni chiamata carica un modello).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "app_live"))
import autocheck as ac  # noqa: E402
import make_scenari as ms  # noqa: E402

ORCH = ac.ORCH
HEAVY = os.environ.get("AC_HEAVY") == "1"


def _rng_img(seed, size=(256, 256)):
    r = np.random.default_rng(seed)
    base = np.kron(r.integers(40, 220, (16, 16, 3)), np.ones((16, 16, 1))).astype(np.uint8)   # blocchi + rumore fine
    return Image.fromarray(np.clip(base + r.integers(-25, 25, base.shape), 0, 255).astype(np.uint8))


def test_hash_pixel_non_metadati(tmp_path):
    im = _rng_img(1)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    im.save(a)
    from PIL import PngImagePlugin
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "seed 2")
    im.save(b, pnginfo=meta)
    assert a.read_bytes() != b.read_bytes()
    assert ac.file_hash(str(a)) == ac.file_hash(str(b))
    _rng_img(2).save(tmp_path / "c.png")
    assert ac.file_hash(str(a)) != ac.file_hash(str(tmp_path / "c.png"))


def test_unchanged_rosso_se_hash_di_un_altro_giro(tmp_path):
    p = tmp_path / "o.png"
    _rng_img(3).save(p)
    h = ac.file_hash(str(p))
    hist = {"caso_x": [{"hash": h, "run": "run_vecchio"}]}
    r = ac.check_unchanged("caso_x", str(p), "run_nuovo", hist)
    assert r["esito"] == "ROSSO" and "run_vecchio" in r["motivo"]
    assert ac.check_unchanged("caso_x", str(p), "run_vecchio", hist)["esito"] == "ok"        # stesso giro: non e un «invariato»
    assert ac.check_unchanged("altro", str(p), "run_nuovo", hist)["esito"] == "ok"


def test_pasted_incollata_vs_rigenerata(tmp_path):
    comp = _rng_img(4)
    mask = np.zeros((256, 256), np.uint8)
    mask[40:220, 60:200] = 255
    Image.fromarray(mask).save(tmp_path / "m.png")
    comp.save(tmp_path / "comp.png")
    # incollata: stessi dettagli, solo tonalita cambiate (+25 sul canale rosso e contrasto)
    a = np.asarray(comp).astype(np.float32)
    toned = np.clip(a * 1.15 + np.array([25, 0, -10]), 0, 255).astype(np.uint8)
    Image.fromarray(toned).save(tmp_path / "toned.png")
    m1 = ac.pasted_measures(str(tmp_path / "toned.png"), str(tmp_path / "comp.png"), str(tmp_path / "m.png"))
    assert m1["hp_ncc"] >= ac.THRESH["integra_hp_ncc"], m1
    assert m1["diff_media"] > 15 and m1["quota_diff_lt8"] < 0.2           # la sola differenza media NON basta a vederla incollata
    # rigenerata: dettagli diversi
    _rng_img(99).save(tmp_path / "regen.png")
    m2 = ac.pasted_measures(str(tmp_path / "regen.png"), str(tmp_path / "comp.png"), str(tmp_path / "m.png"))
    assert m2["hp_ncc"] < ac.THRESH["integra_hp_ncc"], m2


def test_apply_mette_rosso_e_baseline_no(tmp_path, monkeypatch):
    p = tmp_path / "o.png"
    _rng_img(5).save(p)
    hist = {"c1": [{"hash": ac.file_hash(str(p)), "run": "vecchio"}], "c2": [{"hash": ac.file_hash(str(p)), "run": "vecchio"}]}
    r1 = {"id": "c1", "esito": "ok", "uscite": [str(p)]}
    ac.apply({"id": "c1"}, r1, "nuovo", hist)
    assert r1["esito"] == "rosso" and r1["motivo"].startswith("[controllo automatico]")
    r2 = {"id": "c2", "esito": "ok", "uscite": [str(p)]}
    ac.apply({"id": "c2", "baseline": True}, r2, "nuovo", hist)
    assert r2["esito"] == "ok" and r2["autocheck"][0]["esito"] == "ROSSO(baseline)"
    r3 = {"id": "c3", "esito": "errore", "uscite": [str(p)]}
    ac.apply({"id": "c3"}, r3, "nuovo", hist)
    assert "autocheck" not in r3                                           # i casi gia falliti non si controllano
    r4 = {"id": "c4", "esito": "ok", "dry": True, "uscite": [str(p)]}
    ac.apply({"id": "c4"}, r4, "nuovo", hist)
    assert "autocheck" not in r4                                           # le prove a secco usano immagini segnaposto


def test_controllo_che_non_riesce_a_girare_non_blocca(tmp_path, monkeypatch):
    p = tmp_path / "o.png"
    _rng_img(6).save(p)
    monkeypatch.setattr(ac, "clip_scores", lambda items: (_ for _ in ()).throw(RuntimeError("venv assente")))
    Image.fromarray(np.full((256, 256), 255, np.uint8)).save(tmp_path / "mask.png")
    case = {"id": "i1", "kind": "inpaint", "autocheck": {"kind": "inpaint", "oggetto": "a compass"}}
    (tmp_path / "x_frame1_mask.png").write_bytes((tmp_path / "mask.png").read_bytes())
    r = {"id": "i1", "esito": "ok", "uscite": [str(p)]}
    ac.apply(case, r, "nuovo", {})
    assert r["esito"] == "ok" and any(x["esito"] == "n/d" for x in r["autocheck"])


# ── scenari ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_scenari_completi_e_diversi():
    assert len(ms.SCENARI) >= 5
    for sc, d in ms.SCENARI.items():
        assert {"scena", "sfondo", "soggetti", "oggetto", "ellisse", "msr"} <= set(d), sc
        assert len(d["soggetti"]) >= 3 and "A" in d["soggetti"] and "B" in d["soggetti"]
        assert "adult" in d["scena"].lower() and "nobody" in d["sfondo"].lower()
        for k, v in d["soggetti"].items():
            assert "adult" in v, (sc, k)
    assert len({d["oggetto"] for d in ms.SCENARI.values()}) == len(ms.SCENARI)             # un oggetto diverso per scenario
    for bad in ("nude", "naked", "topless", "lingerie"):
        assert not any(bad in json.dumps(d).lower() for d in ms.SCENARI.values()), bad     # tutto SFW


def test_casi_fixture_e_esperimenti():
    fx = ms.casi_fixture()
    assert len(fx["cases"]) == 5 * len(ms.SCENARI)
    assert all(c["fn"] == "create_image" and c["fixture_dest"].startswith(ms.FIX) for c in fx["cases"])
    assert len({c["seed"] for c in fx["cases"]}) == len(fx["cases"])
    for spec in (ms.esperimenti_integra(), ms.esperimenti_inpaint(), ms.esperimenti_msr(), ms.esperimenti_krea2()):
        assert 1 <= len(spec["cases"]) <= 12                                                # al piu 12 job per turno
        assert ms.verifica_rotazione(spec) == [], spec["nome"]
        assert len({c["id"] for c in spec["cases"]}) == len(spec["cases"])
        for c in spec["cases"]:
            assert c["scenario"] in ms.SCENARI and c.get("autocheck"), c["id"]
    integ = ms.esperimenti_integra()["cases"]
    assert any(c.get("baseline") for c in integ)                                            # il confronto col default c'e
    assert any(c.get("approach") == "armonizza" and c["expect_params"] == {"mask_ref": False} for c in integ)   # e il backend deve riceverlo


# ── sui file veri del 03/11 (CPU, modelli gia sul disco) ──────────────────────────────────────────────────────────────
def _res(run, cid):
    p = os.path.join(ORCH, run, "results.jsonl")
    if not os.path.isfile(p):
        pytest.skip("giro non presente")
    for ln in open(p, encoding="utf-8"):
        d = json.loads(ln)
        if d["id"] == cid:
            return d
    pytest.skip("caso non presente")


@pytest.mark.skipif(not HEAVY, reason="AC_HEAVY=1: carica CLIP/ArcFace da disco (CPU)")
def test_file_veri_inpaint_clip():
    ok = _res("run_commenti_A_0311", "modifica_inpaint_Flux_00")           # Fill: compare la bussola
    bad = _res("run_t9b_1103_151127", "modifica_inpaint_Flux_04")          # FLUX base: «una chiazza»
    mk = lambda r: next(iter(__import__("glob").glob(os.path.dirname(r["uscite"][0]) + "/*_frame1_mask.png")))   # noqa: E731
    assert ac.check_inpaint(ok["uscite"][0], mk(ok), "brass compass")[0]["esito"] == "ok"
    assert ac.check_inpaint(bad["uscite"][0], mk(bad), "brass compass")[0]["esito"] == "ROSSO"


@pytest.mark.skipif(not HEAVY, reason="AC_HEAVY=1")
def test_file_veri_integra_identita_e_incollata():
    ref = os.path.join(ORCH, "fixtures", "ritratto_960x1024.png")
    qw = _res("run_t9b_1103_151127", "modifica_integra_qwen-image-2-1_00")
    kx = _res("run_t9b_1103_151127", "modifica_integra_kontext-flux-1-_00")
    for r, atteso in ((qw, "integra_incollata"), (kx, "integra_identita")):
        f = os.path.dirname(r["uscite"][0])
        comp = next(iter(__import__("glob").glob(f + "/*integrate_input.png")))
        pos = next(iter(__import__("glob").glob(f + "/posizione*.png")))
        rossi = [x["nome"] for x in ac.check_integra(r["uscite"][0], comp, pos, ref) if x["esito"] == "ROSSO"]
        assert atteso in rossi, (r["id"], rossi)
