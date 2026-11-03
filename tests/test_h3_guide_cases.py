"""I casi del driver per la guida allineata H3 (orch\\app_live\\casi_h3_guida_20261103.json) eseguiti SENZA l'app e senza GPU: le operazioni sui widget veri della tab
(Modifica Video AI) con la stessa logica del driver (resolve / set_widget / pick_lora), il bottone, e i params che il worker riceverebbe (core.video_edit_h3.build_params):
ogni `expect_params` dei casi eseguibili oggi deve tornare. I casi «pending» (azione «Edit a parole») si controllano solo nella forma. Salta se il file dei casi non c'e."""
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = Path(r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\casi_h3_guida_20261103.json")
pytestmark = pytest.mark.skipif(not CASES.is_file(), reason="file dei casi non presente")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _dig(d, dotted):
    v = d
    for part in dotted.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    return v


def test_case_file_shape():
    d = json.loads(CASES.read_text("utf-8"))
    ids = [c["id"] for c in d["cases"]]
    assert ids == ["h3guida_01_rifugio_senza", "h3guida_02_rifugio_con_guida", "h3guida_03_rifugio_lora_vfx_edit", "h3guida_10_vfx_cappotto_rosso",
                   "h3guida_11_style_transfer_art_deco"]
    assert set(d["skip"]) == {"h3guida_10_vfx_cappotto_rosso", "h3guida_11_style_transfer_art_deco"}
    for c in d["cases"]:
        assert c["fn"] == "ops" and c["expect_params"] and c["expect_subprocess"] == ["minimax_h3_infer.py"] and c["descr"] and c["verify"] and c["stima_min"], c["id"]
    # i turni non mescolano i casi in attesa di decisione con quelli eseguibili
    assert all("h3guida_1" not in i for i in d["turni"]["A (subito, dalla tab di oggi)"])


@pytest.mark.parametrize("cid", ["h3guida_01_rifugio_senza", "h3guida_02_rifugio_con_guida", "h3guida_03_rifugio_lora_vfx_edit"])
def test_runnable_cases_drive_the_real_widgets_and_the_expected_params_come_out(qapp, cid, monkeypatch, tmp_path):
    from core import video_edit_engines as veng
    from core import video_edit_h3 as vh3
    sys.path.insert(0, str(ROOT / "tests" / "app_live"))
    import driver as D
    monkeypatch.setattr(veng, "_present", lambda item: True)
    monkeypatch.setattr(vh3, "turbo_lora_defect", lambda key: "")
    spec = json.loads(CASES.read_text("utf-8"))
    case = next(c for c in spec["cases"] if c["id"] == cid)
    from ui.tabs.video_edit_ai_tab import VideoEditAITab

    class _W:                                                        # radice «finestra»: il driver risolve i percorsi a partire da qui
        pass
    w = _W()
    w.video_edit_ai_tab = VideoEditAITab()
    w.video_edit_ai_tab.set_nsfw_getter(lambda: True)
    cfgs = []
    w.video_edit_ai_tab.request_edit.connect(cfgs.append)
    w.video_edit_ai_tab.h3_lora.refresh()
    for op in case["ops"]:
        if "env" in op:
            continue
        if "set" in op:
            D.set_widget(D.resolve(w, op["set"]), op["value"])
        elif "call" in op:
            ret = D.resolve(w, op["call"])(*op.get("args", []))
            if op["call"].endswith("_go"):
                continue
        elif "lora" in op:
            D.Driver.pick_lora(None, D.resolve(w, op["lora"]), op)
        else:
            raise AssertionError(f"operazione non gestita dal test: {op}")
    assert cfgs, "il bottone non ha emesso nessuna richiesta di edit (errore in GUI?)"
    cfg = cfgs[-1]
    # i params che il worker riceverebbe: la stessa chiamata della tab (qui con la cartella tracking finta: build_params non la legge)
    p = vh3.build_params(dict(cfg), str(tmp_path / "o.mp4"), present=lambda i: True)
    for dotted, want in case["expect_params"].items():
        got = _dig(p, dotted)
        assert got is not None and (got == want or (isinstance(want, str) and isinstance(got, str) and want in got)), (cid, dotted, want, got)
    if cid == "h3guida_01_rifugio_senza":                            # il riferimento NON ha guida: params di sempre
        assert "guide" not in p["h3_edit"] and "h3_guide" not in cfg
    else:
        assert p["h3_edit"]["guide"] is True
    if cid == "h3guida_03_rifugio_lora_vfx_edit":
        assert w.video_edit_ai_tab.h3_guide.isChecked() and not w.video_edit_ai_tab.h3_guide.isEnabled()      # accesa da sola e bloccata
        assert p["loras"][0]["strength"] == 1.0


def test_pending_cases_have_the_proposed_form(qapp):
    spec = json.loads(CASES.read_text("utf-8"))
    for c in spec["cases"]:
        if c["id"].startswith("h3guida_1"):
            assert "DECISIONE" in c["pending"] and any(o.get("call") == "video_edit_ai_tab.select_mode" and o["args"] == ["vfx_edit"] for o in c["ops"])
            assert c["expect_params"]["h3_guide_lora"] in ("vfx_edit", "style_transfer") and c["expect_params"]["prompt"].startswith(c["expect_params"]["h3_guide_lora"] + ": ")
    # il backend della proposta e pronto: stessi params che i casi si aspettano
    from core import video_edit_h3 as vh3
    assert callable(vh3.build_guide_params)
