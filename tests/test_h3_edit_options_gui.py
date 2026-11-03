"""Modifica Video, motore H3: i widget delle opzioni di velocita e modello (widget VERI della tab) arrivano al cfg dell'edit e ai params del worker.
Default invariati; i pesi non sul disco offrono il download (mai in silenzio) e la scelta torna a quella di prima se si rifiuta; i widget sono visibili SOLO col motore H3."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(qapp, tmp_path, monkeypatch):
    from core import video_edit_engines as veng
    monkeypatch.setattr(veng, "_present", lambda item: item not in ("mmh3-ref2va-int8", "mmh3-ref2va-fp8"))   # come il disco di oggi: nvfp4/int4 + DiT personalizzati
    from core import video_edit_h3 as vh3
    monkeypatch.setattr(vh3, "turbo_lora_defect", lambda key: "")             # il difetto della turbo 4 passi sul disco di oggi ha il suo test
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    t = VideoEditAITab()
    t.set_nsfw_getter(lambda: True)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    t._video = str(clip)
    t.select_mode("change_background")
    t.obj.setText("person")
    t.prompt.setPlainText("a mountain refuge at night")
    t._cfgs = []
    t.request_edit.connect(t._cfgs.append)
    return t


def _go(t):
    t._cfgs.clear()
    t._go()
    return t._cfgs[-1] if t._cfgs else None


def test_options_are_visible_only_with_the_h3_engine_and_defaults_are_unchanged(tab):
    assert tab.engine.currentData() == "h3_edit" and tab.h3_opts_box.isVisibleTo(tab) and tab.h3_profile.isVisibleTo(tab)
    cfg = _go(tab)
    assert cfg["h3_turbo"] == "v4" and "h3_veda" not in cfg and "h3_lightvae" not in cfg and "loras" not in cfg      # default = oggi
    assert cfg["h3_profile"] == "nvfp4" and not tab.h3_veda.isChecked() and not tab.h3_lightvae.isChecked()
    assert tab.h3_turbo_info.text().startswith("8") and "12" in tab.h3_turbo_info.text()                          # v4: 8 passi, shift 12
    tab.engine.setCurrentIndex(tab.engine.findData("ltx_inpaint")) if tab.engine.findData("ltx_inpaint") >= 0 else None
    if tab.engine.currentData() != "h3_edit":
        assert not tab.h3_opts_box.isVisibleTo(tab) and not tab.h3_profile.isVisibleTo(tab)


def test_model_menu_lists_profiles_and_custom_dits_with_missing_ones_marked(qapp, tab, monkeypatch):
    from core import video_edit_engines as veng
    monkeypatch.setattr(veng, "_present", lambda item: item not in ("mmh3-ref2va-int8", "mmh3-ref2va-fp8"))
    tab._fill_h3_models()
    ids = [tab.h3_profile.itemData(i) for i in range(tab.h3_profile.count())]
    assert ids == ["int8", "fp8", "nvfp4", "int4", "mmh3-dit-pinkcherry18", "mmh3-dit-10erosmax-b6", "mmh3-dit-dasiwa-v4", "mmh3-dit-rockerboo-nvfp4"]
    assert "non installato" in tab.h3_profile.itemText(0).lower() and "non installato" not in tab.h3_profile.itemText(2).lower()
    assert tab.h3_profile.currentData() == "nvfp4"                                       # come prima: il primo profilo installato


def test_picking_a_missing_model_asks_for_the_download_and_reverts_if_refused(tab, monkeypatch):
    from core import video_edit_engines as veng
    monkeypatch.setattr(veng, "_present", lambda item: item != "mmh3-ref2va-int8")
    tab._fill_h3_models()
    asked = []
    monkeypatch.setattr(tab, "_ensure_ids", lambda ids: (asked.append(list(ids)), False)[1])         # l'utente rifiuta il dialog
    tab.h3_profile.setCurrentIndex(tab.h3_profile.findData("nvfp4"))
    tab._h3_model_prev = "nvfp4"
    tab.h3_profile.setCurrentIndex(tab.h3_profile.findData("int8")); tab._on_h3_model_picked(0)
    assert asked == [["mmh3-ref2va-int8"]]                                                       # mai in silenzio: il dialog dei pesi con l'ID del catalogo
    assert tab.h3_profile.currentData() == "nvfp4"                                               # rifiutato: resta la scelta di prima


def test_turbo_veda_lightvae_lora_and_model_reach_the_edit_cfg_and_the_worker_params(tab):
    from core import video_edit_h3 as vh3
    from core import minimax_h3_spec as S
    tab.h3_turbo.setCurrentIndex(tab.h3_turbo.findData("4"))
    assert "4 passi" in tab.h3_turbo_info.text() and "12" in tab.h3_turbo_info.text()
    tab.h3_veda.setChecked(True)
    tab.h3_lightvae.setChecked(True)
    tab.h3_profile.setCurrentIndex(tab.h3_profile.findData("mmh3-dit-dasiwa-v4"))
    ok_rows = [(n, r) for n, r in tab.h3_lora._rows.items() if r["cb"].isEnabled()]
    assert ok_rows, "nessuna LoRA H3 selezionabile (catalogo vuoto?)"
    name, row = ok_rows[0]
    row["cb"].setChecked(True); row["sp"].setValue(0.7)
    cfg = _go(tab)
    assert cfg is not None, "il lancio e stato bloccato: " + tab.log.toPlainText()[-300:]
    assert cfg["h3_turbo"] == "4" and cfg["h3_veda"] is True and cfg["h3_lightvae"] == "dec" and cfg["h3_profile"] == "mmh3-dit-dasiwa-v4"
    assert cfg["loras"] == [{"name": row["entry"].comfy_name, "strength": 0.7}]
    p = vh3.build_params(dict(cfg), "o.mp4", present=lambda i: True)                                 # = cio che arriva al worker (main_window._run_h3)
    assert p["h3_turbo"] == "4" and p["h3_veda"] and p["h3_lightvae"] == "dec" and p["h3_dit"] == "mmh3-dit-dasiwa-v4" and p["loras"] == cfg["loras"]
    ids = S.weight_ids({**p, "references": [{"kind": "image"}]})
    assert ids["dit"] == "mmh3-dit-dasiwa-v4" and ids["turbo"] == "mmh3-turbo-lora-ref2v" and ids["veda"] == S.VEDA_R2VA and "vae_dec_lite" in ids


def test_turbo_without_ref2va_version_is_listed_but_disabled_with_the_reason(tab):
    en = {tab.h3_turbo.itemData(i): tab.h3_turbo.model().item(i).isEnabled() for i in range(tab.h3_turbo.count())}
    assert en == {"v4": True, "4": True, "r4v10": True, "r8v11": True, "8": False, "dmad": False}
    i8 = tab.h3_turbo.findData("8")
    assert "FL2VA" in tab.h3_turbo.model().item(i8).toolTip() or tab.h3_turbo.model().item(i8).toolTip()
    assert tab.h3_turbo.itemData(0) == "v4"                                                          # la predefinita in testa


def test_turbo_fused_dit_disables_the_turbo_menu_and_shows_its_own_steps(tab):
    tab.h3_profile.setCurrentIndex(tab.h3_profile.findData("mmh3-dit-10erosmax-b6")); tab._sync_h3_turbo_info()
    assert not tab.h3_turbo.isEnabled() and "8" in tab.h3_turbo_info.text() and "10" in tab.h3_turbo_info.text()


def test_lora_rows_are_checked_by_header_keys_and_fl2va_only_ones_are_disabled(tab):
    rows = tab.h3_lora._rows
    assert rows and all(r["entry"].path.lower().endswith(".safetensors") for r in rows.values())
    from core import video_edit_h3 as vh3
    for n, r in rows.items():
        ok, _ = vh3.lora_edit_check(r["entry"].path, r["entry"].comfy_name)
        assert r["cb"].isEnabled() == ok, n
    off = [n for n, r in rows.items() if not r["cb"].isEnabled()]
    assert all("⛔" in rows[n]["cb"].text() and rows[n]["cb"].toolTip() for n in off)                  # disattivate con il motivo


def test_veda_checkbox_asks_for_the_download_when_the_predictor_is_missing(tab, monkeypatch):
    from core import video_edit_engines as veng
    monkeypatch.setattr(veng, "_present", lambda item: item != "mmh3-veda-r2va")
    asked = []
    monkeypatch.setattr(tab, "_ensure_ids", lambda ids: (asked.append(list(ids)), False)[1])
    tab.h3_veda.setChecked(True)
    assert asked == [["mmh3-veda-r2va"]] and not tab.h3_veda.isChecked()
    tab.h3_lightvae.setChecked(True)
    assert asked[-1] == ["mmh3-lightvae-dec"] or tab.h3_lightvae.isChecked()


def test_defective_turbo_on_disk_is_disabled_in_the_menu_with_the_reason(qapp, monkeypatch):
    from core import video_edit_engines as veng
    from core import video_edit_h3 as vh3
    monkeypatch.setattr(veng, "_present", lambda item: True)
    monkeypatch.setattr(vh3, "turbo_lora_defect", lambda key: "adaln" if key == "4" else "")
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    t = VideoEditAITab()
    en = {t.h3_turbo.itemData(i): t.h3_turbo.model().item(i).isEnabled() for i in range(t.h3_turbo.count())}
    assert en == {"v4": True, "4": False, "r4v10": True, "r8v11": True, "8": False, "dmad": False}
    assert "AdaLN" in t.h3_turbo.model().item(t.h3_turbo.findData("4")).toolTip()


def test_text_encoder_menu_defaults_to_the_profile_and_reaches_the_cfg(tab):
    assert tab.h3_te.currentData() == "" and "h3_te" not in _go(tab)
    tab.h3_te.setCurrentIndex(tab.h3_te.findData("mmh3-te-nvfp4"))
    assert _go(tab)["h3_te"] == "mmh3-te-nvfp4"
