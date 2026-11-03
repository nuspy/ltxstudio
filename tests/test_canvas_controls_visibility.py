"""Canvas (utente, 03/11 sera): le due maschere e i loro controlli compaiono SOLO con i motori di Integra che le usano (nascosti, non disattivati);
i cursori di trasparenza stanno a 5 px l'uno dall'altro; «Scontorna» / «Sfondo» solo dove i layer importati vengono composti sulla scena.
Widget veri di Frame1Tab/MaskCanvas, offscreen, nessuna GPU."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PILImage = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(qapp, tmp_path):
    from PIL import Image, ImageDraw
    Image.new("RGB", (800, 600), (90, 120, 150)).save(tmp_path / "scena.png")
    subj = Image.new("RGBA", (200, 300), (0, 0, 0, 0))
    ImageDraw.Draw(subj).ellipse((10, 10, 190, 290), fill=(200, 140, 120, 255))
    subj.save(tmp_path / "soggetto.png")
    from ui.tabs.frame1_tab import Frame1Tab
    t = Frame1Tab()
    t.canvas.set_aspect(800, 600)
    t.canvas.load_image(str(tmp_path / "scena.png"), "canvas")
    t.sub.setCurrentIndex(t._inpaint_tab_index)                  # pagina Modifica = modalita Integra
    t._tmp = tmp_path
    return t


def test_every_integrate_engine_is_in_the_caps_table_with_both_flags():
    from ui.tabs.frame1_tab import INTEGRATE_ENGINE_CAPS, integrate_engine_composes_layers, integrate_engine_supports_context_mask
    assert set(INTEGRATE_ENGINE_CAPS) == {"kontext", "flux2", "krea2", "qwen_image21", "flux3"}
    assert [e for e, c in INTEGRATE_ENGINE_CAPS.items() if not c["context_mask"]] == ["krea2"]
    assert all(c["composes_layers"] for c in INTEGRATE_ENGINE_CAPS.values())
    assert integrate_engine_composes_layers("motore_futuro") and integrate_engine_supports_context_mask("motore_futuro")


def test_engine_selector_in_the_gui_maps_to_the_caps_table(tab):
    """Il combo motore del pannello (indici 0-4) arriva davvero alle righe della tabella: nessuno scambio di indici."""
    names = {0: "kontext", 1: "flux2", 2: "krea2", 3: "qwen_image21", 4: "flux3"}
    for idx, name in names.items():
        tab.integrate_engine.setCurrentIndex(idx)
        assert tab._engine_val(tab.integrate_engine.currentIndex()) == name
        assert tab.canvas.integrate_ctx_enabled == (name != "krea2"), name


def test_manual_mode_rows_hidden_with_krea_and_back_with_other_engines(tab):
    assert tab.canvas.add_layer(str(tab._tmp / "soggetto.png"))
    tab.integrate_engine.setCurrentIndex(0)
    tab.integrate_ctx_mode.setCurrentIndex(tab.integrate_ctx_mode.findData("manual"))
    assert not tab._ctx_manual_row.isHidden() and tab._ctx_pct_row.isHidden()
    assert not tab.canvas._draw_target_cb.isHidden()
    tab.integrate_engine.setCurrentIndex(2)                                           # Krea 2 Edit: tutto nascosto, anche il pennello contesto
    assert tab._ctx_manual_row.isHidden() and tab._ctx_pct_row.isHidden() and tab.integrate_ctx_mode.isHidden()
    assert tab.canvas._draw_target_cb.isHidden() and tab.canvas.draw_target_lbl.isHidden() and tab.canvas.draw_banner.isHidden()
    tab.integrate_engine.setCurrentIndex(3)                                           # Qwen: tornano, nello stato "manuale" di prima
    assert not tab._ctx_manual_row.isHidden() and tab._ctx_pct_row.isHidden()
    assert not tab.canvas._draw_target_cb.isHidden() and not tab.canvas.draw_banner.isHidden()


def test_transparency_sliders_are_five_pixels_apart_and_tight(qapp, tab):
    assert tab.canvas.add_layer(str(tab._tmp / "soggetto.png"))
    from ui.mask_canvas import OVL_TRANSP_GAP
    assert OVL_TRANSP_GAP == 5
    cv = tab.canvas
    cv.resize(1100, 700); cv.show(); qapp.processEvents()
    a, b = cv.ovl_transp_pos_sl, cv.ovl_transp_ctx_sl
    ga = a.mapTo(cv, a.rect().topLeft()); gb = b.mapTo(cv, b.rect().topLeft())
    gap = gb.x() - (ga.x() + a.width())
    assert gap == 5, f"distanza fra i cursori {gap} px"
    assert a.width() <= 18 and b.width() <= 18                                      # niente colonne larghe attorno al cursore
    lay = cv.ovl_transp_box.layout()
    assert lay.contentsMargins().left() == 0 and lay.contentsMargins().right() == 0
    assert abs(a.y() - b.y()) <= 1                                                  # affiancati, stessa altezza
    cv.hide()


def test_transparency_value_labels_follow_the_sliders_and_keep_a_letter(tab):
    cv = tab.canvas
    cv.ovl_transp_pos_sl.setValue(95); cv.ovl_transp_ctx_sl.setValue(5)
    assert cv.ovl_transp_val["position"].text().endswith(" 95%") and cv.ovl_transp_val["context"].text().endswith(" 5%")
    assert cv.ovl_transp_val["position"].text()[0] != cv.ovl_transp_val["context"].text()[0]       # il colore non e l'unico veicolo d'informazione
    assert abs(cv._ovl_pos.opacity() - 0.05) < 1e-6 and abs(cv._ovl_ctx.opacity() - 0.95) < 1e-6


def test_cutout_buttons_follow_layers_page_and_engine(tab):
    cv = tab.canvas
    b_cut, b_rst = cv._cutout_btns
    tab.sub.setCurrentIndex(tab._inpaint_tab_index)
    assert b_cut.isHidden() and b_rst.isHidden()                                    # nessun layer importato: niente da scontornare
    assert cv.add_layer(str(tab._tmp / "soggetto.png"))
    assert not b_cut.isHidden() and not b_rst.isHidden()                            # layer + pagina Inpaint/Integra
    for idx in range(5):                                                            # tutti i motori di Integra compongono i layer
        tab.integrate_engine.setCurrentIndex(idx)
        assert not b_cut.isHidden() and not b_rst.isHidden(), idx
    for page in range(tab.sub.count()):                                             # Genera / Colore / Upscale: lavorano sul file della foto di base
        tab.sub.setCurrentIndex(page)
        want = page == tab._inpaint_tab_index
        assert b_cut.isHidden() != want and b_rst.isHidden() != want, page
    tab.sub.setCurrentIndex(tab._inpaint_tab_index)
    assert not b_cut.isHidden()
    cv._drop_layers()
    assert b_cut.isHidden() and b_rst.isHidden()


def test_cutout_buttons_hidden_for_an_engine_that_does_not_compose_layers(tab, monkeypatch):
    from ui.tabs import frame1_tab as F
    monkeypatch.setitem(F.INTEGRATE_ENGINE_CAPS, "krea2", {"context_mask": False, "composes_layers": False})
    cv = tab.canvas
    assert cv.add_layer(str(tab._tmp / "soggetto.png"))
    tab.sub.setCurrentIndex(tab._inpaint_tab_index)
    tab.integrate_engine.setCurrentIndex(0)
    assert not cv._cutout_btns[0].isHidden()
    tab.integrate_engine.setCurrentIndex(2)
    assert cv._cutout_btns[0].isHidden() and cv._cutout_btns[1].isHidden()


def test_driver_visibility_helper_sees_hidden_rows_and_canvas_controls(tab):
    assert tab.canvas.add_layer(str(tab._tmp / "soggetto.png"))
    """Lo stesso criterio che usano i casi dal vivo (tests/app_live/driver.widget_hidden): righe nascoste = widget nascosti anche se annidati."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app_live"))
    from driver import widget_hidden
    tab.integrate_engine.setCurrentIndex(0)
    assert not widget_hidden(tab.integrate_ctx_spin) and not widget_hidden(tab.canvas.draw_banner) and not widget_hidden(tab.canvas.ovl_transp_pos_sl)
    tab.integrate_engine.setCurrentIndex(2)
    for w in (tab.integrate_ctx_mode, tab.integrate_ctx_spin, tab.integrate_ctx_slider, tab.canvas.draw_banner, tab.canvas.ovl_legend,
              tab.canvas.ovl_transp_pos_sl, tab.canvas.ovl_transp_ctx_sl):
        assert widget_hidden(w), w


def test_masks_are_visible_only_in_integrate_mode_with_a_supporting_engine_and_layers(tab):
    """Richiesta utente 03/11 sera: le due maschere SOLO nella modalita Integra (pagina Modifica) con un motore che le usa; senza layer da integrare niente."""
    cv = tab.canvas
    shown = lambda: [not w.isHidden() for w in (cv.draw_banner, cv.ovl_legend, cv.ovl_transp_box)]
    tab.integrate_engine.setCurrentIndex(0)
    assert shown() == [False, False, False] and not cv.masks_visible()                 # pagina Modifica, motore ok, ma nessun layer
    assert cv.add_layer(str(tab._tmp / "soggetto.png"))
    assert shown() == [True, True, True] and cv.masks_visible()
    for page in range(tab.sub.count()):                                                 # fuori dalla pagina Modifica: nascoste anche con layer e motore ok
        tab.sub.setCurrentIndex(page)
        assert cv.masks_visible() == (page == tab._inpaint_tab_index), page
        assert shown() == [page == tab._inpaint_tab_index] * 3, page
    tab.sub.setCurrentIndex(tab._inpaint_tab_index)
    assert shown() == [True, True, True]
    tab.integrate_engine.setCurrentIndex(2)                                             # Krea 2 Edit: nascoste anche in Integra con layer
    assert shown() == [False, False, False] and cv._ovl_pos.pixmap().isNull() and cv._ovl_ctx.pixmap().isNull()
    tab.integrate_engine.setCurrentIndex(3)
    assert shown() == [True, True, True]
    cv._drop_layers()                                                                   # tolti i layer: niente da integrare
    assert shown() == [False, False, False]
