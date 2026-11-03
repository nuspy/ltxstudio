"""Integra «Zona di contesto» (utente, 03/11): selettore Automatica / Manuale, margine libero oltre il 60 %, due maschere (posizione rossa,
contesto gialla) sempre visibili con trasparenza propria, scritta «Stai disegnando». I test usano i widget veri di Frame1Tab e leggono la maschera
che export_integrate (cioe quella inviata al backend) scrive su disco."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
np = pytest.importorskip("numpy")
PILImage = pytest.importorskip("PIL.Image")


def _rel_lum(hexcol: str) -> float:
    c = [int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(qapp, tmp_path):
    """Frame1Tab vero con una scena sintetica e un soggetto con trasparenza (niente u2net: la sagoma viene dall'alpha)."""
    from PIL import Image, ImageDraw
    scene = Image.new("RGB", (1024, 768), (90, 120, 150))
    scene.save(tmp_path / "scena.png")
    subj = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    ImageDraw.Draw(subj).ellipse((20, 20, 280, 380), fill=(200, 140, 120, 255))
    subj.save(tmp_path / "soggetto.png")
    from ui.tabs.frame1_tab import Frame1Tab
    t = Frame1Tab()
    t.canvas.set_aspect(1024, 768)
    t.canvas.load_image(str(tmp_path / "scena.png"), "canvas")
    assert t.canvas.add_layer(str(tmp_path / "soggetto.png"))
    t.canvas.refresh_integrate_overlay(force=True)
    t._tmp = tmp_path
    return t


def _export(t, name):
    d = t._tmp / name
    r = t.canvas.export_integrate(d)
    ctx = np.asarray(PILImage.open(r["mask"]).convert("L")) > 127
    pos = np.asarray(PILImage.open(r["position_mask"]).convert("L")) > 127
    return r, ctx, pos


def test_dilation_beyond_100_percent_is_fast_exact_and_bounded_by_the_image():
    from PySide6.QtGui import QImage, QColor
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from ui.mask_canvas import MaskCanvas
    img = QImage(600, 400, QImage.Format_ARGB32); img.fill(0)
    for y in range(180, 220):
        for x in range(280, 320):
            img.setPixelColor(x, y, QColor(255, 255, 255, 255))
    areas = []
    t0 = time.time()
    for px in (0, 10, 100, 300, 5000):
        out = MaskCanvas._dilate_alpha(img, px)
        a = np.asarray([[(out.pixel(x, y) >> 24) & 255 for x in range(0, 600, 4)] for y in range(0, 400, 4)]) > 0
        areas.append(int(a.sum()))
    assert time.time() - t0 < 10
    assert areas == sorted(areas) and areas[0] < areas[1] < areas[2] < areas[3]
    assert areas[-1] == (600 // 4) * (400 // 4)                 # raggio enorme: tutta l'immagine, nessun errore (il limite e il bordo)


def test_banner_text_and_colour_follow_the_selector(tab):
    from ui.i18n_manager import tr
    from ui.mask_canvas import BANNER_BG, BANNER_CTX_FG, BANNER_POS_FG
    cv = tab.canvas
    assert _contrast(BANNER_POS_FG, BANNER_BG) >= 4.5 and _contrast(BANNER_CTX_FG, BANNER_BG) >= 4.5
    tab.integrate_ctx_spin.setValue(80)
    assert cv.draw_banner.text() == tr("canvas.banner_auto", "Contesto automatico: {n} %").format(n=80)
    assert cv.draw_banner_color == BANNER_CTX_FG and BANNER_CTX_FG in cv.draw_banner.styleSheet() and BANNER_BG in cv.draw_banner.styleSheet()
    tab.integrate_ctx_mode.setCurrentIndex(tab.integrate_ctx_mode.findData("manual"))
    assert cv.draw_banner.text() == tr("canvas.banner_pos", "Stai disegnando: POSIZIONE")
    assert cv.draw_banner_color == BANNER_POS_FG and BANNER_POS_FG in cv.draw_banner.styleSheet()
    cv._draw_target_cb.setCurrentIndex(cv._draw_target_cb.findData("context"))
    assert cv.draw_banner.text() == tr("canvas.banner_ctx", "Stai disegnando: CONTESTO")
    assert cv.draw_banner_color == BANNER_CTX_FG and BANNER_CTX_FG in cv.draw_banner.styleSheet()
    cv._draw_target_cb.setCurrentIndex(cv._draw_target_cb.findData("position"))
    assert cv.draw_banner_color == BANNER_POS_FG
    tab.integrate_ctx_mode.setCurrentIndex(tab.integrate_ctx_mode.findData("auto"))      # torna automatico: scritta gialla col valore
    assert cv.draw_banner_color == BANNER_CTX_FG and "80" in cv.draw_banner.text()
    assert not cv._draw_target_cb.isVisibleTo(cv)                                         # il selettore c'e solo in manuale


def test_auto_margin_widgets_change_the_mask_that_is_sent_and_have_no_cap(tab):
    areas, pos_areas, px = {}, {}, {}
    for pct in (0, 25, 60, 150, 400):
        tab.integrate_ctx_spin.setValue(pct)                          # il CAMPO e libero: oltre il massimo del cursore (300) accetta lo stesso
        r, ctx, pos = _export(tab, f"p{pct}")
        areas[pct], pos_areas[pct], px[pct] = int(ctx.sum()), int(pos.sum()), r["context_px"]
        assert not (pos & ~ctx).any(), "la posizione sta dentro il contesto"
    assert tab.integrate_ctx_slider.value() == 300 and tab.integrate_ctx_spin.value() == 400
    assert len(set(pos_areas.values())) == 1                           # la posizione non cambia mai
    assert areas[0] < areas[25] < areas[60] < areas[150]
    assert areas[400] >= areas[150]                                    # oltre: limitato solo dal bordo
    assert px[0] == 0 and px[25] > 0 and px[150] == pytest.approx(px[60] * 150 / 60, rel=0.05)    # i pixel di margine sono proporzionali al %
    assert areas[400] <= 1024 * 768


def test_manual_mode_prefills_from_auto_and_the_context_stroke_is_in_the_sent_mask(tab):
    cv = tab.canvas
    tab.integrate_ctx_spin.setValue(60)
    _, auto_ctx, _ = _export(tab, "auto60")
    tab.integrate_ctx_mode.setCurrentIndex(tab.integrate_ctx_mode.findData("manual"))
    _, man_ctx, man_pos = _export(tab, "manual0")
    assert abs(int(man_ctx.sum()) - int(auto_ctx.sum())) < 0.02 * auto_ctx.sum()      # precompilato con la zona automatica del momento
    tab.integrate_ctx_clear_btn.click()                                                # «Svuota»: resta la sola sagoma
    _, empty_ctx, empty_pos = _export(tab, "manual_vuoto")
    assert (empty_ctx == empty_pos).all() and empty_ctx.any()
    # tratto di pennello CONTESTO in un punto lontano dal soggetto (in alto a destra)
    cv._draw_target_cb.setCurrentIndex(cv._draw_target_cb.findData("context"))
    cv.set_mode("brush"); cv.set_brush(60)
    from PySide6.QtCore import QPointF
    cv.push_undo(); cv.paint_at(QPointF(900, 80)); cv.paint_at(QPointF(930, 80))
    _, ctx2, pos2 = _export(tab, "manual_tratto")
    assert ctx2[80, 910] and not empty_ctx[80, 910]                                    # il tratto e nella maschera inviata
    assert (pos2 == empty_pos).all()                                                   # la posizione non e stata toccata
    cv.undo()                                                                          # annulla agisce sul contesto
    _, ctx3, _ = _export(tab, "manual_annulla")
    assert not ctx3[80, 910]
    cv.redo()
    _, ctx4, _ = _export(tab, "manual_ripeti")
    assert ctx4[80, 910]


def test_eraser_acts_only_on_the_selected_mask(tab):
    from PySide6.QtCore import QPointF
    cv = tab.canvas
    tab.integrate_ctx_mode.setCurrentIndex(tab.integrate_ctx_mode.findData("manual"))
    # 1) disegno una maschera di POSIZIONE (rosso, esistente) sul soggetto selezionato
    cv._draw_target_cb.setCurrentIndex(cv._draw_target_cb.findData("position"))
    ent = cv._entries[cv._active]
    c = ent["item"].sceneBoundingRect().center()
    cv.set_mode("brush"); cv.set_brush(50); cv.push_undo(); cv.paint_at(c)
    pos_before = cv.mask_image.copy()
    ctx_before = cv._ctx_manual.copy()
    # 2) con «Stai disegnando: Contesto» la gomma cancella SOLO il contesto
    cv._draw_target_cb.setCurrentIndex(cv._draw_target_cb.findData("context"))
    cv.set_mode("erase"); cv.set_brush(120); cv.push_undo(); cv.paint_at(c)
    assert cv.mask_image == pos_before                                                 # la posizione e intatta
    assert cv._ctx_manual != ctx_before                                                # il contesto e cambiato
    # 3) con «Posizione» la gomma cancella SOLO la posizione
    ctx_now = cv._ctx_manual.copy()
    cv._draw_target_cb.setCurrentIndex(cv._draw_target_cb.findData("position"))
    cv.push_undo(); cv.paint_at(c)
    assert cv._ctx_manual == ctx_now
    assert cv.mask_image != pos_before


def test_each_transparency_slider_changes_only_its_overlay_and_never_the_sent_mask(tab):
    cv = tab.canvas
    tab.integrate_ctx_spin.setValue(40)
    r0, ctx0, pos0 = _export(tab, "t0")
    im0 = PILImage.open(r0["image"]).tobytes()
    op_pos0, op_ctx0 = cv._ovl_pos.opacity(), cv._ovl_ctx.opacity()
    cv.ovl_transp_pos_sl.setValue(90)
    assert cv._ovl_pos.opacity() == pytest.approx(0.1) and cv._ovl_ctx.opacity() == pytest.approx(op_ctx0)
    cv.ovl_transp_ctx_sl.setValue(20)
    assert cv._ovl_ctx.opacity() == pytest.approx(0.8) and cv._ovl_pos.opacity() == pytest.approx(0.1)
    assert cv._ovl_pos.opacity() != op_pos0
    r1, ctx1, pos1 = _export(tab, "t1")
    assert (ctx1 == ctx0).all() and (pos1 == pos0).all()                               # le maschere inviate sono identiche
    assert PILImage.open(r1["image"]).tobytes() == im0                                 # e le sovrapposizioni non finiscono nell'immagine esportata
    cv.ovl_show_cb.setChecked(False)                                                   # interruttore mostra/nascondi
    assert cv._ovl_pos.pixmap().isNull() and cv._ovl_ctx.pixmap().isNull()
    r2, ctx2, _ = _export(tab, "t2")
    assert (ctx2 == ctx0).all() and PILImage.open(r2["image"]).tobytes() == im0
    cv.ovl_show_cb.setChecked(True)
    assert not cv._ovl_pos.pixmap().isNull() and not cv._ovl_ctx.pixmap().isNull()


def test_overlay_is_live_with_the_slider(tab):
    cv = tab.canvas
    tab.integrate_ctx_spin.setValue(0); a0 = cv._ovl_ctx.pixmap().toImage()
    tab.integrate_ctx_spin.setValue(100); a1 = cv._ovl_ctx.pixmap().toImage()
    def area(img):
        return sum(1 for y in range(0, img.height(), 4) for x in range(0, img.width(), 4) if (img.pixel(x, y) >> 24) & 255)
    assert area(a1) > area(a0) > 0


def test_context_selector_is_disabled_with_a_reason_for_engines_without_context_mask(tab):
    from ui.tabs.frame1_tab import INTEGRATE_ENGINE_CAPS, integrate_engine_supports_context_mask
    assert integrate_engine_supports_context_mask("qwen_image21") and integrate_engine_supports_context_mask("flux2")
    assert integrate_engine_supports_context_mask("kontext") and integrate_engine_supports_context_mask("flux3")
    assert not integrate_engine_supports_context_mask("krea2")
    assert integrate_engine_supports_context_mask("motore_futuro_xyz")                 # un motore nuovo la eredita
    assert INTEGRATE_ENGINE_CAPS["krea2"]["context_mask"] is False
    for idx, engine in ((0, "kontext"), (1, "flux2"), (3, "qwen_image21"), (4, "flux3")):
        tab.integrate_engine.setCurrentIndex(idx)
        assert tab.integrate_ctx_mode.isEnabled() and tab.integrate_ctx_spin.isEnabled(), engine
    tab.integrate_engine.setCurrentIndex(2)                                            # Krea 2 Edit
    assert not tab.integrate_ctx_mode.isEnabled() and not tab.integrate_ctx_spin.isEnabled() and not tab.integrate_ctx_slider.isEnabled()
    assert tab.integrate_ctx_mode.toolTip().strip() and tab.integrate_ctx_spin.toolTip().strip()
    assert not tab.canvas.integrate_ctx_enabled and tab.canvas._ovl_ctx.pixmap().isNull()    # niente giallo se il motore non la usa
    tab.integrate_engine.setCurrentIndex(0)
    assert tab.integrate_ctx_mode.isEnabled() and tab.canvas.integrate_ctx_enabled


def test_default_export_is_unchanged_when_nothing_is_touched(tab):
    """Valori di default (automatica, 0 %): la maschera e quella di sempre (riquadro del soggetto + 5 % della scena)."""
    r, ctx, pos = _export(tab, "default")
    assert r["context_px"] == 0 and tab.canvas.integrate_context_pct is None
    ys, xs = np.nonzero(ctx)
    grow = max(24, int(0.05 * 1024))
    bbox = tab.canvas._entries[1]["item"].sceneBoundingRect()
    assert xs.min() == max(0, int(bbox.left()) - grow) or abs(xs.min() - (bbox.left() - grow)) <= 2
    assert (ctx.sum()) >= pos.sum()
