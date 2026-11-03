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
    t.sub.setCurrentIndex(t._inpaint_tab_index)                  # modalita Integra: pagina Modifica del pannello
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
    assert cv.draw_banner.text() == tr("canvas.banner_auto", "Contesto automatico: {n} px").format(n=80) and "px" in cv.draw_banner.text()
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
    for v in (0, 40, 100, 300, 900):
        tab.integrate_ctx_spin.setValue(v)                            # il CAMPO e in pixel e libero: oltre il massimo del cursore accetta lo stesso
        r, ctx, pos = _export(tab, f"p{v}")
        areas[v], pos_areas[v], px[v] = int(ctx.sum()), int(pos.sum()), r["context_px"]
        assert not (pos & ~ctx).any(), "la posizione sta dentro il contesto"
        assert px[v] == v                                              # i pixel inviati sono ESATTAMENTE quelli del campo
    assert tab.integrate_ctx_slider.value() == tab.integrate_ctx_slider.maximum() and tab.integrate_ctx_spin.value() == 900
    assert tab.integrate_ctx_slider.maximum() == max(tab.canvas.aspect_w, tab.canvas.aspect_h) // 2     # il cursore arriva a meta del lato lungo
    assert len(set(pos_areas.values())) == 1                           # la posizione non cambia mai
    assert areas[0] < areas[40] < areas[100] < areas[300]
    assert areas[900] >= areas[300]                                    # oltre: limitato solo dal bordo
    assert areas[900] <= 1024 * 768


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


def test_context_selector_is_hidden_for_engines_without_context_mask(tab):
    """03/11 sera: con i motori che non usano le due maschere i controlli sono NASCOSTI (non piu disattivati con tooltip)."""
    from ui.tabs.frame1_tab import INTEGRATE_ENGINE_CAPS, integrate_engine_supports_context_mask
    assert integrate_engine_supports_context_mask("qwen_image21") and integrate_engine_supports_context_mask("flux2")
    assert integrate_engine_supports_context_mask("kontext") and integrate_engine_supports_context_mask("flux3")
    assert not integrate_engine_supports_context_mask("krea2")
    assert integrate_engine_supports_context_mask("motore_futuro_xyz")                 # un motore nuovo la eredita
    assert INTEGRATE_ENGINE_CAPS["krea2"]["context_mask"] is False
    for idx, engine in ((0, "kontext"), (1, "flux2"), (3, "qwen_image21"), (4, "flux3")):
        tab.integrate_engine.setCurrentIndex(idx)
        assert not tab.integrate_ctx_mode.isHidden() and not tab._ctx_mode_lbl.isHidden(), engine
        assert tab.canvas.integrate_ctx_enabled and not tab.canvas.draw_banner.isHidden(), engine
    tab.integrate_engine.setCurrentIndex(2)                                            # Krea 2 Edit
    for w in (tab.integrate_ctx_mode, tab._ctx_mode_lbl, tab._ctx_pct_row, tab._ctx_pct_lbl, tab._ctx_manual_row, tab._ctx_manual_lbl):
        assert w.isHidden(), w                                                         # slider/campo/% stanno dentro _ctx_pct_row
    cv = tab.canvas
    assert not cv.integrate_ctx_enabled and cv._ovl_ctx.pixmap().isNull() and cv._ovl_pos.pixmap().isNull()   # niente giallo ne rosso
    for w in (cv.draw_banner, cv.ovl_legend, cv.ovl_transp_box, cv.draw_target_lbl, cv._draw_target_cb):
        assert w.isHidden(), w
    tab.integrate_engine.setCurrentIndex(0)
    assert not tab.integrate_ctx_mode.isHidden() and cv.integrate_ctx_enabled
    for w in (cv.draw_banner, cv.ovl_legend, cv.ovl_transp_box):
        assert not w.isHidden(), w


def test_margin_is_in_pixels_around_the_true_silhouette_and_zero_means_the_silhouette(tab):
    """Decisione dell'utente 03/11: base = la maschera VERA (sagoma), margine in PIXEL tutto intorno con la sua forma; 0 px = contesto = sagoma;
    valore iniziale 87 px (circa l'area di prima: riquadro + 5 %); accanto la % equivalente sulla larghezza della sagoma (sola lettura)."""
    from ui.mask_canvas import DEFAULT_CTX_PX
    assert DEFAULT_CTX_PX == 87 and tab.integrate_ctx_spin.value() == 87 and tab.integrate_ctx_slider.value() == 87    # valore iniziale
    tab.integrate_ctx_spin.setValue(0)
    r0, ctx0, pos0 = _export(tab, "w0")
    assert r0["context_px"] == 0 and pos0.any()
    assert abs(int(ctx0.sum()) - int(pos0.sum())) <= 0.01 * pos0.sum()                              # 0 px = solo la sagoma
    ys, xs = np.nonzero(pos0)
    width = int(xs.max() - xs.min() + 1)
    assert abs(tab.canvas.silhouette_width() - width) <= 1
    for px in (30, 87, 100, 250):
        tab.integrate_ctx_spin.setValue(px)
        r, ctx, pos = _export(tab, f"w{px}")
        assert r["context_px"] == px and (pos == pos0).all() and not (pos & ~ctx).any()
        want = int(round(100.0 * px / tab.canvas.silhouette_width()))
        assert f"{want}" in tab.integrate_ctx_pct_lbl.text(), (px, width, tab.integrate_ctx_pct_lbl.text())    # es. larga 200: 100 px = 50 %
    # la zona NON e il riquadro: a 30 px la forma segue la sagoma (ellisse) e non riempie gli angoli del riquadro che la contiene
    tab.integrate_ctx_spin.setValue(30)
    _, ctx30, _ = _export(tab, "w30b")
    box = ctx30[ys.min() - 5:ys.min() + 5, xs.min() - 5:xs.min() + 5]                              # angolo alto-sinistro del riquadro della sagoma
    assert not box.all()
    tab.canvas.set_aspect(512, 512)                                                                 # il cursore segue le proporzioni dell'immagine
    assert tab.integrate_ctx_slider.maximum() == 256
