"""Casi GUI (offscreen, nessuna GPU) dei metadati di generazione nei file (task 21):
- Impostazioni: la casella «Includi i dati di generazione nei file» esiste, è ON di default e persiste;
- Galleria: «Salva copia senza metadati» produce una copia senza dati e lascia intatto l'originale.
"""
from __future__ import annotations

import harness as H


def case_embed_meta_setting(w, spy):
    from core import user_prefs
    from ui.settings_dialog import SettingsDialog
    old = user_prefs.get("embed_gen_metadata", None)
    try:
        user_prefs.set("embed_gen_metadata", True)
        dlg = SettingsDialog(w.engine, w.driver, getattr(w, "_overrides", {}), w)
        cb = dlg.cb_embed_meta
        assert cb.isChecked(), "default ON"
        assert "prompt" in cb.toolTip().lower() and len(cb.toolTip()) > 80, "il tooltip deve avvisare che il prompt è visibile"
        cb.setChecked(False)
        dlg._on_accept()
        assert user_prefs.get("embed_gen_metadata", True) is False
        from core import gen_metadata
        assert gen_metadata.enabled() is False
        dlg2 = SettingsDialog(w.engine, w.driver, getattr(w, "_overrides", {}), w)
        assert not dlg2.cb_embed_meta.isChecked()
    finally:
        if old is None:
            try:
                user_prefs.set("embed_gen_metadata", True)
            except Exception:                                  # noqa: BLE001
                pass
        else:
            user_prefs.set("embed_gen_metadata", old)


def case_navigator_export_without_metadata(w, spy):
    import numpy as np
    from PIL import Image
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from core import gen_metadata
    from ui.navigator_dialog import NavigatorDialog
    d = H.OUT_DIR / "_t21_nav"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "gen.png"
    Image.fromarray((np.random.default_rng(2).random((40, 60, 3)) * 255).astype("uint8")).save(src)
    gen_metadata.embed(src, {"prompt": "secret prompt xyz", "seed": 3, "steps": 9}, respect_pref=False)
    dst = d / "gen_clean.png"
    if dst.exists():
        dst.unlink()
    dlg = NavigatorDialog(output_dir=d, apply_callback=lambda f: None, parent=w)
    # seleziona il file nella lista
    sel = None
    for i in range(dlg.file_list.count()):
        it = dlg.file_list.item(i)
        p = it.data(0x0100)
        if p is not None and getattr(p, "name", "") == "gen.png":
            sel = it
            break
    assert sel is not None, "gen.png non in lista"
    dlg.file_list.setCurrentItem(sel)
    assert dlg.btn_export_clean.isEnabled()
    orig_fd, orig_mb = QFileDialog.getSaveFileName, QMessageBox.information
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(dst), ""))
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        dlg._on_export_clean()
    finally:
        QFileDialog.getSaveFileName, QMessageBox.information = orig_fd, orig_mb
    assert dst.exists() and gen_metadata.read(dst) is None
    assert b"secret prompt xyz" not in dst.read_bytes()
    assert gen_metadata.read(src) is not None, "l'originale non deve cambiare"
    # lettura inversa: senza sidecar la Galleria legge i dati dal file
    from core.metadata_saver import MetadataSaver
    meta = MetadataSaver.load_sidecar(src)
    assert meta and meta.get("seed") == 3


CASES = [
    ("embed_meta_setting", "Impostazioni: casella metadati ON di default, tooltip di avviso, persiste.",
     case_embed_meta_setting),
    ("navigator_export_without_metadata", "Galleria: copia senza metadati; originale intatto; lettura dal file.",
     case_navigator_export_without_metadata),
]
