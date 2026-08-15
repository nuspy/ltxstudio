"""TEST LIVE end-to-end DALLA GUI — Lucky Director, film da 3 minuti, 3 soggetti coerenti.

Percorso REALE utente→GUI→backend (regola [[test-end-to-end-from-gui]]):
  1) costruisce la MainWindow vera (offscreen);
  2) apre la finestra AI Film Studio come fa la voce di menu (_open_film_studio);
  3) crea il LuckyDirectorDialog ESATTAMENTE come il bottone 🍀 (LuckyDirectorDialog(film_win));
  4) imposta i widget: prompt con 3 personaggi @Nome, Coerenza volto = Krea 2 Edit, Immagine = Krea 2,
     Video = LingBot-Video, durata 3 min, 360p 16:9;
  5) CLICCA il bottone 🍀 «Generate film» (QTest.mouseClick su dlg.btn_go) → make_film REALE;
  6) fa girare il loop Qt fino a ok/fail del worker, loggando fase/percentuale con timestamp.

La coerenza volto vera arriva da cast_auto (crea gli asset+ritratti per i @Nome) + Krea 2 Edit negli shot,
con il worker Krea 2 persistente (:8191) che tiene il pipe caldo tra le decine di keyframe.

Avvio (python GUI con PySide6):
  QT_QPA_PLATFORM=offscreen <gui_python> tests/gui_regression/live_lucky_3min.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))       # per `import harness`
faulthandler.enable()

LOG = Path(__file__).resolve().parents[2] / "output" / "DIAG_lucky_3min_gui.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# Prompt CREATIVO con 3 personaggi ricorrenti nominati (regola test-prompts-creative: mai gattini/cani).
PROMPT = (
    "In a neon-drenched floating bazaar above a storm, three unlikely allies pull off an impossible heist. "
    "@Kaira is a sharp-eyed sky-pirate captain with a silver scar and a red coat. "
    "@Dorian is a nervous clockwork engineer with round glasses and copper gloves. "
    "@Vesna is a calm oracle with white braids and glowing tattoos. "
    "They meet, argue, split up across the market, and reunite on the airship deck as lightning cracks. "
    "Keep @Kaira, @Dorian and @Vesna clearly recognizable and consistent in every scene they appear in."
)


def main() -> int:
    from PySide6.QtCore import QEventLoop, Qt, QTimer
    from PySide6.QtTest import QTest
    import harness

    LOG.write_text("", encoding="utf-8")
    log("=== BUILD MainWindow (offscreen) ===")
    w = harness.build_window()

    log("=== APRO AI Film Studio (percorso menu _open_film_studio) ===")
    w._open_film_studio()
    fsw = w._film_win
    assert fsw is not None, "FilmStudioWindow non aperta"

    log("=== CREO LuckyDirectorDialog come il bottone 🍀 ===")
    from ui.film_studio.lucky_director_dialog import LuckyDirectorDialog
    dlg = LuckyDirectorDialog(fsw)

    # imposta i widget della GUI
    dlg.prompt.setPlainText(PROMPT)
    _set(dlg.coherence, "krea2_edit")
    _set(dlg.img_arch, "krea2")
    _set(dlg.vid_arch, "lingbot")
    _set(dlg.aspect, "16:9")
    _set_res(dlg, 360)                                   # 360p per velocità
    dlg.h.setValue(0); dlg.m.setValue(3); dlg.s.setValue(0)   # 3 minuti
    dlg.music.setChecked(False)                          # focus coerenza cast (musica riattivabile poi)
    log(f"spec GUI: coherence={dlg.coherence.currentData()} img={dlg.img_arch.currentData()} "
        f"vid={dlg.vid_arch.currentData()} dur={dlg.m.value()}m {dlg.aspect.currentData()} 360p music=OFF")

    if os.environ.get("LUCKY_DRYRUN") == "1":
        log("=== DRYRUN: wiring GUI ok, esco prima del click ===")
        return 0

    # CLIC sul bottone della GUI
    log("=== CLICK 🍀 Generate film ===")
    QTest.mouseClick(dlg.btn_go, Qt.LeftButton)
    worker = dlg._worker
    assert worker is not None, "worker non partito dopo il click"

    loop = QEventLoop()
    result = {"status": None, "res": None, "err": None}

    def _ok(res):
        result["status"] = "ok"; result["res"] = res; loop.quit()

    def _fail(msg):
        result["status"] = "fail"; result["err"] = msg; loop.quit()

    worker.ok.connect(_ok)
    worker.fail.connect(_fail)

    # heartbeat: logga la fase corrente ogni 20s (dalla status label della GUI)
    t0 = time.time()
    hb = QTimer()
    hb.setInterval(20000)
    hb.timeout.connect(lambda: log(f"[{int(time.time()-t0)}s] {dlg.status.text()[:160]}"))
    hb.start()

    # safety timeout 5h (film 3 min con decine di shot)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.setInterval(5 * 3600 * 1000)
    guard.timeout.connect(lambda: (result.update(status="timeout"), loop.quit()))
    guard.start()

    log("=== RUN (make_film reale in QThread) — attendo ok/fail ===")
    loop.exec()

    dt = int(time.time() - t0)
    if result["status"] == "ok":
        res = result["res"] or {}
        pid = res.get("project_id"); master = res.get("master", "")
        log(f"=== OK in {dt}s — project_id={pid} master={master} ===")
        _report_cast(pid)
        ok = bool(master and os.path.exists(master))
        log(f"master esiste: {ok} ({master})")
        return 0 if ok else 2
    if result["status"] == "fail":
        log(f"=== FAIL in {dt}s: {result['err']} ===")
        return 3
    log(f"=== TIMEOUT dopo {dt}s ===")
    return 4


def _set(combo, data) -> None:
    i = combo.findData(data)
    assert i >= 0, f"opzione {data!r} assente nel combo"
    combo.setCurrentIndex(i)


def _set_res(dlg, px: int) -> None:
    for i in range(dlg.res.count()):
        if int(dlg.res.itemData(i)) == px:
            dlg.res.setCurrentIndex(i); return
    dlg.res.setCurrentIndex(0)


def _report_cast(pid) -> None:
    """Elenca gli asset del cast e le loro foto (per il giudizio a occhio dell'utente sulla coerenza)."""
    try:
        from core.film_studio import db
        assets = db.query("asset", "project_id=? OR scope='project'", (pid,)) if pid else []
        # gli asset creati da cast_auto hanno scope='project'; elenchiamo quelli con foto
        for a in db.query("asset", "type='character'", ()):
            imgs = db.query("image", "owner_kind='asset' AND owner_id=?", (a["id"],))
            if imgs:
                log(f"  CAST @{a['name']}: {len(imgs)} foto → {imgs[0].get('file','')}")
    except Exception as e:                                    # noqa: BLE001
        log(f"  (report cast non disponibile: {e})")


if __name__ == "__main__":
    raise SystemExit(main())
