"""Accesso a SAM 3.1 (facebook/sam3.1, gated): promemoria «ho chiesto l'accesso», controllo leggero ogni 30 minuti, esiti, rimanda / annulla, avvio del download.
Nessuna rete (HEAD iniettato), nessun download, nessuna GPU."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core import sam31_access as AC
from core import segmentation_engine as SE


@pytest.fixture()
def prefs(monkeypatch, tmp_path):
    store = {}
    from core import user_prefs
    monkeypatch.setattr(user_prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(user_prefs, "set", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr(SE, "models_dir", lambda: tmp_path / "vuoto")            # pesi di Meta assenti
    monkeypatch.setattr(SE, "comfy_dir", lambda: tmp_path / "vuoto_comfy")       # copia Comfy assente
    monkeypatch.setattr(AC, "token", lambda: store.get("hf_token") or None)
    return store


def test_probe_maps_http_status_to_outcomes_without_leaking_the_token():
    seen = []

    def head(code):
        def _h(url, headers):
            seen.append((url, headers))
            return code
        return _h
    assert AC.probe("hf_SEGRETO", head=head(200))[0] == "granted"
    assert AC.probe("hf_SEGRETO", head=head(302))[0] == "granted"               # redirect verso il CDN = accesso concesso
    assert AC.probe("hf_SEGRETO", head=head(403))[0] == "pending"
    assert AC.probe("hf_SEGRETO", head=head(401))[0] == "bad_token"
    assert AC.probe("hf_SEGRETO", head=head(None))[0] == "offline"
    assert AC.probe("hf_SEGRETO", head=head(500))[0] == "error"
    url, headers = seen[0]
    assert url == "https://huggingface.co/facebook/sam3.1/resolve/main/config.json"                  # file piccolo del repo gated, nessun peso
    assert headers["Authorization"] == "Bearer hf_SEGRETO"
    for code in (200, 403, 401, None, 500):
        assert "hf_SEGRETO" not in "".join(AC.probe("hf_SEGRETO", head=head(code)))             # il token non finisce nei messaggi


def test_probe_without_token_does_not_touch_the_network():
    def boom(url, headers):
        raise AssertionError("rete usata senza token")
    assert AC.probe("", head=boom)[0] == "no_token"


def test_button_appears_only_after_confirmation_and_goes_away_on_cancel(prefs):
    assert not AC.requested() and not AC.watching() and not AC.button_visible()
    AC.confirm_requested(now=1000.0)
    assert AC.requested() and AC.watching() and AC.button_visible()
    AC.cancel_watch()
    assert not AC.watching() and not AC.button_visible() and AC.requested()
    AC.confirm_requested(now=2000.0)                                              # dalla guida si puo riattivare
    assert AC.button_visible()


def test_check_runs_every_30_minutes_and_snooze_postpones(prefs):
    t0 = 1_000_000.0
    assert not AC.due(t0)                                                         # nessuna conferma: nessun controllo
    AC.confirm_requested(now=t0)
    assert AC.due(t0 + 1)                                                         # mai controllato: subito
    AC.record("pending", now=t0)
    assert not AC.due(t0 + 29 * 60) and AC.due(t0 + 30 * 60)
    assert AC.CHECK_INTERVAL_S == 1800
    until = AC.snooze(2 * 3600, now=t0 + 30 * 60)
    assert until == t0 + 30 * 60 + 7200
    assert not AC.due(t0 + 60 * 60) and not AC.due(until - 1) and AC.due(until)
    assert AC.next_check_at() == until
    AC.cancel_watch()
    assert not AC.due(until + 10_000)


def test_no_check_and_no_button_when_the_weights_are_complete(prefs, monkeypatch):
    AC.confirm_requested(now=1.0)
    assert AC.button_visible()
    monkeypatch.setattr(SE, "weights_state", lambda base=None, comfy_base=None: "ok")
    assert not AC.button_visible() and not AC.due(10 ** 9)


def test_describe_covers_every_outcome():
    for r in ("granted", "pending", "no_token", "bad_token", "offline", "error"):
        assert AC.describe(r)
    assert "alcune ore" in AC.describe("pending")


def test_save_token_is_local_and_empty_removes_it(prefs):
    AC.save_token("  hf_abc  ")
    assert prefs["hf_token"] == "hf_abc"
    AC.save_token("")
    assert prefs["hf_token"] == ""


# ── Qt: il controllo in thread e l'avvio del download ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(qapp, cond, timeout=5.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    return cond()


def test_watcher_downloads_only_when_access_is_granted(qapp, prefs):
    from ui.sam31_watch import Sam31Watcher
    answers = iter([("pending", "403"), ("pending", "403"), ("granted", "200")])
    downloads, msgs = [], []
    w = Sam31Watcher(None, probe=lambda: next(answers), start_download=lambda: downloads.append(1), tick_ms=10_000_000)
    w.message.connect(msgs.append)
    changed = []
    w.changed.connect(lambda: changed.append(1))
    w.confirm_requested()                                                         # primo controllo subito: ancora in attesa
    assert _wait(qapp, lambda: AC.last_result() == "pending")
    assert downloads == [] and AC.watching() and any("alcune ore" in m for m in msgs)
    w.check_now()
    assert _wait(qapp, lambda: not w._thread.isRunning())
    qapp.processEvents()
    assert downloads == []
    w.check_now()
    assert _wait(qapp, lambda: AC.last_result() == "granted")
    assert downloads == [1] and changed


def test_watcher_snooze_and_cancel_signal_the_gui(qapp, prefs):
    from ui.sam31_watch import Sam31Watcher
    w = Sam31Watcher(None, probe=lambda: ("pending", ""), start_download=lambda: None, tick_ms=10_000_000)
    seen = []
    w.changed.connect(lambda: seen.append("c"))
    AC.confirm_requested(now=time.time() - 10_000)
    w.snooze()
    assert AC.snoozed_until() > time.time() + 7000 and not AC.due()
    w.cancel()
    assert not AC.watching() and len(seen) == 2


def test_tick_checks_only_when_due(qapp, prefs):
    from ui.sam31_watch import Sam31Watcher
    calls = []
    w = Sam31Watcher(None, probe=lambda: (calls.append(1), ("pending", ""))[1], start_download=lambda: None, tick_ms=10_000_000)
    w._tick()
    assert _wait(qapp, lambda: True) and calls == []                              # nessuna conferma: niente richiesta
    AC.confirm_requested(now=time.time() - 3600)
    w._tick()
    assert _wait(qapp, lambda: calls == [1])
    assert _wait(qapp, lambda: not w._thread.isRunning())
    qapp.processEvents()
    w._tick()
    assert _wait(qapp, lambda: True) and calls == [1]                             # appena controllato: non ripete prima di 30 minuti


def test_default_download_uses_the_catalog_entry_and_comfy_alternative(qapp, prefs, monkeypatch):
    from ui import download_manager as DM
    from ui.sam31_watch import Sam31Watcher
    started = []

    class _Mgr:
        def active(self):
            return []

        def start(self, ids, feature="", extra=None):
            started.append((tuple(ids), feature))
            return 7

        class _S:
            def connect(self, *a):
                pass
        finished = _S()
        changed = _S()
    monkeypatch.setattr(DM, "manager", lambda: _Mgr())
    w = Sam31Watcher(None, probe=lambda: ("granted", ""), tick_ms=10_000_000)
    w._default_download()
    assert started == [(("sam3.1-multiplex",), "SAM 3.1")] and w.downloading_job == 7
    w.download_comfy()
    assert started[1][0] == ("sam31-comfy-multiplex",)
