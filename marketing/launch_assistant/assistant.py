# -*- coding: utf-8 -*-
"""
AI Tools Forge — Copilota di Lancio.

Un'app pensata per accompagnare, un passo alla volta, una persona sotto forte stress
e con poche energie. Mostra UN solo prossimo passo, con dettagli a scomparsa, bottoni-link
belli, un bottone "Procedi" che fa l'azione dove possibile (e mostra il risultato da
verificare), premi/traguardi, una linea del tempo dei progressi, e un compagno (Hermes)
che spiega e incoraggia.

Avvio: marketing.ps1  (oppure: python assistant.py)
"""
from __future__ import annotations

import os
import sys
import json
import random
import threading
import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QUrl, QEvent
from PySide6.QtGui import QGuiApplication, QDesktopServices, QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QScrollArea, QFrame, QProgressBar, QTextEdit, QLineEdit, QSpinBox,
    QCheckBox, QDialog, QSizePolicy, QGridLayout, QPlainTextEdit
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEB = True
except Exception:
    HAS_WEB = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan_data as P                     # noqa: E402
from copy_texts import COPY               # noqa: E402
from hermes import HermesClient           # noqa: E402

MARKETING_DIR = HERE.parent
CONFIG_PATH = HERE / "config.json"
PROGRESS_PATH = HERE / "progress.json"


# ─────────────────────────────────────────────────────────────────────────────
# Util async (Hermes senza bloccare la GUI)
# ─────────────────────────────────────────────────────────────────────────────
class _Bridge(QObject):
    done = Signal(object)


def run_async(window, fn, callback):
    """Esegue fn() in un thread e richiama callback(result) nel thread GUI."""
    bridge = _Bridge()
    bridge.done.connect(callback)
    window._bridges.append(bridge)   # tieni un riferimento

    def work():
        try:
            res = fn()
        except Exception as e:
            res = f"[errore: {e}]"
        bridge.done.emit(res)
    threading.Thread(target=work, daemon=True).start()


def today_str():
    return datetime.date.today().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Persistenza progressi
# ─────────────────────────────────────────────────────────────────────────────
class Storage:
    def __init__(self):
        self.data = {"done": {}, "sales": 0, "active_dates": [], "badges": []}
        self.load()

    def load(self):
        try:
            if PROGRESS_PATH.exists():
                d = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    self.data.update(d)
        except Exception:
            pass

    def save(self):
        try:
            tmp = PROGRESS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, PROGRESS_PATH)
        except Exception:
            pass

    def is_done(self, tid):
        return tid in self.data["done"]

    def _touch_today(self):
        t = today_str()
        if t not in self.data["active_dates"]:
            self.data["active_dates"].append(t)

    def mark_done(self, tid):
        self.data["done"][tid] = today_str()
        self._touch_today()
        self.save()

    def unmark(self, tid):
        self.data["done"].pop(tid, None)
        self.save()

    def add_sale(self):
        self.data["sales"] = int(self.data.get("sales", 0)) + 1
        self._touch_today()
        self.save()

    def daily_done(self):
        t = today_str()
        return sum(1 for v in self.data["done"].values() if v == t)

    def tu_done_count(self):
        return sum(1 for tid in self.data["done"]
                   if (P.task_by_id(tid) or {}).get("who") == "TU")

    def streak(self):
        dates = set(self.data.get("active_dates", []))
        d = datetime.date.today()
        if d.isoformat() not in dates:
            d = d - datetime.timedelta(days=1)
        n = 0
        while d.isoformat() in dates:
            n += 1
            d = d - datetime.timedelta(days=1)
        return n


def evaluate_badges(storage):
    """Ritorna i badge appena conquistati (non ancora registrati)."""
    done = set(storage.data["done"].keys())
    tu_done = storage.tu_done_count()
    sales = int(storage.data.get("sales", 0))
    streak = storage.streak()
    new = []
    for b in P.BADGES:
        if b["id"] in storage.data["badges"]:
            continue
        c = b["check"]
        ok = False
        if c["type"] == "task":
            ok = c["id"] in done
        elif c["type"] == "all":
            ok = all(i in done for i in c["ids"])
        elif c["type"] == "count":
            ok = tu_done >= c["n"]
        elif c["type"] == "sales":
            ok = sales >= c["n"]
        elif c["type"] == "streak":
            ok = streak >= c["n"]
        if ok:
            new.append(b)
    return new


# ─────────────────────────────────────────────────────────────────────────────
# Stylesheet (caldo, arrotondato, gratificante)
# ─────────────────────────────────────────────────────────────────────────────
QSS = """
QMainWindow, QWidget { background: #0f1320; color: #e8ecf5; font-size: 14px; }
#side { background: #161b2e; border: none; }
#sideBtn { text-align: left; padding: 12px 14px; border: none; border-radius: 10px;
           background: transparent; color: #c7cee0; font-size: 15px; }
#sideBtn:hover { background: #20273f; }
#sideBtn:checked { background: #283153; }
#sideBtn[active="true"] { background: #6c5ce7; color: white; font-weight: 600; }
#header { background: #161b2e; border-radius: 14px; }
#brand { font-size: 20px; font-weight: 700; color: #ffffff; }
#why { color: #9aa3bd; font-style: italic; }
#motiv { color: #ffd479; font-size: 15px; font-weight: 600; }
#card { background: #161b2e; border: 1px solid #232a44; border-radius: 16px; }
#cardBig { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1b2340, stop:1 #151a2e);
           border: 1px solid #2e3a66; border-radius: 18px; }
#title { font-size: 19px; font-weight: 700; color: #ffffff; }
#detail { color: #b9c0d6; }
#chip { background: #232a44; color: #aeb6d0; border-radius: 8px; padding: 3px 9px; font-size: 12px; }
QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #8e7bff);
    color: white; border: none; border-radius: 12px; padding: 11px 16px; font-weight: 700; font-size: 15px; }
QPushButton#primary:hover { background: #7d6cf0; }
QPushButton#success { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #16a34a, stop:1 #22c55e);
    color: white; border: none; border-radius: 12px; padding: 11px 16px; font-weight: 700; font-size: 15px; }
QPushButton#success:hover { background: #1eb257; }
QPushButton#link { background: #202a4a; color: #cdd6ff; border: 1px solid #34406e;
    border-radius: 11px; padding: 9px 14px; font-weight: 600; }
QPushButton#link:hover { background: #283566; }
QPushButton#ghost { background: transparent; color: #aeb6d0; border: 1px solid #2c3450;
    border-radius: 11px; padding: 9px 14px; }
QPushButton#ghost:hover { background: #1c2238; color: #fff; }
QPushButton#sale { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #f59e0b, stop:1 #fbbf24);
    color: #1a1300; border: none; border-radius: 12px; padding: 10px 14px; font-weight: 800; }
QProgressBar { background: #232a44; border: none; border-radius: 9px; height: 18px; text-align: center;
    color: white; font-weight: 700; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22c55e, stop:1 #6c5ce7);
    border-radius: 9px; }
QTextEdit, QPlainTextEdit, QLineEdit, QSpinBox { background: #0e1220; border: 1px solid #2a3252;
    border-radius: 10px; padding: 8px; color: #e8ecf5; }
QScrollArea { border: none; }
#result { background: #0e1a12; border: 1px solid #1f6b3a; border-radius: 12px; color: #d8ffe6; }
#badge { background: #161b2e; border: 1px solid #232a44; border-radius: 14px; }
#badgeOn { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2a2150, stop:1 #1c2a22);
           border: 1px solid #6c5ce7; border-radius: 14px; }
#toast { background: #22c55e; color: #04210f; border-radius: 12px; padding: 12px 18px;
         font-weight: 700; font-size: 15px; }
"""
# (alcuni nomi colore "creativi" sopra sono refusi innocui ignorati da Qt; i fallback restano validi)


# ─────────────────────────────────────────────────────────────────────────────
# Sezione a scomparsa (accordion) per i dettagli
# ─────────────────────────────────────────────────────────────────────────────
class Collapsible(QWidget):
    def __init__(self, text, open_default=False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = QPushButton(("▾ Nascondi dettagli" if open_default else "▸ Mostra dettagli"))
        self.btn.setObjectName("ghost")
        self.btn.setCheckable(True)
        self.btn.setChecked(open_default)
        self.body = QLabel(text)
        self.body.setObjectName("detail")
        self.body.setWordWrap(True)
        self.body.setVisible(open_default)
        self.btn.toggled.connect(self._toggle)
        lay.addWidget(self.btn)
        lay.addWidget(self.body)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.btn.setText("▾ Nascondi dettagli" if on else "▸ Mostra dettagli")


# ─────────────────────────────────────────────────────────────────────────────
# Linea del tempo dei progressi
# ─────────────────────────────────────────────────────────────────────────────
class Timeline(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.c = controller
        self.setMinimumHeight(96)

    def paintEvent(self, ev):
        days = list(range(0, 15))
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width(); h = self.height()
        m = 28
        y = h // 2
        n = len(days)
        step = (w - 2 * m) / max(1, n - 1)
        done = set(self.c.storage.data["done"].keys())
        cur = self.c.current_day()
        # linea base
        p.setPen(QPen(QColor("#2a3252"), 4))
        p.drawLine(m, y, w - m, y)
        for i, d in enumerate(days):
            x = int(m + i * step)
            tasks = [t for t in P.TASKS if t["who"] == "TU" and t["day"] == d]
            if tasks:
                ndone = sum(1 for t in tasks if t["id"] in done)
                if ndone == len(tasks):
                    col = QColor("#22c55e")
                elif ndone > 0:
                    col = QColor("#f59e0b")
                else:
                    col = QColor("#39406a")
            else:
                col = QColor("#39406a")
            r = 11
            if d == cur:
                p.setPen(QPen(QColor("#8e7bff"), 3))
                p.setBrush(QColor("#6c5ce7"))
                p.drawEllipse(x - r - 3, y - r - 3, 2 * (r + 3), 2 * (r + 3))
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(x - r, y - r, 2 * r, 2 * r)
            p.setPen(QColor("#cdd6ff"))
            f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
            p.drawText(x - 8, y + 4, str(d))
            p.setPen(QColor("#7b84a3"))
            f2 = QFont(); f2.setPointSize(7); p.setFont(f2)
            p.drawText(x - 14, y + 26, f"G{d}")
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Card di un passo
# ─────────────────────────────────────────────────────────────────────────────
class TaskCard(QFrame):
    def __init__(self, controller, task, big=False):
        super().__init__()
        self.c = controller
        self.t = task
        self.setObjectName("cardBig" if big else "card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)

        # header
        top = QHBoxLayout()
        emoji = {"account": "🧩", "content": "🎬", "sales": "💳", "community": "📣",
                 "checkpoint": "📊", "selfcare": "🌿", "tech": "🤖"}.get(task["kind"], "•")
        title = QLabel(f"{emoji}  {task['title']}")
        title.setObjectName("title")
        title.setWordWrap(True)
        top.addWidget(title, 1)
        chip = QLabel(P.phase_title(task["phase"]))
        chip.setObjectName("chip")
        top.addWidget(chip, 0, Qt.AlignTop)
        lay.addLayout(top)

        # dettaglio a scomparsa
        lay.addWidget(Collapsible(task.get("detail", ""), open_default=big))

        # bottoni azione
        row = QHBoxLayout()
        row.setSpacing(8)
        if task.get("url"):
            b = QPushButton("🌐  " + task.get("url_label", "Apri la pagina"))
            b.setObjectName("link")
            b.clicked.connect(lambda: self.c.open_url(task["url"]))
            row.addWidget(b)
        if task.get("copy_key"):
            b = QPushButton("📋  Copia il testo pronto")
            b.setObjectName("link")
            b.clicked.connect(lambda: self.c.copy_text(task["copy_key"]))
            row.addWidget(b)
        if task.get("doc"):
            b = QPushButton("📄  Apri documento")
            b.setObjectName("ghost")
            b.clicked.connect(lambda: self.c.open_doc(task))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        # riga "fai per me" + hermes + fatto
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        if self._has_auto():
            b = QPushButton("✨  Procedi (fallo per me)")
            b.setObjectName("primary")
            b.clicked.connect(self._procedi)
            row2.addWidget(b)
        if task.get("hermes", False):
            b = QPushButton("💬  Chiedi a Hermes")
            b.setObjectName("ghost")
            b.clicked.connect(self._ask_hermes)
            row2.addWidget(b)
        row2.addStretch(1)
        if task["who"] == "TU":
            self.done_btn = QPushButton("✓  Fatto!" if not self.c.storage.is_done(task["id"]) else "✓ Completato")
            self.done_btn.setObjectName("success")
            self.done_btn.clicked.connect(self._toggle_done)
            row2.addWidget(self.done_btn)
        else:
            tag = QLabel("🤖 lo faccio io — tu non devi fare nulla")
            tag.setObjectName("detail")
            row2.addWidget(tag)
        lay.addLayout(row2)

        # area risultato (nascosta finché non serve)
        self.result = QLabel("")
        self.result.setObjectName("result")
        self.result.setWordWrap(True)
        self.result.setVisible(False)
        self.result.setContentsMargins(12, 10, 12, 10)
        lay.addWidget(self.result)

        self.confirm_row = QWidget()
        cr = QHBoxLayout(self.confirm_row)
        cr.setContentsMargins(0, 0, 0, 0)
        self.confirm_btn = QPushButton("✓  Confermo: è fatto")
        self.confirm_btn.setObjectName("success")
        self.confirm_btn.clicked.connect(self._toggle_done)
        cr.addWidget(self.confirm_btn)
        cr.addStretch(1)
        self.confirm_row.setVisible(False)
        lay.addWidget(self.confirm_row)

    def _has_auto(self):
        t = self.t
        return bool(t.get("url") or t.get("copy_key") or t["id"] == "prep_materials"
                    or t.get("hermes"))

    def _show_result(self, text, allow_confirm=True):
        self.result.setText(text)
        self.result.setVisible(True)
        self.confirm_row.setVisible(allow_confirm and self.t["who"] == "TU"
                                    and not self.c.storage.is_done(self.t["id"]))

    def _procedi(self):
        t = self.t
        actions = []
        if t["id"] == "prep_materials":
            path = self.c.make_lancio_folder()
            actions.append(f"Ho creato la cartella «{path}» e l'ho aperta.")
        if t.get("url"):
            self.c.open_url(t["url"])
            actions.append(f"Ho aperto la pagina giusta ({t.get('url_label','la pagina')}).")
        if t.get("copy_key"):
            self.c.copy_text(t["copy_key"], silent=True)
            actions.append("Ho copiato il testo pronto: incollalo con Ctrl+V dove serve.")
        if actions:
            self._show_result("✅ Fatto per te:\n• " + "\n• ".join(actions)
                              + "\n\nControlla che sia tutto ok, poi conferma qui sotto.")
        elif t.get("hermes"):
            self._show_result("⏳ Chiedo a Hermes di prepararti il risultato...", allow_confirm=False)
            run_async(self.c.win, lambda: self.c.hermes.do_task(t), self._hermes_done)

    def _ask_hermes(self):
        self._show_result("⏳ Hermes sta pensando a come aiutarti...", allow_confirm=False)
        run_async(self.c.win, lambda: self.c.hermes.help_task(self.t), self._hermes_done)

    def _hermes_done(self, text):
        self._show_result("💬 Hermes:\n" + str(text))

    def _toggle_done(self):
        tid = self.t["id"]
        if self.c.storage.is_done(tid):
            self.c.storage.unmark(tid)
        else:
            self.c.complete_task(tid)
        self.c.refresh_all()


# ─────────────────────────────────────────────────────────────────────────────
# Finestra principale / controller
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.win = self
        self._bridges = []
        self.cfg = self._load_cfg()
        self.storage = Storage()
        self.hermes = HermesClient(self.cfg)
        self.controller = self  # le card chiamano self.*
        self.c = self

        self.setWindowTitle("AI Tools Forge — Copilota di Lancio")
        self.resize(1180, 820)
        self.setStyleSheet(QSS)

        self._session_start = datetime.datetime.now()
        self._last_interaction = datetime.datetime.now()
        self._break_reminded = False

        self._build()
        self._start_timers()
        QApplication.instance().installEventFilter(self)
        self._greet()

    # — config —
    def _load_cfg(self):
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"brand": "AI Tools Forge",
                    "patreon_url": "https://www.patreon.com/c/AIToolsForge",
                    "why": "", "daily_goal": 3, "hermes": {}}

    def _save_cfg(self):
        try:
            CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # — costruzione UI —
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # sidebar
        side = QFrame(); side.setObjectName("side"); side.setFixedWidth(210)
        sl = QVBoxLayout(side); sl.setContentsMargins(10, 14, 10, 14); sl.setSpacing(6)
        logo = QLabel("🛠️  AI Tools Forge"); logo.setObjectName("brand")
        logo.setWordWrap(True)
        sl.addWidget(logo)
        sub = QLabel("Copilota di Lancio"); sub.setObjectName("why")
        sl.addWidget(sub); sl.addSpacing(10)

        # prova a costruire il browser interno; se fallisce, si ripiega su quello esterno
        self._browser_page = None
        self._web_ok = False
        if HAS_WEB:
            try:
                self._browser_page = self._build_browser()
                self._web_ok = True
            except Exception:
                self._web_ok = False

        self.nav = []
        items = [("⭐ Adesso", 0), ("📅 Oggi", 1), ("🗺️ Piano", 2),
                 ("🏆 Premi", 3), ("💬 Compagno", 4)]
        if self._web_ok:
            self._browser_idx = 5
            items.append(("🌐 Browser", 5))
        self.settings_idx = 6 if self._web_ok else 5
        items.append(("⚙️ Impostazioni", self.settings_idx))
        for label, idx in items:
            b = QPushButton(label); b.setObjectName("sideBtn"); b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=idx: self._go(i))
            sl.addWidget(b); self.nav.append((b, idx))
        sl.addStretch(1)

        stuck = QPushButton("🆘 Mi sento bloccato")
        stuck.setObjectName("link")
        stuck.clicked.connect(self._struggle)
        sl.addWidget(stuck)
        sale = QPushButton("💶 Ho fatto una vendita!")
        sale.setObjectName("sale")
        sale.clicked.connect(self._register_sale)
        sl.addWidget(sale)
        root.addWidget(side)

        # area destra
        right = QVBoxLayout(); right.setSpacing(12)
        right.addWidget(self._build_header())
        self.stack = QStackedWidget()
        self.page_adesso = QScrollArea(); self.page_adesso.setWidgetResizable(True)
        self.page_oggi = QScrollArea(); self.page_oggi.setWidgetResizable(True)
        self.page_piano = QScrollArea(); self.page_piano.setWidgetResizable(True)
        self.page_premi = QScrollArea(); self.page_premi.setWidgetResizable(True)
        self.page_comp = self._build_companion()
        self.stack.addWidget(self.page_adesso)   # 0
        self.stack.addWidget(self.page_oggi)     # 1
        self.stack.addWidget(self.page_piano)    # 2
        self.stack.addWidget(self.page_premi)    # 3
        self.stack.addWidget(self.page_comp)     # 4
        if self._web_ok and self._browser_page is not None:
            self.stack.addWidget(self._browser_page)     # 5
        self.stack.addWidget(self._build_settings())     # 6 (o 5 se no web)
        right.addWidget(self.stack, 1)
        root.addLayout(right, 1)

        # toast overlay
        self.toast = QLabel("", self)
        self.toast.setObjectName("toast")
        self.toast.setAlignment(Qt.AlignCenter)
        self.toast.setVisible(False)

        self.refresh_all()
        self._go(0)

    def _build_header(self):
        h = QFrame(); h.setObjectName("header")
        v = QVBoxLayout(h); v.setContentsMargins(18, 14, 18, 14); v.setSpacing(8)
        row = QHBoxLayout()
        self.h_motiv = QLabel(random.choice(P.MOTIVATION)); self.h_motiv.setObjectName("motiv")
        row.addWidget(self.h_motiv, 1)
        self.h_streak = QLabel("🔥 0"); self.h_streak.setObjectName("chip")
        self.h_daily = QLabel("Oggi: 0"); self.h_daily.setObjectName("chip")
        row.addWidget(self.h_daily); row.addWidget(self.h_streak)
        v.addLayout(row)
        self.h_bar = QProgressBar(); self.h_bar.setMaximum(100)
        v.addWidget(self.h_bar)
        self.h_why = QLabel(self.cfg.get("why", "")); self.h_why.setObjectName("why")
        self.h_why.setWordWrap(True)
        v.addWidget(self.h_why)
        return h

    def _build_companion(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6)
        t = QLabel("💬 Hermes — il tuo compagno di lancio")
        t.setObjectName("title"); v.addWidget(t)
        info = QLabel("Chiedimi come fare un passo, oppure dimmi solo come ti senti. "
                      "Sono qui. (Se non rispondo, configura l'endpoint in Impostazioni.)")
        info.setObjectName("detail"); info.setWordWrap(True); v.addWidget(info)
        self.chat_log = QTextEdit(); self.chat_log.setReadOnly(True); v.addWidget(self.chat_log, 1)
        row = QHBoxLayout()
        self.chat_in = QLineEdit(); self.chat_in.setPlaceholderText("Scrivi a Hermes...")
        self.chat_in.returnPressed.connect(self._chat_send)
        send = QPushButton("Invia"); send.setObjectName("primary"); send.clicked.connect(self._chat_send)
        row.addWidget(self.chat_in, 1); row.addWidget(send)
        v.addLayout(row)
        row2 = QHBoxLayout()
        b1 = QPushButton("🌿 Qual è il passo più piccolo?"); b1.setObjectName("link")
        b1.clicked.connect(self._tiny_step); row2.addWidget(b1)
        b2 = QPushButton("🆘 Mi sento bloccato"); b2.setObjectName("link")
        b2.clicked.connect(self._struggle); row2.addWidget(b2)
        row2.addStretch(1)
        v.addLayout(row2)
        return w

    def _build_browser(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.url_bar = QLineEdit(); self.url_bar.setPlaceholderText("https://...")
        self.url_bar.returnPressed.connect(lambda: self._web_load(self.url_bar.text()))
        go = QPushButton("Vai"); go.setObjectName("link"); go.clicked.connect(lambda: self._web_load(self.url_bar.text()))
        row.addWidget(self.url_bar, 1); row.addWidget(go)
        v.addLayout(row)
        self.web = QWebEngineView()
        self.web.setUrl(QUrl(self.cfg.get("patreon_url", "https://www.patreon.com/c/AIToolsForge")))
        v.addWidget(self.web, 1)
        return w

    def _build_settings(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(10)
        t = QLabel("⚙️ Impostazioni"); t.setObjectName("title"); v.addWidget(t)
        v.addWidget(QLabel("Il tuo «perché» (appare in alto, ti tiene ancorato):"))
        self.set_why = QPlainTextEdit(self.cfg.get("why", "")); self.set_why.setFixedHeight(90)
        v.addWidget(self.set_why)
        r = QHBoxLayout(); r.addWidget(QLabel("Obiettivo passi al giorno:"))
        self.set_goal = QSpinBox(); self.set_goal.setRange(1, 12); self.set_goal.setValue(int(self.cfg.get("daily_goal", 3)))
        r.addWidget(self.set_goal); r.addStretch(1); v.addLayout(r)
        v.addWidget(QLabel("— Hermes Agent (il tuo compagno) —"))
        hz = self.cfg.get("hermes", {})
        self.set_henabled = QCheckBox("Hermes attivo"); self.set_henabled.setChecked(bool(hz.get("enabled", True)))
        v.addWidget(self.set_henabled)
        self.set_hmode = QCheckBox("Usa il tuo Hermes Agent in WSL (consigliato — nessun token)")
        self.set_hmode.setChecked(str(hz.get("mode", "cli")).lower() == "cli")
        v.addWidget(self.set_hmode)
        v.addWidget(QLabel("Solo per modalità HTTP (gateway :8642 o LM Studio :1234):"))
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Endpoint:"))
        self.set_hurl = QLineEdit(hz.get("base_url", "http://localhost:8642/v1")); r2.addWidget(self.set_hurl, 1)
        v.addLayout(r2)
        r3 = QHBoxLayout(); r3.addWidget(QLabel("Modello:"))
        self.set_hmodel = QLineEdit(hz.get("model", "default")); r3.addWidget(self.set_hmodel, 1)
        v.addLayout(r3)
        r4 = QHBoxLayout(); r4.addWidget(QLabel("Token (solo HTTP):"))
        self.set_htoken = QLineEdit(hz.get("api_key", "")); self.set_htoken.setEchoMode(QLineEdit.Password)
        r4.addWidget(self.set_htoken, 1)
        v.addLayout(r4)
        rb = QHBoxLayout()
        save = QPushButton("💾 Salva"); save.setObjectName("primary"); save.clicked.connect(self._save_settings)
        test = QPushButton("🔌 Testa Hermes"); test.setObjectName("link"); test.clicked.connect(self._test_hermes)
        rb.addWidget(save); rb.addWidget(test); rb.addStretch(1); v.addLayout(rb)
        self.set_status = QLabel(""); self.set_status.setObjectName("detail"); self.set_status.setWordWrap(True)
        v.addWidget(self.set_status)
        v.addStretch(1)
        return w

    # — navigazione —
    def _go(self, idx):
        self.stack.setCurrentIndex(idx)
        for b, i in self.nav:
            b.setChecked(i == idx)
            b.setProperty("active", "true" if i == idx else "false")
            b.style().unpolish(b); b.style().polish(b)
        self._touch()

    # — refresh —
    def refresh_all(self):
        self._refresh_header()
        self._refresh_adesso()
        self._refresh_oggi()
        self._refresh_piano()
        self._refresh_premi()

    def _refresh_header(self):
        tu_total = sum(1 for t in P.TASKS if t["who"] == "TU")
        tu_done = self.storage.tu_done_count()
        pct = int(100 * tu_done / max(1, tu_total))
        self.h_bar.setValue(pct)
        self.h_bar.setFormat(f"{tu_done}/{tu_total} passi  ·  {pct}%")
        self.h_streak.setText(f"🔥 {self.storage.streak()} giorni")
        goal = int(self.cfg.get("daily_goal", 3))
        self.h_daily.setText(f"Oggi: {self.storage.daily_done()}/{goal}")
        self.h_why.setText(self.cfg.get("why", ""))

    def _clear(self, area):
        host = QWidget(); lay = QVBoxLayout(host)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(12)
        area.setWidget(host)
        return lay

    def next_task(self):
        done = set(self.storage.data["done"].keys())
        for t in P.TASKS:
            if t["who"] != "TU":
                continue
            if t["id"] in done:
                continue
            if all(d in done for d in t.get("deps", [])):
                return t
        return None

    def current_day(self):
        t = self.next_task()
        return t["day"] if t else 14

    def _refresh_adesso(self):
        lay = self._clear(self.page_adesso)
        head = QLabel("⭐ Il tuo prossimo passo — solo questo, niente altro")
        head.setObjectName("title"); lay.addWidget(head)
        t = self.next_task()
        if t is None:
            done_all = QFrame(); done_all.setObjectName("cardBig")
            dl = QVBoxLayout(done_all)
            msg = QLabel("🏁 Hai completato tutti i passi del lancio.\n\nFermati un momento e rendi onore a "
                         "quello che hai fatto: partendo da zero, in difficoltà, hai costruito e lanciato. "
                         "Sei più forte di quanto pensi.")
            msg.setObjectName("title"); msg.setWordWrap(True); dl.addWidget(msg)
            lay.addWidget(done_all); lay.addStretch(1); return
        lay.addWidget(TaskCard(self, t, big=True))
        # anteprima del passo successivo
        done = set(self.storage.data["done"].keys()) | {t["id"]}
        nxt = None
        for cand in P.TASKS:
            if cand["who"] == "TU" and cand["id"] not in done and all(d in done for d in cand.get("deps", [])):
                nxt = cand; break
        if nxt:
            prev = QLabel(f"⏭️  Dopo questo:  {nxt['title']}")
            prev.setObjectName("why"); lay.addWidget(prev)
        lay.addStretch(1)

    def _refresh_oggi(self):
        lay = self._clear(self.page_oggi)
        head = QLabel("📅 Oggi"); head.setObjectName("title"); lay.addWidget(head)
        cur = self.current_day()
        todays = [t for t in P.TASKS if t["who"] == "TU" and t["day"] == cur]
        if not todays:
            todays = [self.next_task()] if self.next_task() else []
        sub = QLabel(f"Concentrati sul Giorno {cur}. Bastano {self.cfg.get('daily_goal',3)} passi per dire "
                     f"'oggi ho fatto la mia parte'.")
        sub.setObjectName("detail"); sub.setWordWrap(True); lay.addWidget(sub)
        for t in todays:
            if t:
                lay.addWidget(TaskCard(self, t, big=False))
        lay.addStretch(1)

    def _refresh_piano(self):
        lay = self._clear(self.page_piano)
        head = QLabel("🗺️ Tutto il piano"); head.setObjectName("title"); lay.addWidget(head)
        tl = Timeline(self); lay.addWidget(tl)
        for key, title, emoji, days in P.PHASES:
            ph = QLabel(f"{emoji}  {title}   ·   {days}"); ph.setObjectName("title")
            lay.addWidget(ph)
            for t in [x for x in P.TASKS if x["phase"] == key]:
                lay.addWidget(self._mini_row(t))
        lay.addStretch(1)

    def _mini_row(self, t):
        f = QFrame(); f.setObjectName("card")
        h = QHBoxLayout(f); h.setContentsMargins(12, 8, 12, 8)
        done = self.storage.is_done(t["id"])
        cb = QCheckBox(); cb.setChecked(done)
        if t["who"] == "TU":
            cb.toggled.connect(lambda on, tid=t["id"]: self._set_done(tid, on))
        else:
            cb.setEnabled(True)
            cb.toggled.connect(lambda on, tid=t["id"]: self._set_done(tid, on))
        h.addWidget(cb)
        lbl = QLabel(("✅ " if done else "") + t["title"])
        lbl.setWordWrap(True)
        if t["who"] == "IO":
            lbl.setText("🤖 " + t["title"] + "  (lo faccio io)")
        h.addWidget(lbl, 1)
        if t.get("url"):
            b = QPushButton("🌐"); b.setObjectName("ghost"); b.setFixedWidth(40)
            b.clicked.connect(lambda _=False, u=t["url"]: self.open_url(u)); h.addWidget(b)
        if t.get("doc"):
            b = QPushButton("📄"); b.setObjectName("ghost"); b.setFixedWidth(40)
            b.clicked.connect(lambda _=False, tt=t: self.open_doc(tt)); h.addWidget(b)
        return f

    def _set_done(self, tid, on):
        if on and not self.storage.is_done(tid):
            self.complete_task(tid)
        elif not on and self.storage.is_done(tid):
            self.storage.unmark(tid)
        self.refresh_all()

    def _refresh_premi(self):
        lay = self._clear(self.page_premi)
        head = QLabel("🏆 I tuoi premi"); head.setObjectName("title"); lay.addWidget(head)
        sub = QLabel("Ogni traguardo che sblocchi è reale e te lo sei guadagnato.")
        sub.setObjectName("detail"); lay.addWidget(sub)
        grid_host = QWidget(); grid = QGridLayout(grid_host); grid.setSpacing(12)
        earned = set(self.storage.data["badges"])
        for i, b in enumerate(P.BADGES):
            card = QFrame(); card.setObjectName("badgeOn" if b["id"] in earned else "badge")
            cl = QVBoxLayout(card)
            em = QLabel(b["emoji"] if b["id"] in earned else "🔒")
            em.setStyleSheet("font-size: 34px;"); em.setAlignment(Qt.AlignCenter)
            cl.addWidget(em)
            ti = QLabel(b["title"]); ti.setObjectName("title"); ti.setAlignment(Qt.AlignCenter); ti.setWordWrap(True)
            cl.addWidget(ti)
            de = QLabel(b["desc"] if b["id"] in earned else "Da sbloccare")
            de.setObjectName("detail"); de.setAlignment(Qt.AlignCenter); de.setWordWrap(True)
            cl.addWidget(de)
            grid.addWidget(card, i // 3, i % 3)
        lay.addWidget(grid_host); lay.addStretch(1)

    # — azioni —
    def complete_task(self, tid):
        self.storage.mark_done(tid)
        praise = random.choice(["Bravo. Un altro mattone.", "Fatto! Vai così.",
                                "Ottimo, sei più avanti di prima.", "Sì! Piccolo passo, grande effetto.",
                                "Ce l'hai fatta. Respira e festeggia un attimo."])
        self._show_toast(f"✓ {praise}")
        for b in evaluate_badges(self.storage):
            self.storage.data["badges"].append(b["id"]); self.storage.save()
            self._badge_dialog(b)
        # obiettivo giornaliero raggiunto
        if self.storage.daily_done() == int(self.cfg.get("daily_goal", 3)):
            self._show_toast("🎯 Obiettivo di oggi raggiunto! Hai fatto la tua parte. Puoi staccare sereno.")

    def _register_sale(self):
        self.storage.add_sale()
        dlg = QDialog(self); dlg.setWindowTitle("🎉")
        v = QVBoxLayout(dlg)
        e = QLabel("💶🎉"); e.setStyleSheet("font-size: 56px;"); e.setAlignment(Qt.AlignCenter); v.addWidget(e)
        m = QLabel(f"Vendita registrata! Totale: {self.storage.data['sales']}.\n\n"
                   "Questa è la prova che il tuo lavoro vale. Qualcuno ha pagato per ciò che hai creato.\n"
                   "Tienilo a mente quando arrivano i momenti duri.")
        m.setWordWrap(True); m.setAlignment(Qt.AlignCenter); v.addWidget(m)
        ok = QPushButton("Continuo 💪"); ok.setObjectName("success"); ok.clicked.connect(dlg.accept); v.addWidget(ok)
        for b in evaluate_badges(self.storage):
            self.storage.data["badges"].append(b["id"]); self.storage.save()
        dlg.exec()
        self.refresh_all()

    def _badge_dialog(self, b):
        dlg = QDialog(self); dlg.setWindowTitle("Nuovo traguardo!")
        v = QVBoxLayout(dlg)
        e = QLabel(b["emoji"]); e.setStyleSheet("font-size: 64px;"); e.setAlignment(Qt.AlignCenter); v.addWidget(e)
        t = QLabel("🏆  " + b["title"]); t.setObjectName("title"); t.setAlignment(Qt.AlignCenter); v.addWidget(t)
        d = QLabel(b["desc"]); d.setWordWrap(True); d.setAlignment(Qt.AlignCenter); v.addWidget(d)
        ok = QPushButton("Evvai! 🎉"); ok.setObjectName("primary"); ok.clicked.connect(dlg.accept); v.addWidget(ok)
        dlg.exec()

    def open_url(self, url):
        self._touch()
        if getattr(self, "_web_ok", False):
            try:
                self._web_load(url)
                idx = self._browser_idx
                self.stack.setCurrentIndex(idx)
                for b, i in self.nav:
                    act = (i == idx)
                    b.setChecked(act); b.setProperty("active", "true" if act else "false")
                    b.style().unpolish(b); b.style().polish(b)
                return
            except Exception:
                pass
        QDesktopServices.openUrl(QUrl(url))

    def _web_load(self, url):
        if not getattr(self, "_web_ok", False):
            QDesktopServices.openUrl(QUrl(url)); return
        if not url.startswith("http"):
            url = "https://" + url
        self.url_bar.setText(url)
        self.web.setUrl(QUrl(url))

    def open_doc(self, task):
        self._touch()
        key = task.get("doc")
        fname = P.DOCS.get(key, key)
        path = MARKETING_DIR / fname
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self._show_toast("Documento non trovato: " + str(fname))

    def copy_text(self, key, silent=False):
        self._touch()
        txt = COPY.get(key, "")
        QGuiApplication.clipboard().setText(txt)
        if not silent:
            self._show_toast("📋 Testo copiato! Incollalo con Ctrl+V dove serve.")

    def make_lancio_folder(self):
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            desktop = Path.home()
        folder = desktop / "LANCIO"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception:
            pass
        return str(folder)

    # — compagno / hermes —
    def _chat_send(self):
        txt = self.chat_in.text().strip()
        if not txt:
            return
        self.chat_in.clear()
        self.chat_log.append(f"<b>Tu:</b> {txt}")
        self.chat_log.append("<i>Hermes sta scrivendo...</i>")
        run_async(self, lambda: self.hermes.chat(txt), self._chat_reply)
        self._touch()

    def _chat_reply(self, text):
        # rimuovi il "sta scrivendo"
        doc = self.chat_log.toPlainText()
        self.chat_log.append(f"<b>Hermes:</b> {text}")

    def _tiny_step(self):
        t = self.next_task()
        self._go(4)
        if t:
            self.chat_log.append(f"<b>🌿 Il passo più piccolo adesso:</b> {t['title']}")
            self.chat_log.append(f"<i>{t.get('detail','')}</i>")
        else:
            self.chat_log.append("<b>🌿</b> Non c'è nessun passo rimasto. Hai finito. Riposa.")

    def _struggle(self):
        self._go(4)
        msg = random.choice(P.STRUGGLE)
        self.chat_log.append(f"<b>🆘</b> {msg}")
        t = self.next_task()
        if t:
            self.chat_log.append(f"<b>Micro-passo proposto:</b> {t['title']}")
        if self.hermes.enabled:
            ctx = f"{self.storage.tu_done_count()} passi fatti, prossimo: {t['title'] if t else 'nessuno'}"
            self.chat_log.append("<i>Sento cosa dice Hermes...</i>")
            run_async(self, lambda: self.hermes.encourage(ctx), self._chat_reply)
        self._touch()

    # — impostazioni —
    def _save_settings(self):
        self.cfg["why"] = self.set_why.toPlainText()
        self.cfg["daily_goal"] = self.set_goal.value()
        self.cfg.setdefault("hermes", {})
        self.cfg["hermes"]["enabled"] = self.set_henabled.isChecked()
        self.cfg["hermes"]["mode"] = "cli" if self.set_hmode.isChecked() else "http"
        self.cfg["hermes"]["base_url"] = self.set_hurl.text().strip()
        self.cfg["hermes"]["model"] = self.set_hmodel.text().strip()
        self.cfg["hermes"]["api_key"] = self.set_htoken.text()
        self._save_cfg()
        self.hermes.update(mode=self.cfg["hermes"]["mode"],
                           base_url=self.cfg["hermes"]["base_url"],
                           model=self.cfg["hermes"]["model"],
                           api_key=self.cfg["hermes"]["api_key"],
                           enabled=self.cfg["hermes"]["enabled"])
        self.set_status.setText("✅ Salvato.")
        self._refresh_header()

    def _test_hermes(self):
        self.set_status.setText("⏳ Provo a contattare Hermes...")
        self.hermes.update(mode="cli" if self.set_hmode.isChecked() else "http",
                           base_url=self.set_hurl.text().strip(),
                           model=self.set_hmodel.text().strip(),
                           api_key=self.set_htoken.text(),
                           enabled=self.set_henabled.isChecked())
        run_async(self, lambda: ("ok" if self.hermes.available() else "ko"),
                  lambda r: self.set_status.setText(
                      "✅ Hermes c'è! Sei collegato al tuo agente." if r == "ok"
                      else "⚠️ Non raggiungo Hermes. In modalità WSL verifica che l'agente sia installato; "
                           "in modalità HTTP controlla endpoint/token."))

    # — toast / timers / nudge —
    def _show_toast(self, text):
        self.toast.setText(text)
        self.toast.adjustSize()
        self.toast.move((self.width() - self.toast.width()) // 2, self.height() - 80)
        self.toast.setVisible(True)
        self.toast.raise_()
        QTimer.singleShot(4200, lambda: self.toast.setVisible(False))

    def _start_timers(self):
        self._mt = QTimer(self); self._mt.timeout.connect(self._rotate_motiv); self._mt.start(20000)
        self._nt = QTimer(self); self._nt.timeout.connect(self._check_idle); self._nt.start(60000)

    def _rotate_motiv(self):
        self.h_motiv.setText(random.choice(P.MOTIVATION))

    def _check_idle(self):
        now = datetime.datetime.now()
        idle = (now - self._last_interaction).total_seconds() / 60.0
        if idle >= 6:
            self._show_toast(random.choice(P.NUDGES))
            self._last_interaction = now  # non ripetere subito
        sess = (now - self._session_start).total_seconds() / 60.0
        if sess >= 50 and not self._break_reminded:
            self._break_reminded = True
            self._show_toast("🌿 Stai lavorando da un po'. Una pausa vera ora ti renderà più lucido dopo.")

    def _touch(self):
        self._last_interaction = datetime.datetime.now()

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.MouseButtonPress, QEvent.KeyPress):
            self._last_interaction = datetime.datetime.now()
        return super().eventFilter(obj, ev)

    def _greet(self):
        tu_done = self.storage.tu_done_count()
        if tu_done == 0:
            self.chat_log.append("<b>Hermes:</b> Ciao. So che è un momento difficile. "
                                 "Non devi fare tutto: devi fare solo il prossimo piccolo passo, e l'ho già "
                                 "preparato per te nella scheda ⭐ Adesso. Sono qui con te.")
        else:
            self.chat_log.append(f"<b>Hermes:</b> Bentornato. Hai già fatto {tu_done} passi — non è poco. "
                                 "Riprendiamo piano, dal prossimo. ⭐")


def main():
    # necessario quando è presente il browser interno (QtWebEngine), innocuo altrimenti
    try:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("AI Tools Forge — Copilota di Lancio")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
