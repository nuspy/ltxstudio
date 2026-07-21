"""Tema scuro moderno per LTX Studio (QSS globale). Applicato in app.py."""
from __future__ import annotations

from PySide6.QtGui import QFont


def style_generate(btn) -> None:
    """Uniforma un bottone 'Genera' al colore della generazione video (#3ECF8E): verde pieno, testo scuro,
    grassetto. Applicato ai bottoni-azione principali di tutte le pipeline (immagine/video/musica/film/lora…)
    per coerenza visiva. Idempotente."""
    btn.setStyleSheet(
        "QPushButton { background: #3ecf8e; color: #16181d; font-weight: bold; border: none;"
        " border-radius: 8px; padding: 8px 14px; }"
        " QPushButton:hover { background: #4fdc9c; }"
        " QPushButton:disabled { background: #2f3441; color: #6b7280; }")

# palette
BG       = "#16181d"   # sfondo finestra
SURFACE  = "#1f222a"   # pannelli / group box
ELEV     = "#272b35"   # input / aree elevate
ELEV2    = "#2f3441"   # hover input
BORDER   = "#363b47"
BORDER_HI= "#454b59"
TEXT     = "#e7e9ef"
MUTED    = "#9aa1b0"
ACCENT   = "#5a8dee"
ACCENT_HI= "#6f9bf0"
ACCENT_LO= "#4878d6"
GREEN    = "#3ecf8e"
GREEN_HI = "#4fdc9c"

STYLESHEET = f"""
* {{ outline: 0; }}
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
    selection-background-color: {ACCENT};
    selection-color: #fff;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{
    background: {ELEV2}; color: {TEXT}; border: 1px solid {BORDER_HI};
    border-radius: 6px; padding: 6px 8px;
}}

/* ---- tabs ---- */
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 10px; top: -1px; background: {SURFACE}; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent; color: {MUTED};
    padding: 9px 18px; margin-right: 4px;
    border: none; border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

/* ---- group box ---- */
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER}; border-radius: 10px;
    margin-top: 14px; padding: 10px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: {ACCENT_HI};
}}

/* ---- buttons ---- */
QPushButton {{
    background: {ELEV}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 12px; font-weight: 600;
}}
QPushButton:hover {{ background: {ELEV2}; border-color: {BORDER_HI}; }}
QPushButton:pressed {{ background: {ACCENT_LO}; border-color: {ACCENT}; color: #fff; }}
QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
QPushButton:disabled {{ color: {MUTED}; background: {SURFACE}; border-color: {BORDER}; }}
QPushButton#tool {{
    font-size: 22px; border-radius: 11px; background: {ELEV}; border: 1px solid {BORDER};
}}
QPushButton#tool:hover {{ background: {ELEV2}; border-color: {ACCENT}; }}
QPushButton#tool:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
QPushButton#generate {{
    background: {GREEN}; color: #08130d; border: none; border-radius: 9px;
    padding: 11px; font-size: 15px; font-weight: 800;
}}
QPushButton#generate:hover {{ background: {GREEN_HI}; }}
QPushButton#generate:pressed {{ background: #2fb87b; }}

/* ---- text inputs ---- */
QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {ELEV}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px; padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {ACCENT}; }}

/* ---- combo / spin ---- */
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {ELEV}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px; padding: 4px 8px; min-height: 22px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER_HI}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid {MUTED}; margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {ELEV};
    border: 1px solid {BORDER_HI}; border-radius: 8px;
    selection-background-color: {ACCENT}; selection-color: #fff; outline: 0;
}}
/* NB: NIENTE `color:` qui. Un `color` sul popup fa sì che QStyleSheetStyle ignori il
   Qt.ForegroundRole per-item → i colori VRAM verde/giallo/rosso dei modelli sparivano.
   Il colore di default degli item eredita da `QWidget {{ color }}`; il per-item lo sovrascrive. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {ELEV2}; border: none; width: 16px; border-radius: 4px; margin: 1px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {ACCENT_LO}; }}

/* ---- sliders ---- */
QSlider::groove:horizontal {{ height: 6px; background: {ELEV2}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: #fff; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
    border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{ border-color: {ACCENT_HI}; }}

/* ---- progress ---- */
QProgressBar {{
    background: {ELEV}; border: 1px solid {BORDER}; border-radius: 8px;
    text-align: center; color: {TEXT}; height: 18px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 7px; }}

/* ---- checkbox / radio ---- */
QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 17px; height: 17px; }}
QCheckBox::indicator {{ border: 1px solid {BORDER_HI}; border-radius: 5px; background: {ELEV}; }}
QRadioButton::indicator {{ border: 1px solid {BORDER_HI}; border-radius: 9px; background: {ELEV}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}

/* ---- lists ---- */
QListWidget, QListView {{
    background: {ELEV}; border: 1px solid {BORDER}; border-radius: 8px; padding: 2px;
}}
QListWidget::item {{ border-radius: 6px; padding: 3px; }}
QListWidget::item:hover {{ background: {ELEV2}; }}
QListWidget::item:selected {{ background: {ACCENT_LO}; color: #fff; }}

/* ---- scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_HI}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_HI}; border-radius: 5px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- menu / status ---- */
QMenuBar {{ background: {BG}; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {ELEV2}; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER_HI}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #fff; }}
QStatusBar {{ background: {BG}; color: {MUTED}; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
"""


def apply_theme(app):
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)
