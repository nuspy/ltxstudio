"""Sistema i18n per LTX Studio.

Uso:
    from ui.i18n_manager import tr, i18n
    label = QLabel(tr("tab.images"))
    i18n.language_changed.connect(self.retranslateUi)

La lingua è salvata in data/gui_lang.json e applicata all'avvio.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal

_I18N_DIR = Path(__file__).parent / "i18n"
_LANG_FILE = Path(__file__).resolve().parents[1] / "data" / "gui_lang.json"

LANGUAGES: dict[str, str] = {
    "en": "English",
    "it": "Italiano",
    "ru": "Русский",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "zh": "中文",
    "ko": "한국어",
    "uk": "Українська",
    "pt": "Português",
    "hu": "Magyar",
    "he": "עברית",
    "el": "Ελληνικά",
    "ar": "العربية",
    "hi": "हिन्दी",
}

# Flag emoji per ogni lingua (visualizzato nel menu)
LANG_FLAGS: dict[str, str] = {
    "en": "🇬🇧",
    "it": "🇮🇹",
    "ru": "🇷🇺",
    "de": "🇩🇪",
    "fr": "🇫🇷",
    "es": "🇪🇸",
    "zh": "🇨🇳",
    "ko": "🇰🇷",
    "uk": "🇺🇦",
    "pt": "🇵🇹",
    "hu": "🇭🇺",
    "he": "🇮🇱",
    "el": "🇬🇷",
    "ar": "🇸🇦",
    "hi": "🇮🇳",
}


class I18nManager(QObject):
    """Singleton che gestisce il caricamento e il cambio di lingua."""

    language_changed = Signal()

    _instance: "I18nManager | None" = None

    @classmethod
    def instance(cls) -> "I18nManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._lang: str = "it"   # default: italiano (già tradotto nell'app)
        self._dict: dict[str, str] = {}
        self._fallback: dict[str, str] = {}
        self._load_saved_lang()
        self._load()

    # ── persistenza ──────────────────────────────────────────────────────
    def _load_saved_lang(self):
        try:
            data = json.loads(_LANG_FILE.read_text("utf-8"))
            lang = data.get("lang", "it")
            if lang in LANGUAGES:
                self._lang = lang
        except Exception:
            pass

    def _save_lang(self):
        try:
            _LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LANG_FILE.write_text(
                json.dumps({"lang": self._lang}, ensure_ascii=False, indent=2),
                "utf-8",
            )
        except Exception:
            pass

    # ── caricamento dizionario ────────────────────────────────────────────
    def _load(self):
        f = _I18N_DIR / f"{self._lang}.json"
        if not f.exists():
            f = _I18N_DIR / "it.json"   # fallback: italiano
        try:
            self._dict = json.loads(f.read_text("utf-8"))
        except Exception:
            self._dict = {}
        # fallback = italiano (usato se una chiave non è ancora tradotta)
        fb = _I18N_DIR / "it.json"
        try:
            self._fallback = json.loads(fb.read_text("utf-8"))
        except Exception:
            self._fallback = {}

    # ── API pubblica ──────────────────────────────────────────────────────
    def tr(self, key: str, fallback: str = "") -> str:
        """Traduce una chiave. Cade su it.json, poi sul fallback testuale."""
        v = self._dict.get(key)
        if v is not None:
            return v
        v = self._fallback.get(key)
        if v is not None:
            return v
        return fallback if fallback else key

    def current_lang(self) -> str:
        return self._lang

    def set_language(self, lang_code: str) -> bool:
        """Cambia lingua, salva, emette il segnale. Ritorna True se la lingua è cambiata."""
        if lang_code not in LANGUAGES or lang_code == self._lang:
            return False
        self._lang = lang_code
        self._load()
        self._save_lang()
        self.language_changed.emit()
        return True


# ── modulo-livello convenienza ───────────────────────────────────────────────
i18n: I18nManager = I18nManager.instance()


def tr(key: str, fallback: str = "") -> str:
    """Shortcut globale per la traduzione."""
    return i18n.tr(key, fallback)
