"""Tab LoRA — elenco dinamico LoRA compatibili LTX-2 con descrizione, trigger e preview.

- Catalogo ricaricato all'avvio + pulsante Ricarica (legge sempre catalog.json più recente).
- Solo LoRA compatibili (architecture=="LTX").
- Selezione multipla con strength; verifica trigger-word nel prompt (avviso non bloccante).
- Preview automatica della LoRA selezionata (immagine scaricata da Civitai; per i .mp4 mostra avviso).
"""
from __future__ import annotations

import urllib.request
from pathlib import PureWindowsPath

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QLabel,
    QDoubleSpinBox, QPushButton, QLineEdit, QCheckBox, QSplitter, QPlainTextEdit,
    QToolButton,
)

from core.lora_catalog import LoraCatalog, LoraEntry
from core import user_prefs
from ui.i18n_manager import tr


class _ThumbLoader(QThread):
    loaded = Signal(bytes)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                self.loaded.emit(r.read())
        except Exception:
            self.loaded.emit(b"")


class LoraTab(QWidget):
    def __init__(self, allow_nsfw_getter=lambda: True, embedded: bool = False, arch: str | None = "LTX"):
        super().__init__()
        self.allow_nsfw_getter = allow_nsfw_getter
        self._embedded = embedded            # se True: filtro NSFW pilotato dal selettore di sezione
        self._arch_label = arch or "LTX"
        self.catalog = LoraCatalog(compatible_arch=arch) if arch else LoraCatalog()
        self._extra_entries: list[LoraEntry] = []  # LoRA viste da ComfyUI ma mancanti nel catalogo
        self._rows: dict[str, dict] = {}      # name -> {item, cb, sp, entry}
        # name -> (checked, strength, entry) — sopravvive ai cicli di filtro
        self._persistent_selection: dict[str, tuple[bool, float, "LoraEntry | None"]] = {}
        # LoRA da ri-selezionare al prossimo refresh (restore da gui_state/preset, per comfy_name):
        # serve perché l'inventario LoRA si popola in modo asincrono (come i modelli).
        self._restore_pending: dict[str, float] | None = None
        self._thumb: _ThumbLoader | None = None
        self._build()
        self.refresh()

    def set_arch(self, new_arch: str):
        """Cambia l'architettura del catalogo LoRA e ricarica la lista."""
        self._arch_label = new_arch or "LTX"
        self.catalog = LoraCatalog(compatible_arch=new_arch) if new_arch else LoraCatalog()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText(tr("lora.search.ph"))
        self.search.textChanged.connect(self.refresh)
        self.hide_nsfw = QCheckBox(tr("lora.hide_nsfw"))
        self.hide_nsfw.stateChanged.connect(self.refresh)
        if self._embedded:
            self.hide_nsfw.setVisible(False)     # filtro pilotato dal selettore NSFW della sezione
        self.auto_triggers = QCheckBox(tr("lora.auto_triggers"))
        self.auto_triggers.setChecked(True)
        self.auto_triggers.setToolTip(tr("lora.auto_triggers.tt"))
        self.reload_btn = QPushButton(tr("lora.reload.btn"))
        self.reload_btn.clicked.connect(self._reload)
        top.addWidget(self.search, 1); top.addWidget(self.auto_triggers)
        top.addWidget(self.hide_nsfw); top.addWidget(self.reload_btn)
        root.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)
        split.addWidget(self.list)

        right = QWidget(); rl = QVBoxLayout(right)
        self.preview = QLabel(tr("lora.preview.lbl")); self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(260, 260)
        self.preview.setStyleSheet("background:#111;color:#666;border:1px solid #333;")
        rl.addWidget(self.preview)
        self.info = QPlainTextEdit(); self.info.setReadOnly(True); self.info.setMaximumHeight(160)
        rl.addWidget(self.info)
        split.addWidget(right)
        split.setSizes([520, 320])
        root.addWidget(split, 1)

        self.count_lbl = QLabel("")
        root.addWidget(self.count_lbl)

    def _reload(self):
        self.catalog.load()
        self.refresh()

    def refresh(self):
        self.catalog.reload_if_changed()
        allow = self.allow_nsfw_getter() if self._embedded else not self.hide_nsfw.isChecked()
        q = self.search.text().strip().lower()
        # Aggiorna il dict persistente con lo stato corrente dei widget visibili PRIMA del clear.
        # In questo modo le LoRA temporaneamente nascoste dal filtro mantengono la loro selezione.
        for name, r in self._rows.items():
            self._persistent_selection[name] = (r["cb"].isChecked(), r["sp"].value(), r["entry"])
        self.list.clear(); self._rows.clear()
        loras = self._merged_loras(allow)
        # Ordina: PREFERITE in cima (poi ordine catalogo già ordinato per downloads/nome)
        loras = sorted(loras, key=lambda e: (not user_prefs.is_fav_lora(e.comfy_name),))
        shown = 0
        for e in loras:
            hay = (e.name + " " + " ".join(e.tags) + " " + " ".join(e.triggers)).lower()
            if q and q not in hay:
                continue
            is_fav = user_prefs.is_fav_lora(e.comfy_name)
            it = QListWidgetItem(self.list)
            it.setData(Qt.UserRole, e)
            badge = " 🔞" if e.nsfw else ""
            trig = f"  ⟦{', '.join(e.triggers)}⟧" if e.triggers else ""
            # checkbox REALE dentro la riga (selezione multipla affidabile)
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(4, 1, 4, 1)
            # stella preferiti (oro se preferita)
            star = QToolButton(); star.setCheckable(True); star.setChecked(is_fav)
            star.setText("★" if is_fav else "☆")
            star.setStyleSheet(
                "QToolButton{border:none;font-size:16px;color:%s;}" % ("#f5c518" if is_fav else "#777"))
            star.setToolTip(tr("lora.fav.tt"))
            star.clicked.connect(lambda _c, ent=e: self._toggle_fav(ent))
            h.addWidget(star)
            cb = QCheckBox(f"{e.name}{badge}{trig}")
            # Ripristina dal dict persistente (sopravvive ai cicli di filtro);
            # default strength = ultima salvata a mano per questa LoRA (user_prefs).
            default_strength = user_prefs.get_lora_strength(e.comfy_name, 0.6)
            _sel = self._persistent_selection.get(e.name, (False, default_strength, None))
            prev_checked, prev_strength = _sel[0], _sel[1]
            # restore da gui_state/preset (per comfy_name): forza selezione + strength
            if self._restore_pending and e.comfy_name in self._restore_pending:
                prev_checked = True
                prev_strength = self._restore_pending[e.comfy_name]
            cb.setChecked(prev_checked)
            cb.clicked.connect(lambda _c, ent=e: self._show_entry(ent))
            h.addWidget(cb, 1)
            h.addWidget(QLabel(tr("lora.str_lbl")))
            sp = QDoubleSpinBox(); sp.setRange(0, 2); sp.setSingleStep(0.05); sp.setDecimals(2)
            sp.setValue(prev_strength)
            sp.setMinimumWidth(96); sp.setMaximumWidth(110)
            # quando l'utente cambia la strength a mano → diventa il nuovo default per quella LoRA
            sp.valueChanged.connect(lambda val, ent=e: user_prefs.set_lora_strength(ent.comfy_name, val))
            h.addWidget(sp)
            it.setSizeHint(w.sizeHint())
            self.list.setItemWidget(it, w)
            self._rows[e.name] = {"item": it, "cb": cb, "sp": sp, "entry": e, "star": star}
            shown += 1
        # consuma i restore-pending già applicati (le LoRA non ancora presenti restano in attesa)
        if self._restore_pending:
            applied = {r["entry"].comfy_name for r in self._rows.values()}
            for cn in [c for c in self._restore_pending if c in applied]:
                del self._restore_pending[cn]
            if not self._restore_pending:
                self._restore_pending = None
        # Conta selezioni visibili + nascoste dal filtro corrente
        n_sel_visible = sum(1 for r in self._rows.values() if r["cb"].isChecked())
        n_sel_hidden = sum(1 for name, (chk, _str, _ent) in self._persistent_selection.items()
                          if chk and name not in self._rows)
        n_sel = n_sel_visible + n_sel_hidden
        hidden_note = (tr("lora.count.hidden").format(n=n_sel_hidden)) if n_sel_hidden else ""
        self.count_lbl.setText(tr("lora.count.lbl").format(shown=shown, arch=self._arch_label, n_sel=n_sel, hidden_note=hidden_note))

    def set_available_loras(self, names: list[str]):
        """Aggiunge al catalogo UI le LoRA LTX presenti in ComfyUI ma non catalogate.

        Il catalogo resta la fonte primaria per descrizioni/trigger/preview; questo fallback
        evita che un file installato e usabile in ComfyUI sparisca solo perché scan.py non lo ha
        ancora indicizzato o lo ha classificato male.
        """
        catalog_entries = self.catalog.loras(allow_nsfw=True)
        known = {e.comfy_name.replace("/", "\\").lower() for e in catalog_entries}
        known_files = {e.filename.lower() for e in catalog_entries}
        extra: list[LoraEntry] = []
        for raw in names or []:
            comfy = str(raw).replace("/", "\\")
            low = comfy.lower()
            filename = PureWindowsPath(comfy).name
            if low in known or filename.lower() in known_files:
                continue
            if "ltx" not in low:
                continue
            nsfw = any(k in low for k in (
                "nsfw", "xxx", "adult", "nude", "nudity", "naked", "erotic", "porn",
                "sex", "deepthroat", "penile", "pussy", "anal", "blowjob",
            ))
            extra.append(LoraEntry(
                sha256=f"comfy:{low}", filename=filename,
                path=f"E:\\AI\\Models\\LoRA\\{comfy}", architecture="LTX",
                nsfw=nsfw, tags=["ComfyUI"], description=tr("lora.extra.desc"),
            ))
        self._extra_entries = extra
        self.refresh()

    def _merged_loras(self, allow_nsfw: bool) -> list[LoraEntry]:
        out = self.catalog.loras(allow_nsfw=allow_nsfw)
        seen = {e.comfy_name.replace("/", "\\").lower() for e in out}
        for e in self._extra_entries:
            if not allow_nsfw and e.nsfw:
                continue
            key = e.comfy_name.replace("/", "\\").lower()
            if key not in seen:
                out.append(e)
                seen.add(key)
        return out

    def _toggle_fav(self, e: "LoraEntry"):
        """Inverte il preferito; se diventa preferita salva la strength corrente come default."""
        row = self._rows.get(e.name)
        strength = row["sp"].value() if row else None
        # sincronizza la selezione persistente prima del refresh (riordina)
        for name, r in self._rows.items():
            self._persistent_selection[name] = (r["cb"].isChecked(), r["sp"].value(), r["entry"])
        user_prefs.toggle_fav_lora(e.comfy_name, strength)
        self.refresh()

    def _on_select(self, cur, prev):
        if cur:
            self._show_entry(cur.data(Qt.UserRole))

    def _show_entry(self, e: "LoraEntry"):
        lines = [tr("lora.info.name").format(name=e.name),
                 tr("lora.info.file").format(file=e.comfy_name)]
        if e.triggers:
            lines.append(tr("lora.info.trigger") + ": " + ", ".join(e.triggers))
        else:
            lines.append(tr("lora.info.no_trigger"))
        if e.downloads:
            lines.append(tr("lora.info.downloads").format(n=e.downloads))
        if e.description:
            lines.append("\n" + e.description[:600])
        self.info.setPlainText("\n".join(lines))
        # preview
        self.preview.setText(tr("lora.preview.loading"))
        url = self.catalog.preview_url_for(e, allow_nsfw=self.allow_nsfw_getter())
        if not url:
            self.preview.setText(tr("lora.preview.none"))
            return
        if e.preview_is_video:
            self.preview.setText(tr("lora.preview.video"))
            return
        self._thumb = _ThumbLoader(url)
        self._thumb.loaded.connect(self._set_thumb)
        self._thumb.start()

    def _set_thumb(self, data: bytes):
        if not data:
            self.preview.setText(tr("lora.preview.failed"))
            return
        pix = QPixmap()
        if pix.loadFromData(data):
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.setText(tr("lora.preview.unsupported"))

    # API per il controller
    def selected_loras(self) -> list[dict]:
        """Ritorna tutte le LoRA selezionate, incluse quelle nascoste dal filtro corrente."""
        out = []
        seen: set[str] = set()
        # LoRA visibili e selezionate
        for name, r in self._rows.items():
            if r["cb"].isChecked():
                e: LoraEntry = r["entry"]
                out.append({"name": e.comfy_name, "strength": r["sp"].value(), "_entry": e})
                seen.add(name)
        # LoRA selezionate ma nascoste dal filtro corrente (incluse nella generazione).
        # IMPORTANTE: salta i nomi che hanno una riga VISIBILE (`name in self._rows`): per quelle
        # lo stato vivo della checkbox è autoritativo (già gestito sopra). Senza questo controllo,
        # una riga deselezionata a mano rientrava qui perché `_persistent_selection` (aggiornato
        # solo da refresh(), non al toggle) conservava lo snapshot stantio chk=True → la LoRA
        # deselezionata veniva ri-aggiunta e usata davvero nella generazione.
        for name, (chk, strength, entry) in self._persistent_selection.items():
            if chk and name not in seen and name not in self._rows and entry is not None:
                out.append({"name": entry.comfy_name, "strength": strength, "_entry": entry})
        return out

    # ---------- persistenza (gui_state + preset) ----------
    def gui_get_state(self) -> dict:
        """Stato serializzabile: LoRA selezionate (comfy_name + strength) + auto-trigger.
        Usato da core.gui_state (riapertura) e dai preset video."""
        return {
            "loras": [{"name": s["name"], "strength": s["strength"]} for s in self.selected_loras()],
            "auto_triggers": self.auto_triggers.isChecked(),
        }

    def gui_set_state(self, st) -> None:
        """Ripristina la selezione LoRA (restore AUTORITATIVO). Robusto all'inventario asincrono:
        le LoRA non ancora presenti restano in `_restore_pending` e vengono applicate al caricamento."""
        if not isinstance(st, dict):
            return
        if "auto_triggers" in st:
            self.auto_triggers.setChecked(bool(st["auto_triggers"]))
        want = {d["name"]: float(d.get("strength", 1.0)) for d in st.get("loras", [])
                if isinstance(d, dict) and d.get("name")}
        # righe visibili: seleziona le volute, deseleziona le altre (restore autoritativo)
        for r in self._rows.values():
            cn = r["entry"].comfy_name
            if cn in want:
                r["cb"].setChecked(True); r["sp"].setValue(want[cn])
            else:
                r["cb"].setChecked(False)
        # togli dalla selezione persistente (LoRA nascoste dal filtro) quelle non volute
        self._persistent_selection = {n: v for n, v in self._persistent_selection.items()
                                      if v[2] is not None and v[2].comfy_name in want}
        # volute non ancora presenti → applicate quando l'inventario le carica
        self._restore_pending = want or None
        self.refresh()

    def auto_insert_triggers(self) -> bool:
        return self.auto_triggers.isChecked()

    def missing_trigger_words(self, prompt: str) -> list[str]:
        """Trigger word (prima per LoRA) mancanti dal prompt, per inserimento automatico."""
        out = []
        for sel in self.selected_loras():
            e = sel["_entry"]
            if e.triggers and e.triggers[0].lower() not in prompt.lower():
                out.append(e.triggers[0])
        return out

    def trigger_warnings(self, prompt: str) -> list[str]:
        """Per le LoRA selezionate con trigger reali assenti dal prompt, ritorna messaggi."""
        warns = []
        for sel in self.selected_loras():
            e: LoraEntry = sel["_entry"]
            miss = self.catalog.missing_triggers_in_prompt(e, prompt)
            if miss:
                warns.append(tr("lora.trigger.warn").format(name=e.name, triggers=', '.join(miss)))
        return warns
