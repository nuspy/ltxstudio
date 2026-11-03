"""Finestra NON modale «Come ottenere SAM 3.1»: avviso (Meta approva in alcune ore, nel frattempo SAM2), cosa fa e quando sceglierlo, passi per chiedere l'accesso
(link, login Hugging Face, token), «Ho chiesto l'accesso» (da quel momento l'app controlla da sola ogni 30 minuti e scarica), rimanda / annulla, la copia
Comfy-Org non gated come alternativa, e il pulsante per tornare a SAM2.

Si apre dal menu «Segmentazione» quando SAM 3.1 non e utilizzabile (non installato, download incompleto, caricatore non pronto) e quando un lavoro
scelto con SAM 3.1 non puo partire. Mai un ripiego silenzioso su SAM2: la finestra dice cosa fare e l'utente decide.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from core import sam31_access as AC
from core import segmentation_engine as SE
from ui.i18n_manager import tr

_open: list = []
TOKENS_URL = "https://huggingface.co/settings/tokens"


def intro_text(status: str) -> str:
    return {
        "missing": tr("seg.help.intro_missing", "SAM 3.1 non e installato. Nel frattempo puoi usare SAM2 (e gia pronto e funziona ovunque)."),
        "partial": tr("seg.help.intro_partial", "Il download di SAM 3.1 e incompleto (file mancanti o troncati). Riprendilo dal tab «Pesi modelli» o usa SAM2."),
        "loader": tr("seg.help.intro_loader", "SAM 3.1 e installato, ma il caricatore nativo non e ancora pronto: per ora usa SAM2."),
        "ok": tr("seg.help.intro_ok", "SAM 3.1 e pronto."),
    }.get(status, "")


def warning_text() -> str:
    return tr("sam31.warn", "Attenzione: Meta deve approvare la tua richiesta e di solito servono alcune ore. Nel frattempo usa SAM2: e gia pronto e funziona ovunque.")


def about_text() -> str:
    return "\n".join([
        tr("sam31.help.what", "SAM 3.1 (Meta) trova gli oggetti da una descrizione scritta («persona», «auto rossa») e ne disegna la maschera, anche per piu oggetti "
                              "insieme. Nei video li segue nel tempo, tutti insieme."),
        tr("sam31.help.choose_sam31", "Scegli SAM 3.1 quando selezioni per nome: selezione da testo, «Trova persone», tracking di un oggetto descritto a parole."),
        tr("sam31.help.choose_sam2", "Scegli SAM2 per clic, lazo e box sulla foto e quando la memoria della scheda e poca: e piu leggero e funziona ovunque."),
    ])


def steps_text() -> str:
    return "\n".join([
        tr("sam31.steps_title", "Come chiedere l'accesso, passo per passo"),
        tr("sam31.step1", "1. Accedi a Hugging Face (o crea un account gratuito): huggingface.co."),
        tr("sam31.step2", "2. Apri la pagina del modello {url}, compila il modulo e accetta la licenza SAM di Meta (pulsante «Submit»).").format(url=SE.SAM31_URL),
        tr("sam31.step3", "3. Crea un token di accesso (tipo «Read») su {tok} e incollalo qui sotto: resta solo su questo PC.").format(tok=TOKENS_URL),
        tr("sam31.step4", "4. Premi «Ho chiesto l'accesso»: l'app controlla da sola ogni 30 minuti e, appena Meta approva, scarica SAM 3.1 in background "
                          "(circa 3,5 GB) senza bloccare nulla. Puoi rimandare o annullare il controllo quando vuoi."),
    ])


def build_text(status: str) -> str:
    """Testo completo (traducibile) per lo stato: 'missing' | 'partial' | 'loader' | 'ok'."""
    lic = tr("seg.help.license",
             "Licenza SAM (Meta, 19/11/2025): uso commerciale consentito. Vietati gli usi militari, nucleari, di spionaggio e il reverse engineering. "
             "Il testo completo e nel file LICENSE della cartella del modello e su {url}.").format(url=SE.SAM31_URL)
    parts = [intro_text(status)]
    if status in ("missing", "partial"):
        parts += [warning_text(), about_text(), steps_text()]
    else:
        parts.append(about_text())
    parts.append(lic)
    return "\n\n".join(p for p in parts if p)


def watch_text(now: float | None = None) -> str:
    """Riga di stato del controllo automatico."""
    if not AC.watching():
        return tr("sam31.watch_none", "Controllo automatico non attivo.")
    res = AC.last_result()
    res_txt = tr(f"sam31.res.{res}", AC.describe(res)) if res else tr("sam31.res.none", "non ancora controllato")
    nxt = AC.next_check_at()
    when = time.strftime("%H:%M", time.localtime(max(nxt, now or 0)))
    return tr("sam31.watch_active", "Controllo automatico attivo. Ultimo esito: {res} Prossimo controllo: {when}.").format(res=res_txt, when=when)


def comfy_offer_visible(status: str) -> bool:
    """L'alternativa Comfy-Org si offre a chi non ha i pesi di Meta, finche il confronto non l'ha dichiarata NON equivalente."""
    return status in ("missing", "partial") and SE.COMFY_EQUIVALENT is not False and SE.comfy_status()[0] != "ok"


def show(parent, status: str | None = None, on_use_sam2=None, watcher=None) -> QDialog:
    """`watcher` = ui.sam31_watch.Sam31Watcher (opzionale): senza, la finestra mostra solo la guida."""
    st = status
    if st is None:
        s = SE.weights_state()
        st = "ok" if (s == "ok" and SE.LOADER_READY) else ("loader" if s == "ok" else s)
    dlg = QDialog(parent)
    dlg.setWindowTitle(tr("seg.help.title", "SAM 3.1 — come ottenerlo"))
    dlg.setModal(False)
    dlg.setAttribute(Qt.WA_DeleteOnClose, True)
    dlg.setMinimumWidth(560)
    lay = QVBoxLayout(dlg)
    lbl = QLabel(build_text(st))
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lay.addWidget(lbl)
    dlg.text_label = lbl

    need_access = st in ("missing", "partial")
    dlg.token_edit = None
    if need_access:
        row_t = QHBoxLayout()
        dlg.token_edit = QLineEdit()
        dlg.token_edit.setEchoMode(QLineEdit.Password)
        dlg.token_edit.setPlaceholderText(tr("sam31.token_ph", "hf_… (resta solo su questo PC)"))
        dlg.token_edit.setAccessibleName(tr("sam31.token_lbl", "Token Hugging Face"))
        dlg.token_edit.setMinimumHeight(44)
        b_save = QPushButton(tr("sam31.token_save", "Salva token"))
        b_save.setMinimumHeight(44)
        row_t.addWidget(QLabel(tr("sam31.token_lbl", "Token Hugging Face")))
        row_t.addWidget(dlg.token_edit, 1)
        row_t.addWidget(b_save)
        lay.addLayout(row_t)
        dlg.token_state = QLabel(tr("sam31.token_present", "Un token e gia salvato su questo PC.") if AC.token() else "")
        lay.addWidget(dlg.token_state)

        def _save():
            AC.save_token(dlg.token_edit.text())
            dlg.token_edit.clear()
            dlg.token_state.setText(tr("sam31.token_saved", "Token salvato.") if AC.token() else "")
            if watcher:
                watcher.refresh()
        b_save.clicked.connect(_save)

    dlg.watch_label = QLabel(watch_text())
    dlg.watch_label.setWordWrap(True)
    lay.addWidget(dlg.watch_label)

    def _refresh_watch():
        try:
            dlg.watch_label.setText(watch_text())
        except RuntimeError:                                  # finestra gia chiusa e distrutta
            pass

    row = QHBoxLayout()
    b_page = QPushButton(tr("sam31.btn_page", "Apri la pagina del modello"))
    b_page.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(SE.SAM31_URL)))
    b_tok = QPushButton(tr("sam31.btn_tokens", "Apri la pagina dei token"))
    b_tok.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(TOKENS_URL)))
    for b in (b_page, b_tok):
        b.setMinimumHeight(44)
        row.addWidget(b)
    lay.addLayout(row)

    row2 = QHBoxLayout()
    dlg.btn_requested = QPushButton(tr("sam31.btn_requested", "Ho chiesto l'accesso"))
    dlg.btn_check = QPushButton(tr("sam31.btn_check_now", "Controlla ora"))
    dlg.btn_snooze = QPushButton(tr("sam31.btn_snooze", "Rimanda di 2 ore"))
    dlg.btn_cancel = QPushButton(tr("sam31.btn_cancel", "Annulla il controllo"))
    for b in (dlg.btn_requested, dlg.btn_check, dlg.btn_snooze, dlg.btn_cancel):
        b.setMinimumHeight(44)
        row2.addWidget(b)
    lay.addLayout(row2)

    # alternativa non gated: la copia Comfy-Org (stesso modello, fp16), col normale download del catalogo
    dlg.btn_comfy = QPushButton(tr("sam31.btn_comfy", "Oppure scarica la copia Comfy-Org (non gated, 1,75 GB)"))
    dlg.btn_comfy.setMinimumHeight(44)
    dlg.btn_comfy.setVisible(bool(watcher) and comfy_offer_visible(st))
    if watcher:
        dlg.btn_comfy.clicked.connect(lambda: watcher.download_comfy())
    lay.addWidget(dlg.btn_comfy)

    def _sync_buttons():
        try:
            w = AC.watching()
            dlg.btn_requested.setVisible(bool(watcher) and need_access and not w)
            for b in (dlg.btn_check, dlg.btn_snooze, dlg.btn_cancel):
                b.setVisible(bool(watcher) and need_access and w)
            dlg.btn_comfy.setVisible(bool(watcher) and comfy_offer_visible(SE.weights_state()))
        except RuntimeError:                                  # finestra gia chiusa e distrutta
            return
        _refresh_watch()
    if watcher:
        dlg.btn_requested.clicked.connect(lambda: (watcher.confirm_requested(), _sync_buttons()))
        dlg.btn_check.clicked.connect(lambda: watcher.check_now())
        dlg.btn_snooze.clicked.connect(lambda: (watcher.snooze(), _sync_buttons()))
        dlg.btn_cancel.clicked.connect(lambda: (watcher.cancel(), _sync_buttons()))
        watcher.changed.connect(_sync_buttons)
    else:
        for b in (dlg.btn_requested, dlg.btn_check, dlg.btn_snooze, dlg.btn_cancel):
            b.setVisible(False)
    _sync_buttons()

    row3 = QHBoxLayout()
    b_sam2 = QPushButton(tr("seg.help.btn_use_sam2", "Usa SAM2 ora"))
    b_sam2.setMinimumHeight(44)

    def _use_sam2():
        SE.set_current(SE.SAM2)
        if on_use_sam2:
            on_use_sam2()
        dlg.close()
    b_sam2.clicked.connect(_use_sam2)
    b_close = QPushButton(tr("seg.help.btn_close", "Chiudi"))
    b_close.setMinimumHeight(44)
    b_close.clicked.connect(dlg.close)
    for b in (b_sam2, b_close):
        row3.addWidget(b)
    lay.addLayout(row3)
    dlg.destroyed.connect(lambda *_: _open.remove(dlg) if dlg in _open else None)
    _open.append(dlg)
    dlg.show()
    return dlg
