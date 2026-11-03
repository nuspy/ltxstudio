"""Accesso a SAM 3.1 (repo gated facebook/sam3.1): promemoria «ho chiesto l'accesso», controllo leggero ogni 30 minuti, esito.

Logica SENZA Qt (testabile): lo stato sta in user_prefs, il controllo e UNA richiesta HEAD sul file `config.json` del repo col token HF dell'utente
(nessun download: HEAD non scarica nulla). Lo schedulatore e il download in background stanno in ui/sam31_watch.py.

Esiti di `probe`:
  granted   accesso approvato (HEAD risponde 200/3xx)            -> si puo scaricare
  pending   403: richiesta inviata ma non ancora approvata da Meta (o non ancora inviata)
  no_token  nessun token HF salvato                              -> serve il token (passo 3 della guida)
  bad_token 401 con un token                                     -> token non valido o scaduto
  offline   rete assente / timeout                               -> si riprova al prossimo giro
  error     qualunque altra risposta                             -> si riprova al prossimo giro
Il token non finisce mai in un log, in un messaggio d'errore o in un file del repo.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request

from core import segmentation_engine as SE

CHECK_INTERVAL_S = 30 * 60                 # controllo automatico ogni 30 minuti
SNOOZE_DEFAULT_S = 2 * 3600                # «Rimanda»: 2 ore
PROBE_FILE = "config.json"
PROBE_URL = f"https://huggingface.co/{SE.SAM31_REPO}/resolve/main/{PROBE_FILE}"

K_REQUESTED = "sam31_access_requested_at"      # epoch di quando l'utente ha confermato di aver chiesto l'accesso (0 = no)
K_CANCELLED = "sam31_watch_cancelled"          # True = l'utente ha annullato il controllo automatico
K_SNOOZE = "sam31_watch_snooze_until"          # epoch fino a cui non controllare
K_LAST = "sam31_watch_last_check"              # epoch dell'ultimo controllo
K_RESULT = "sam31_watch_last_result"           # ultimo esito di probe


def _prefs():
    from core import user_prefs
    return user_prefs


def _get(key, default=0):
    try:
        return _prefs().get(key, default)
    except Exception:                                        # noqa: BLE001
        return default


def _set(key, value) -> None:
    try:
        _prefs().set(key, value)
    except Exception:                                        # noqa: BLE001
        pass


def token() -> str | None:
    """Token HF dell'utente (env -> user_prefs -> `hf auth login`). Mai stampato."""
    try:
        from core import downloader
        return downloader.resolve_token()
    except Exception:                                        # noqa: BLE001
        return None


def save_token(value: str) -> None:
    """Salva il token SOLO in locale (data/user_prefs.json), come la chiave Civitai. Vuoto = lo toglie."""
    _set("hf_token", (value or "").strip())


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):                     # noqa: D401
        return None


def _head(url: str, headers: dict, timeout: float = 15.0) -> int | None:
    """Stato HTTP di una HEAD senza seguire i redirect (un 302 verso il CDN = accesso concesso). None = rete assente."""
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    op = urllib.request.build_opener(_NoRedirect)
    try:
        with op.open(req, timeout=timeout) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:                                        # noqa: BLE001 - DNS, timeout, TLS: non e un verdetto
        return None


def probe(tok: str | None = None, *, head=_head) -> tuple[str, str]:
    """(esito, dettaglio breve senza dati sensibili). `head(url, headers) -> int|None` iniettabile nei test."""
    tok = (tok if tok is not None else token()) or ""
    if not tok:
        return "no_token", "nessun token Hugging Face salvato"
    code = head(PROBE_URL, {"Authorization": f"Bearer {tok}", "User-Agent": "LTXStudio-sam31-check"})
    if code is None:
        return "offline", "rete non raggiungibile"
    if 200 <= code < 400:
        return "granted", f"HTTP {code}"
    if code == 403:
        return "pending", "HTTP 403: richiesta in attesa di approvazione da Meta"
    if code == 401:
        return "bad_token", "HTTP 401: token non valido"
    return "error", f"HTTP {code}"


# ── stato del promemoria ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def requested() -> bool:
    return float(_get(K_REQUESTED, 0) or 0) > 0


def cancelled() -> bool:
    return bool(_get(K_CANCELLED, False))


def confirm_requested(now: float | None = None) -> None:
    """L'utente dice «ho chiesto l'accesso»: parte il controllo (ogni 30 minuti) e compare il bottone «Controlla e scarica SAM 3.1»."""
    _set(K_REQUESTED, float(now if now is not None else time.time()))
    _set(K_CANCELLED, False)
    _set(K_SNOOZE, 0)


def cancel_watch() -> None:
    """«Annulla»: niente piu controlli automatici e il bottone sparisce (si riattiva dalla guida con «Ho chiesto l'accesso»)."""
    _set(K_CANCELLED, True)


def snooze(seconds: float = SNOOZE_DEFAULT_S, now: float | None = None) -> float:
    """«Rimanda»: nessun controllo automatico fino a `now + seconds`. Ritorna l'epoch di ripresa."""
    until = float(now if now is not None else time.time()) + float(seconds)
    _set(K_SNOOZE, until)
    return until


def snoozed_until() -> float:
    return float(_get(K_SNOOZE, 0) or 0)


def watching(base=None) -> bool:
    """Il controllo e attivo: richiesta confermata, non annullata, pesi (di Meta o copia Comfy equivalente) non ancora utilizzabili."""
    return requested() and not cancelled() and SE.weights_state(base) != "ok"


def button_visible(base=None) -> bool:
    """Il bottone «Controlla e scarica SAM 3.1» compare da solo dopo la conferma e sparisce a pesi completi o dopo «Annulla»."""
    return watching(base)


def due(now: float | None = None, base=None) -> bool:
    """E ora del controllo automatico? Sorvegliato, non rimandato, e passati 30 minuti dall'ultimo controllo."""
    now = float(now if now is not None else time.time())
    if not watching(base):
        return False
    if now < snoozed_until():
        return False
    return now - float(_get(K_LAST, 0) or 0) >= CHECK_INTERVAL_S


def next_check_at() -> float:
    return max(snoozed_until(), float(_get(K_LAST, 0) or 0) + CHECK_INTERVAL_S)


def record(result: str, now: float | None = None) -> None:
    _set(K_LAST, float(now if now is not None else time.time()))
    _set(K_RESULT, result)


def last_result() -> str:
    return str(_get(K_RESULT, "") or "")


def describe(result: str) -> str:
    """Testo (italiano, la GUI lo traduce con la chiave `sam31.res.<esito>`) per l'esito del controllo."""
    return {
        "granted": "Accesso approvato: scarico SAM 3.1.",
        "pending": "Meta non ha ancora approvato la richiesta (di solito servono alcune ore). Riprovo fra 30 minuti.",
        "no_token": "Manca il token Hugging Face: incollalo nella guida «Come ottenere SAM 3.1».",
        "bad_token": "Il token Hugging Face non e valido o e scaduto: creane uno nuovo (tipo Read).",
        "offline": "Rete non raggiungibile: riprovo fra 30 minuti.",
        "error": "Risposta inattesa da Hugging Face: riprovo fra 30 minuti.",
    }.get(result, "")
