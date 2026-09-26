"""CA di sistema per la verifica TLS.

Python (e quindi aiohttp/requests/huggingface_hub) verifica i certificati contro il bundle di
`certifi`. Su Windows il traffico passa spesso da un intermediario TLS — antivirus, firewall,
proxy aziendale — la cui CA è installata SOLO nello store di Windows: certifi non la conosce e
ogni connessione muore con `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.
Nel binario compilato è la norma, perché certifi viene impacchettato insieme all'app.

`truststore` rimpiazza la verifica di Python con quella nativa del sistema operativo (store di
Windows, Keychain su macOS, ca-certificates su Linux), che è anche il comportamento che l'utente
si aspetta: quello del browser.

I SOTTOPROCESSI non ereditano l'iniezione di truststore: il downloader dei pesi (lanciato dalla GUI
come QProcess), l'API e soprattutto il pip di un venv appena creato (senza truststore) fallivano dietro
l'antivirus (visto il 26/09/2026: «Avast Web/Mail Shield Root»). Per loro si genera `data/ca_bundle.pem`
= certifi + certificati ROOT/CA dello store di Windows, esportato in SSL_CERT_FILE / REQUESTS_CA_BUNDLE /
PIP_CERT. La verifica resta sempre attiva: si aggiungono solo le CA di cui il sistema già si fida.
"""
from __future__ import annotations

import os
import ssl
import sys
import time
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1] / "data" / "ca_bundle.pem"
_BUNDLE_MAX_AGE_S = 7 * 24 * 3600
_ENV_KEYS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT")
_SERVER_AUTH = "1.3.6.1.5.5.7.3.1"


def _windows_pems() -> list[str]:
    out: list[str] = []
    if sys.platform != "win32":
        return out
    for store in ("ROOT", "CA"):
        try:
            for cert, enc, trust in ssl.enum_certificates(store):
                if enc == "x509_asn" and (trust is True or (trust and _SERVER_AUTH in trust)):
                    out.append(ssl.DER_cert_to_PEM_cert(cert))
        except Exception:                                    # noqa: BLE001
            continue
    return out


def ensure_bundle(force: bool = False) -> str:
    """Bundle PEM combinato (certifi + store di Windows), rigenerato se più vecchio di 7 giorni.
    Ritorna il path, o "" se non è stato possibile crearlo."""
    try:
        if not force and BUNDLE.is_file() and time.time() - BUNDLE.stat().st_mtime < _BUNDLE_MAX_AGE_S:
            return str(BUNDLE)
        parts: list[str] = []
        try:
            import certifi
            parts.append(Path(certifi.where()).read_text("ascii", errors="ignore"))
        except Exception:                                    # noqa: BLE001
            pass
        parts.extend(_windows_pems())
        if not parts:
            return ""
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        tmp = BUNDLE.with_suffix(".tmp")
        tmp.write_text("\n".join(parts), "ascii", errors="ignore")
        os.replace(tmp, BUNDLE)
        return str(BUNDLE)
    except Exception:                                        # noqa: BLE001
        return str(BUNDLE) if BUNDLE.is_file() else ""


def child_env() -> dict:
    """Variabili per i sottoprocessi che non importano questo modulo (es. pip di un venv nuovo)."""
    b = ensure_bundle()
    return {k: b for k in _ENV_KEYS} if b else {}


def use_system_ca(export_env: bool = True, inject: bool = True) -> bool:
    """Instrada la verifica TLS sullo store di certificati del sistema. Ritorna True se truststore è attivo.

    Va chiamata all'avvio, prima di aprire connessioni. Se `truststore` manca non è fatale: si
    resta su certifi (comportamento precedente), che funziona su una rete senza intermediari.
    Con export_env=True esporta anche il bundle combinato per i sottoprocessi (senza sovrascrivere
    variabili già impostate dall'utente). inject=False = SOLO bundle, niente truststore: nel downloader
    dei pesi, dove `urllib3-future` (se installato al posto di urllib3) con truststore iniettato segnalava
    le connessioni come «Unverified HTTPS request»; col bundle la verifica è quella standard di OpenSSL."""
    if export_env:
        for k, v in child_env().items():
            os.environ.setdefault(k, v)
    if not inject:
        return False
    try:
        import truststore
        truststore.inject_into_ssl()
        return True
    except ImportError:
        return False
    except Exception as e:                                   # noqa: BLE001 — mai bloccare l'avvio
        print(f"[tls] store di sistema non attivabile ({type(e).__name__}: {e}) → si usa certifi")
        return False
