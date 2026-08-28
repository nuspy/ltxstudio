"""CA di sistema per la verifica TLS.

Python (e quindi aiohttp/requests/huggingface_hub) verifica i certificati contro il bundle di
`certifi`. Su Windows il traffico passa spesso da un intermediario TLS — antivirus, firewall,
proxy aziendale — la cui CA è installata SOLO nello store di Windows: certifi non la conosce e
ogni connessione muore con `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.
Nel binario compilato è la norma, perché certifi viene impacchettato insieme all'app.

`truststore` rimpiazza la verifica di Python con quella nativa del sistema operativo (store di
Windows, Keychain su macOS, ca-certificates su Linux), che è anche il comportamento che l'utente
si aspetta: quello del browser.
"""
from __future__ import annotations


def use_system_ca() -> bool:
    """Instrada la verifica TLS sullo store di certificati del sistema. Ritorna True se attivo.

    Va chiamata all'avvio, prima di aprire connessioni. Se `truststore` manca non è fatale: si
    resta su certifi (comportamento precedente), che funziona su una rete senza intermediari."""
    try:
        import truststore
        truststore.inject_into_ssl()
        return True
    except ImportError:
        return False
    except Exception as e:                                   # noqa: BLE001 — mai bloccare l'avvio
        print(f"[tls] store di sistema non attivabile ({type(e).__name__}: {e}) → si usa certifi")
        return False
