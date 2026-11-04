"""Connessioni TCP locali affidabili su Windows (difetto trovato il 03-04/11).

Su questo PC un filtro di rete locale (WFP) scarta le connessioni TCP in loopback la cui porta SORGENTE ha
`(porta & 0x0F00) == 0x0500` (una porta effimera su sedici). La `connect` non fallisce: il SYN sparisce e il chiamante resta in attesa.
Conseguenze viste:
  - asyncio crea la self-pipe con un socketpair emulato (listen + connect non bloccante + accept): se la connect si perde, `accept` aspetta
    per sempre e l'InferenceServer non apre mai la porta 8189 (bloccato 3 volte: turno 30 alle 22:21, turno 16a alle 00:16 e alle 01:49);
  - i client locali (app -> :8189, driver di test -> :8765, API) restano in SYN_SENT fino al timeout.

Rimedio: non fidarsi della porta scelta dal sistema. Le connessioni verso 127.0.0.1 si fanno da una porta sorgente «buona» scelta qui,
con nuovo tentativo se e occupata; il socketpair di asyncio e sostituito da una versione con timeout e nuovo tentativo.
`install()` e idempotente e non fa nulla fuori da Windows.
"""
from __future__ import annotations

import errno
import random
import socket
import sys

LOOPBACK = "127.0.0.1"
_PORTA_MIN, _PORTA_MAX = 49152, 65535            # intervallo delle porte dinamiche di Windows
_ERRORI_PORTA_OCCUPATA = {errno.EADDRINUSE, errno.EACCES, 10048, 10013}      # WSAEADDRINUSE, WSAEACCES
_TENTATIVI_PORTA = 30
_TIMEOUT_COPPIA_S = 3.0
_originale: dict = {}


def is_bad_source_port(porta: int) -> bool:
    """True se il filtro locale scarta le connessioni in uscita da questa porta."""
    return (porta & 0x0F00) == 0x0500


def pick_source_port(rng: random.Random | None = None) -> int:
    """Una porta sorgente dinamica che il filtro non scarta (non verifica che sia libera: ci pensa il nuovo tentativo)."""
    generatore = rng or random
    while True:
        porta = generatore.randint(_PORTA_MIN, _PORTA_MAX)
        if not is_bad_source_port(porta):
            return porta


def _is_loopback(address) -> bool:
    return isinstance(address, tuple) and len(address) >= 2 and address[0] == LOOPBACK


def create_connection_from_good_port(address, *args, **kwargs):
    """`socket.create_connection` che per le destinazioni 127.0.0.1 parte da una porta sorgente buona.
    Se il chiamante indica gia un `source_address` (anche posizionale: http.client lo passa come None), lo rispetta."""
    originale = _originale["create_connection"]
    timeout_args = args[:1]                                  # (timeout,) oppure ()
    sorgente_data = kwargs.get("source_address") if "source_address" in kwargs else (args[1] if len(args) >= 2 else None)
    if sorgente_data is not None or not _is_loopback(address):
        return originale(address, *args, **kwargs)
    extra = {key: value for key, value in kwargs.items() if key != "source_address"}
    ultimo_errore: OSError | None = None
    for _ in range(_TENTATIVI_PORTA):
        try:
            return originale(address, *timeout_args, source_address=(LOOPBACK, pick_source_port()), **extra)
        except OSError as errore:
            if errore.errno not in _ERRORI_PORTA_OCCUPATA:
                raise
            ultimo_errore = errore                   # porta sorgente occupata: se ne prova un'altra
    raise ultimo_errore if ultimo_errore else OSError("nessuna porta sorgente libera")


def socketpair_with_retry(attempts: int = _TENTATIVI_PORTA, timeout: float = _TIMEOUT_COPPIA_S):
    """Sostituto di `socket.socketpair()` (su Windows asyncio usa quello emulato): connessione da una porta sorgente buona, con
    timeout su connect e accept e nuovo tentativo. Ritorna (socket del server, socket del client) come l'originale."""
    ultimo_errore: Exception | None = None
    for _ in range(attempts):
        ascolto = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            ascolto.bind((LOOPBACK, 0))
            ascolto.listen(1)
            client.bind((LOOPBACK, pick_source_port()))
            client.settimeout(timeout)
            client.connect(ascolto.getsockname())
            ascolto.settimeout(timeout)
            server, _ = ascolto.accept()
            server.setblocking(True)
            client.setblocking(True)
            return server, client
        except (OSError, socket.timeout) as errore:      # porta occupata o SYN perso: coppia nuova
            ultimo_errore = errore
            client.close()
        finally:
            ascolto.close()
    raise OSError(f"socketpair: nessuna coppia riuscita in {attempts} tentativi ({ultimo_errore!r})")


def install() -> bool:
    """Attiva i due rimedi nel processo corrente. Ritorna True se ha installato qualcosa (solo Windows)."""
    if sys.platform != "win32" or _originale:
        return False
    _originale["create_connection"] = socket.create_connection
    _originale["socketpair"] = socket.socketpair
    socket.create_connection = create_connection_from_good_port
    socket.socketpair = socketpair_with_retry
    return True


def uninstall() -> None:
    """Ripristina le funzioni originali (per i test)."""
    if _originale:
        socket.create_connection = _originale.pop("create_connection")
        socket.socketpair = _originale.pop("socketpair")
