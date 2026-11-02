"""Messaggio d'errore COMPLETO di un worker lanciato come sottoprocesso (senza Qt, senza torch).

Prima la GUI mostrava gli ultimi 200 caratteri di stdout + gli ultimi 300 di stderr e poi tagliava a 300: della causa restava
«re\\src\\ltx_core\\utils.py … No files mat» (02/11, inpaint video con GGUF LTX-2.5). Qui si tiene la riga dell'eccezione
INTERA (che dice quale file cercava e dove) e la coda del traceback.
"""
from __future__ import annotations

import re

_EXC_LINE = re.compile(r"^(?:[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)|RuntimeError|AssertionError)\b.*")


def explain(stdout: str | None, stderr: str | None, *, limit: int = 4000, return_code: int | None = None) -> str:
    """Testo per l'utente: prima la CAUSA (ultima riga d'eccezione, intera), poi la coda del log (traceback compreso).

    `limit` = tetto dei caratteri della coda (non della causa, che resta sempre intera)."""
    err = (stderr or "").strip()
    out = (stdout or "").strip()
    text = err if "Traceback" in err or not out else (out + "\n" + err).strip()
    if "Traceback" not in text and "Traceback" in out:
        text = out
    lines = [ln.rstrip() for ln in text.splitlines()]
    cause = ""
    for ln in reversed(lines):
        s = ln.strip()
        if s and _EXC_LINE.match(s):
            cause = s
            break
    if not cause:
        for ln in reversed(lines):
            if ln.strip():
                cause = ln.strip()
                break
    i = text.rfind("Traceback (most recent call last)")
    tail = text[i:] if i >= 0 else text
    if len(tail) > limit:
        tail = "…" + tail[-limit:]
    head = cause or (f"il worker e terminato senza output (codice {return_code})" if return_code is not None else "errore sconosciuto")
    if tail.strip().endswith(head) and len(tail.strip()) <= len(head) + 2:
        return head
    return head + "\n\n" + tail
