"""Motore di segmentazione scelto dall'utente: SAM2 (default, con GroundingDINO) oppure SAM 3.1 (Meta, Object Multiplex).

Vale OVUNQUE oggi si usa SAM2: clic e lazo sulla foto, maschere dell'editing immagini, tracking in «Modifica Video AI», tab «Scambia».
La scelta e ricordata (user_prefs `segmentation_engine`). Nessuna scelta automatica: se l'utente sceglie SAM 3.1 e non e utilizzabile, il motore NON
ripiega di nascosto su SAM2 — l'interfaccia lo dice (finestra non modale con come ottenere l'accesso) e l'utente sceglie.

SAM 3.1 (facebook/sam3.1, SAM License del 19/11/2025, repo con accesso a richiesta): un checkpoint nativo `sam3.1_multiplex.pt` (3 502 755 717 byte,
da caricare con torch.load(weights_only=True)) + 9 file piccoli (config, tokenizer, LICENSE...). Chiavi: `detector.*` (1166, ~855 M parametri,
architettura SAM 3) e `tracker.model.*` (457, ~20 M, tracker Object Multiplex); nessuna integrazione transformers (README del repo).
"""
from __future__ import annotations

import os
from pathlib import Path

SAM2 = "sam2"
SAM31 = "sam31"
ENGINES = (SAM2, SAM31)
DEFAULT = SAM2
PREF_KEY = "segmentation_engine"

SAM31_REPO = "facebook/sam3.1"
SAM31_URL = "https://huggingface.co/facebook/sam3.1"
SAM31_CATALOG_ID = "sam3.1-multiplex"
SAM31_DIRNAME = "sam3.1"
SAM31_CHECKPOINT = "sam3.1_multiplex.pt"
# dimensioni ESATTE (byte) dei file del repo: un download a pezzi (.part) o troncato non conta come installato
SAM31_FILES = {
    "sam3.1_multiplex.pt": 3_502_755_717,
    "config.json": 25_843,
    "processor_config.json": 1_712,
    "tokenizer.json": 3_642_073,
    "tokenizer_config.json": 799,
    "vocab.json": 862_328,
    "merges.txt": 524_619,
    "special_tokens_map.json": 588,
    "LICENSE": 7_352,
}

# Il checkpoint e nel formato nativo Meta. Finche il caricatore nativo non e pronto e provato, SAM 3.1 si vede nel menu ma non e selezionabile.
# Si accende SOLO quando esiste un caricatore che ha superato la prova dalla GUI (nessuna promessa a vuoto).
LOADER_READY = False

LICENSE_SUMMARY_IT = (
    "Licenza SAM (Meta, 19/11/2025): uso commerciale consentito. Vietati gli usi militari, nucleari, di spionaggio e il reverse engineering. "
    "Il testo completo e nel file LICENSE della cartella del modello e su " + SAM31_URL + ".")


def models_dir() -> Path:
    try:
        from core import weights_catalog as wc
        for e in wc.load():
            if e.get("id") == SAM31_CATALOG_ID:
                return Path(wc.dest_for(e))
    except Exception:                                        # noqa: BLE001
        pass
    return Path(os.environ.get("LTX_MODELS_ROOT", r"E:\AI\Models")) / "Segmentation" / SAM31_DIRNAME


def sam31_status(base: Path | None = None) -> tuple[str, list[str]]:
    """('ok' | 'partial' | 'missing', file mancanti o di dimensione diversa). 'partial' = c'e qualcosa (o un .part) ma non e completo."""
    d = Path(base) if base else models_dir()
    if not d.is_dir():
        return "missing", list(SAM31_FILES)
    bad = [n for n, size in SAM31_FILES.items() if not (d / n).is_file() or (d / n).stat().st_size != size]
    if not bad:
        return "ok", []
    anything = any((d / n).exists() for n in SAM31_FILES) or any(p.suffix == ".part" for p in d.glob("*"))
    return ("partial" if anything else "missing"), bad


def is_installed(engine: str, base: Path | None = None) -> bool:
    if engine == SAM2:
        try:
            from inference_server.backends import image_segment as seg
            return os.path.isdir(seg.SAM2_DIR) and os.path.isdir(seg.GDINO_DIR)
        except Exception:                                    # noqa: BLE001
            return True                                      # il backend dara l'errore chiaro se manca
    return sam31_status(base)[0] == "ok"


def is_selectable(engine: str, base: Path | None = None) -> bool:
    """Si puo scegliere dal menu: SAM2 sempre; SAM 3.1 solo se i file ci sono E il caricatore nativo e pronto."""
    if engine == SAM2:
        return True
    return LOADER_READY and sam31_status(base)[0] == "ok"


def current() -> str:
    """Motore scelto (default SAM2). Un valore sconosciuto o SAM 3.1 non selezionabile ->  quello salvato resta, e` il chiamante a dire l'errore."""
    try:
        from core import user_prefs
        v = str(user_prefs.get(PREF_KEY, DEFAULT) or DEFAULT).lower()
    except Exception:                                        # noqa: BLE001
        v = DEFAULT
    return v if v in ENGINES else DEFAULT


def set_current(engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"motore di segmentazione sconosciuto: {engine!r}")
    from core import user_prefs
    user_prefs.set(PREF_KEY, engine)


def menu_entries(base: Path | None = None) -> list[dict]:
    """Voci del menu «Segmentazione»: id, testo italiano (la GUI lo traduce con la chiave), stato e se e selezionabile."""
    st, _bad = sam31_status(base)
    out = [{"id": SAM2, "key": "seg.engine.sam2", "label": "SAM2 + GroundingDINO (predefinito)", "installed": is_installed(SAM2, base),
            "selectable": True, "state": "ok"}]
    if st != "ok":
        state = "partial" if st == "partial" else "missing"
        label = "SAM 3.1 (non installato)" if st == "missing" else "SAM 3.1 (download incompleto)"
    elif not LOADER_READY:
        state, label = "loader", "SAM 3.1 (installato; caricatore nativo in preparazione)"
    else:
        state, label = "ok", "SAM 3.1 (Meta, multi-oggetto)"
    out.append({"id": SAM31, "key": "seg.engine.sam31", "label": label, "installed": st == "ok", "selectable": is_selectable(SAM31, base), "state": state})
    return out


def describe_error(text: str) -> str:
    """Spiega l'errore di accesso a SAM 3.1 (download del repo gated): 403 = richiesta in attesa di approvazione da Meta."""
    t = (text or "").lower()
    if "403" in t or "gated" in t or "access" in t and "sam3" in t:
        return ("Richiesta di accesso in attesa di approvazione da Meta: l'approvazione di facebook/sam3.1 puo richiedere giorni. "
                "Nel frattempo usa SAM2.")
    if "401" in t:
        return "Serve accedere a Hugging Face (token) e accettare la licenza SAM su " + SAM31_URL + "."
    return text


def resolve_for_job(engine: str | None = None, base: Path | None = None) -> tuple[str, str]:
    """(motore, errore). `errore` non vuoto = il lavoro NON parte e la GUI lo mostra; mai un ripiego silenzioso su SAM2."""
    eng = engine or current()
    if eng == SAM2:
        return SAM2, ""
    if eng != SAM31:
        return DEFAULT, f"motore di segmentazione sconosciuto: {eng!r}"
    st, bad = sam31_status(base)
    if st != "ok":
        return SAM31, ("SAM 3.1 non e installato" if st == "missing" else "Il download di SAM 3.1 e incompleto (" + ", ".join(bad[:3]) + ")") + \
            f". Per averlo: account Hugging Face, accetta la licenza SAM su {SAM31_URL} (approvazione di Meta, puo richiedere giorni). " \
            "Nel frattempo usa SAM2 dal menu Segmentazione."
    if not LOADER_READY:
        return SAM31, "SAM 3.1 e installato ma il caricatore nativo non e ancora pronto: scegli SAM2 dal menu Segmentazione."
    return SAM31, ""
