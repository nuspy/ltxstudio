"""Motore di segmentazione scelto dall'utente: SAM2 (default, con GroundingDINO) oppure SAM 3.1 (Meta).

Vale OVUNQUE oggi si usa GroundingDINO per trovare un oggetto da TESTO: selezione da testo sulla foto, tracking in «Modifica Video AI» (oggetto descritto a
parole), «Trova persone» del tab «Scambia». Clic, lazo e box restano di SAM2 (il modello immagine di SAM 3.1 e guidato da testo e box, non da punti), e la
PROPAGAZIONE nel tempo resta di SAM2: SAM 3.1 trova l'oggetto da testo sul primo fotogramma al posto di GroundingDINO, SAM2 lo segue.
La scelta e ricordata (user_prefs `segmentation_engine`). Nessuna scelta automatica: se l'utente sceglie SAM 3.1 e non e utilizzabile, il motore NON ripiega
di nascosto su SAM2 — l'interfaccia lo dice e l'utente sceglie.

Pesi (decisione dell'utente 03/11: niente repo gated di Meta, nessuna approvazione): copie NON gated, normale download del catalogo pesi.
  · unsloth/sam3.1 (id `sam31-unsloth`, cartella Segmentation/sam3.1): formato transformers `Sam3Model`, SOLO IMMAGINE (testo/box -> maschere, box, punteggi),
    BF16 1,68 GB; e quello che usa il programma (core/sam31_loader.py).
  · Comfy-Org/sam3.1 (id `sam31-comfy-multiplex`, cartella Segmentation/sam3.1_multiplex): single-file con il tracker video Object Multiplex. NON usato:
    transformers 5.17 non ha quel tracker (nessuna traccia di `multiplex` / `output_valid_embed` nei modelli sam3, sam3_video, sam3_tracker_video) e non converte
    i checkpoint Meta; farlo girare richiede il pacchetto `sam3` di Meta. Vedi MULTIPLEX_TRACKER_USABLE.
Licenza SAM (Meta, 19/11/2025): uso commerciale consentito; vietati usi militari, nucleari, di spionaggio e reverse engineering.
"""
from __future__ import annotations

import os
from pathlib import Path

SAM2 = "sam2"
SAM31 = "sam31"
ENGINES = (SAM2, SAM31)
DEFAULT = SAM2
PREF_KEY = "segmentation_engine"

SAM31_REPO = "unsloth/sam3.1"
SAM31_URL = "https://huggingface.co/unsloth/sam3.1"
SAM31_CATALOG_ID = "sam31-unsloth"
SAM31_DIRNAME = "sam3.1"
SAM31_CHECKPOINT = "model.safetensors"
# dimensioni ESATTE (byte) dei file del repo: un download a pezzi (.part) o troncato non conta come installato. None = basta che il file esista.
SAM31_FILES = {
    "model.safetensors": 1_680_935_372,          # BF16 1,68 GB
    "config.json": 6_517,
    "processor_config.json": 613,
    "tokenizer.json": 3_642_073,
    "tokenizer_config.json": 799,
    "vocab.json": 862_328,
    "merges.txt": 524_619,
    "special_tokens_map.json": 588,
    "LICENSE": 7_352,
}

# Tracker video Object Multiplex (Comfy-Org): scaricabile dal catalogo ma NON utilizzabile senza il pacchetto Meta (vedi docstring del modulo).
MULTIPLEX_CATALOG_ID = "sam31-comfy-multiplex"
MULTIPLEX_TRACKER_USABLE = False

# Il caricatore (transformers Sam3Model, core/sam31_loader.py) e provato su CPU con un fotogramma vero (03/11): SAM 3.1 si puo scegliere dal menu quando i file ci sono.
# Si accende SOLO dopo la prova su un fotogramma vero (nessuna promessa a vuoto).
LOADER_READY = True

LICENSE_SUMMARY_IT = (
    "Licenza SAM (Meta, 19/11/2025): uso commerciale consentito. Vietati gli usi militari, nucleari, di spionaggio e il reverse engineering. "
    "Il testo completo e nel file LICENSE della cartella del modello.")


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
    bad = [n for n, size in SAM31_FILES.items() if not (d / n).is_file() or (size and (d / n).stat().st_size != size)]
    if not bad:
        return "ok", []
    anything = any((d / n).exists() for n in SAM31_FILES) or any(p.suffix in (".part", ".chunk") for p in d.glob("*"))
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
    """Si puo scegliere dal menu: SAM2 sempre; SAM 3.1 solo se i file ci sono E il caricatore e provato."""
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
    """Spiega un errore di download di SAM 3.1 (repo non gated: nessun account ne approvazione)."""
    t = (text or "").lower()
    if "401" in t or "403" in t or "gated" in t:
        return ("Il download di SAM 3.1 ha risposto 401/403: la copia unsloth/sam3.1 non e gated. Controlla la connessione e riprova dal tab «Pesi modelli»; "
                "nel frattempo usa SAM2.")
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
            ". Scaricalo dal tab «Pesi modelli» (voce «SAM 3.1 immagine», circa 1,8 GB, senza account ne approvazioni) oppure scegli SAM2 dal menu Segmentazione."
    if not LOADER_READY:
        return SAM31, "SAM 3.1 e installato ma il caricatore nativo non e ancora pronto: scegli SAM2 dal menu Segmentazione."
    return SAM31, ""
