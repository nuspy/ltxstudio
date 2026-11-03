"""Motore di segmentazione scelto dall'utente: SAM2 (default, con GroundingDINO) oppure SAM 3.1 (Meta, Object Multiplex).

Vale OVUNQUE oggi si usa SAM2 / GroundingDINO per trovare un oggetto da TESTO: selezione da testo sulla foto, tracking in «Modifica Video AI» (oggetto descritto a
parole), «Trova persone» del tab «Scambia». Clic, lazo e box restano di SAM2 (il percorso immagine di SAM 3.1 e guidato da testo e box).
La scelta e ricordata (user_prefs `segmentation_engine`). Nessuna scelta automatica: se l'utente sceglie SAM 3.1 e non e utilizzabile, il motore NON ripiega
di nascosto su SAM2 — l'interfaccia lo dice (finestra non modale con come ottenere l'accesso) e l'utente sceglie.

SAM 3.1 (facebook/sam3.1, SAM License del 19/11/2025, repo con accesso a richiesta): un checkpoint nativo `sam3.1_multiplex.pt` (3 502 755 717 byte,
da caricare con torch.load(weights_only=True)) + 8 file piccoli (config, tokenizer, LICENSE...). Chiavi: `detector.*` (1166, ~855 M parametri,
architettura SAM 3) e `tracker.model.*` (457, ~20 M, tracker Object Multiplex); nessuna integrazione transformers (README del repo).

Verificato 03/11 (transformers 5.17): transformers NON converte il .pt di Meta e NON ha il tracker multiplex (nessuna traccia di `multiplex` /
`output_valid_embed` nei modelli sam3, sam3_video, sam3_tracker_video). L'unica strada completa (immagine + video multiplex) e il pacchetto `sam3` di Meta
(E:/AI/Tools/sam3_src, revisionato il 03/11), eseguito SOLO in un processo figlio con le garanzie del lanciatore E:/AI/Tools/sam3_run.sh (core/sam31_detect.py,
workers/sam3_jobs). Il percorso transformers/unsloth e stato scartato (solo immagine).

Copia alternativa NON gated: Comfy-Org/sam3.1 `checkpoints/sam3.1_multiplex_fp16.safetensors` (catalogo `sam31-comfy-multiplex`, cartella
Segmentation/sam3.1_multiplex): stesso modello in fp16. Si usa al posto del .pt di Meta SOLO se il confronto tensore per tensore la dichiara equivalente
(`COMFY_EQUIVALENT`, scripts/sam31_compare_checkpoints.py).
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

# Copia Comfy-Org (non gated): stesso modello in fp16, single-file.
COMFY_CATALOG_ID = "sam31-comfy-multiplex"
COMFY_DIRNAME = "sam3.1_multiplex"
COMFY_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
COMFY_FILES = {COMFY_CHECKPOINT: 1_745_546_848}
# None = confronto non ancora fatto; True/False = esito di scripts/sam31_compare_checkpoints.py. Solo True la rende usabile al posto del .pt di Meta.
# Esito 03/11 (scripts/sam31_compare_checkpoints.py + job sam31_compare_ckpts_image.py): 1590 tensori in comune con le stesse forme, errore massimo 6,7e-3 e
# medio 3,8e-6 (arrotondamento fp16, massimo relativo 4,8e-4), mancano solo i 32 freqs_cis (RoPE ricalcolata) e text_projection (non usata dal testo del modello);
# stesso fotogramma e stesso testo: IoU delle maschere >= 0,9998 e punteggi entro 0,0008. Il tracker multiplex non e verificabile finche il video non gira.
COMFY_EQUIVALENT: bool | None = True

# I percorsi sono eseguiti dal codice di Meta in un processo figlio (core/sam31_detect.py): si accendono SOLO dopo la prova (permesso al lanciatore sam3_run.sh).
# Provato 03/11 su CPU con un fotogramma vero (job sam31_image_text.py via sam3_run.sh): testo -> riquadri/maschere con Meta e con la copia Comfy. Nei video SAM 3.1
# trova l'oggetto da testo sul fotogramma 0 e SAM2 lo propaga (percorso provato). Il tracking multiplex nativo (sam31_video_text.py) trova gli oggetti sul fotogramma 0
# ma su CPU la propagazione fallisce (tensori di batch 0 nel tracker, codice di Meta pensato per GPU): VIDEO_READY resta spento finche non gira (serve GPU).
LOADER_READY = True                    # cancello della scelta nel menu
IMAGE_READY = True                     # testo -> maschere su una foto
VIDEO_READY = False                    # tracking multiplex nativo da testo

LICENSE_SUMMARY_IT = (
    "Licenza SAM (Meta, 19/11/2025): uso commerciale consentito. Vietati gli usi militari, nucleari, di spionaggio e il reverse engineering. "
    "Il testo completo e nel file LICENSE della cartella del modello e su " + SAM31_URL + ".")


def _entry_dir(entry_id: str, dirname: str) -> Path:
    try:
        from core import weights_catalog as wc
        for e in wc.load():
            if e.get("id") == entry_id:
                return Path(wc.dest_for(e))
    except Exception:                                        # noqa: BLE001
        pass
    return Path(os.environ.get("LTX_MODELS_ROOT", r"E:\AI\Models")) / "Segmentation" / dirname


def models_dir() -> Path:
    return _entry_dir(SAM31_CATALOG_ID, SAM31_DIRNAME)


def comfy_dir() -> Path:
    return _entry_dir(COMFY_CATALOG_ID, COMFY_DIRNAME)


def _status(files: dict, d: Path) -> tuple[str, list[str]]:
    if not d.is_dir():
        return "missing", list(files)
    bad = [n for n, size in files.items() if not (d / n).is_file() or (size and (d / n).stat().st_size != size)]
    if not bad:
        return "ok", []
    anything = any((d / n).exists() for n in files) or any(p.suffix in (".part", ".chunk") for p in d.glob("*"))
    return ("partial" if anything else "missing"), bad


def sam31_status(base: Path | None = None) -> tuple[str, list[str]]:
    """('ok' | 'partial' | 'missing', file mancanti o di dimensione diversa) per i pesi di META. 'partial' = c'e qualcosa (o un .part) ma non e completo."""
    return _status(SAM31_FILES, Path(base) if base else models_dir())


def comfy_status(base: Path | None = None) -> tuple[str, list[str]]:
    return _status(COMFY_FILES, Path(base) if base else comfy_dir())


def comfy_usable(base: Path | None = None) -> bool:
    """La copia Comfy-Org e scaricata E dichiarata equivalente a quella di Meta."""
    return bool(COMFY_EQUIVALENT) and comfy_status(base)[0] == "ok"


def weights_state(base: Path | None = None, comfy_base: Path | None = None) -> str:
    """'ok' se SAM 3.1 si puo usare (pesi di Meta completi, oppure copia Comfy-Org equivalente); altrimenti lo stato dei pesi di Meta."""
    st = sam31_status(base)[0]
    return "ok" if (st == "ok" or comfy_usable(comfy_base)) else st


def checkpoint_path(base: Path | None = None, comfy_base: Path | None = None) -> Path:
    """Il checkpoint da passare ai job: quello di Meta se completo, altrimenti la copia Comfy-Org (solo se equivalente), altrimenti quello di Meta (il job
    dara l'errore chiaro)."""
    if sam31_status(base)[0] == "ok":
        return (Path(base) if base else models_dir()) / SAM31_CHECKPOINT
    if comfy_usable(comfy_base):
        return (Path(comfy_base) if comfy_base else comfy_dir()) / COMFY_CHECKPOINT
    return (Path(base) if base else models_dir()) / SAM31_CHECKPOINT


def is_installed(engine: str, base: Path | None = None) -> bool:
    if engine == SAM2:
        try:
            from inference_server.backends import image_segment as seg
            return os.path.isdir(seg.SAM2_DIR) and os.path.isdir(seg.GDINO_DIR)
        except Exception:                                    # noqa: BLE001
            return True                                      # il backend dara l'errore chiaro se manca
    return weights_state(base) == "ok"


def is_selectable(engine: str, base: Path | None = None) -> bool:
    """Si puo scegliere dal menu: SAM2 sempre; SAM 3.1 solo se i pesi ci sono E il caricatore e provato."""
    if engine == SAM2:
        return True
    return LOADER_READY and weights_state(base) == "ok"


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
    st = weights_state(base)
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
        return ("Richiesta di accesso in attesa di approvazione da Meta: l'approvazione di facebook/sam3.1 richiede di solito alcune ore. "
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
    st = weights_state(base)
    if st != "ok":
        bad = sam31_status(base)[1]
        head = "SAM 3.1 non e installato" if st == "missing" else "Il download di SAM 3.1 e incompleto (" + ", ".join(bad[:3]) + ")"
        return SAM31, (head + f". Per averlo: account Hugging Face, accetta la licenza SAM su {SAM31_URL} (approvazione di Meta: di solito alcune ore) e scarica dal tab "
                       "«Pesi modelli»; in alternativa la copia Comfy-Org non gated (stesso tab). Nel frattempo usa SAM2 dal menu Segmentazione.")
    if not LOADER_READY:
        return SAM31, "SAM 3.1 e installato ma il caricatore nativo non e ancora pronto: scegli SAM2 dal menu Segmentazione."
    return SAM31, ""
