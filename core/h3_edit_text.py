"""Testo dell'utente nell'edit H3 (Modifica Video, cambia-sfondo / sostituisci): traduzione in inglese + arricchimento, UNA chiamata per job, PRIMA dei clip.

Il prompt H3 e uno schema inglese (core/h3_edit_prompt.py): il luogo, la descrizione della foto e il soggetto scritti in italiano ci finivano tali e quali.
Decisione dell'utente (03/11): «Traduzione + arricchimento» con l'enhancer locale gia in uso (core/prompt_enhance, Gemma).
 · campi DESCRITTIVI (prompt = luogo/ambiente, picture_desc, shots): tradotti e arricchiti;
 · `object` (il soggetto: e anche l'etichetta di SAM2/SAM 3.1): tradotto in modo FEDELE e corto, MAI arricchito («persona» -> «person»);
 · testo gia inglese: nessuna chiamata; cache per hash del testo; enhancer assente o fallito: avviso e invio del testo ORIGINALE, senza bloccare;
 · l'originale e quello inviato al modello vanno nel log e nel meta del job (`h3_edit.text_info`).
Modulo senza Qt/torch: la chiamata all'LLM e iniettabile (test)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

# chiave del cfg -> nome del campo nello schema dell'enhancer
FIELDS = {"object": "subject", "prompt": "setting", "picture_desc": "picture_desc", "shots": "shots"}
LABEL_MAX_WORDS = 8

# Parole che in inglese non esistono (italiano, spagnolo, francese, tedesco, portoghese): basta una per dire «non e inglese».
_FOREIGN = frozenset("""il lo la le gli uno una del della dei delle degli nel nella nei nelle sul sulla sui sulle che per con sono molto
persona persone uomo donna ragazza ragazzo bambino bambina vestito vestita rosso rossa rossi blu verde giallo nero nera bianco bianca montagna rifugio notte giorno
neve camino acceso accesa sera mattina mare spiaggia citta strada mercato tavolo cielo luce fuoco bosco foresta casa stanza
el los las una unos unas del con una mujer hombre noche
une des du les est dans pour avec femme homme nuit
und der die das ein eine mit nicht von zu im
os uma uns umas mulher noite""".split())

_KEEP_DICT = {                                               # etichette di segmentazione comuni: niente LLM
    "persona": "person", "la persona": "person", "una persona": "person", "uomo": "man", "l'uomo": "man", "un uomo": "man", "donna": "woman", "la donna": "woman",
    "una donna": "woman", "ragazza": "girl", "la ragazza": "girl", "ragazzo": "boy", "il ragazzo": "boy", "bambino": "child", "il bambino": "child",
    "bambina": "girl", "la bambina": "girl", "persone": "people", "le persone": "people", "personaggio": "person", "il personaggio": "person",
    "persona": "person", "hombre": "man", "mujer": "woman", "personne": "person", "homme": "man", "femme": "woman", "mensch": "person",
}


def is_english(text: str) -> bool:
    """Euristica senza modelli: vuoto, o solo ASCII e nessuna parola tipica di altre lingue latine. Il testo con lettere accentate/non ASCII non e inglese."""
    t = (text or "").strip()
    if not t:
        return True
    if any(ord(c) > 127 and c.isalpha() for c in t):
        return False
    words = re.findall(r"[a-zA-Z']+", t.lower())
    return not any(w in _FOREIGN for w in words)


def _cache_path() -> Path:
    base = os.environ.get("LTX_TEXT_CACHE_DIR") or str(Path.home() / ".ltxstudio")
    return Path(base) / "h3_text_cache.json"


def _cache_get(key: str):
    try:
        return json.loads(_cache_path().read_text("utf-8")).get(key)
    except (OSError, ValueError):
        return None


def _cache_put(key: str, value: dict) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        try:
            d = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            pass
        d[key] = value
        if len(d) > 300:                                      # cache piccola: si butta la meta piu vecchia
            d = dict(list(d.items())[-150:])
        p.write_text(json.dumps(d, ensure_ascii=False), "utf-8")
    except OSError:
        pass


def parse_reply(reply: str, wanted: list[str]) -> dict | None:
    """Estrae l'oggetto JSON dalla risposta dell'LLM; None se non e valido o manca una chiave richiesta."""
    a, b = (reply or "").find("{"), (reply or "").rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(reply[a:b + 1])
    except ValueError:
        return None
    if not isinstance(d, dict) or any(not isinstance(d.get(k), str) or not d[k].strip() for k in wanted):
        return None
    return {k: " ".join(d[k].split()) for k in wanted}


def apply_to_cfg(cfg: dict, *, log=print, enhance=None, available=None) -> dict:
    """Se `cfg["h3_translate"]`: traduce/arricchisce i campi non inglesi del cfg (in place) e ritorna
    {"status": off|english|dict|cache|ok|unavailable|failed, "seconds", "original": {...}, "sent": {...}}. Non solleva mai."""
    info = {"status": "off", "seconds": 0.0, "original": {}, "sent": {}}
    if not cfg.get("h3_translate"):
        return info
    orig = {k: str(cfg.get(k) or "").strip() for k in FIELDS if str(cfg.get(k) or "").strip()}
    info["original"] = dict(orig)
    sent = dict(orig)
    need = {k: v for k, v in orig.items() if not is_english(v)}
    if not need:
        info.update(status="english", sent=sent)
        return info
    # etichetta del soggetto con una parola nota: nessuna chiamata
    if "object" in need and need["object"].lower().strip(" .") in _KEEP_DICT:
        sent["object"] = _KEEP_DICT[need["object"].lower().strip(" .")]
        need.pop("object")
        log(f"[h3text] soggetto «{orig['object']}» -> «{sent['object']}» (vocabolario, senza LLM)")
        info["status"] = "dict"
    if need:
        t0 = time.time()
        key = hashlib.sha256(json.dumps(need, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        got = _cache_get(key)
        if got and all(k in got for k in need):
            for k in need:
                sent[k] = got[k]
            info["status"] = "cache"
            log("[h3text] traduzione dalla cache (stesso testo gia elaborato)")
        else:
            try:
                if enhance is None or available is None:
                    from core import prompt_enhance as pe
                    enhance, available = enhance or pe.enhance, available or pe.available
                if not available():
                    raise RuntimeError("enhancer non disponibile (pesi o VRAM)")
                payload = json.dumps({FIELDS[k]: v for k, v in need.items()}, ensure_ascii=False)
                log(f"[h3text] traduzione + arricchimento con l'enhancer locale ({', '.join(need)})…")
                reply = enhance(payload, kind="h3_fields", max_new_tokens=420)
                rev = {v: k for k, v in FIELDS.items()}
                out = parse_reply(reply, [FIELDS[k] for k in need]) if reply and reply.strip() != payload.strip() else None
                if out is None:
                    raise RuntimeError("risposta non valida dall'enhancer")
                res = {rev[n]: v for n, v in out.items()}
                if "object" in res and len(res["object"].split()) > LABEL_MAX_WORDS:
                    log(f"[h3text] AVVISO: la traduzione del soggetto e troppo lunga ({len(res['object'].split())} parole): si invia il soggetto originale")
                    res.pop("object")
                sent.update(res)
                _cache_put(key, {k: sent[k] for k in need if k in sent})
                info["status"] = "ok"
            except Exception as ex:                          # noqa: BLE001 — mai bloccare l'edit
                info["status"] = "unavailable" if "non disponibile" in str(ex) else "failed"
                log(f"[h3text] AVVISO: {ex}. Invio il testo ORIGINALE (non tradotto) al modello.")
        info["seconds"] = round(time.time() - t0, 1)
    for k, v in sent.items():
        cfg[k] = v
    info["sent"] = sent
    for k in orig:
        if sent.get(k) != orig[k]:
            log(f"[h3text] {k}: «{orig[k]}» -> «{sent[k]}»")
    log(f"[h3text] {info['status']} in {info['seconds']} s")
    return info


def summary_line(info: dict) -> str:
    """Riga per l'etichetta della GUI: cosa e arrivato al modello (vuota se non e cambiato nulla)."""
    ch = {k: v for k, v in (info.get("sent") or {}).items() if v != (info.get("original") or {}).get(k)}
    if not ch:
        return ""
    return "[h3text] " + " | ".join(f"{k}: {v}" for k, v in ch.items())
