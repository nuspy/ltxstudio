"""Orchestratore LoRA IMMAGINE → VIDEO per un asset del Film Studio, con GATE umani (richiesta utente).

Catena: addestra la LoRA IMMAGINE del personaggio → genera foto-prova → [GATE: l'utente sceglie le foto buone]
→ addestra la LoRA VIDEO usando quelle foto come `real_media` (diventano clip i2v) → REGISTRA entrambe nel DB
del Film Studio con binding all'asset (chiude le LoRA orfane, B6).

Riusa il codice esistente senza re-implementare nulla:
- core.lora_image_service.make_lora / core.lora_video_service.make_lora (dataset+train)
- core.lora_video_service.Subject (con `real_media` che ingloba le foto come clip i2v)
- core.film_studio.gen_service.gen_image (foto-prova col la LoRA immagine appena addestrata)

Macchina a stati PERSISTITA (tabella lora_pipeline) → ripresa dopo crash. mode='supervised' si ferma ai gate;
mode='auto' auto-approva (le foto-prova diventano tutte selezionate). Gli step lunghi (training) vanno chiamati
da un worker; qui la logica è sincrona e idempotente."""
from __future__ import annotations

import json
import os
import time

from core.film_studio import db

# Stati della pipeline
NEW = "new"; IMG_TRAINING = "img_training"; REVIEW = "review"; VID_TRAINING = "vid_training"
REGISTERED = "registered"; ERROR = "error"


def register_lora(asset_id: int, flavor: str, file: str, *, variation_id: int = 0, trigger: str = "",
                  dataset: str = "", base_arch: str = "", scope: str = "episode", weight: float = 0.9) -> int:
    """INSERT di una LoRA addestrata nel DB del Film Studio con binding all'asset/variazione (B6: prima i
    wizard copiavano il file ma NON registravano → LoRA orfane). Idempotente sul path file."""
    ex = db.query("lora", "asset_id=? AND flavor=? AND file=?", (asset_id, flavor, file))
    if ex:
        return ex[0]["id"]
    ver = 1 + max([r["version"] for r in db.query("lora", "asset_id=? AND flavor=?", (asset_id, flavor))
                   if r.get("version")] or [0])
    return db.insert("lora", asset_id=asset_id, variation_id=variation_id or 0, flavor=flavor,
                     trigger=trigger or "", scope=scope, version=ver, file=file, dataset=dataset,
                     weight=weight, base_arch=base_arch)


def create_pipeline(asset_id: int, name: str, spec: dict, *, variation_id: int = 0,
                    mode: str = "supervised") -> int:
    """Crea una pipeline LoRA img→video per un asset. spec = {photos:[...], case_ids:[...], customs:[...],
    face_focus:bool, trigger:str}. mode: 'supervised' (gate manuali) | 'auto'."""
    return db.insert("lora_pipeline", asset_id=asset_id, variation_id=variation_id or 0, name=name,
                     state=NEW, mode=mode, spec=json.dumps(spec, ensure_ascii=False), img_lora="",
                     img_dataset="", review_photos="[]", selected_photos="[]", vid_lora="", vid_dataset="",
                     error="", updated=time.time())


def _set(pid: int, **fields) -> None:
    fields["updated"] = time.time()
    db.update("lora_pipeline", pid, **fields)


def _subject(spec: dict, real_media: list | None = None):
    """Costruisce un core.lora_video_service.Subject dallo spec (riusato anche dal servizio immagini)."""
    from core.lora_video_service import Subject
    return Subject(name=spec.get("trigger") or spec.get("name") or "char",
                   type=spec.get("type", "person"), photos=list(spec.get("photos") or []),
                   case_ids=list(spec.get("case_ids") or []), customs=list(spec.get("customs") or []),
                   real_media=list(real_media or []), face_focus=bool(spec.get("face_focus", True)))


def step(pid: int, *, on_log=None, review_count: int = 6) -> str:
    """Esegue l'azione dello stato corrente e AVANZA. Ritorna il nuovo stato. Chiamare da un worker (gli step
    di training sono lunghi). Si ferma su REVIEW finché non arriva select_photos(). Idempotente/riprendibile."""
    p = db.get("lora_pipeline", pid)
    if not p:
        return ERROR
    log = on_log or (lambda *_: None)
    spec = json.loads(p.get("spec") or "{}")
    out_dir = str(db.project_dir(0).parent / "loras" / (p["name"] or f"lora_{pid}"))
    os.makedirs(out_dir, exist_ok=True)
    try:
        st = p["state"]
        if st in (NEW, IMG_TRAINING):
            log("Training LoRA IMMAGINE…")
            _set(pid, state=IMG_TRAINING)
            from core import lora_image_service as LI
            res = LI.make_lora([_subject(spec)], p["name"], out_dir,
                               progress=lambda m: log(str(m)))     # {dataset, lora}
            _set(pid, img_lora=res.get("lora", ""), img_dataset=res.get("dataset", ""))
            # foto-prova col la LoRA immagine appena addestrata
            log(f"Genero {review_count} foto-prova…")
            from core.film_studio import gen_service as GS
            photos = []
            for i in range(review_count):
                out = os.path.join(out_dir, f"review_{i}.png")
                try:
                    GS.gen_image(f"portrait of {spec.get('trigger','the character')}, neutral expression, "
                                 f"studio light", out, seed=1000 + i,
                                 loras=[{"file": res.get("lora", ""), "trigger": spec.get("trigger", ""),
                                         "weight": 0.9}] if res.get("lora") else None)
                    if os.path.exists(out):
                        photos.append(out)
                except Exception as _e:                          # noqa: BLE001
                    log(f"foto-prova {i} fallita: {_e}")
            _set(pid, review_photos=json.dumps(photos, ensure_ascii=False), state=REVIEW)
            if p["mode"] == "auto":                              # auto: seleziona tutte e prosegui
                return select_photos(pid, photos, on_log=on_log)
            log("GATE: seleziona le foto buone, poi continua.")
            return REVIEW
        if st == VID_TRAINING:
            return _train_video(pid, spec, out_dir, log)
        return st
    except Exception as e:                                        # noqa: BLE001
        _set(pid, state=ERROR, error=str(e)[:500])
        log(f"ERRORE: {e}")
        return ERROR


def select_photos(pid: int, photos: list, *, on_log=None) -> str:
    """GATE: l'utente ha scelto le foto buone → alimentano il dataset VIDEO come real_media e parte il training
    video. Ritorna il nuovo stato."""
    log = on_log or (lambda *_: None)
    _set(pid, selected_photos=json.dumps(list(photos or []), ensure_ascii=False), state=VID_TRAINING)
    p = db.get("lora_pipeline", pid)
    return _train_video(pid, json.loads(p.get("spec") or "{}"),
                        str(db.project_dir(0).parent / "loras" / (p["name"] or f"lora_{pid}")), log)


def _train_video(pid: int, spec: dict, out_dir: str, log) -> str:
    """Addestra la LoRA VIDEO con le foto selezionate come real_media, poi REGISTRA entrambe nel DB."""
    p = db.get("lora_pipeline", pid)
    sel = json.loads(p.get("selected_photos") or "[]")
    log("Training LoRA VIDEO (foto scelte come clip i2v)…")
    from core import lora_video_service as LV
    res = LV.make_lora([_subject(spec, real_media=sel)], p["name"] + "_video", out_dir,
                       progress=lambda m: log(str(m)))
    _set(pid, vid_lora=res.get("lora", ""), vid_dataset=res.get("dataset", ""))
    # REGISTER: entrambe le LoRA nel DB del Film Studio, bindate all'asset (chiude le orfane)
    trig = spec.get("trigger", "")
    if p.get("img_lora"):
        register_lora(p["asset_id"], "image", p["img_lora"], variation_id=p.get("variation_id", 0),
                      trigger=trig, dataset=p.get("img_dataset", ""), base_arch="flux")
    if res.get("lora"):
        register_lora(p["asset_id"], "video", res["lora"], variation_id=p.get("variation_id", 0),
                      trigger=trig, dataset=res.get("dataset", ""), base_arch="ltx2")
    _set(pid, state=REGISTERED)
    log("REGISTRATE: LoRA immagine + video bindate all'asset.")
    return REGISTERED
