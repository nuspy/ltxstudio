"""Costruzione del job IC-LoRA del tab «Modifica Video AI» — logica CONDIVISA da GUI e REST API.

Modulo senza Qt e senza torch. Dato un `cfg` (stesse chiavi del segnale `request_edit` del tab) produce i
`params` per `workers/ltx2_infer.py` (+ file intermedi + post-processo), cosi' GUI (`_VideoEditAIWorker`) e
`POST /video/ic_lora` si comportano allo stesso modo (parita').

cfg (oltre alle chiavi storiche del tab): mode, video, prompt, strength, conditioning_attention, keep_audio,
res_mode (source|half|custom) + res_long, out_format (16:9|9:16|1:1|4:3|3:4|21:9|custom) + out_anchor +
out_margins (sopra,destra,sotto,sinistra) + out_dark + out_paste + out_feather (px, default 24; 0 = taglio netto)
+ out_min_bar (0 = spento; barre minime in frazione del canvas, rimpicciolisce il sorgente), style_image, cam_ref/cam_img/cam_img_strength/
cam_seconds/width/height (cameraman), model / gguf_transformer / ltx_release (checkpoint ESATTAMENTE come scelto
dall'utente: mai sostituito qui).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from core import ltx_release as _rel
from core import video_edit_modes as vem
from core import video_ic_prep as prep


class IcJobError(Exception):
    """Errore di validazione/preflight con messaggio per l'utente (mai degrado silenzioso)."""


# ── checkpoint scelto nel tab Video ───────────────────────────────────────────────────────────────
OTHER_ARCH_KINDS = ("krea_rt", "minimax_h3", "lingbot")


def distilled_lora_path(release: str | None = None) -> str:
    """Percorso della distilled-LoRA ufficiale (quella che il worker aggiunge ai checkpoint dev)."""
    return str(_rel.get(release or "2.3").distilled_lora)


def distilled_lora_ok(release: str | None = None) -> bool:
    """True se la distilled-LoRA e' su disco (nessun download automatico: se manca la GUI avvisa)."""
    return os.path.exists(distilled_lora_path(release))


def checkpoint_info(vals: dict | None) -> dict:
    """Interpreta `video_tab.get_values()` SENZA scegliere nulla. Ritorna:
    state: distilled | dev | auto | release25 | other_arch | unknown
    name: nome file (o ""), distilled: bool, release: str|None, params: {model|gguf_transformer, ltx_release}
    `auto`/`other_arch` => il worker usera' il suo default (dev): per l'avviso valgono come NON distilled.
    Col dev il worker AGGIUNGE la distilled-LoRA (ricetta ufficiale Lightricks, strength 0.5, stage 1+2):
    `dlora_ok` dice se il file e' su disco (assente: la GUI avvisa e rimanda al tab Pesi, nessun download)."""
    info = _checkpoint_info(vals)
    info["dlora_ok"] = distilled_lora_ok(info.get("release") or "2.3")
    info["dlora_path"] = distilled_lora_path(info.get("release") or "2.3")
    return info


def _checkpoint_info(vals: dict | None) -> dict:
    if not vals:
        return {"state": "unknown", "name": "", "distilled": False, "release": None, "params": {}}
    kind = vals.get("model_kind") or ""
    params: dict = {}
    if kind in OTHER_ARCH_KINDS:
        return {"state": "other_arch", "name": kind, "distilled": False, "release": None, "params": {}}
    path = ""
    if kind == "gguf" and vals.get("model_gguf"):
        path = str(vals["model_gguf"]); params["gguf_transformer"] = path
    elif vals.get("model") and vals.get("model") not in ("", "auto"):
        path = str(vals["model"]); params["model"] = path
    release = _rel.detect_release(path) if path else None
    if release:
        params["ltx_release"] = release
    name = os.path.basename(path.replace("\\", "/"))
    if not path:
        return {"state": "auto", "name": "", "distilled": False, "release": None, "params": params}
    if release and not _rel.ic_lora_supported(release):
        return {"state": "release25", "name": name, "distilled": _rel.is_distilled(path),
                "release": release, "params": params}
    d = _rel.is_distilled(path)
    return {"state": "distilled" if d else "dev", "name": name, "distilled": d,
            "release": release, "params": params}


def weight_path(spec: vem.ModeSpec) -> str:
    """Path del peso IC-LoRA della modalita' (registro release 2.3), '' se non e' una modalita' a registro."""
    if not spec.kind:
        return ""
    return str(_rel.get("2.3").ic_lora.get(spec.kind, ""))


def missing_weight(spec: vem.ModeSpec) -> str:
    """Messaggio di errore se il peso della modalita' manca su disco, altrimenti ''."""
    if not spec.kind:
        return ""
    p = weight_path(spec)
    if p and os.path.exists(p):
        return ""
    return (f"Peso «{spec.kind}» mancante: {p or '(non configurato)'}\n"
            f"Scaricalo dal tab «Pesi modelli» (id catalogo {spec.catalog_id}).")


# ── lip dub (testo → nuova voce + labbra; ricetta ufficiale Lightricks, vedi video_edit_modes) ──────
# LINGUE PARLABILI. NON esiste un elenco ufficiale Lightricks delle lingue che LTX-2.3 sa parlare (verificato il
# 17/10/2026: model card, paper LTX-2 «multilingual text encoder... performance varies across languages», paper
# JUST-DUB-IT, workflow ufficiale). Qui sta l'insieme con almeno UNA prova, con la prova accanto: e' una PROPOSTA in
# attesa di conferma dell'utente (DECISIONE RICHIESTA nel report). Per restringerlo basta togliere righe da questa tabella.
#   card = model card HF Lightricks/LTX-2.3, tag `language:` (en de es fr ja ko zh it pt; il testo dice «Language(s): English»)
#   eval = paper JUST-DUB-IT: valutazione «English-to-Foreign (covering Russian, Spanish, German, and French)»
#   wf   = workflow ufficiale Lipdub: «Use the alphabet of the language ... Cyrillic for Russian or Chinese characters for Chinese»
LIPDUB_SPOKEN_LANGS: tuple[tuple[str, str, str], ...] = (
    ("en", "English", "card+eval"), ("es", "Spanish", "card+eval"), ("fr", "French", "card+eval"),
    ("de", "German", "card+eval"), ("ru", "Russian", "eval+wf"), ("zh", "Chinese", "card+wf"),
    ("it", "Italian", "card"), ("pt", "Portuguese", "card"), ("ja", "Japanese", "card"), ("ko", "Korean", "card"),
)
SPOKEN_CODES: tuple[str, ...] = tuple(c for c, _n, _e in LIPDUB_SPOKEN_LANGS)
# voce vuota (= nessuna lingua nel prompt) + le lingue parlabili: il dropdown della GUI usa SOLO queste
LIPDUB_LANGUAGES: tuple[tuple[str, str], ...] = (("", ""),) + tuple((c, n) for c, n, _e in LIPDUB_SPOKEN_LANGS)


def lipdub_language_name(lang: str) -> str:
    """Nome inglese della lingua dal codice breve (it, es, ...); un nome libero passa invariato; '' = nessuna."""
    lang = (lang or "").strip()
    return dict(LIPDUB_LANGUAGES).get(lang.lower(), lang)


def compose_lipdub_prompt(line: str, language: str = "", scene: str = "") -> str:
    """Prompt nel formato dell'esempio ufficiale: `<scena>, speaking in <Lingua> saying: "<battuta>"`.
    La battuta va nell'alfabeto della lingua parlata (il modello NON traduce). Senza scena: «A person»."""
    line = " ".join((line or "").split())
    if not line:
        return ""
    scene = (scene or "").strip().rstrip(",.")
    lang = lipdub_language_name(language)
    head = scene if scene else "A person"
    where = " in " + lang if lang else ""
    return head + ", speaking" + where + ' saying: "' + line.replace('"', "'") + '"'


def compose_lipdub_audio_prompt(line: str = "", language: str = "", scene: str = "") -> str:
    """Prompt della variante da audio: battuta (trascrizione) opzionale; senza, solo la scena + «speaking»."""
    full = compose_lipdub_prompt(line, language, scene)
    if full:
        return full
    scene = (scene or "").strip().rstrip(",.")
    return (scene if scene else "A person") + ", speaking"


def estimate_speech_seconds(text: str) -> float:
    """Stima GROSSOLANA (euristica, non misurata) della durata di una battuta parlata: ~2,5 parole/s per le
    lingue alfabetiche, ~4,5 caratteri/s per cinese/giapponese/coreano. Serve solo ad avvisare."""
    t = " ".join((text or "").split())
    if not t:
        return 0.0
    cjk = sum(1 for c in t if "\u3040" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af")
    if cjk >= 0.3 * len(t.replace(" ", "")):
        return cjk / 4.5
    return len(t.split()) / 2.5


LIPDUB_TOO_SHORT, LIPDUB_TOO_LONG = 0.6, 1.25


def lipdub_text_fit(line: str, video_seconds: float) -> tuple[str, float]:
    """('ok'|'short'|'long'|'unknown', stima_s). Ufficiale: testo troppo lungo = parole saltate, troppo corto =
    voce lenta e innaturale; l'uscita dura sempre quanto il video sorgente."""
    est = estimate_speech_seconds(line)
    if est <= 0 or video_seconds <= 0:
        return "unknown", est
    r = est / video_seconds
    return ("short" if r < LIPDUB_TOO_SHORT else "long" if r > LIPDUB_TOO_LONG else "ok"), est


# ── job ───────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class IcJob:
    params: dict
    spec: vem.ModeSpec
    work_dir: str
    mux_audio_from: str = ""                 # sorgente dell'audio originale ('' = nessun mux)
    layout: object | None = None             # OutpaintLayout (outpaint)
    gamma: float = 1.0
    paste_original: bool = False
    feather: int = 0                         # outpaint: px di sfumatura del bordo nel paste (0 = taglio netto)
    source: str = ""
    notes: list = field(default_factory=list)
    segments: int = 1


def _scale_to(w: int, h: int, mode: str, long_edge: int) -> tuple[int, int]:
    if mode == "half":
        return max(64, w // 2), max(64, h // 2)
    if mode == "custom" and long_edge > 0:
        s = long_edge / float(max(w, h))
        return max(64, int(round(w * s))), max(64, int(round(h * s)))
    return w, h


def build_job(cfg: dict, out_dir: str, stamp: int, output: str, log=None) -> IcJob:
    """Prepara params + file intermedi per una modalita' a registro IC-LoRA. Solleva IcJobError."""
    log = log or (lambda s: None)
    mode = cfg.get("mode", "")
    try:
        spec = vem.get_spec(mode)
    except KeyError as e:
        raise IcJobError(str(e)) from None
    if not spec.enabled:
        raise IcJobError(f"modalita' «{mode}» non ancora disponibile (fase {spec.phase})")
    if not spec.kind:
        raise IcJobError(f"modalita' «{mode}» non usa il registro IC-LoRA")
    msg = missing_weight(spec)
    if msg:
        raise IcJobError(msg)
    rel = cfg.get("ltx_release")
    if rel and not _rel.ic_lora_supported(rel):
        raise IcJobError(f"Le IC-LoRA locali sono per LTX 2.3: il checkpoint scelto e' release {rel}.")
    src = cfg.get("cam_ref") if spec.key == "cameraman" else cfg.get("video")
    if not src or not os.path.exists(src):
        raise IcJobError("video sorgente mancante" if spec.key != "cameraman"
                         else "video di riferimento del movimento mancante")
    if spec.needs_audio_in_video and not prep.has_audio(src):
        raise IcJobError("il video non ha una traccia audio")
    ext_audio = ""
    if spec.needs_audio_file:
        ext_audio = str(cfg.get("lipdub_audio") or "")
        if not ext_audio or not os.path.exists(ext_audio):
            raise IcJobError("audio mancante: scegli il file audio da usare (lipdub da audio)")
        if not prep.has_audio(ext_audio):
            raise IcJobError("il file scelto non contiene una traccia audio")
        if cfg.get("lipdub_audio_ref") == "original" and not prep.has_audio(src):
            raise IcJobError("riferimento voce «originale»: il video non ha una traccia audio")
    work = os.path.join(out_dir, f"_vedit_{spec.key}_{stamp}")
    os.makedirs(work, exist_ok=True)
    info = prep.probe(src, count_frames=True)
    notes: list[str] = []
    total = int(info.nb_frames or 0)
    if total < 9:
        raise IcJobError(f"video troppo corto ({total} frame, minimo 9)")
    fps = int(round(info.fps or 25))
    prompt = (cfg.get("prompt") or "").strip()
    if spec.kind == "lipdub" and not prompt:                        # prompt libero vince; altrimenti si compone
        _compose = compose_lipdub_audio_prompt if spec.needs_audio_file else compose_lipdub_prompt
        prompt = _compose(cfg.get("lipdub_line", ""), cfg.get("lipdub_lang", ""), cfg.get("lipdub_scene", ""))
    if spec.needs_prompt == vem.NEEDS_REQUIRED and not prompt:
        raise IcJobError("battuta/prompt mancante" if spec.kind == "lipdub" else "prompt mancante")

    # lipdub: forza del riferimento fissa (workflow ufficiale: 1.0), non regolabile
    _strength = spec.default_strength if spec.kind == "lipdub" else float(cfg.get("strength", spec.default_strength))
    p: dict = {"prompt": prompt, "negative": "worst quality, blurry", "control_type": spec.kind,
               "edit_lora": "", "control_strength": float(_strength),
               "conditioning_attention": float(cfg.get("conditioning_attention", 1.0)),
               "fps": fps, "seed": int(cfg.get("seed", 42)), "output": output, "_strict_ic": True}
    for k in ("model", "gguf_transformer", "ltx_release"):          # checkpoint come scelto, mai sostituito
        if cfg.get(k):
            p[k] = cfg[k]

    n_total = total
    guide = src
    mux = ""
    layout = None
    gamma = 1.0
    paste = False
    feather = 0

    if spec.preprocess == vem.PRE_OUTPAINT:
        fmt = cfg.get("out_format", "16:9")
        kw = {"anchor": cfg.get("out_anchor", "center"), "max_long_edge": spec.max_long_edge}
        min_bar = float(cfg.get("out_min_bar", 0.0) or 0.0)
        if min_bar > 0 and fmt != "custom":
            kw["min_bar"] = min_bar                                    # opt-in: sorgente rimpicciolito, barre >= min_bar
        if fmt == "custom":
            kw["margins"] = tuple(int(v) for v in cfg.get("out_margins", (0, 0, 0, 0)))
        else:
            kw["aspect"] = fmt
        try:
            layout = prep.compute_outpaint_layout(info.width, info.height, **kw)
        except ValueError as e:
            raise IcJobError(str(e)) from None
        if layout.new_area <= 0.001:
            raise IcJobError("nessuna area nuova da generare: scegli un formato/margini diversi")
        if layout.too_much_new_area:
            notes.append(f"area nuova {layout.new_area:.0%} > {prep.MAX_NEW_AREA:.0%}: il modello inventera' molto")
        if layout.narrow_bar:
            notes.append(f"barra nuova stretta ({layout.narrow_bar_fraction:.0%} del canvas < {prep.MIN_BAR_FRACTION:.0%}): "
                         "il modello potrebbe NON riempirla; usa margini piu' larghi")
        if layout.clamped:
            notes.append(f"canvas ridotto al tetto {spec.max_long_edge}px")
        if cfg.get("out_dark"):
            gamma = 2.0
            notes.append("sorgente scura: gamma 2.0 in ingresso, 0.5 sul risultato")
        guide = os.path.join(work, "canvas.mp4")
        log(f"[outpaint] canvas {layout.canvas_w}x{layout.canvas_h} (area nuova {layout.new_area:.0%})")
        prep.build_outpaint_canvas(src, guide, layout, gamma)
        p["width"], p["height"] = layout.canvas_w, layout.canvas_h
        paste = bool(cfg.get("out_paste", True))
        feather = max(0, min(int(cfg.get("out_feather", prep.DEFAULT_FEATHER)), prep.MAX_FEATHER)) if paste else 0
        mux = src if cfg.get("keep_audio", True) and info.has_audio else ""
    else:
        if spec.key == "cameraman":
            w0 = int(cfg.get("width") or info.width)
            h0 = int(cfg.get("height") or info.height)
            sec = float(cfg.get("cam_seconds", 0) or 0)
            if sec > 0:
                n_total = min(total, int(sec * fps))
            if cfg.get("cam_img"):
                p["image_path"] = cfg["cam_img"]
                p["ref_strength"] = float(cfg.get("cam_img_strength", 0.7))
        else:
            w0, h0 = _scale_to(info.width, info.height, cfg.get("res_mode", "source"),
                               int(cfg.get("res_long", 0) or 0))
            mux = src if (spec.keep_audio and cfg.get("keep_audio", True) and info.has_audio) else ""
        w, h, red = vem.clamp_resolution(w0, h0, spec)
        if red:
            notes.append(f"Sorgente {w0}x{h0} ridotto a {w}x{h} (tetto {spec.max_long_edge}px)")
        p["width"], p["height"] = w, h
        if spec.min_long_edge and max(w, h) < spec.min_long_edge:
            notes.append(f"risoluzione {w}x{h} sotto il minimo consigliato ({spec.min_long_edge}px lato lungo)")

    if spec.needs_audio_file:
        # Variante da audio (sperimentale): l'AUDIO comanda la durata. Piu' corto del video → il video e' tagliato alla
        # sua durata; piu' lungo → l'audio e' tagliato alla durata del video. Il latente audio fisso deve avere
        # esattamente la lunghezza del video, quindi l'audio e' preparato (wav stereo 48 kHz) a quella durata.
        vsec = n_total / float(fps)
        adur = prep.audio_duration(ext_audio)
        target = vsec
        if adur and adur + 0.3 < vsec:
            target = adur
            n_total = min(n_total, max(9, int(target * fps)))
            notes.append(f"l'audio ({adur:.1f} s) e' piu' corto del video ({vsec:.1f} s): il video viene tagliato a "
                         f"{n_total / fps:.1f} s (l'audio comanda la durata)")
        elif adur and adur > vsec + 0.3:
            notes.append(f"l'audio ({adur:.1f} s) e' piu' lungo del video ({vsec:.1f} s): verra' tagliato a "
                         f"{vsec:.1f} s (il video non si allunga)")
        n_total = vem.snap_frames(n_total)
        if n_total < total:
            guide = prep.cut_video(src, os.path.join(work, "guide_cut.mp4"), n_total)
        wav = prep.prepare_audio(ext_audio, os.path.join(work, "ext_audio.wav"), n_total / float(fps))
        p["lipdub_audio"] = wav
        p["lipdub_audio_ref"] = "original" if cfg.get("lipdub_audio_ref") == "original" else "external"
        notes.append("SPERIMENTALE: audio dell'utente tenuto fisso nei latenti, si rigenera solo il video; qualita' non "
                     "garantita (per canzoni/voci collaudate usa Talk/Canta del tab Video)")
    elif spec.kind == "lipdub":
        vsec = n_total / float(fps)
        if cfg.get("lipdub_line") and not (cfg.get("prompt") or "").strip():
            fit, est = lipdub_text_fit(cfg.get("lipdub_line", ""), vsec)
            if fit == "long":
                notes.append(f"battuta lunga (~{est:.1f} s di parlato per un video di {vsec:.1f} s): il modello "
                             "potrebbe saltare delle parole; accorcia il testo")
            elif fit == "short":
                notes.append(f"battuta corta (~{est:.1f} s di parlato per un video di {vsec:.1f} s): la voce potrebbe "
                             "risultare lenta e innaturale; allunga il testo")
        adur = prep.audio_duration(src)
        if adur and vsec - adur > 0.5:
            notes.append(f"la traccia audio ({adur:.1f} s) e' piu' corta del video ({vsec:.1f} s): il riferimento "
                         "della voce e' incompleto")
        notes.append(f"la durata dell'uscita e' quella del video sorgente ({vsec:.1f} s); la voce e' generata dal "
                     "modello (non si fornisce un audio esterno)")

    p["num_frames"] = vem.snap_frames(n_total)
    nseg = vem.segments_needed(p["num_frames"], spec)
    if p["num_frames"] > spec.max_frames_per_segment and not spec.chunkable:
        raise IcJobError(f"{p['num_frames']} frame oltre il tetto {spec.max_frames_per_segment} e la "
                         "modalita' non si puo' spezzare in segmenti: accorcia il clip")
    if nseg > 1:
        p["chunk_frames"] = spec.max_frames_per_segment
        p["overlap_frames"] = vem.DEFAULT_OVERLAP
        notes.append(f"{nseg} segmenti da {spec.max_frames_per_segment} frame")

    if spec.preprocess == vem.PRE_RESTYLE_FRAME0:
        img = cfg.get("style_image") or cfg.get("ref_image")
        if not img or not os.path.exists(img):
            raise IcJobError("immagine di stile mancante")
        if nseg > 1:
            p["ic_frame0_image"] = img                              # il worker la mette su OGNI segmento
        else:
            guide = os.path.join(work, "guide_f0.mp4")
            prep.replace_first_frame(src, img, guide)
    p["control_video"] = guide
    return IcJob(params=p, spec=spec, work_dir=work, mux_audio_from=mux, layout=layout, gamma=gamma,
                 paste_original=paste, feather=feather, source=src, notes=notes, segments=nseg)


def mux_audio(video: str, audio_src: str) -> str:
    """Rimette l'audio originale (copia video, aac sull'audio). Ritorna il file nuovo o `video` se fallisce."""
    import subprocess
    out = video[:-4] + "_audio.mp4"
    try:
        subprocess.run([prep.ffmpeg_path(), "-y", "-i", video, "-i", audio_src, "-c:v", "copy",
                        "-map", "0:v:0", "-map", "1:a:0?", "-shortest", out],
                       capture_output=True, text=True, timeout=1800)
    except Exception:                                               # noqa: BLE001
        return video
    return out if os.path.exists(out) and os.path.getsize(out) > 0 else video


def finalize(job: IcJob, result: str, log=None, cleanup: bool = True) -> str:
    """Post-processo del risultato del worker: outpaint (gamma inverso + originale incollato), mux audio.
    Ritorna il path finale. Cancella solo la cartella di lavoro creata da `build_job`."""
    log = log or (lambda s: None)
    out = result
    if job.layout is not None:
        cur = out
        if abs(job.gamma - 1.0) > 1e-6:
            inv = out[:-4] + "_invgamma.mp4"
            prep.inverse_gamma(cur, inv, job.gamma)
            cur = inv
        if job.paste_original:
            pasted = out[:-4] + "_paste.mp4"
            log("[outpaint] rimetto l'originale al centro"
                + (f" (bordo sfumato {job.feather} px)…" if job.feather > 0 else "…"))
            prep.paste_original(cur, job.source, pasted, job.layout, feather=job.feather)
            cur = pasted
        out = cur
    if job.mux_audio_from and os.path.exists(job.mux_audio_from):
        out = mux_audio(out, job.mux_audio_from)
    if cleanup:
        shutil.rmtree(job.work_dir, ignore_errors=True)
    return out
