"""Inpaint VIDEO (SAM2 + LoRA inpaint_masked) — logica CONDIVISA da GUI e REST API. Senza Qt, senza torch.

Due LoRA, scelte dall'UTENTE (con la foto di riferimento oppure senza), mai in automatico:
- **t2v** `ltx23_inpaint_masked_t2v_rank128_v1_10000steps`: l'area mascherata si descrive a parole (prompt);
- **r2v** `ltx23_inpaint_masked_r2v_rank32_v1_3000steps`: in piu' una FOTO di riferimento dell'oggetto/persona da inserire
  (reference-to-video). La foto va al worker come immagine di condizionamento (`params["image_path"]`, frame 0, forza
  `ref_strength`), cioe' `pipe.inpaint(images=[...])`. Verificato sul file: la LoRA non ha metadata (solo
  `reference_downscale_factor=1`), come l'autore la addestra a ricevere il riferimento NON e' documentato: per questo
  la modalita' di aggancio e' un parametro (`r2v_mode`, oggi solo «frame0») e il test dal vivo la valuta.

Il tracking SAM2 gira nel venv inference_server (`inference_server/backends/video_segment.py`) e salva frame e maschere in
`<video senza estensione>_track/`; il worker LTX (venv ltx2) li legge da `params["inpaint_masks_dir"]`.

Video lunghi: la maschera e' per-frame, quindi il worker spezza guida E maschere negli stessi segmenti
(`workers/ltx2_infer.py::_run_chunked_ic`, `mask_offset`); la foto di riferimento vale per OGNI segmento
(e' l'identita' dell'oggetto, non un keyframe del primo segmento).
"""
from __future__ import annotations

import os


LORA_DIR = r"E:\AI\Models\LoRA"
INPAINT_T2V_LORA = "LTX-2/ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors"
INPAINT_R2V_LORA = "LTX-2/ltx23_inpaint_masked_r2v_rank32_v1_3000steps.safetensors"
R2V_CATALOG_ID = "ltx23-inpaint-r2v"
INF_PY = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"

# Modalita' di aggancio della foto (DECISIONE: ipotesi piu' semplice, come replace_char). Cambia in una riga.
R2V_MODE = "frame0"
R2V_FIT = "pad"                   # "pad" = foto intera con bordi (default) | "crop" = ritaglio centrale del worker
R2V_FRAME = 0                     # frame del clip a cui si aggancia la foto (0 = primo; -1 = ultimo); env LTX_R2V_FRAME
R2V_DEFAULT_STRENGTH = 0.8        # = default del campo «Forza identita'» del tab (replace_char); da tarare dal vivo

# Segmentazione dei clip lunghi: stesso tetto della spec (97 frame = 8*12+1, ~3,9 s a 25 fps) e overlap standard.
SEGMENT_FRAMES = 97
SEGMENT_OVERLAP = 9


def has_reference(cfg: dict) -> bool:
    return bool(str(cfg.get("ref_image") or "").strip())


def lora_for(cfg: dict) -> str:
    """LoRA dell'inpaint: r2v se l'utente ha dato la foto, altrimenti la t2v (o `edit_lora` esplicita di `cfg`)."""
    if has_reference(cfg):
        return INPAINT_R2V_LORA
    return cfg.get("edit_lora") or INPAINT_T2V_LORA


def lora_abspath(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(LORA_DIR, rel.replace("/", os.sep))


def missing_weight(cfg: dict) -> str:
    """Messaggio d'errore se manca un peso necessario (LoRA r2v/t2v o foto), altrimenti ''. Nessun download."""
    msgs = []
    lp = lora_abspath(lora_for(cfg))
    if not os.path.exists(lp):
        if has_reference(cfg):
            msgs.append(f"Peso «Inpaint con foto di riferimento (r2v)» mancante: {lp}\n"
                        f"Prendilo dal tab «Pesi modelli» (id catalogo {R2V_CATALOG_ID}, circa 0,33 GB) "
                        "oppure togli la foto per usare l'inpaint a parole.")
        else:
            msgs.append(f"Peso dell'inpaint mancante: {lp}")
    if has_reference(cfg) and not os.path.isfile(str(cfg["ref_image"])):
        msgs.append(f"Foto di riferimento non trovata: {cfg['ref_image']}")
    return "\n".join(msgs)


def inpaint_params(cfg: dict, out: str) -> dict:
    """`params` per `workers/ltx2_infer.py` (senza il tracking SAM2, che aggiunge `attach_tracking`).

    cfg: prompt, guide (forza della guida), cfg_scale, mask_dilate, mask_blur, width, height,
    num_frames, fps, seed, ref_image, ref_strength, model/gguf_transformer/ltx_release (checkpoint come scelto)."""
    r2v = has_reference(cfg)
    p = {"prompt": cfg.get("prompt", ""), "negative": "worst quality, blurry",
         "edit_lora": lora_for(cfg),
         "control_strength": float(cfg.get("guide", 0.4)),
         "cfg_scale": float(cfg.get("cfg_scale", 6.0)),
         "mask_dilate": int(cfg.get("mask_dilate", 24)), "mask_blur": float(cfg.get("mask_blur", 6.0)),
         "width": int(cfg.get("width", 512)), "height": int(cfg.get("height", 320)),
         "num_frames": int(cfg.get("num_frames", 25)), "fps": int(cfg.get("fps", 25)),
         # niente offload forzato: 'cpu' = block-streaming = pinned memory => SEGFAULT col pagefile da 16GB (13/07)
         "seed": int(cfg.get("seed", 42)), "output": out, "skip_stage_2": False}
    for k in ("model", "gguf_transformer", "ltx_release"):     # checkpoint ESATTAMENTE come scelto nel tab Video
        if cfg.get(k):
            p[k] = cfg[k]
    if r2v:
        p["image_path"] = str(cfg["ref_image"])
        p["ref_strength"] = float(cfg.get("ref_strength", R2V_DEFAULT_STRENGTH))
        p["r2v_mode"] = R2V_MODE
        p["r2v_frame"] = int(os.environ.get("LTX_R2V_FRAME", R2V_FRAME))
        p["r2v_every_segment"] = True                          # la foto vale per ogni segmento dei clip lunghi
    if p["num_frames"] > SEGMENT_FRAMES:                       # clip lungo: segmenti di guida+maschere, foto su ognuno
        p["chunk_frames"] = SEGMENT_FRAMES
        p["overlap_frames"] = SEGMENT_OVERLAP
    return p


def track_dir_of(video: str) -> str:
    """Cartella in cui `video_segment.py` salva `frames/` e `masks/` (accanto al video sorgente)."""
    return os.path.splitext(video)[0] + "_track"


def attach_tracking(params: dict, src: str, obj: str, guide_out: str, dilate: int, *, run=None, log=None) -> dict:
    """Esegue il tracking SAM2 dell'oggetto `obj` su `src` e aggancia maschere e guida ai `params`.

    `run` (iniettabile nei test) = subprocess.run. Errore chiaro (RuntimeError) se il tracking fallisce."""
    import subprocess as _sp
    run = run or _sp.run
    if log:
        log("[edit] SAM2: traccio l'oggetto sul clip…")
    seg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "inference_server", "backends", "video_segment.py")
    # il dilate della GUI va anche al TRACKING: le maschere del guide devono combaciare con quelle che il worker usera'
    r = run([INF_PY, "-u", seg, src, obj, guide_out, str(int(dilate))],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = next((ln for ln in (r.stdout or "").splitlines() if ln.startswith("GUIDE:")), "")
    if not line:
        raise RuntimeError("tracking SAM2 fallito: " + (r.stderr or "")[-300:])
    td = track_dir_of(src)
    params["original_frames_dir"] = os.path.join(td, "frames")
    params["inpaint_masks_dir"] = os.path.join(td, "masks")
    params["control_video"] = src          # guide = video sorgente: senza, il worker degrada a ti2v (bug c6f8529, 14/10)
    if log:
        log("[edit] maschera tracciata → latent-inpaint")
    return params


def fit_reference(params: dict, out_dir: str, stamp, *, mode: str | None = None) -> str:
    """Adatta la FOTO di riferimento all'aspetto del video e la sostituisce in `params["image_path"]`. Ritorna una nota ('' = nulla).

    Il worker porta l'immagine al quadro con resize+ritaglio centrale (cover): una foto 1:1 su un clip 16:9 perde il 44% in
    altezza e un oggetto alto (ombrello, persona) esce tagliato. Con `mode="pad"` (DEFAULT, `R2V_FIT`) l'oggetto resta intero:
    la foto e' rimpicciolita a stare nel quadro e i lati liberi sono riempiti col colore mediano del bordo. `mode="crop"` = come
    prima (nessuna modifica). Aspetti entro il 15% restano com'erano."""
    mode = (mode or R2V_FIT).lower()
    src = params.get("image_path")
    if not src or mode != "pad":
        return ""
    from PIL import Image
    import numpy as np
    W, H = int(params.get("width", 0)), int(params.get("height", 0))
    im = Image.open(src).convert("RGB")
    iw, ih = im.size
    if not (W and H and iw and ih) or abs((iw / ih) / (W / H) - 1.0) <= 0.15:
        return ""
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    col = tuple(int(v) for v in np.median(border, axis=0))
    k = min(W / iw, H / ih)
    nw, nh = max(1, round(iw * k)), max(1, round(ih * k))
    canvas = Image.new("RGB", (W, H), col)
    canvas.paste(im.resize((nw, nh), Image.LANCZOS), ((W - nw) // 2, (H - nh) // 2))
    out = os.path.join(out_dir, f"_r2vref_{stamp}.png")
    canvas.save(out)
    params["image_path"] = out
    return (f"foto {iw}x{ih} adattata al quadro {W}x{H} con bordi (l'oggetto resta intero, nessun ritaglio): {os.path.basename(out)}")


def segments_info(num_frames: int) -> int:
    """Numero di segmenti con cui verra' girato un clip di `num_frames` frame."""
    n = int(num_frames)
    if n <= SEGMENT_FRAMES:
        return 1
    step = max(1, SEGMENT_FRAMES - SEGMENT_OVERLAP)
    return 1 + -(-(n - SEGMENT_FRAMES) // step)

