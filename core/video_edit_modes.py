"""Fonte UNICA di verita' delle modalita' del tab «Modifica Video AI» (Fase 1 IC-LoRA LTX).

Modulo PURO (nessun Qt, nessun torch, nessuna I/O): lo usano tab, worker GUI, API REST, worker di
inferenza (tetti) e test. Specifica: scratchpad orch02/f1/DESIGN.md §2.1.

Concetti
- `ModeSpec.key`        id della modalita' (stesso `key` storico per le 5 voci del vecchio MODES del tab).
- `ModeSpec.kind`       kind IC-LoRA del worker (`control_type`, registro `IC_LORA` di workers/ltx2_infer.py);
                        "" per le modalita' che passano da `edit_lora` (edit/inpaint/stile) invece che dal registro.
- `ModeSpec.edit_lora`  percorso RELATIVO a E:\\AI\\Models\\LoRA (solo per le modalita' storiche: stessi valori di prima).
- `ModeSpec.distilled`  "required" (la funzione non esiste senza checkpoint distilled) | "recommended"
                        (col dev la distilled-LoRA vale solo per lo stage 1: qualita' ridotta, vedi memoria
                        iclora-distilled-lora-mancante). La GUI avvisa/blocca, MAI cambia il modello da sola.
- `phase` / `enabled`   le modalita' non ancora implementate (lipdub 1b, hdr 1c) esistono come spec ma
                        `enabled=False`: non vanno mostrate come selezionabili.

TETTI (`max_long_edge`, `max_frames_per_segment`) — PRUDENTI, NON MISURATI dal vivo (DESIGN §7: i test live
li fissano). Ipotesi di lavoro: GPU 32 GB, checkpoint distilled fp8 residente (~23 GB) + stage 2 a risoluzione
piena, quindi attivazioni di una sola finestra corta:
- 1280 px sul lato lungo (720p 16:9 = 1280x704): oltre, lo stage 2 a 2x del latente stage-1 sfora la VRAM
  residua (oggi il tab non limita e un sorgente 1080p/4K va in OOM).
- 97 frame per segmento (~3,9 s a 25 fps; 97 = 8*12+1): meta' di 193, stima conservativa con attivazioni
  di un 704x1280 a 4 s; si scende a 73 per detailer (rank 256, LoRA 2,6 GB) e a 49/97 per modalita' che
  non si possono spezzare.
- `chunkable=False` dove il video non si puo' tagliare in segmenti (inpaint con maschere tracciate, lipdub
  con audio-reference, hdr): il tetto di frame vale per l'intero clip e la GUI deve RIFIUTARE (non troncare).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── famiglie (ordine = ordine nel combo «Famiglia») ───────────────────────────────────────────────
FAMILY_EDIT = "edit"
FAMILY_STYLE = "style"
FAMILY_REPAIR = "repair"
FAMILY_OUTPAINT = "outpaint"
FAMILY_LIPDUB = "lipdub"
FAMILY_CAMERA = "camera"
FAMILIES: tuple[str, ...] = (FAMILY_EDIT, FAMILY_STYLE, FAMILY_REPAIR, FAMILY_OUTPAINT,
                             FAMILY_LIPDUB, FAMILY_CAMERA)
FAMILY_I18N: dict[str, str] = {f: f"vedit.family.{f}" for f in FAMILIES}   # chiavi i18n famiglia

# valori dei campi enumerati
NEEDS_REQUIRED, NEEDS_OPTIONAL, NEEDS_NONE = "required", "optional", "none"
DISTILLED_REQUIRED, DISTILLED_RECOMMENDED = "required", "recommended"
PRE_NONE, PRE_OUTPAINT, PRE_RESTYLE_FRAME0 = "none", "outpaint", "restyle_frame0"

# Overlap di default fra segmenti (frame): >= 8 per il restauro (DESIGN §4.3)
DEFAULT_OVERLAP = 9


@dataclass(frozen=True)
class ModeSpec:
    key: str                                  # id modalita' (storico per le prime 5)
    family: str                               # una di FAMILIES
    i18n_key: str                             # etichetta modalita' (vedit.mode.<key>)
    kind: str = ""                            # control_type del worker ("" = via edit_lora)
    edit_lora: str = ""                       # relativo a E:\AI\Models\LoRA (solo modalita' storiche)
    catalog_id: str = ""                      # id del peso IC-LoRA nel registro (weights_catalog)
    note_key: str = ""                        # nota della modalita' (vedit.note_<key>)
    cap: str = ""                             # capability matrix (cap.*) per il gate; "" = non gated
    needs_prompt: str = NEEDS_OPTIONAL
    needs_ref_image: bool = False             # immagine di riferimento/stile obbligatoria
    needs_object: bool = False                # oggetto testuale (SAM2) obbligatorio
    needs_audio_in_video: bool = False        # il video DEVE avere una traccia audio
    needs_video: bool = True                  # False: il video principale e' opzionale (cameraman)
    distilled: str = DISTILLED_RECOMMENDED
    default_strength: float = 1.0
    default_prompt_key: str = ""              # chiave i18n del prompt precompilato ("" = nessuno)
    max_long_edge: int = 1280                 # tetto lato lungo (px); multiplo di 64
    max_frames_per_segment: int = 97          # 8n+1
    chunkable: bool = True                    # il video si puo' spezzare in segmenti (split+crossfade)
    preprocess: str = PRE_NONE
    keep_audio: bool = True                   # rimettere l'audio originale dopo la generazione (_mux_audio_from)
    phase: str = "1a"                         # fase di implementazione: 1a | 1b | 1c | 1d
    enabled: bool = True                      # False: spec presente ma funzione NON implementata
    min_long_edge: int = 0                    # risoluzione minima consigliata (cameraman: 960 sul lato lungo)
    experimental_key: str = ""                # chiave i18n di un avviso «sperimentale» (detailer)
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_ic_registry(self) -> bool:
        """True se il peso e' risolto dal registro IC-LoRA del worker (kind) e non da edit_lora."""
        return bool(self.kind)

    @property
    def available(self) -> bool:
        return self.enabled


def _s(**kw) -> ModeSpec:
    return ModeSpec(**kw)


_RESTORE = dict(family=FAMILY_REPAIR, needs_prompt=NEEDS_OPTIONAL,
                default_prompt_key="vedit.restore.prompt_default",
                max_long_edge=1280, max_frames_per_segment=97, chunkable=True, keep_audio=True)

MODE_SPECS: dict[str, ModeSpec] = {s.key: s for s in (
    # ── famiglia Modifica: le 3 storiche, stessi key/edit_lora del vecchio MODES del tab ──────────
    _s(key="edit", family=FAMILY_EDIT, i18n_key="vedit.mode.edit",
       edit_lora="LTX-2/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors",
       note_key="vedit.note_edit", needs_prompt=NEEDS_REQUIRED,
       max_long_edge=1280, max_frames_per_segment=121),
    _s(key="inpaint", family=FAMILY_EDIT, i18n_key="vedit.mode.inpaint",
       edit_lora="LTX-2/ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors",
       note_key="vedit.note_inpaint", needs_prompt=NEEDS_REQUIRED, needs_object=True,
       max_long_edge=1024, max_frames_per_segment=97, chunkable=False,   # maschere SAM2 per l'intero clip
       tags=("sam2", "masks")),
    _s(key="replace_char", family=FAMILY_EDIT, i18n_key="vedit.mode.replace_char",
       kind="union", catalog_id="ltx-iclora-union",
       edit_lora="LTX-2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
       note_key="vedit.note_replace_char", needs_prompt=NEEDS_REQUIRED, needs_ref_image=True,
       default_strength=0.6, max_long_edge=1280, max_frames_per_segment=97,
       tags=("canny", "ref_image")),
    # ── famiglia Stile ────────────────────────────────────────────────────────────────────────────
    _s(key="anime2real", family=FAMILY_STYLE, i18n_key="vedit.mode.anime2real",
       edit_lora="LTX-2/ltx23_anime2real_rank64_v1_4500.safetensors",
       note_key="vedit.note_style", needs_prompt=NEEDS_REQUIRED,
       max_long_edge=1280, max_frames_per_segment=121),
    _s(key="real2anime", family=FAMILY_STYLE, i18n_key="vedit.mode.real2anime",
       edit_lora="LTX-2/ltx23_real2anime_rank64_v1_5000.safetensors",
       note_key="vedit.note_style", needs_prompt=NEEDS_REQUIRED,
       max_long_edge=1280, max_frames_per_segment=121),
    _s(key="restyle", family=FAMILY_STYLE, i18n_key="vedit.mode.restyle",
       kind="restyle", catalog_id="ltx-iclora-restyle", note_key="vedit.note_restyle",
       cap="cap.video_restyle", needs_prompt=NEEDS_OPTIONAL, needs_ref_image=True,
       preprocess=PRE_RESTYLE_FRAME0, default_strength=1.0,       # card: forza 1.0 fissa
       max_long_edge=1280, max_frames_per_segment=97, tags=("style_image", "frame0")),
    # ── famiglia Ripara video (guida = sorgente, nessun preprocess) ───────────────────────────────
    _s(key="refocus", kind="refocus", catalog_id="ltx-iclora-refocus", i18n_key="vedit.mode.refocus",
       note_key="vedit.note_refocus", cap="cap.video_restore_iclora", **_RESTORE),
    _s(key="uncompress", kind="uncompress", catalog_id="ltx-iclora-uncompress",
       i18n_key="vedit.mode.uncompress", note_key="vedit.note_uncompress",
       cap="cap.video_restore_iclora", **_RESTORE),
    _s(key="deinterlace", kind="deinterlace", catalog_id="ltx-iclora-deinterlace",
       i18n_key="vedit.mode.deinterlace", note_key="vedit.note_deinterlace",
       cap="cap.video_restore_iclora", **_RESTORE),                  # card: «preview, poco testato»
    _s(key="motiondeblur", kind="motiondeblur", catalog_id="ltx-iclora-motiondeblur",
       i18n_key="vedit.mode.motiondeblur", note_key="vedit.note_motiondeblur",
       cap="cap.video_restore_iclora", **_RESTORE),
    _s(key="detailer", kind="detailer", catalog_id="ltx-iclora-detailer", i18n_key="vedit.mode.detailer",
       note_key="vedit.note_detailer", cap="cap.video_restore_iclora",
       experimental_key="vedit.note_detailer",                       # LoRA LTX-2 19B (rank 256) usata su 2.3
       **{**_RESTORE, "max_long_edge": 1024, "max_frames_per_segment": 73}),
    _s(key="hdr", kind="hdr", catalog_id="ltx-iclora-hdr", i18n_key="vedit.mode.hdr",
       note_key="vedit.note_hdr", cap="cap.video_hdr_ic", family=FAMILY_REPAIR,
       needs_prompt=NEEDS_NONE, distilled=DISTILLED_REQUIRED, max_long_edge=1024,
       max_frames_per_segment=97, chunkable=False, keep_audio=True,
       phase="1c", enabled=False, tags=("hdr_pipeline", "scene_emb")),
    # ── famiglia Outpaint video ───────────────────────────────────────────────────────────────────
    _s(key="outpaint", family=FAMILY_OUTPAINT, i18n_key="vedit.mode.outpaint", kind="outpaint",
       catalog_id="ltx-iclora-outpaint", note_key="vedit.note_outpaint", cap="cap.video_outpaint",
       needs_prompt=NEEDS_OPTIONAL, preprocess=PRE_OUTPAINT,
       max_long_edge=1280, max_frames_per_segment=97),               # tetto sul CANVAS finale
    # ── famiglia Lip dub (fase 1b: non implementata) ──────────────────────────────────────────────
    _s(key="lipdub", family=FAMILY_LIPDUB, i18n_key="vedit.mode.lipdub", kind="lipdub",
       catalog_id="ltx-iclora-lipdub", note_key="vedit.note_lipdub", cap="cap.video_lipdub",
       needs_prompt=NEEDS_REQUIRED, needs_audio_in_video=True, distilled=DISTILLED_REQUIRED,
       max_long_edge=1024, max_frames_per_segment=249, chunkable=False, keep_audio=False,
       phase="1b", enabled=False, tags=("lipdub_pipeline",)),
    # ── famiglia Camera / Movimento ───────────────────────────────────────────────────────────────
    _s(key="cameraman", family=FAMILY_CAMERA, i18n_key="vedit.mode.cameraman", kind="cameraman",
       catalog_id="ltx-iclora-cameraman", note_key="vedit.note_cameraman", cap="cap.video_camera_ic",
       needs_prompt=NEEDS_OPTIONAL, needs_video=False, default_strength=1.0,
       min_long_edge=960, max_long_edge=1280, max_frames_per_segment=121,
       chunkable=True, keep_audio=False, tags=("reference_video", "camera")),
)}

# i kind che il worker accetta per control_type con `_strict_ic` (esclusi canny/depth/pose, nel tab ControlNet)
ICLORA_VIDEO_KINDS: tuple[str, ...] = tuple(sorted(
    {s.kind for s in MODE_SPECS.values() if s.kind and s.enabled} | {"motion-track"}))


# ── API pubblica ──────────────────────────────────────────────────────────────────────────────────
def get_spec(key: str) -> ModeSpec:
    """Spec della modalita' `key`; KeyError con messaggio chiaro se sconosciuta."""
    try:
        return MODE_SPECS[key]
    except KeyError:
        raise KeyError(f"modalita' video-edit sconosciuta: {key!r} (note: {sorted(MODE_SPECS)})") from None


def spec_for_kind(kind: str) -> ModeSpec | None:
    """Spec abilitata che usa il kind IC-LoRA `kind` (control_type del worker), o None.
    Se piu' modalita' condividono il kind, vince la prima in ordine di definizione."""
    for s in MODE_SPECS.values():
        if s.kind == kind and s.enabled:
            return s
    return None


def modes_for_family(family: str, *, include_disabled: bool = False) -> list[ModeSpec]:
    """Modalita' di una famiglia nell'ordine di definizione. Le spec `enabled=False` (lipdub/hdr finche'
    non implementati) sono escluse salvo `include_disabled=True`."""
    return [s for s in MODE_SPECS.values()
            if s.family == family and (include_disabled or s.enabled)]


def families(*, include_empty: bool = False) -> list[str]:
    """Famiglie nell'ordine di FAMILIES; di default solo quelle con almeno una modalita' abilitata
    (cosi' «Lip dub» non compare finche' la fase 1b non e' fatta)."""
    return [f for f in FAMILIES if include_empty or modes_for_family(f)]


def family_of(key: str) -> str:
    return get_spec(key).family


def legacy_modes() -> list[tuple[str, str, str]]:
    """Le voci storiche `(key, i18n_key, edit_lora)` del vecchio `MODES` di ui/tabs/video_edit_ai_tab.py,
    nello stesso ordine di prima (edit, inpaint, anime2real, real2anime, replace_char): il tab puo'
    derivarle da qui e i casi GUI esistenti restano validi."""
    order = ("edit", "inpaint", "anime2real", "real2anime", "replace_char")
    return [(k, MODE_SPECS[k].i18n_key, MODE_SPECS[k].edit_lora) for k in order]


def clamp_resolution(width: int, height: int, spec: ModeSpec | str,
                     *, multiple: int = 64) -> tuple[int, int, bool]:
    """Riduce (mai ingrandisce) `width`x`height` al tetto `max_long_edge` mantenendo l'aspetto e
    arrotondando a multipli di `multiple` (default 64, richiesto dal two-stage). Ritorna
    `(w, h, ridotto)`; `ridotto` e' True se il risultato differisce dall'input arrotondato 64.
    Un sorgente gia' dentro il tetto viene solo arrotondato al multiplo piu' vicino (min 64)."""
    sp = get_spec(spec) if isinstance(spec, str) else spec
    w, h = max(1, int(width)), max(1, int(height))
    long_edge = max(w, h)
    scale = min(1.0, sp.max_long_edge / long_edge)

    def _r(x: float) -> int:
        return max(multiple, int(round(x / multiple)) * multiple)

    w2, h2 = _r(w * scale), _r(h * scale)
    # l'arrotondamento puo' riportare il lato lungo sopra il tetto (es. tetto 1280, 1290 -> 1280 ok;
    # ma 1.4*scale ...): se succede, scendi di un multiplo sul lato lungo
    while max(w2, h2) > sp.max_long_edge and max(w2, h2) > multiple:
        if w2 >= h2:
            w2 -= multiple
        else:
            h2 -= multiple
    return w2, h2, (scale < 1.0)


def snap_frames(n: int) -> int:
    """Frame validi LTX: il piu' grande 8k+1 <= n (min 9). 8k+1 = 9, 17, 25, ... 97, 121, 249."""
    n = int(n)
    if n < 9:
        return 9
    return ((n - 1) // 8) * 8 + 1


def segments_needed(total_frames: int, spec: ModeSpec | str, overlap: int = DEFAULT_OVERLAP) -> int:
    """Quanti segmenti servono per `total_frames` con il tetto della modalita' (1 se entra in uno)."""
    sp = get_spec(spec) if isinstance(spec, str) else spec
    seg = sp.max_frames_per_segment
    if total_frames <= seg or not sp.chunkable:
        return 1
    step = max(1, seg - max(0, int(overlap)))
    return 1 + -(-(int(total_frames) - seg) // step)
