"""MiniMax H3: GUIDA DI LATENTI ALLINEATA — parte pura (senza torch, senza Qt): quali LoRA la usano, trigger nel prompt, aritmetica dei fotogrammi.

Cos'e (README di Alissonerdx/Minimax-H3-ComfyUI, «Common setup»): le LoRA LMS (nitidezza), Style Transfer, VFX Edit, Head Swap e Motion Transfer «guide latents»
non ricevono il video sorgente come riferimento nativo (`<Video 1>`, con coordinate di tempo SUE) ma come blocco di latenti VAE PULITI che occupa la STESSA
griglia spaziale e temporale del video generato (in ComfyUI: nodo `MiniMaxH3AddGuide`, `frame_idx = 0`). Il lavoro sul modello sta in workers/h3_guide.py.

Qui: la tabella delle LoRA (trigger / caption fissa / se vogliono una foto), `apply_trigger` (il trigger entra nel prompt da solo: l'utente non lo scrive),
e i numeri della griglia (17n+5 fotogrammi, passi di latente 1,4,4,4,4). Licenza: tutto scritto qui da zero; il comportamento del nodo e stato letto
(sola lettura) da README/workflow di Alissonerdx (assets con licenze per file, v. LICENSES.md) e dal nodo di ComfyUI (GPL-3.0: NESSUN codice copiato).
"""
from __future__ import annotations

import math
import os

FPS = 24
FRAMES_PER_LATENT = (1, 4, 4, 4, 4)            # fotogrammi-pixel per passo di latente (ciclo di 5): 17n+5 fotogrammi -> 5n+2 passi

# caption FISSA della LoRA di nitidezza (docs/lms.md): e tutto il prompt, nessun trigger
LMS_CAPTION = "Enhance this video with sharp, crisp details while preserving a natural photorealistic appearance."

# token del nome file -> scheda (la PRIMA voce che combacia vince: «motion_transfer_ffp» sta prima di «motion_transfer»). `kind` e il nome mostrato; `picture`: "no" | "optional" | "required" (foto nel canale nativo <Picture 1>);
# `trigger`: prefisso del prompt; `caption`: prompt fisso (LMS). L'ordine conta: «tara» va controllato PRIMA di «motion_transfer».
GUIDE_LORAS = {
    "vfx_edit": {"kind": "VFX Edit", "trigger": "vfx_edit:", "picture": "optional", "label_key": "h3.guide.kind.vfx_edit"},
    "style_transfer": {"kind": "Style Transfer", "trigger": "style_transfer:", "picture": "optional", "label_key": "h3.guide.kind.style_transfer"},
    "head_swap": {"kind": "Head Swap", "trigger": "head_swap:", "picture": "required", "label_key": "h3.guide.kind.head_swap"},
    # FFP: prima immagine modificata (foto nel canale nativo) che si propaga a tutto il video; trigger vfx_edit: (README)
    "motion_transfer_ffp": {"kind": "Motion Transfer FFP", "trigger": "vfx_edit:", "picture": "required", "label_key": "h3.guide.kind.motion_transfer_ffp"},
    "motion_transfer": {"kind": "Motion Transfer", "trigger": "motion_transfer:", "picture": "required", "label_key": "h3.guide.kind.motion_transfer"},
    "_lms_": {"kind": "LMS (nitidezza)", "trigger": "", "caption": LMS_CAPTION, "picture": "no", "label_key": "h3.guide.kind.lms"},
}
# NON supportate (diverso meccanismo): Motion Transfer TARA = riferimento VIDEO nativo con le coordinate di tempo rimappate e una penalita di attenzione
# (patch del modello, `BFSMiniMaxH3NativeTemporal`), non una guida di latenti. Le fusioni con SolarRenderer (licenza MiniMax H3 Community con limiti
# territoriali) non sono cosa nostra da attivare in automatico.
NOT_GUIDE_TOKENS = ("_tara_", "motion_transfer_tara", "solarrenderer")


def guide_info(name: str) -> dict | None:
    """Scheda della LoRA di guida dal NOME del file (le LoRA H3 di guida non dichiarano la guida nelle chiavi dell'header: Ref2VA/FL2VA/guida
    hanno le stesse chiavi), oppure None se non e una LoRA di guida. `{"id", "kind", "trigger", "caption", "picture", "label_key"}`."""
    low = os.path.basename(str(name or "")).lower()
    if any(t in low for t in NOT_GUIDE_TOKENS):
        return None
    for tok, meta in GUIDE_LORAS.items():
        if tok in low:
            return {"id": tok.strip("_"), "kind": meta["kind"], "trigger": meta["trigger"], "caption": meta.get("caption", ""),
                    "picture": meta["picture"], "label_key": meta["label_key"]}
    return None


def is_tara(name: str) -> bool:
    low = os.path.basename(str(name or "")).lower()
    return "_tara_" in low or "motion_transfer_tara" in low


def guide_of_loras(loras: list | None) -> dict | None:
    """Scheda della (unica) LoRA di guida tra quelle scelte, o None. Due LoRA di guida DIVERSE insieme non hanno senso (trigger diversi): ValueError."""
    found: dict[str, dict] = {}
    for lo in loras or []:
        gi = guide_info(str((lo or {}).get("name") or (lo or {}).get("path") or ""))
        if gi:
            found[gi["id"]] = gi
    if len(found) > 1:
        raise ValueError("Due LoRA di guida diverse insieme (" + " + ".join(g["kind"] for g in found.values()) + "): ognuna vuole il suo trigger. Scegline una.")
    return next(iter(found.values()), None)


def apply_trigger(prompt: str, info: dict | None) -> str:
    """Il prompt con il trigger della LoRA di guida in testa (una sola volta, anche se l'utente l'ha scritto) o, per LMS, la caption fissa.
    Senza LoRA di guida il prompt resta IDENTICO."""
    if not info:
        return prompt
    if info.get("caption"):
        return info["caption"]
    trig = info.get("trigger") or ""
    body = (prompt or "").strip()
    if not trig:
        return prompt
    if body.lower().startswith(trig.lower()):
        return prompt if prompt.startswith(trig) else trig + body[len(trig):]
    return f"{trig} {body}".strip()


# ── griglia dei fotogrammi ───────────────────────────────────────────────────────────────────────
def grid_floor(frames: int) -> int:
    """Il clip H3 (17n+5) piu lungo contenuto in `frames` fotogrammi (0 se meno di 5)."""
    return 0 if frames < 5 else (frames - 5) // 17 * 17 + 5


def latent_steps(pixel_frames: int) -> int:
    """Passi di latente di `pixel_frames` fotogrammi: 1 fotogramma -> 1 passo; 17n+5 -> 5n+2."""
    if pixel_frames == 1:
        return 1
    if pixel_frames != grid_floor(pixel_frames):
        raise ValueError(f"{pixel_frames} fotogrammi non sono 1 ne della forma 17n+5")
    return (pixel_frames - 5) // 17 * 5 + 2


def latent_start_frames(total_frames: int) -> list[int]:
    """Indici-pixel a cui comincia un passo di latente (0, 1, 5, 9, 13, 17, 18, 22, ...): frame_idx deve cadere qui (il nodo non aggancia)."""
    out, acc, k = [], 0, 0
    while acc < total_frames:
        out.append(acc)
        acc += FRAMES_PER_LATENT[k % 5]
        k += 1
    return out


def plan_guide(source_frames: int, *, frame_idx: int = 0, max_frames: int = 0, target_frames: int = 0) -> dict:
    """Quanti fotogrammi della sorgente entrano come guida e quanti ne genera il video.

    · `target_frames` 0 (default, uso LMS/VFX/stile: «stessa durata della sorgente»): il video generato ha i fotogrammi della guida (17n+5, entro `max_frames` se dato);
    · `target_frames` > 0: il video e piu lungo/corto di cosi e la guida occupa `frame_idx...` (la guida non puo uscire dal video).
    Ritorna {"guide_frames", "frames", "frame_idx", "steps"}; ValueError col motivo se non e possibile."""
    n = int(source_frames)
    if max_frames:
        n = min(n, int(max_frames))
    gf = grid_floor(n)
    if gf < 5:
        raise ValueError(f"la guida ha {n} fotogrammi: ne servono almeno 5 (H3 usa clip da 17n+5)")
    frames = int(target_frames) if target_frames else gf
    if frames != grid_floor(frames):
        raise ValueError(f"il video da generare ({frames} fotogrammi) non e della forma 17n+5")
    fi = int(frame_idx)
    if fi < 0:
        fi = frames + fi                                           # negativo = dalla fine (come AddGuide)
    if fi not in latent_start_frames(frames):
        raise ValueError(f"frame_idx {frame_idx} non cade all'inizio di un passo di latente (validi: {latent_start_frames(frames)[:12]}...)")
    gf = min(gf, grid_floor(frames - fi)) if frames - fi >= 5 else 0
    if gf < 5:
        raise ValueError(f"la guida non entra nel video: frame_idx {fi}, video di {frames} fotogrammi")
    return {"guide_frames": gf, "frames": frames, "frame_idx": fi, "steps": latent_steps(gf)}


def guide_rows(steps: int, height: int, width: int, patch: tuple = (1, 2, 2), spatial: int = 16) -> int:
    """Righe (token) che la guida aggiunge alla sequenza: passi di latente x patch della griglia."""
    return int(steps) * math.ceil(height / spatial / patch[1]) * math.ceil(width / spatial / patch[2])


def spec_key(spec: dict | None) -> tuple:
    """Parte della chiave dei checkpoint dell'edit dovuta alla guida: VUOTA se non c'e (chiave e risultati di sempre invariati)."""
    if not spec:
        return ()
    return (("guide_v1", int(spec.get("frame_idx") or 0)),)
