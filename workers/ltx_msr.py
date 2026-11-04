"""MSR — Multiple Subject Reference per LTX (riferimenti multipli SENZA posizione temporale).

Le LoRA `LiconStudio/LTX-2.{3,5}-Multiple-Subject-Reference` fanno vedere al modello 2–5 immagini
di riferimento. **Le due release usano due meccanismi DIVERSI**, accertato aprendo i file e le
implementazioni di riferimento (17/08):

· **2.3 → pseudo-video** (plugin `ComfyUI-Licon-MSR`, nodo `LiconMSR`). La LoRA è una IC-LoRA
  normale: 960 tensori lora_A/lora_B, ZERO slot embedding. Le immagini diventano una sequenza
  video sintetica — i soggetti (adattati proporzionalmente e centrati su TELA BIANCA, mai
  ritagliati) occupano i primi frame latenti, il resto è lo sfondo — che entra come VIDEO-GUIDA
  IC-LoRA: esattamente il meccanismo che il worker usa già per canny/depth/pose. Lo sfondo è
  OBBLIGATORIO. In coda ComfyUI ritaglia i frame di guida (`LTXVCropGuides`).
· **2.5 → slot embedding** (plugin `ComfyUI-LTX2.5-MSR`). Lì la LoRA porta dentro una Fourier-MLP
  sotto `reference_slot_embedding.*`: ogni riferimento è codificato col VAE, sommato al proprio
  embedding di slot e appeso al latente a una posizione temporale NEGATIVA.

`strategy_for()` sceglie guardando il FILE, non la release: se un giorno esce una 2.3 con gli slot
embedding (o viceversa) il codice fa la cosa giusta senza modifiche.

⚠ Nessuna delle due è ancora stata validata con una generazione completa.
"""
from __future__ import annotations

import os

SLOT_PREFIX = "reference_slot_embedding."
MAX_REFERENCES = 5                      # 4 soggetti + 1 sfondo (2.3) · 5 slot (2.5)
MAX_SUBJECTS = 4                        # nel nodo LiconMSR gli ingressi soggetto sono 4
FRAME_COUNTS = (17, 25, 33, 41, 49, 57, 65)      # lunghezze ammesse della pseudo-sequenza


# ── quale meccanismo usa questa LoRA ──────────────────────────────────────────
def slot_state_dict(lora_path: str) -> dict:
    """Pesi della Fourier-MLP degli slot (chiavi senza prefisso). Vuoto se la LoRA non li ha."""
    from safetensors import safe_open
    out: dict = {}
    with safe_open(lora_path, framework="pt") as h:
        for k in h.keys():
            i = k.find(SLOT_PREFIX)
            if i >= 0:
                out[k[i + len(SLOT_PREFIX):]] = h.get_tensor(k)
    return out


def strategy_for(lora_path: str) -> str:
    """'slot_embedding' se il file porta gli slot appresi, altrimenti 'pseudo_video'."""
    return "slot_embedding" if "frequencies" in slot_state_dict(lora_path) else "pseudo_video"


# ── 2.3: pseudo-video (ricetta del nodo LiconMSR, replicata 1:1) ──────────────
def ref_latent_frames(frame_count: int) -> int:
    """Frame latenti coperti da una sequenza di `frame_count` pixel-frame (VAE temporale ×8)."""
    if frame_count <= 1:
        return max(1, frame_count)
    return int(round((frame_count - 1) / 8.0)) + 1


def latent_to_frame_range(latent_start: int, latent_end: int) -> tuple[int, int]:
    """Intervallo di pixel-frame coperto dai latenti [start, end]. Il latente 0 è il solo frame 0."""
    start = 0 if latent_start <= 0 else 1 + (int(latent_start) - 1) * 8
    end = 0 if latent_end <= 0 else int(latent_end) * 8
    return start, end


def allocate_subject_latents(num_subjects: int, budget: int) -> list[int]:
    """Quanti frame latenti a ciascun soggetto: 1 a testa, il primo ha priorità sugli extra."""
    counts = [1] * num_subjects
    extra = max(0, int(budget) - num_subjects)
    if extra > 0:
        counts[0] += 1
        extra -= 1
    i = 1
    while extra > 0 and num_subjects > 1 and any(c < 2 for c in counts[1:]):
        if counts[i] < 2:
            counts[i] += 1
            extra -= 1
        i = i + 1 if i + 1 < num_subjects else 1
    if extra > 0 and counts[0] < 3:
        counts[0] += 1
        extra -= 1
    i = 0
    while extra > 0:
        counts[i] += 1
        extra -= 1
        i = (i + 1) % num_subjects
    return counts


def frame_layout(num_subjects: int, frame_count: int) -> list[int]:
    """Per ogni pixel-frame, l'indice del soggetto che lo occupa (-1 = sfondo).
    È la funzione `_expand_frames` del nodo, in forma verificabile senza immagini."""
    layout = [-1] * frame_count
    if num_subjects <= 0:
        return layout
    latent_count = ref_latent_frames(frame_count)
    budget = max(0, latent_count - 1)
    if budget >= num_subjects:
        cursor = 0
        for idx, count in enumerate(allocate_subject_latents(num_subjects, budget)):
            start, end = latent_to_frame_range(cursor, cursor + count - 1)
            cursor += count
            for f in range(max(0, start), min(frame_count - 1, end) + 1):
                layout[f] = idx
    else:
        span = max(1, latent_to_frame_range(0, max(0, latent_count - 2))[1] + 1)
        for idx in range(num_subjects):
            start = int(idx * span / num_subjects)
            end = int((idx + 1) * span / num_subjects) - 1
            if idx == num_subjects - 1:
                end = span - 1
            for f in range(start, min(frame_count - 1, max(start, end)) + 1):
                layout[f] = idx
    return layout


def fit_on_white(img, width: int, height: int):
    """Soggetto: scalato per stare tutto dentro e centrato su bianco. MAI ritagliato (perderebbe
    parti del personaggio, che è esattamente ciò che il riferimento deve conservare)."""
    import numpy as np
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return np.ascontiguousarray(img)
    scale = min(width / w, height / h)
    rw = max(1, min(width, round(w * scale)))
    rh = max(1, min(height, round(h * scale)))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    top, left = (height - rh) // 2, (width - rw) // 2
    canvas[top:top + rh, left:left + rw] = _resize(img, rw, rh)
    return np.ascontiguousarray(canvas)


def cover_crop(img, width: int, height: int):
    """Sfondo: scalato fino a coprire e ritagliato al centro (nessuna banda bianca)."""
    import numpy as np
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return np.ascontiguousarray(img)
    scale = max(width / w, height / h)
    rw, rh = max(width, round(w * scale)), max(height, round(h * scale))
    r = _resize(img, rw, rh)
    top, left = (rh - height) // 2, (rw - width) // 2
    return np.ascontiguousarray(r[top:top + height, left:left + width])


def _resize(img, width: int, height: int):
    import numpy as np
    try:
        import cv2
        interp = cv2.INTER_AREA if (width < img.shape[1] or height < img.shape[0]) else cv2.INTER_LANCZOS4
        return cv2.resize(img, (width, height), interpolation=interp)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((width, height), Image.LANCZOS))


def _load_rgb(path: str):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def best_frame_count(num_frames: int) -> int:
    """Lunghezza ammessa più lunga che ENTRA nel video da generare.

    La pipeline IC-LoRA legge la guida con `frame_cap=num_frames`: una guida più lunga viene
    TRONCATA e gli ultimi soggetti spariscono senza un errore. Quindi la guida non deve mai
    superare il video."""
    ammesse = [f for f in FRAME_COUNTS if f <= max(1, int(num_frames))]
    return max(ammesse) if ammesse else min(FRAME_COUNTS)


def build_pseudo_video(subject_paths: list[str], background_path: str, out_path: str, *,
                       width: int = 736, height: int = 1280, frame_count: int = 41,
                       fps: int = 24, log=print) -> str:
    """Costruisce la pseudo-sequenza dei riferimenti e la scrive come mp4 (video-guida IC-LoRA)."""
    import numpy as np
    if not background_path:
        raise ValueError("MSR 2.3: serve un'immagine di SFONDO (ingresso 'background' del nodo "
                         "LiconMSR è obbligatorio).")
    if frame_count not in FRAME_COUNTS:
        frame_count = min(FRAME_COUNTS, key=lambda v: abs(v - frame_count))
    subs = [fit_on_white(_load_rgb(p), width, height) for p in subject_paths[:MAX_SUBJECTS]]
    bg = cover_crop(_load_rgb(background_path), width, height)
    layout = frame_layout(len(subs), frame_count)
    frames = np.stack([bg if i < 0 else subs[i] for i in layout])
    log(f"[msr] pseudo-video {width}x{height} {frame_count}f: "
        f"{len(subs)} soggetti su frame {[layout.index(i) for i in range(len(subs))]}, resto sfondo")
    _write_mp4(frames, out_path, fps)
    return out_path


def crop_guides_padding(guide_frames: int) -> int:
    """Fotogrammi da generare IN PIU e poi scartare dall'inizio perche la guida non finisca nell'uscita (`msr_crop_guides`, prova del 04/11).
    La guida pseudo-video e allineata nel tempo ai primi `guide_frames`: nei turni 33-34 l'uscita ricopiava per ~2,7 s i riferimenti sul fondo dello studio e finita la guida
    l'identita spariva (somiglianza al volto da 0,47 a 0,0). Con `guide_frames - 1` fotogrammi in piu (lunghezza ancora 8k+1) il primo fotogramma tenuto e l'ultimo della guida
    (lo sfondo) e il video vero comincia dopo i soggetti, come nel taglio `LTXVCropGuides` di ComfyUI."""
    return max(0, int(guide_frames) - 1)


def drop_leading_frames(video, count: int):
    """Toglie i primi `count` fotogrammi da un tensore [F,H,W,C] o da un iteratore di blocchi [f,H,W,C] (la forma che `encode_video` accetta). Ritorna lo stesso tipo."""
    import torch
    if count <= 0:
        return video
    if torch.is_tensor(video):
        return video[count:]

    def chunks():
        left = count
        for chunk in video:
            if left >= chunk.shape[0]:
                left -= chunk.shape[0]
                continue
            yield chunk[left:]
            left = 0
    return chunks()


def _write_mp4(frames, out_path: str, fps: int) -> None:
    import av
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.height, stream.width = int(frames.shape[1]), int(frames.shape[2])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "12"}                   # guida: meglio poca compressione
    for f in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ── 2.5: slot embedding a posizioni temporali negative ───────────────────────
def slot_embedding(state: dict, slot_id: int, device, dtype):
    """Fourier features (slot/16 → sin,cos) → Linear+SiLU → Linear. Ricetta del plugin 2.5."""
    import torch

    def t(key):
        v = state.get(key)
        return None if v is None else v.to(device=device, dtype=torch.float32)

    freq = t("frequencies")
    scaled = torch.tensor(float(slot_id), device=device, dtype=torch.float32) / 16.0
    phases = scaled * freq
    feats = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
    hidden = torch.nn.functional.silu(
        torch.nn.functional.linear(feats, t("net.0.weight"), t("net.0.bias")))
    emb = torch.nn.functional.linear(hidden, t("net.2.weight"), t("net.2.bias"))
    return emb.to(dtype=dtype)


def frame_offsets(num_slots: int, spacing: int = 1) -> list[int]:
    """Posizioni dei riferimenti: slot i → -(N-i)·spacing, cioè fuori dalla finestra del video.

    `spacing=1` è la ricetta del plugin: -N … -1, gli istanti IMMEDIATAMENTE precedenti al primo
    fotogramma. Le posizioni non sono indici ma coordinate temporali continue (keyframe_cond:
    `positions += frame_idx` poi `/= fps`), quindi con spacing 1 i riferimenti risultano contigui
    alla scena e il modello genera una transizione fluida fra loro e il video — il fading
    incrociato con i soggetti che compare a inizio o fine clip. Distanziandoli si verifica se
    quella contiguità è la causa.
    """
    return [-(num_slots - i) * max(1, int(spacing)) for i in range(num_slots)]


def references_to_inputs(references: list[dict], strength: float = 1.0, crf: int = 33,
                         spacing: int = 1) -> list:
    """`references` del pannello → ImageConditioningInput a frame NEGATIVI (solo 2.5)."""
    from ltx_pipelines.utils.args import ImageConditioningInput
    refs = [r for r in references if r.get("kind", "image") == "image"][:MAX_REFERENCES]
    offs = frame_offsets(len(refs), spacing)
    return [ImageConditioningInput(r["path"], off, float(r.get("strength", strength)), crf)
            for r, off in zip(refs, offs)]


def install_slot_embedding(lora_path: str, *, reference_frames: int = 1, log=print) -> bool:
    """Innesta lo slot embedding nella costruzione dei conditioning (solo LoRA che li contengono).

    Avvolge `combined_image_conditionings` — la funzione che entrambi gli stage di
    TI2VidTwoStagesPipeline usano — e intercetta le voci a frame NEGATIVO: le codifica, ci somma
    l'embedding dello slot e le appende come keyframe. Le altre immagini passano all'originale.
    La pipeline nativa regge gli indici negativi senza modifiche:
    `VideoConditionByKeyframeIndex.apply_to` fa `positions += frame_idx` e concatena, senza clamp.
    """
    state = slot_state_dict(lora_path)
    if "frequencies" not in state:
        log(f"[msr] {os.path.basename(lora_path)} non contiene '{SLOT_PREFIX}*' → "
            f"non è la variante a slot embedding")
        return False

    import ltx_pipelines.ti2vid_two_stages as _t2
    from ltx_core.conditioning import VideoConditionByKeyframeIndex
    from ltx_pipelines.utils.media_io.decode import load_image_and_preprocess

    if getattr(_t2, "_msr_patched", False):                  # idempotente (2 stage, stessa run)
        _t2._msr_state = state
        return True
    original = _t2.combined_image_conditionings

    def patched(images, height, width, video_encoder, dtype, device, color_space=None):
        normal = [im for im in images if im.frame_idx >= 0]
        refs = [im for im in images if im.frame_idx < 0]
        out = list(original(normal, height, width, video_encoder, dtype, device, color_space)) \
            if normal else []
        n = len(refs)
        for im in sorted(refs, key=lambda x: x.frame_idx):
            slot = n + im.frame_idx                          # -N→0, -1→N-1
            image = load_image_and_preprocess(
                image_path=im.path, height=height, width=width, dtype=dtype, device=device,
                crf=im.crf if im.crf is not None else 33, color_space=color_space)
            if reference_frames > 1:
                image = _repeat_frames(image, reference_frames)
            lat = video_encoder(image)
            lat = lat + slot_embedding(_t2._msr_state, slot, lat.device, lat.dtype).view(1, -1, 1, 1, 1)
            out.append(VideoConditionByKeyframeIndex(
                keyframes=lat, frame_idx=im.frame_idx, strength=im.strength))
        return out

    _t2.combined_image_conditionings = patched
    _t2._msr_patched = True
    _t2._msr_state = state
    log(f"[msr] slot embedding attivo ({len(state)} tensori da {os.path.basename(lora_path)})")
    return True


def _repeat_frames(image, n: int):
    """Ripete l'immagine su `n` frame lungo l'asse temporale (5-D: B,C,F,H,W; 4-D: B,C,H,W)."""
    if image.dim() == 5:
        return image.repeat_interleave(n, dim=2) if image.shape[2] == 1 else image
    return image.unsqueeze(2).repeat_interleave(n, dim=2)
