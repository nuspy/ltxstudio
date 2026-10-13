"""Tooltip esplicativi dei pesi (weights-unification T) — cosa fa un modello e quando preferirlo.

Composizione, non testo libero: ogni voce di catalogo porta `use` = lista di TAG da un
VOCABOLARIO CONTROLLATO (~28 frasi). Il tooltip = frasi dei tag + righe strutturate (dimensione,
dipendenze auto-scaricate, gated, 18+, destinazione). Vantaggi: i18n sostenibile (si traducono
28 frasi ×15 lingue, non 114 testi), coerenza tra i TRE spazi di download (tab Pesi/wizard,
download-on-select nei dropdown, dialog primo-uso) e curatela nel CRUD (tools/catalog_manager).

Le frasi qui sotto sono i FALLBACK italiani; le chiavi i18n sono `weights.use.<tag>` e
`weights.tt.*` (15 lingue in ui/i18n).
"""
from __future__ import annotations

# tag → frase italiana (fallback; i18n key = weights.use.<tag>)
USE_TAGS: dict[str, str] = {
    "ckpt_video": "Checkpoint video: è il modello che genera i video.",
    "ckpt_image": "Checkpoint immagini: è il modello che genera le foto.",
    "distilled": "Distillato: veloce (8 step) — la scelta consigliata per l'uso quotidiano.",
    "dev_quality": "Dev (non distillato): massima fedeltà al prompt, più lento; richiede la distilled-LoRA.",
    "gguf_small": "Quantizzato GGUF: pesa molto meno e gira su GPU piccole; qualità leggermente inferiore.",
    "q_extreme": "Quantizzazione estrema: per VRAM molto scarsa; qualità da valutare.",
    "nvfp4": "FP4 Blackwell: residente e veloce sulle RTX 50xx (kernel FP4).",
    "int8_comfy": "Formato comfy (int8/fp8_scaled): per il motore ComfyUI, non per quello nativo.",
    "te": "Text encoder: traduce il prompt per il modello — componente necessario.",
    "te_quant": "Text encoder quantizzato: stessa funzione del bf16 con molta meno VRAM e disco.",
    "vae": "VAE video: converte i latenti in pixel — componente necessario.",
    "audio_vae": "VAE audio + vocoder: genera la traccia audio nativa dei video.",
    "upscaler_latent": "Upscaler latente x2: raddoppia la risoluzione durante la generazione.",
    "upscaler_temporal": "Upscaler temporale x2: raddoppia i fotogrammi (più fluidità).",
    "distill_lora": "LoRA di distillazione: OBBLIGATORIA coi checkpoint dev, inutile coi distilled.",
    "lora_style": "LoRA: aggiunge uno stile o una capacità al checkpoint (pannello LoRA).",
    "iclora_control": "IC-LoRA di controllo: guida la generazione con canny/depth/pose/riferimenti.",
    "iclora_edit": "IC-LoRA effetto: editing e restauro video (detailer, outpaint, HDR, lip-dub…).",
    "upscale_video": "Super-risoluzione video: migliora un video già generato.",
    "upscale_image": "Upscaler immagini (ESRGAN): ingrandisce e rifinisce le foto.",
    "faceres": "Restauro volti: corregge i volti generati o rovinati.",
    "face_tools": "Strumenti volto: face-swap e identità (detector + swapper).",
    "segment": "Segmentazione: trova e maschera oggetti da testo (SAM2 + GroundingDINO).",
    "depth_ctrl": "Preprocessore depth: mappa di profondità per il ControlNet.",
    "tts": "Voce: sintesi vocale e clonazione della voce da un campione.",
    "music": "Musica: generazione di canzoni e audio.",
    "threed": "3D: mesh e mondi generati da un'immagine.",
    "caption": "Caption VLM: descrive le immagini per i dataset di training LoRA.",
    "multisubject": "Riferimenti multi-soggetto: più personaggi coerenti nello stesso video.",
    "big_quality": "Qualità massima ma file enorme: nell'uso normale preferisci le varianti quantizzate.",
    "av_joint": "Video e audio stereo generati insieme in una sola passata: già sincronizzati.",
    "ref_omni": "Riferimenti multipli: fino a 9 immagini, 3 video e 3 clip audio per guidare "
                "identità, stile, movimento e voce.",
    "q_int8": "Quantizzazione int8 (ConvRot): circa metà della memoria del bf16, qualità vicina.",
    "fp8": "FP8: circa metà della memoria del bf16; veloce sulle RTX 40xx/50xx.",
    "avatar": "Avatar parlante: anima una foto sulla voce, con labiale sincronizzato.",
    "audio_enc": "Encoder audio: estrae dal parlato i segnali che guidano il labiale.",
    "multiref10": "Fino a 10 immagini di riferimento (persone, prodotti, stili) e trasparenza RGBA nativa.",
    "uncensored": "Senza censura (18+): nudo ed esplicito senza filtri; visibile solo in modalità NSFW.",
    "prompt_enhancer": "Arricchitore di prompt: un LLM riscrive la tua richiesta in un prompt dettagliato prima di generare.",
}


def _tr(key: str, fallback: str) -> str:
    try:
        from ui.i18n_manager import tr
        return tr(key, fallback)
    except Exception:                                    # noqa: BLE001 — contesti headless/CLI
        return fallback


def tooltip_for(entry: dict, *, with_dest: bool = True) -> str:
    """Tooltip completo per una voce di catalogo: frasi d'uso + righe strutturate.
    Usato identico nei tre spazi di download (coerenza) e nel CRUD (anteprima)."""
    from core import weights_catalog as wc
    lines: list[str] = [entry.get("label", entry.get("id", ""))]
    for tag in entry.get("use", []):
        base = USE_TAGS.get(tag)
        if base:
            lines.append("• " + _tr(f"weights.use.{tag}", base))
    note = entry.get("note", "")
    if note:
        lines.append("• " + note)
    gb = entry.get("approx_gb")
    if gb:
        lines.append(_tr("weights.tt.size", "Dimensione: ~{gb} GB").format(gb=gb))
    deps = entry.get("deps") or []
    if deps:
        lines.append(_tr("weights.tt.deps",
                         "Richiede: {deps} — selezionati e scaricati automaticamente")
                     .format(deps=", ".join(d.replace("|", " / ") for d in deps)))
    if entry.get("gated"):
        lines.append(_tr("weights.tt.gated", "Repo gated: accetta la licenza su HF e imposta il token."))
    if entry.get("sensitivity") == "adult":
        lines.append(_tr("weights.tt.adult", "Solo edizioni 18+."))
    if entry.get("no_remote"):
        lines.append(_tr("weights.tt.no_remote", "Senza fonte remota: usa Collega/Importa."))
    if wc.install_role(entry) == "base":
        lines.append(_tr("weights.tt.base", "Peso BASE: l'installer lo scarica per la tua edizione."))
    if with_dest:
        lines.append(_tr("weights.tt.dest", "Destinazione: {path}").format(path=wc.dest_for(entry)))
    return "\n".join(lines)


def short_line(entry: dict) -> str:
    """Riga breve (dialog primo-uso / CLI installer): label + primo tag d'uso + GB."""
    tags = entry.get("use", [])
    first = ""
    if tags:
        base = USE_TAGS.get(tags[0], "")
        first = " — " + _tr(f"weights.use.{tags[0]}", base).split(":", 1)[-1].strip()
    return f"{entry.get('label', entry.get('id'))} (~{entry.get('approx_gb', '?')}GB){first}"
