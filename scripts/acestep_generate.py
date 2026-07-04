"""Runner ACE-Step nativo (text/tags + lyrics → canzone) — niente ComfyUI. Repo ace-step/ACE-Step.

Espone TUTTI i parametri del pipeline: qualità/aderenza (scheduler, cfg_type, omega, guidance shaping,
ERG, Double Condition Guidance), output (format, seed, batch), performance (quantized, overlapped_decode,
torch_compile) e i task di editing (retake/repaint/edit/audio2audio/extend) guidati da --task.
"""
import argparse, os
_FB = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FB):
    os.environ["PATH"] = _FB + os.pathsep + os.environ.get("PATH", "")
    try: os.add_dll_directory(_FB)
    except (AttributeError, OSError): pass


def _seeds(s):
    """Parsea "12,34" → [12,34]; vuoto/None → None (seed casuale)."""
    if not s:
        return None
    try:
        return [int(x) for x in str(s).replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)              # stile/genere/strumenti (prompt)
    ap.add_argument("--lyrics", default="")               # testo (vuoto = strumentale)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--guidance", type=float, default=15.0)
    # Double Condition Guidance: peso SEPARATO di testo-stile e liriche (attivo solo se >1.0).
    ap.add_argument("--gs-text", dest="gs_text", type=float, default=0.0)
    ap.add_argument("--gs-lyric", dest="gs_lyric", type=float, default=0.0)
    # Qualità / campionamento
    ap.add_argument("--scheduler", default="euler", choices=["euler", "heun", "pingpong"])
    ap.add_argument("--cfg-type", dest="cfg_type", default="apg", choices=["apg", "cfg", "cfg_star"])
    ap.add_argument("--omega", type=float, default=10.0)
    ap.add_argument("--guidance-interval", dest="guidance_interval", type=float, default=0.5)
    ap.add_argument("--guidance-interval-decay", dest="guidance_interval_decay", type=float, default=0.0)
    ap.add_argument("--min-guidance", dest="min_guidance", type=float, default=3.0)
    ap.add_argument("--no-erg-tag", dest="erg_tag", action="store_false")
    ap.add_argument("--no-erg-lyric", dest="erg_lyric", action="store_false")
    ap.add_argument("--no-erg-diffusion", dest="erg_diffusion", action="store_false")
    # Output / riproducibilità
    ap.add_argument("--format", default="wav", choices=["wav", "mp3", "ogg", "flac"])
    ap.add_argument("--seeds", default="")                # "12,34" → riproducibile; vuoto = casuale
    ap.add_argument("--batch", type=int, default=1)
    # Performance / VRAM
    ap.add_argument("--lazy", action="store_true", help="low-VRAM: CPU offload dei componenti")
    ap.add_argument("--quantized", action="store_true", help="modello quantizzato (low-VRAM, resta su GPU)")
    ap.add_argument("--overlapped-decode", dest="overlapped_decode", action="store_true")
    ap.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    # LoRA
    ap.add_argument("--lora-path", dest="lora_path", default="none")
    ap.add_argument("--lora-weight", dest="lora_weight", type=float, default=1.0)
    # Task di editing (retake/repaint/edit/audio2audio/extend)
    ap.add_argument("--task", default="text2music",
                    choices=["text2music", "retake", "repaint", "edit", "extend", "audio2audio"])
    ap.add_argument("--src-audio", dest="src_audio", default="")     # audio sorgente per repaint/edit/extend
    ap.add_argument("--retake-variance", dest="retake_variance", type=float, default=0.5)
    ap.add_argument("--repaint-start", dest="repaint_start", type=float, default=0.0)
    ap.add_argument("--repaint-end", dest="repaint_end", type=float, default=0.0)
    ap.add_argument("--edit-target-prompt", dest="edit_target_prompt", default="")
    ap.add_argument("--edit-target-lyrics", dest="edit_target_lyrics", default="")
    ap.add_argument("--ref-audio", dest="ref_audio", default="")     # audio2audio: riferimento di stile
    ap.add_argument("--ref-strength", dest="ref_strength", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import torch
    # GPU by default: l'offload su CPU (lento) si attiva SOLO se la VRAM è insufficiente (VRAM dinamica).
    _cuda = torch.cuda.is_available()
    _free = (torch.cuda.mem_get_info()[0] / (1024 ** 3)) if _cuda else 0.0
    need_offload = (not _cuda) or (_free < 12.0) or (bool(a.lazy) and _free < 24.0)
    print(f"[acestep] cuda={_cuda} free={_free:.1f}GB task={a.task} sched={a.scheduler} "
          f"quant={a.quantized} → cpu_offload={need_offload}", flush=True)
    from acestep.pipeline_ace_step import ACEStepPipeline
    pipe = ACEStepPipeline(dtype="bfloat16", torch_compile=bool(a.torch_compile),
                           cpu_offload=need_offload, quantized=bool(a.quantized),
                           overlapped_decode=bool(a.overlapped_decode))
    kw = dict(
        prompt=a.tags, lyrics=a.lyrics, audio_duration=a.duration,
        infer_step=a.steps, guidance_scale=a.guidance,
        guidance_scale_text=a.gs_text, guidance_scale_lyric=a.gs_lyric,
        scheduler_type=a.scheduler, cfg_type=a.cfg_type, omega_scale=a.omega,
        guidance_interval=a.guidance_interval, guidance_interval_decay=a.guidance_interval_decay,
        min_guidance_scale=a.min_guidance,
        use_erg_tag=a.erg_tag, use_erg_lyric=a.erg_lyric, use_erg_diffusion=a.erg_diffusion,
        format=a.format, manual_seeds=_seeds(a.seeds), batch_size=max(1, a.batch),
        lora_name_or_path=a.lora_path or "none", lora_weight=a.lora_weight,
        task=a.task, save_path=a.output,
    )
    # parametri specifici del task
    if a.task in ("repaint", "edit", "extend") and a.src_audio:
        kw["src_audio_path"] = a.src_audio
    if a.task in ("repaint", "extend"):
        kw["repaint_start"] = a.repaint_start; kw["repaint_end"] = a.repaint_end
    if a.task == "retake":
        kw["retake_variance"] = a.retake_variance
    if a.task == "edit":
        kw["edit_target_prompt"] = a.edit_target_prompt or a.tags
        kw["edit_target_lyrics"] = a.edit_target_lyrics or a.lyrics
    if a.task == "audio2audio" and a.ref_audio:
        kw["audio2audio_enable"] = True; kw["ref_audio_input"] = a.ref_audio
        kw["ref_audio_strength"] = a.ref_strength

    pipe(**kw)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
