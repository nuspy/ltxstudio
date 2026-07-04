"""Runner ACE-Step nativo (text/tags + lyrics → canzone) — niente ComfyUI. Repo ace-step/ACE-Step."""
import argparse, os
_FB = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FB):
    os.environ["PATH"] = _FB + os.pathsep + os.environ.get("PATH", "")
    try: os.add_dll_directory(_FB)
    except (AttributeError, OSError): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)              # stile/genere/strumenti (prompt)
    ap.add_argument("--lyrics", default="")               # testo (vuoto = strumentale)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=27)
    ap.add_argument("--guidance", type=float, default=15.0)
    ap.add_argument("--lazy", action="store_true", help="low-VRAM: CPU offload dei componenti")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    import torch
    # GPU by default: l'offload su CPU (lento) si attiva SOLO se la VRAM è insufficiente (VRAM dinamica),
    # NON per un flag. ACE-Step 3.5B bf16 ≈ 8GB + attivazioni → serve ~12GB liberi per il full-GPU.
    _cuda = torch.cuda.is_available()
    _free = (torch.cuda.mem_get_info()[0] / (1024 ** 3)) if _cuda else 0.0
    need_offload = (not _cuda) or (_free < 12.0) or (bool(a.lazy) and _free < 24.0)
    print(f"[acestep] cuda={_cuda} free={_free:.1f}GB lazy={bool(a.lazy)} → cpu_offload={need_offload}", flush=True)
    from acestep.pipeline_ace_step import ACEStepPipeline
    pipe = ACEStepPipeline(dtype="bfloat16", torch_compile=False, cpu_offload=need_offload)
    pipe(prompt=a.tags, lyrics=a.lyrics, audio_duration=a.duration,
         infer_step=a.steps, guidance_scale=a.guidance, save_path=a.output)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
