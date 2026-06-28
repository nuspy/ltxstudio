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
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    from acestep.pipeline_ace_step import ACEStepPipeline
    pipe = ACEStepPipeline(dtype="bfloat16", torch_compile=False)
    pipe(prompt=a.tags, lyrics=a.lyrics, audio_duration=a.duration,
         infer_step=a.steps, guidance_scale=a.guidance, save_path=a.output)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
