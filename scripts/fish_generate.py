"""Runner Fish-Speech nativo (2 stadi: text2semantic LLM + firefly-gan decode). NB: API da validare
al primo uso. DLL FFmpeg shared per torchcodec (vedi f5tts_generate)."""
import argparse, os, subprocess, sys
from pathlib import Path
_FB = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FB):
    os.environ["PATH"] = _FB + os.pathsep + os.environ.get("PATH", "")
    try: os.add_dll_directory(_FB)
    except (AttributeError, OSError): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True); ap.add_argument("--ref_audio", required=True)
    ap.add_argument("--ref_text", default=""); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    # fish-speech CLI a 2 stadi (vedi repo README): tools.llama.generate → codes; tools.vqgan/firefly → wav.
    # Wrapper best-effort dell'inference one-shot del repo (da validare/affinare al primo test live).
    repo = Path(r"E:\AI\Tools\fish-speech")
    py = sys.executable
    subprocess.run([py, "-m", "tools.api_client", "--text", a.text, "--reference-audio", a.ref_audio,
                    "--reference-text", a.ref_text, "--output", a.output], cwd=str(repo), check=True)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
