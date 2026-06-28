"""Runner XTTS-v2 nativo (Coqui TTS) — voice cloning multilingue, niente ComfyUI.

speaker_wav + text (+ language) → wav nella voce del riferimento. Repo: coqui-ai/TTS (pip TTS).
Licenza modello CC-BY-NC (non commerciale) — uso personale/non-commerciale.
"""
import argparse
import os

# DLL FFmpeg shared per torchcodec/torchaudio su Windows (vedi f5tts_generate).
_FFMPEG_BIN = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FFMPEG_BIN):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_FFMPEG_BIN)
    except (AttributeError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref_audio", required=True)         # speaker_wav
    ap.add_argument("--language", default="it")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import torch
    from TTS.api import TTS                                # import pesante: solo nel venv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    tts.tts_to_file(text=a.text, speaker_wav=a.ref_audio, language=a.language, file_path=a.output)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
