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
    # LOCALE-FIRST (weights-unification C): il caricamento per nome passava dal ModelManager
    # Coqui → auto-download + possibile PROMPT TOS interattivo che bloccava il subprocess.
    # Ora: pesi dal catalogo (TTS\XTTS-v2); assenti = errore CHIARO, niente auto-download.
    _local = r"E:\AI\Models\TTS\XTTS-v2"
    _cfg = os.path.join(_local, "config.json")
    if not os.path.exists(_cfg):
        print(f"ERRORE: pesi XTTS-v2 assenti in {_local} — scaricali dal tab «Pesi modelli» "
              "(voce xtts-v2, ~2GB). Nessun download automatico.", flush=True)
        raise SystemExit(3)
    os.environ.setdefault("COQUI_TOS_AGREED", "1")         # niente prompt interattivo nel subprocess
    tts = TTS(model_path=_local, config_path=_cfg).to(dev)
    tts.tts_to_file(text=a.text, speaker_wav=a.ref_audio, language=a.language, file_path=a.output)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
