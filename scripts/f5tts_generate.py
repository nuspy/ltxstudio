"""Runner F5-TTS nativo (subprocess nel venv isolato) — voice cloning zero-shot, niente ComfyUI.

ref_audio + ref_text + gen_text → wav nella voce del riferimento. Repo: SWivid/F5-TTS (pip f5-tts).
"""
import argparse
import os

# torchcodec (usato da f5-tts/transformers per l'audio) cerca le DLL FFmpeg shared: su Windows le
# forniamo da E:\AI\Tools\ffmpeg-shared\bin (build shared 7.1). Senza, "libtorchcodec_core*.dll" fallisce.
_FFMPEG_BIN = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FFMPEG_BIN):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_FFMPEG_BIN)
    except (AttributeError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)              # testo da sintetizzare
    ap.add_argument("--ref_audio", required=True)         # campione voce (wav)
    ap.add_argument("--ref_text", default="")             # trascrizione del campione (vuoto = ASR auto)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    from f5_tts.api import F5TTS                           # import pesante: solo nel venv
    f5 = F5TTS()                                           # modello default (F5TTS_v1_Base)
    f5.infer(ref_file=a.ref_audio, ref_text=a.ref_text or "", gen_text=a.text, file_wave=a.output)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
