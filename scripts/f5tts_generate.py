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
    # LOCALE-FIRST (weights-unification C): F5TTS() senza argomenti scaricava SWivid/F5-TTS in
    # silenzio nella cache HF al primo uso. Ora: checkpoint dal catalogo (TTS\F5-TTS) se presente;
    # assente = errore CHIARO, niente auto-download (connessione metered).
    _local = r"E:\AI\Models\TTS\F5-TTS"
    _ck = os.path.join(_local, "F5TTS_v1_Base", "model_1250000.safetensors")
    _vb = os.path.join(_local, "F5TTS_v1_Base", "vocab.txt")
    if not os.path.exists(_ck):
        print(f"ERRORE: pesi F5-TTS assenti ({_ck}) — scaricali dal tab «Pesi modelli» "
              "(voce f5-tts, ~1.4GB). Nessun download automatico.", flush=True)
        raise SystemExit(3)
    f5 = F5TTS(ckpt_file=_ck, vocab_file=_vb if os.path.exists(_vb) else "")
    f5.infer(ref_file=a.ref_audio, ref_text=a.ref_text or "", gen_text=a.text, file_wave=a.output)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
