"""Runner F5-TTS nativo (subprocess nel venv isolato) — voice cloning zero-shot, niente ComfyUI.

ref_audio + ref_text + gen_text → wav nella voce del riferimento. Repo: SWivid/F5-TTS (pip f5-tts).
"""
import argparse


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
