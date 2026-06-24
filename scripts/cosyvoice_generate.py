"""Voce-clone CosyVoice2 headless — eseguito nel venv E:\\AI\\Tools\\CosyVoice.

zero-shot: testo + campione vocale (wav) + sua trascrizione → audio nella voce del campione.
Chiamato da core/cosyvoice_service.py via subprocess. Stampa `OK:<wav>`.
API: FunAudioLLM/CosyVoice → CosyVoice2.inference_zero_shot / inference_instruct2.
"""
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--prompt_wav", required=True)
    ap.add_argument("--prompt_text", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--instruct", default="")
    a = ap.parse_args()

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    model_dir = os.environ.get("COSYVOICE_MODEL", r"E:\AI\Models\TTS\CosyVoice2-0.5B")
    cv = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    prompt = load_wav(a.prompt_wav, 16000)

    if a.instruct:                                          # voce + istruzione di stile
        gen = cv.inference_instruct2(a.text, a.instruct, prompt, stream=False)
    else:                                                  # clone zero-shot puro
        gen = cv.inference_zero_shot(a.text, a.prompt_text, prompt, stream=False)

    chunks = [o["tts_speech"] for o in gen]
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    torchaudio.save(a.output, audio, cv.sample_rate)
    print("OK:" + a.output, flush=True)


if __name__ == "__main__":
    main()
