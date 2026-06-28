"""Runner VibeVoice nativo — TTS/voice-clone via repo VibeVoice-Community. NB: API da validare al
primo uso (modello microsoft/VibeVoice-1.5B). DLL FFmpeg shared per torchcodec (vedi f5tts_generate)."""
import argparse, os
_FB = r"E:\AI\Tools\ffmpeg-shared\bin"
if os.path.isdir(_FB):
    os.environ["PATH"] = _FB + os.pathsep + os.environ.get("PATH", "")
    try: os.add_dll_directory(_FB)
    except (AttributeError, OSError): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True); ap.add_argument("--ref_audio", required=True)
    ap.add_argument("--output", required=True); a = ap.parse_args()
    # API VibeVoice-Community: processor + VibeVoiceForConditionalGeneration (HF). Voice sample → cloning.
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    import soundfile as sf
    mdl = "microsoft/VibeVoice-1.5B"
    proc = VibeVoiceProcessor.from_pretrained(mdl)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(mdl, device_map="cuda")
    inp = proc(text=[a.text], voice_samples=[[a.ref_audio]], return_tensors="pt").to("cuda")
    out = model.generate(**inp, tokenizer=proc.tokenizer)
    sf.write(a.output, out.speech_outputs[0].cpu().float().numpy().squeeze(), 24000)
    print(f"OK:{a.output}", flush=True)


if __name__ == "__main__":
    main()
