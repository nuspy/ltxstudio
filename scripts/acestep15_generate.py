"""Runner ACE-Step **1.5** (fondazione F-D) — text2music con planner 5Hz LM + DiT turbo.

SCAFFOLD: gira nel venv dedicato del sorgente OSS `E:\\AI\\Tools\\ACE-Step\\ACE-Step-1.5-src` (requirements.txt
del repo) e con i pesi in `E:\\AI\\Tools\\ACE-Step\\Ace-Step1.5` (acestep-v15-turbo/, acestep-5Hz-lm-1.7B/, vae/,
Qwen3-Embedding-0.6B/). API confermata dal repo (run_generate_test.py / acestep/inference.py):
  AceStepHandler().initialize_service(project_root, config_path="acestep-v15-turbo", device, offload_to_cpu)
  LLMHandler().initialize(checkpoint_dir, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device, offload_to_cpu, dtype)
  generate_music(dit, llm, params=GenerationParams(...), config=GenerationConfig(...), save_dir=...)

NB (da validare live a pesi scaricati): risoluzione dei path checkpoint da parte di initialize_service
(project_root/checkpoints/<config_path> vs dir diretta) — vedi acestep.handler.
"""
import argparse
import os
import sys

# sorgente OSS 1.5 (contiene il package acestep)
SRC_DIR = r"E:\AI\Tools\ACE-Step\ACE-Step-1.5-src"
CKPT_DIR = r"E:\AI\Tools\ACE-Step\Ace-Step1.5"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# CRITICO (connessione metered): indica al handler dove sono i pesi LOCALI, così NON prova ad auto-scaricare
# da HuggingFace (init_service_catalog.py rispetta ACESTEP_CHECKPOINTS_DIR). + offline mode di sicurezza.
os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", CKPT_DIR)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", required=True)               # stile globale (= i nostri "tag")
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--instrumental", action="store_true")
    ap.add_argument("--bpm", type=int, default=0)             # 0 = auto (None)
    ap.add_argument("--keyscale", default="")
    ap.add_argument("--timesignature", default="")
    ap.add_argument("--language", default="unknown")
    ap.add_argument("--duration", type=float, default=-1.0)   # -1 = auto
    ap.add_argument("--steps", type=int, default=8)           # turbo = 8
    ap.add_argument("--guidance", type=float, default=1.0)    # turbo: no CFG (≈1.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--think", default="true")               # planner 5Hz LM (CoT)
    ap.add_argument("--lm-model", dest="lm_model", default="acestep-5Hz-lm-1.7B")
    ap.add_argument("--dit-config", dest="dit_config", default="acestep-v15-turbo")
    ap.add_argument("--format", default="wav")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--output", required=True)               # cartella di salvataggio
    a = ap.parse_args()

    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    cuda = torch.cuda.is_available()
    free = (torch.cuda.mem_get_info()[0] / (1024 ** 3)) if cuda else 0.0
    offload = (not cuda) or (free < 20.0)                    # 1.5 (DiT 4B + LM 1.7B): offload sotto ~20GB liberi
    print(f"[acestep15] cuda={cuda} free={free:.1f}GB offload={offload} dit={a.dit_config} lm={a.lm_model}", flush=True)

    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=CKPT_DIR, config_path=a.dit_config,
                                     device="auto", offload_to_cpu=offload)
    if not ok:
        print(f"ERR: DiT init: {msg}", flush=True); sys.exit(1)

    llm = LLMHandler()
    msg, ok = llm.initialize(checkpoint_dir=CKPT_DIR, lm_model_path=a.lm_model,
                             backend="pt", device="auto", offload_to_cpu=offload, dtype=None)
    if not ok:
        print(f"ERR: LLM init: {msg}", flush=True); sys.exit(1)

    params = GenerationParams(
        task_type="text2music", thinking=(str(a.think).lower() == "true"),
        caption=a.caption, lyrics=("" if a.instrumental else a.lyrics), instrumental=a.instrumental,
        bpm=(a.bpm or None), keyscale=a.keyscale, timesignature=a.timesignature,
        vocal_language=a.language or "unknown", duration=a.duration,
        inference_steps=a.steps, guidance_scale=a.guidance, seed=a.seed,
    )
    config = GenerationConfig(batch_size=max(1, a.batch), audio_format=a.format)

    os.makedirs(a.output, exist_ok=True)
    result = generate_music(dit, llm, params=params, config=config, save_dir=a.output)
    if getattr(result, "success", False):
        for audio in result.audios:
            print(f"OK:{audio.get('path', '')}", flush=True)
    else:
        print(f"ERR: {getattr(result, 'status_message', 'generazione fallita')}", flush=True); sys.exit(1)


if __name__ == "__main__":
    main()
