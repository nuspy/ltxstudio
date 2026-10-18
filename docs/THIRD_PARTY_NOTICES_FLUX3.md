# FLUX.3 Image Klein — attribuzioni

La funzione «FLUX.3 Klein» (tab Immagini; `workers/flux3_infer.py`, `flux3_dit.py`, `flux3_pup.py`, `flux3_gguf.py`,
`core/flux3_*.py`, `ui/widgets/flux3_boxes.py`) usa il modello **FLUX.3 Image Klein** di Black Forest Labs Inc.

## Codice
- `workers/flux3_dit.py` è una riscrittura adattata (batch 1 / due campioni, configurazione parametrica, caricamento senza mmap) della
  rete `src/flux3/dit.py` e del linear fp8 per riga `src/flux3/fp8.py` del repository ufficiale
  <https://github.com/black-forest-labs/flux-3-image-klein> — (c) Black Forest Labs Inc., **Apache-2.0**. La parità numerica con
  `diffusers.Flux3Transformer2DModel` è provata da `tests/dev/flux3_dit_parity.py` (scarto massimo 3e-7).
- `core/flux3_caption.py` (formato dei riquadri, piano dell'upsampler, riconciliazione della risposta) è scritto dalla specifica
  (README HF/Comfy, `src/flux3/bbox.py` e `pup.py`, Apache-2.0): stessa semantica, codice nostro.
- `workers/flux3_gguf.py` applica a un GGUF le trasformazioni inverse di llama.cpp (`convert_hf_to_gguf`, MIT).

## Pesi (non distribuiti con l'app)
- DiT distillato `variants/fp8r` e text encoder Qwen3-VL-4B-Instruct (repo `black-forest-labs/flux-3-image-klein`): DiT sotto
  **FLUX Kommunity License v1.0** (uso commerciale degli output solo per «Qualifying User», ricavi annui lordi con affiliate sotto
  5.000.000 US$, con filtro o revisione dei contenuti e dichiarazione AI; vietato usare modello, derivati o output per addestrare altri
  modelli di immagini); text encoder Apache-2.0. Repo «gated: auto»: l'utente accetta la licenza su Hugging Face con il proprio
  account e scarica i pesi dal tab «Pesi modelli» con avviso e conferma (`license_ack`, `core/license_ack.py`). Nessun peso è
  nell'installer né nelle build.
- VAE: quello di FLUX.2 (`flux2-vae.safetensors`, stesso sha256 del VAE di FLUX.3).
- Prompt upsampler: Qwen3.5-9B (Apache-2.0) fine-tuned da BFL (`flux-3-image-klein-pup`, Apache-2.0); la variante abliterata di terzi
  `webmp3/Sakura-FLUX.3-Image-Klein-PromptUpsampler-Uncensored-GGUF` (Apache-2.0, «not-for-all-audiences») è in catalogo come
  candidata e NON è ancora confermata dall'utente. Dal repo ufficiale dell'upsampler si scaricano solo json/chat template/system prompt.
