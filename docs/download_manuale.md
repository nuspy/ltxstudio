# Download manuale dei pesi (CDN HF lento) — indirizzi e destinazioni

Il backend nativo cerca i pesi nei percorsi sotto. Scarica MANTENENDO la struttura delle
sottocartelle del repo (config + safetensors). Ordinati per **priorità (impatto, più piccoli prima)**.

CLI consigliata (mantiene la struttura): `huggingface-cli download <repo> --local-dir "<dest>"`.
Eseguibile: `E:\AI\Tools\inference_server_venv\Scripts\huggingface-cli.exe`
(oppure `E:\AI\Tools\inference_server_venv\Scripts\python.exe -m huggingface_hub` ).
Mirror alternativo se il CDN è lento: anteporre `set HF_ENDPOINT=https://hf-mirror.com`.

---

## 1. Segmentazione (sblocca A.4) — ~1.6GB, NON gated
**GroundingDINO** — https://huggingface.co/IDEA-Research/grounding-dino-base
→ destinazione: `E:\AI\Models\Segmentation\hf\grounding-dino-base`
```
huggingface-cli download IDEA-Research/grounding-dino-base --local-dir "E:\AI\Models\Segmentation\hf\grounding-dino-base"
```
**SAM2** — https://huggingface.co/facebook/sam2-hiera-large
→ destinazione: `E:\AI\Models\Segmentation\hf\sam2-hiera-large`
```
huggingface-cli download facebook/sam2-hiera-large --local-dir "E:\AI\Models\Segmentation\hf\sam2-hiera-large"
```

## 2. Motore VIDEO nativo (sblocca Fase B) — NON gated
**Text-encoder Gemma-3 fp8** (~12GB) — https://huggingface.co/MISHANM/google-gemma-3-12b-it-fp8
→ destinazione: `E:\AI\Models\diffusers\_te\gemma3_ltx_fp8`
```
huggingface-cli download MISHANM/google-gemma-3-12b-it-fp8 --local-dir "E:\AI\Models\diffusers\_te\gemma3_ltx_fp8"
```
**Componenti LTX-2** (audio_vae/connectors/vocoder/upsampler — il transformer/vae li abbiamo già nel checkpoint locale) — https://huggingface.co/Lightricks/LTX-2
→ destinazione: `E:\AI\Models\diffusers\ltx2`
```
huggingface-cli download Lightricks/LTX-2 --local-dir "E:\AI\Models\diffusers\ltx2" --include "audio_vae/*" "connectors/*" "vocoder/*" "latent_upsampler/*" "tokenizer/*" "scheduler/*" "processor/*" "model_index.json"
```

## 3. FLUX.2 nativo (sblocca #2/#14) — ⚠️ GATED (serve login + accettare licenza)
1) Accetta la licenza sul sito: https://huggingface.co/black-forest-labs/FLUX.2-dev
2) `huggingface-cli login` (col tuo token HF)
3) Scarica SOLO il TE + config (transformer/vae li abbiamo locali) — https://huggingface.co/black-forest-labs/FLUX.2-dev
→ destinazione: `E:\AI\Models\diffusers\flux2`
```
huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir "E:\AI\Models\diffusers\flux2" --include "text_encoder/*" "tokenizer/*" "vae/config.json" "scheduler/*" "model_index.json"
```

## 4. Ernie nativo (sblocca #1) — ~16GB, NON gated
**ERNIE-Image** — https://huggingface.co/baidu/ERNIE-Image
→ destinazione: `E:\AI\Models\diffusers\Ernie\ERNIE-Image`
```
huggingface-cli download baidu/ERNIE-Image --local-dir "E:\AI\Models\diffusers\Ernie\ERNIE-Image"
```

---

### Note
- Riprende da dove interrotto se rilanci lo stesso comando (resume nativo di huggingface-cli).
- Dopo aver scaricato 1 e/o 2, avvisami: testo **segmentazione** e **video** nativi (codice già pronto: `image_segment.py`, `video_ltx.py`).
- I percorsi sopra sono quelli HARDCODED negli handler nativi — se cambi cartella, vanno aggiornati nel codice.
