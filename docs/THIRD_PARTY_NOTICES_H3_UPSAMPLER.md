# H3 Video Upsampler — attribuzioni

La funzione «MiniMax H3 Video Upsampler» (tab Definizione video, `core/h3_upsample_*.py`,
`workers/h3_upsample_*.py`) è un porting NATIVO (diffusers, nessun ComfyUI) della ricetta di
[ComfyUI-H3-Video-Upsampler](https://huggingface.co/sandpies/ComfyUI-H3-Video-Upsampler) di sandpies / dntpi (MIT),
generata da [ComfyUI-Hand-Tie-Clips](https://github.com/dntpi/ComfyUI-Hand-Tie-Clips) (MIT).

## Rete dell'upscaler latente e statistiche
**Autore originale: [LBH-123-AI](https://github.com/LBH-123-AI).**
Codice: `Comfyui_Minimax_h3_latent_Upscaler` (MIT). Pesi: `LBH-123-AI/Minimax_h3_latent_Upscaler` (Apache-2.0).
Nessun peso è distribuito con l'app: li scarica l'utente dal tab «Pesi modelli».
I pesi si caricano solo nel layout con cui sono stati addestrati: la struttura della rete (conv-in, blocchi residui con
embedding di scala, conv temporali depthwise, resize trilineare, conv-out, con i nomi delle chiavi) e le 24 medie /
24 deviazioni standard sono di LBH-123-AI (le medie/deviazioni coincidono con quelle del VAE video di MiniMax H3,
verificato da `tests/test_h3_upsample_plan.py`). Il resto è codice di questo progetto.

## Ricetta (segmenti, tile, giunture, un passo di ri-disegno)
La ricetta (segmenti di 102 fotogrammi con 17 di sovrapposizione, tile uguali nel budget di 70.000 token con
sovrapposizione 128 px, passi fusi insieme con finestra a coseno rialzato, un passo a denoise 0,1–0,2) è quella
dimostrata nel grafo [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)
di bbaudio-2025 (MIT) e implementata in `tiled_upsampler.py` / `video_upsampler.py` di sandpies / dntpi. Nessun codice
è copiato dal nodo di bbaudio-2025.

## Guida h3upscale
La geometria della guida (`target_crop_v2` / `target_grid_stride_v1`: il sorgente ridotto del fattore della LoRA,
impacchettato su ogni fattore-esima patch della griglia del target) è quella del fork ai-toolkit di
[Alissonerdx](https://huggingface.co/Alissonerdx), ramo `minimax-h3-latent-guides` (ai-toolkit: MIT). Nessun codice è
copiato; qui è ricostruita sul layout del pipeline diffusers di H3 e verificata riga per riga contro `diffusers`
(`tests/h3_upsample_diffusers_check.py`).

## `h3upscale_v2_x2.safetensors`
LoRA addestrata da Sandpies con quel fork sul dataset
[Alissonerdx/h3upscale-4k](https://huggingface.co/datasets/Alissonerdx/h3upscale-4k), sulla base MiniMax H3 Ref2VA.
Leggere i termini di entrambi prima di ridistribuirla o usarla commercialmente. Nessun peso è distribuito con l'app.

## MiniMax H3
Licenza «MiniMax H3 Community»: territori esclusi UE, Regno Unito, USA, Corea del Sud; vietato usare gli output per
addestrare altri modelli. Il codice è distribuito senza pesi: li scarica l'utente, che valuta la licenza per il proprio
territorio.
