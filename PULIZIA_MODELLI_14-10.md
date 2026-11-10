# Pulizia modelli — cosa tenere e cosa cancellare (14/10/2026)

Niente è stato ancora cancellato. Questo file elenca le decisioni divise per motivazione; quelle della sezione 1 le hai già prese tu (doppioni), le sezioni 2–3 sono proposte verificate, la sezione 4 aspetta te.

Metodo: inventario di E:\AI\Models (esclusi gli LLM di app separate: vllm, LLM\unsloth…) + i 2 file in Downloads; 236 file non gestiti dal catalogo analizzati da 10 agenti Haiku 4.5 (Haiku 5.5 non esiste), poi verificati da Sonnet 5 col codice reale e ricontrollati da me (grep, header, confronto byte per byte).

## 1. Doppioni che hai indicato (confronto byte per byte fatto)

Dove la cartella o il codice hanno bisogno del file in quel punto (snapshot diffusers/HF, percorsi usati dai worker), il doppione diventa un **hardlink** alla copia che resta: il dato su disco esiste una volta sola (spazio liberato uguale) e nulla si rompe. Dove nessuno lo usa, si cancella davvero. Le coppie «già lo stesso file» erano GIÀ hardlink: cancellarle non libera nulla.

| File doppione | Copia che resta | Azione | GB liberati | Perché |
|---|---|---|---|---|
| `TextEncoders\gemma3\model.safetensors` | `TextEncoders\gemma_3_12B_it.safetensors` | hardlink | 12.77 | core/lyrics_llm.py:20 (cartella HF di riserva) |
| `TextEncoders\Flux\mistral_3_small_flux2.safetensors` | `TextEncoders\Flux\mistral_3_small_flux2_fp4_mixed.safetensors` | cancella | 11.43 | ridiretto nel catalogo e nel codice |
| `diffusers\_te\t5xxl\model.safetensors` | `TextEncoders\t5xxl_fp16.safetensors` | hardlink | 9.12 | inference_server/backends/image_flux.py:742 (T5 di FLUX.1) |
| `diffusers\Krea2\Krea-2-Turbo\text_encoder\model.safetensors` | `diffusers\Krea2\Krea-2-Raw\text_encoder\model.safetensors` | hardlink | 8.27 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `diffusers\_te\qwen3_4b\model.safetensors` | `TextEncoders\Qwen\qwen_3_4b.safetensors` | hardlink | 7.49 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `TextEncoders\HunyuanVideo\HunyuanVideo\text_encoder\model-00002-of-00004.safetensors` | `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00002-of-00004.safetensors` | cancella | 0 | HunyuanVideo abbandonato; era già lo stesso file della cache HF (si libera solo cancellando entrambe, vedi sez. 2) |
| `TextEncoders\HunyuanVideo\HunyuanVideo\text_encoder\model-00001-of-00004.safetensors` | `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00001-of-00004.safetensors` | cancella | 0 | HunyuanVideo abbandonato; era già lo stesso file della cache HF (si libera solo cancellando entrambe, vedi sez. 2) |
| `diffusers\LingBot\lingbot-video-dense-1.3b\text_encoder\model-00001-of-00002.safetensors` | `diffusers\LingBot\lingbot-video-moe-30b-a3b\text_encoder\model-00001-of-00002.safetensors` | hardlink | 4.63 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `TextEncoders\HunyuanVideo\HunyuanVideo\text_encoder\model-00003-of-00004.safetensors` | `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00003-of-00004.safetensors` | cancella | 0 | HunyuanVideo abbandonato; era già lo stesso file della cache HF (si libera solo cancellando entrambe, vedi sez. 2) |
| `diffusers\LingBot\lingbot-video-moe-30b-a3b\text_encoder\model-00002-of-00002.safetensors` | `diffusers\LingBot\lingbot-video-dense-1.3b\text_encoder\model-00002-of-00002.safetensors` | hardlink | 3.64 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `diffusers\LingBot\LingBot-Video-MoE-30B-A3B-SDNQ-uint4-static\text_encoder\model-00002-of-00002.safetensors` | `diffusers\LingBot\lingbot-video-dense-1.3b\text_encoder\model-00002-of-00002.safetensors` | hardlink | 3.64 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `TextEncoders\ltx-2.3_text_projection_bf16.safetensors` | `Checkpoints\LTX\ltx-2.3_text_projection_bf16.safetensors` | lascia | 0 | già lo stesso file (hardlink): 0 GB da guadagnare; il nome è usato dal codice |
| `clip_vision\clip_vit_large_patch14.safetensors` | `TextEncoders\Supir\model.safetensors` | cancella | 1.59 | nessun motore lo usa (MultiTalk cerca CLIP-H in VisionEncoders); anche la copia in TextEncoders\Supir va via con SUPIR |
| `VAE\LTX23_video_vae_bf16.safetensors` | `VAE\LTX23_video_vae_bf16_KJ.safetensors` | lascia | 0 | già lo stesso file (hardlink): 0 GB da guadagnare; il nome è usato dal codice |
| `latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `diffusers\ltx2\ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | cancella | 0.93 | versione 1.0 superata dalla 1.1 a catalogo (via anche la copia in diffusers\ltx2) |
| `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\vae\diffusion_pytorch_model.safetensors` | `VAE\HunyuanVideo\diffusion_pytorch_model.safetensors` | cancella | 0 | HunyuanVideo abbandonato; era già lo stesso file della cache HF (si libera solo cancellando entrambe, vedi sez. 2) |
| `Segmentation\grounding-dino\groundingdino_swinb_cogcoor.pth` | `Segmentation\groundingdino_swinb_cogcoor.pth` | cancella | 0.87 | i motori usano lo snapshot HF grounding-dino-base, non i .pth (via anche l'altra copia) |
| `checkpoint\Wan13B\Wan2.1-T2V-1.3B\Wan2.1_VAE.pth` | `FlashVSR\FlashVSR-v1.1\Wan2.1_VAE.pth` | hardlink | 0.47 | workers/self_forcing_infer.py:61 |
| `VAE\Wan2.1\Wan2.1_VAE.pth` | `FlashVSR\FlashVSR-v1.1\Wan2.1_VAE.pth` | hardlink | 0.47 | workers/self_forcing_infer.py (per nome) |
| `diffusers\Krea2\Krea-2-Turbo\vae\diffusion_pytorch_model.safetensors` | `diffusers\Krea2\Krea-2-Raw\vae\diffusion_pytorch_model.safetensors` | hardlink | 0.47 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `diffusers\LingBot\lingbot-video-dense-1.3b\vae\diffusion_pytorch_model.safetensors` | `diffusers\LingBot\LingBot-Video-MoE-30B-A3B-SDNQ-uint4-static\vae\diffusion_pytorch_model.safetensors` | hardlink | 0.47 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `diffusers\LingBot\lingbot-video-moe-30b-a3b\vae\diffusion_pytorch_model.safetensors` | `diffusers\LingBot\LingBot-Video-MoE-30B-A3B-SDNQ-uint4-static\vae\diffusion_pytorch_model.safetensors` | hardlink | 0.47 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `VAE\LongCat\diffusion_pytorch_model.safetensors` | `diffusers\LingBot\LingBot-Video-MoE-30B-A3B-SDNQ-uint4-static\vae\diffusion_pytorch_model.safetensors` | hardlink | 0.47 | file dentro una cartella-modello (snapshot) che il motore carica intera |
| `VAE\LTX23_audio_vae_bf16.safetensors` | `VAE\LTX23_audio_vae_bf16_KJ.safetensors` | lascia | 0 | già lo stesso file (hardlink): 0 GB da guadagnare; il nome è usato dal codice |
| `Other\HunyuanVideo\model.safetensors` | `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder_2\model.safetensors` | cancella | 0 | HunyuanVideo abbandonato; era già lo stesso file della cache HF (si libera solo cancellando entrambe, vedi sez. 2) |

**Spazio liberato dalla sezione 1: ~67.2 GB.**

## 2. CANCELLARE (proposta verificata) — 64 file, 146.3 GB

### Componente orfano — 16 file, 24.5 GB

Pezzi (text encoder, VAE, adattatori) di modelli che non abbiamo o che l'app non sa caricare.

| File | GB | Motivazione |
|---|---|---|
| `Audio\MelBandRoformer_fp32.safetensors` | 0.85 | Separatore voce MelBandRoformer per i workflow ComfyUI di MultiTalk: compare solo nella mappa nodi Comfy di core/template_engine.py:731. Il lip-sync nativo separa la voce con demucs/htdemucs (core/vocal_separation.py, workers/wan_multitalk.py:_vocals_audio). |
| `Audio\wav2vec2_large_english_fp16.safetensors` | 0.59 | Wav2Vec2 large inglese fp16 (formato Kijai per FantasyTalking/Wan): nessun riferimento nel codice (solo nello storico .oi4.json). MultiTalk usa chinese-wav2vec2-base (workers/wan_multitalk.py:213). |
| `diffusers\ltx2\audio_vae\diffusion_pytorch_model.safetensors` | 0.1 | Audio VAE di uno snapshot diffusers LTX-2 19B (model_index LTX2Pipeline, transformer a 48 layer) incompleto: transformer/ e vae/ contengono solo config.json. Nessun codice usa LTX2Pipeline/AutoencoderKLLTX2Audio; ltx2_infer.py usa i checkpoint ltx-core 2.3/2.5. |
| `diffusers\ltx2\connectors\diffusion_pytorch_model.safetensors` | 2.67 | Connettori testo dello snapshot diffusers LTX-2 19B incompleto (diffusers\ltx2): LTX2TextConnectors non è usato da nessun motore; i connettori LTX-2.3/2.5 vengono dai checkpoint ltx-core o da ltx2-embeddings-connector-distill. |
| `diffusers\ltx2\vocoder\diffusion_pytorch_model.safetensors` | 0.1 | Vocoder dello snapshot diffusers LTX-2 19B incompleto: nessun motore usa LTX2Pipeline; il vocoder LTX-2.3/2.5 viene dai checkpoint ltx-core o da ltx-2.5-audio-vae (+vocoder) a catalogo. |
| `HuggingFace\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594\model.safetensors` | 0.41 | Verificato con grep nel repo ('bert-base', 'bert_base', 'BertModel'): nessun risultato. Nessun motore di LTX Studio usa BERT; residuo di un esperimento o libreria terza che lo ha scaricato in cache HuggingFace. |
| `IPAdapter\Flux\xlabs_ip_adapter_v2.safetensors` | 0.98 | CORREZIONE: docs/upgrades-evaluation.md #105 valuta esplicitamente IP-Adapter per FLUX e decide di SCARTARLO ('skip salvo modelli SDXL'; ridondante con PuLID-FLUX gia' integrato). core/model_selection.py esclude esplicitamente i file 'ipadapter' dalla lista checkpoint (tupla _SKIP). Nessun loader lo carica davvero. |
| `LLM\Qwen-VL\Qwen3-VL-2B-Instruct\model.safetensors` | 3.96 | Nessun codice legge questa cartella: il prompt enhancer (core/prompt_enhance.py) usa Gemma-3 locale via core/lyrics_llm.py, e krea2_edit_pipeline.py prende dal repo HF solo il processor Qwen-VL, non i pesi. |
| `OpticalFlow\Wan2.1\raft-things.pth` | 0.02 | Nessun riferimento nel codice: grep su tutto il repo per 'raft'/'RAFT'/'OpticalFlow' non trova alcun uso. L'unico motore di interpolazione frame e' RIFE (core/rife.py), che carica esplicitamente 'flownet.pkl' (voce di catalogo rife-flownet-v4), non raft-things.pth. File orfano di un flusso ottico mai cablato nell'app nativa. |
| `Other\face_yolov9c.pt` | 0.08 | Nessun riferimento nel codice (grep 'face_yolov9c'/'yolov9' senza risultati). Il motore di rilevamento volti usato dall'app oggi e' InsightFace/antelopev2 (PuLID, face-swap); YOLOv9c-face e' un detector tipico dei nodi ComfyUI Impact-Pack, e l'app non usa piu' ComfyUI. Nessuna feature attuale lo richiama. |
| `Segmentation\DreamO\BEN2_Base.pth` | 1.06 | Rimozione sfondo BEN2 usata solo dalla pipeline DreamO, che l'app non ha; nessun riferimento nel codice (ui/main_window.py esclude anzi DreamO/BEN2 dalla lista SAM). |
| `Segmentation\sam2_hiera_large.pt` | 0.84 | Checkpoint SAM2 originale Meta (.pt): i backend nativi (inference_server/backends/image_segment.py, video_segment.py) caricano la snapshot HF Segmentation\hf\sam2-hiera-large (a catalogo, presente) e ui/main_window.py esclude esplicitamente i sam2_* dal loader SAM v1. |
| `Segmentation\Wan2.1\sam_vit_h_4b8939_fp16.safetensors` | 1.19 | Nessun riferimento nel codice a 'sam_vit_h' o 'vit_h_4b8939' (grep su tutto il repo senza risultati). E' il SAM (v1) ViT-H originale, leftover di vecchi workflow ComfyUI/VACE per Wan; l'app oggi usa SAM2 (transformers Sam2VideoModel) per tracking/maschere, non il SAM v1. |
| `TextEncoders\LTX\ltx2-embeddings-connector-distill.safetensors` | 2.67 | Connettore testo (chiavi model.diffusion_model.*_embeddings_connector) dello split Kijai di LTX-2 19B: nessun codice lo cita, e il transformer LTX-2.3 usato (ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled) contiene già 258 chiavi embeddings_connector. Non è un doppione dello snapshot diffusers\ltx2 (chiavi diverse). |
| `TextEncoders\Qwen\qwen_2.5_vl_7b_fp8_scaled.safetensors` | 8.74 | Text encoder di Qwen-Image 1.0 in formato ComfyUI: l'app ha solo Qwen-Image 2.1 (TE Qwen3-VL-8B a catalogo) e nessun DiT Qwen-Image 1.0; nessun riferimento nel codice. |
| `VAE\Qwen\qwen_image_vae.safetensors` | 0.24 | VAE di Qwen-Image 1.0 (0.24 GB): Qwen-Image 2.1, l'unico motore Qwen dell'app, usa il suo VAE RGBA qwen_image_2.1_vae_bf16.safetensors (0.63 GB, a catalogo); nessun DiT Qwen-Image 1.0 sul disco e nessun codice cita questo file. |

### Doppione — 13 file, 17.8 GB

Copie identiche (stesso hash/header) di un file che resta.

| File | GB | Motivazione |
|---|---|---|
| `LoRA\Flux\FluxCreative_(SwarmUI).safetensors` | 0.57 | Dati dei tensori identici (hash) a LoRA\Flux\FluxCreative.safetensors: la LoRA resta tramite quella copia. |
| `LoRA\Hunyuan\Titfuck_v2_Release_(SwarmUI).safetensors` | 0.15 | Doppione verificato byte-equivalente di Titfuck_v2_Release.safetensors (stessi 640 tensori BF16, stessa dimensione 0.15GB, stesso nome base con suffisso SwarmUI). La LoRA resta comunque disponibile nella copia principale (vedi voce precedente, DECIDE L'UTENTE), quindi il doppione esatto puo' essere rimosso senza perdita di co… |
| `LoRA\Wan\detailz-wan_(SwarmUI).safetensors` | 0.29 | Doppione verificato: stesso sshs_model_hash, stessi 800 tensori F16, stessa dimensione 0.29GB del file detailz-wan.safetensors nella stessa cartella. Si tiene l'originale, si elimina la copia SwarmUI. |
| `LoRA\Wan\masturbation_cumshot_wanI2V480p_v1_(LoRAs).safetensors` | 0.29 | Doppione verificato: stessi 800 tensori BF16, stessa dimensione 0.29GB del file masturbation_cumshot_wanI2V480p_v1.safetensors, solo con metadati Kohya azzerati (variante scaricata da un'altra fonte/'LoRAs'). Si tiene l'originale con metadati completi. |
| `LoRA\Wan\nippleSuckWan_e40_(LoRAs).safetensors` | 0.29 | Doppione verificato: stessi 800 tensori BF16, stessa dimensione 0.29GB di nippleSuckWan_e40.safetensors, solo metadati Kohya azzerati. Si tiene l'originale. |
| `LoRA\Wan\WAN-facefuck-epoch110 fast speed full insertion man has sex with womans mouth_(Framepack).safetensors` | 0.33 | Dati dei tensori identici (hash) alla copia senza suffisso (Framepack) nella stessa cartella LoRA\Wan: la LoRA resta tramite quella. |
| `Segmentation\groundingdino_swinb_cogcoor.pth` | 0.87 | Identico byte per byte (verifica dedup) a Segmentation\grounding-dino\groundingdino_swinb_cogcoor.pth, che resta nella cartella attesa da comfyui_segment_anything. |
| `TextEncoders\ltx-2-3-22b-text_encoder.safetensors` | 2.15 | Dati dei 4 tensori identici (hash) a TextEncoders\ltx-2.3_text_projection_bf16.safetensors, che è il file caricato per nome da workers/ltx_comfy_native.py:230 (clip2). Resta quella copia. |
| `TextEncoders\umt5_xxl_fp16.safetensors` | 10.59 | Dati identici (hash) a TextEncoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors, che nonostante il nome è anch'esso fp16 ed è il default di MultiTalk (workers/wan_multitalk.py:209). Resta quella copia. |
| `Upscale\RealESRGAN_x4plus.pth` | 0.06 | Caricati entrambi con torch: stessi 702 tensori con valori identici a RealESRGAN_x4plus.safetensors, che resta (lo carica il backend spandrel inference_server/backends/image_upscale.py). |
| `VAE\ltx-2-3-22b-audio_vae.safetensors` | 0.34 | Dati dei 1329 tensori identici (hash) a VAE\LTX23_audio_vae_bf16_KJ.safetensors (stessa famiglia del default LTX23_audio_vae_bf16 di workers/ltx_comfy_native.py:341). Resta la copia _KJ. |
| `VAE\ltx-2-3-22b-VAE.safetensors` | 1.65 | Stessi 170 tensori con dati identici (hash) a VAE\LTX23_video_vae_bf16_KJ.safetensors (default di workers/ltx_comfy_native.py:231); il file ha in più ~318 MB di byte in coda non indicizzati dall'header (per questo l'inventario lo segnava 'incompleto'). Resta la copia _KJ. |
| `VAE\Wan2.1\wan_2.1_vae.safetensors` | 0.24 | Dati identici (hash) a VAE\Wan\Wan2_1_VAE_bf16.safetensors, il VAE che MultiTalk carica per nome (workers/wan_multitalk.py:207). Resta quella copia. |

### Download incompleto o corrotto — 1 file, 16.8 GB

File troncati: inutilizzabili.

| File | GB | Motivazione |
|---|---|---|
| `C:\Users\atain\Downloads\h3ErosMax_beta3_int8.WohZOBgP.safetensors.part` | 16.82 | File .part troncato: 18061721600 B su 20975924960 attesi dall'header (86%). Inutilizzabile; a catalogo c'è già la 10Eros-Max H3 beta6 Turbo int8 completa (mmh3-dit-10erosmax-b6, presente). |

### Progetto abbandonato — 24 file, 65.4 GB

Pesi di progetti tolti dall'app (SUPIR, SD1.5/SDXL, HunyuanVideo, Wan 2.2, DreamO/BEN2): nessun motore li carica.

| File | GB | Motivazione |
|---|---|---|
| `Checkpoints\AnimateDiff\v3_sd15_mm.ckpt` | 1.56 | Motion module AnimateDiff v3 per SD1.5 (progetto abbandonato). Nel codice «AnimateDiff» compare solo come valore di formato VHS (core/extend_pipeline.py:150), nessun loader. |
| `Checkpoints\SD\dreamshaper_8.safetensors` | 1.99 | DreamShaper 8, checkpoint SD1.5 non NSFW: SD1.5 è tra i progetti abbandonati e nessun motore lo carica (presente solo nelle cache data/model_detect.json e model_vram_cache.json). |
| `ControlNet\controlnet-canny-sdxl.safetensors` | 2.33 | ControlNet Canny per SDXL (progetto abbandonato); il nome compare solo nello storico .oi4.json, nessun loader nel codice. |
| `ControlNet\controlnet-depth-sdxl.safetensors` | 2.33 | ControlNet Depth per SDXL (progetto abbandonato); nessun riferimento nel codice. |
| `ControlNet\controlnet-openpose-sdxl-xinsir.safetensors` | 2.33 | ControlNet OpenPose per SDXL (xinsir, progetto abbandonato); nessun riferimento nel codice. |
| `ControlNet\Wan2.1\Wan21_Uni3C_controlnet_fp16.safetensors` | 1.86 | ControlNet Uni3C per Wan2.1 14B (camera/posa): nessun motore lo usa (grep Uni3C: solo .oi4.json; workers/wan_multitalk.py non lo carica). Rientra in «Wan non usato da Krea Realtime/MultiTalk/Self-Forcing». |
| `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00001-of-00004.safetensors` | 4.64 | Componente HunyuanVideo nella cache HF: nessun motore Hunyuan nell'app (grep: solo etichette in core/auto_select.py, ui/tabs/video_tab.py, core/model_detect.py). È lo stesso file (hardlink/link) delle copie in TextEncoders\HunyuanVideo, Other\HunyuanVideo e VAE\HunyuanVideo: lo spazio si libera solo cancellando entrambe le co… |
| `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00002-of-00004.safetensors` | 4.66 | Componente HunyuanVideo nella cache HF: nessun motore Hunyuan nell'app (grep: solo etichette in core/auto_select.py, ui/tabs/video_tab.py, core/model_detect.py). È lo stesso file (hardlink/link) delle copie in TextEncoders\HunyuanVideo, Other\HunyuanVideo e VAE\HunyuanVideo: lo spazio si libera solo cancellando entrambe le co… |
| `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00003-of-00004.safetensors` | 4.58 | Componente HunyuanVideo nella cache HF: nessun motore Hunyuan nell'app (grep: solo etichette in core/auto_select.py, ui/tabs/video_tab.py, core/model_detect.py). È lo stesso file (hardlink/link) delle copie in TextEncoders\HunyuanVideo, Other\HunyuanVideo e VAE\HunyuanVideo: lo spazio si libera solo cancellando entrambe le co… |
| `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder\model-00004-of-00004.safetensors` | 0.11 | Componente HunyuanVideo nella cache HF: nessun motore Hunyuan nell'app (grep: solo etichette in core/auto_select.py, ui/tabs/video_tab.py, core/model_detect.py). È lo stesso file (hardlink/link) delle copie in TextEncoders\HunyuanVideo, Other\HunyuanVideo e VAE\HunyuanVideo: lo spazio si libera solo cancellando entrambe le co… |
| `HuggingFace\hub\models--hunyuanvideo-community--HunyuanVideo\snapshots\e8c2aaa66fe3742a32c11a6766aecbf07c56e773\text_encoder_2\model.safetensors` | 0.23 | Componente HunyuanVideo nella cache HF: nessun motore Hunyuan nell'app (grep: solo etichette in core/auto_select.py, ui/tabs/video_tab.py, core/model_detect.py). È lo stesso file (hardlink/link) delle copie in TextEncoders\HunyuanVideo, Other\HunyuanVideo e VAE\HunyuanVideo: lo spazio si libera solo cancellando entrambe le co… |
| `IPAdapter\ip-adapter-plus_sdxl_vit-h.safetensors` | 0.79 | IP-Adapter per SDXL: SDXL e' un motore abbandonato per regola utente, e lo stesso docs/upgrades-evaluation.md #105 conferma lo skip di IP-Adapter salvo reintroduzione di modelli SDXL (non prevista). |
| `LLM\Supir\llava-v1.5-7b\pytorch_model-00001-of-00002.bin` | 9.29 | Componente LLaVA del captioner di SUPIR, progetto esplicitamente elencato come abbandonato in agent_brief.md; verificato con grep ('supir','llava') nel repo: zero riferimenti nel codice. |
| `LLM\Supir\llava-v1.5-7b\pytorch_model-00002-of-00002.bin` | 3.3 | Come la parte 1: componente SUPIR abbandonato, zero riferimenti nel codice. |
| `LLM\Supir\mm_projector.bin` | 0.04 | Projector multimodale di SUPIR (connette LLaVA al resto della pipeline), stesso progetto abbandonato, zero riferimenti nel codice. |
| `TextEncoders\HunyuanVideo\HunyuanVideo\text_encoder\model-00004-of-00004.safetensors` | 0.11 | Parte 4/4 del text encoder (Llama-based) di HunyuanVideo. Come per il checkpoint principale, docs/MODEL_REVIEW_2026-07.md conferma che HunyuanVideo non e' cablato ad alcuna arch GUI ed e' superato da Wan/LTX-2; nessun motore attivo dell'app lo usa. |
| `TextEncoders\llama_3.1_8b_instruct_fp8_scaled.safetensors` | 8.46 | Llama-3.1-8B-Instruct fp8_scaled è il 4° text encoder tipico di HiDream (CLIP-L+CLIP-G+T5+Llama); HiDream è progetto abbandonato (agent_brief). JoyCaption (che userebbe Llama) carica da E:\AI\Models\JoyCaption in nf4, non da questo file in TextEncoders. Nessun grep nel repo. |
| `TextEncoders\long_clip_g_hi_dream.safetensors` | 1.29 | Nome esplicito 'hi_dream': componente del quartetto encoder di HiDream (CLIP-L+CLIP-G+T5-XXL+Llama-3.1), progetto abbandonato. Il Long-CLIP L per FLUX.1 è già a catalogo come file separato (long-clip-hi-dream, riga 89) e resta. Nessun riferimento di codice a questo file G. |
| `TextEncoders\Supir\model.safetensors` | 1.59 | SUPIR è progetto esplicitamente abbandonato (agent_brief); unico riferimento nel repo è il rilevatore generico core/model_detect.py, nessuna pipeline SUPIR attiva. |
| `TextEncoders\Supir\open_clip_pytorch_model.bin` | 9.46 | Componente OpenCLIP di SUPIR, progetto abbandonato; nessun uso nel codice oltre al rilevatore generico. |
| `TextEncoders\Supir\pytorch_model.bin` | 1.59 | Componente SUPIR, progetto abbandonato; nessun uso nel codice oltre al rilevatore generico. |
| `TextEncoders\Supir\pytorch_model_(Supir).bin` | 1.59 | Componente/doppione SUPIR, progetto abbandonato; nessun uso nel codice. |
| `VAE\HunyuanVideo\diffusion_pytorch_model.safetensors` | 0.92 | HunyuanVideo (verificato): unico riferimento nel repo è il rilevatore generico core/model_detect.py e la tabella VRAM core/model_precision.py, NESSUN worker/pipeline di inferenza HunyuanVideo esiste (nessun file hunyuan*_infer.py). Motore non presente nell'app. |
| `VAE\vae-ft-mse-840000-ema-pruned.safetensors` | 0.31 | VAE SD1.5/SDXL (vae-ft-mse-840000), architettura dichiarata abbandonata; l'app non ha alcun motore SD1.5/SDXL attivo. |

### Versione inferiore o superata — 10 file, 21.8 GB

Esiste già una versione migliore o più leggera dello stesso modello, a catalogo e in uso.

| File | GB | Motivazione |
|---|---|---|
| `Caption\clip_model.pt` | 1.6 | Encoder visivo SigLIP di JoyCaption alpha-two; superato da JoyCaption beta-one nf4 usato da core/caption_service.py (E:\AI\Models\JoyCaption). |
| `Caption\image_adapter.pt` | 0.08 | Proiettore immagine→LLM di JoyCaption alpha-two; superato da JoyCaption beta-one (core/caption_service.py). |
| `ControlNet\Z-Image\Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | 2.89 | Prima release del ControlNet Union Z-Image Turbo (136 tensori), superata dalla 2.1-8steps nella stessa cartella (295 tensori, release successiva alibaba-pai). Nessun codice usa nessuna delle due. |
| `Depth\Wan2.1\depth_anything_v2_vitb.pth` | 0.36 | DepthAnything V2 ViT-B (formato .pth originale): variante piccola di qualità inferiore alla ViT-L presente accanto; il codice usa solo la Large (DEPTH_DIR depth-anything-v2-large in core/controlnet_preprocess.py). |
| `diffusers\ltx2\latent_upsampler\diffusion_pytorch_model.safetensors` | 0.93 | Latent upsampler dello snapshot diffusers LTX-2 19B (v2.0): superato dall'upscaler LTX-2.3 x2 1.1 a catalogo (ltx23-spatial-upscaler, presente, ltx2_infer.py:894). Nessun codice usa diffusers\ltx2. |
| `diffusers\ltx2\ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 0.93 | Upscaler spaziale LTX-2.3 x2 v1.0: il codice usa solo la 1.1 (workers/ltx2_infer.py:894, core/ltx_release.py:86-87; ltx23-spatial-upscaler presente). Esiste una copia identica in latent_upscale_models: anche quella è superata. |
| `SeedVR2\SeedVR2-3B\ema_vae.pth` | 0.93 | core/seedvr2_service.py risolve MODEL_DIR dalla voce di catalogo 'seedvr2-3b-fp8' (cartella E:\AI\Models\SeedVR2, file ema_vae_fp16.safetensors, fp16, 0.5GB) e NON referenzia affatto la sottocartella 'SeedVR2-3B' ne' file .pth. Questo e' il VAE originale fp32 nativo (precursore), gia' sostituito dalla versione fp16 effettivam… |
| `SeedVR2\SeedVR2-3B\seedvr2_ema_3b.pth` | 12.63 | Stesso motivo del VAE accanto: core/seedvr2_service.py usa solo E:\AI\Models\SeedVR2\seedvr2_ema_3b_fp8_e4m3fn.safetensors (fp8, 3.2GB, voce catalogo seedvr2-3b-fp8) e non la sottocartella 'SeedVR2-3B'. Questo e' il checkpoint originale fp32 (12.63GB) gia' sostituito dalla versione fp8 effettivamente usata (regola utente: pre… |
| `Upscale\LTX\ltx-2-spatial-upscaler-x2-1.0.safetensors` | 0.93 | Il catalogo ha ltx23-spatial-upscaler v1.1 (ltx-2.3-spatial-upscaler-x2-1.1.safetensors, presente): questo è la v1.0 della stessa famiglia (arch ltx23), superata dalla versione cui punta core/ltx_release.py. |
| `VAE\Wan2.1\Wan2.1_VAE.safetensors` | 0.47 | Stesso VAE Wan 2.1 (stesse 194 chiavi) ma in fp32 (0.47 GB): MultiTalk carica per nome la versione bf16 VAE\Wan\Wan2_1_VAE_bf16.safetensors (workers/wan_multitalk.py:207), Self-Forcing usa Wan2.1_VAE.pth, Krea Realtime lo snapshot diffusers. Nessun codice cita questo file. |

## 3. TENERE — 113 file, 313.5 GB

### Cache generata dall'app — 1 file, 22.2 GB

File che l'app ricrea da sola: cancellarli fa solo rifare il lavoro.

| File | GB | Motivazione |
|---|---|---|
| `diffusion_models\Flux\fuxCapacityNSFWPornFlux_20_(SwarmUI).safetensors.normalized.safetensors` | 22.17 | Copia con i prefissi normalizzati che image_flux.py crea da sé al primo uso del checkpoint (riga 782): se la cancelli si rigenera (22 GB) la prossima volta che usi fuxCapacity. |

### Checkpoint NSFW da integrare — 3 file, 28.3 GB

Regola tua: tutti i checkpoint NSFW si tengono e si integrano (voce di catalogo + menu).

| File | GB | Motivazione |
|---|---|---|
| `C:\Users\atain\Downloads\bigLoveZ_zt1_pruned_fp8.safetensors` | 5.73 | Big Love Z versione ZT1 per Z-Image Turbo, fp8 E4M3, file completo (6154958752 B = 8+header+ultimo tensore; chiavi cap_embedder = Z-Image). È una versione diversa dalla Zora1 bf16 a catalogo (zimage-biglove-zora1, non scaricata, bloccata da Early Access) e già nel formato fp8 preferito. |
| `diffusion_models\Flux\getphat Flux Reality v23 Softcore FP8.safetensors` | 11.08 | Checkpoint FLUX.1 dev NSFW in fp8 (780 tensori F8_E4M3, 11.08GB): formato preferito per memoria/velocita' (regola utente fp8>bf16) e inoltre NSFW (si tiene sempre). |
| `diffusion_models\Z-Image\zimageTurboNSFWBy_2602NSFWBF16.safetensors` | 11.46 | Checkpoint Z-Image Turbo NSFW single-file bf16 (453 tensori), caricabile dallo stesso loader single-file di image_flux.py. Regola utente: i checkpoint NSFW si tengono sempre. |

### Funzionalità aggiuntiva — 22 file, 30.2 GB

Aggiungono qualcosa che oggi l'app non fa (o lo fa peggio). Alcune richiedono codice (loader).

| File | GB | Motivazione |
|---|---|---|
| `Checkpoints\Flux\FLUX_DEV_Redux.safetensors` | 0.12 | FLUX.1 Redux [dev] ufficiale (modelspec Flux.1-redux-dev, 4 tensori): variazioni e riferimento da immagine per FLUX.1. core/model_detect.py:128 lo riconosce come adattatore e frame1_tab.py:504 lo esclude dai menu, ma nessun codice lo carica. |
| `Checkpoints\Unknown\ernie-image-prompt-enhancer.safetensors` | 6.41 | Prompt enhancer ufficiale di ERNIE-Image (LLM, 237 tensori bf16): la cartella E:\AI\Models\diffusers\Ernie\ERNIE-Image\pe ha config e tokenizer ma non i pesi; _assemble_ernie (image_flux.py:928) oggi non lo carica. |
| `ControlNet\Flux\shakker_union_pro_v2.safetensors` | 3.99 | ControlNet Union Pro v2 (Shakker) per FLUX.1 (canny/depth/pose ecc.): l'app oggi non ha ControlNet per FLUX.1 (grep FluxControlNet vuoto), quindi aggiunge una funzione. Non è una LoRA (Haiku la chiamava LoRA). |
| `ControlNet\Openpose\body_pose_model.pth` | 0.19 | Pesi OpenPose body (lllyasviel): core/controlnet_preprocess.py:pose() è ancora uno stub NotImplemented e serve a produrre le mappe di posa per l'IC-LoRA Pose LTX (ltx-iclora-pose, presente). Questo file permette di implementarlo. |
| `ControlNet\Z-Image\Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors` | 6.25 | ControlNet Union 2.1 (8 passi) per Z-Image Turbo, alibaba-pai (modelspec z-image/control-diffpatch). Z-Image è supportato ma nessun codice carica ControlNet Z-Image: aggiunge il controllo strutturale. Non è una LoRA. |
| `DiffPitcher\Diff-Pitcher_transformer_pitch_360.safetensors` | 0.09 | Pitch predictor (Pitchformer, c'è Pitchformer.yaml accanto) del modulo DiffPitcher attivo (core/diffpitcher_service.py, scaffold). Il runner scripts/diffpitcher_run.py oggi carica solo world_fixed_40 e bigvgan, ma il file fa parte dello stesso pacchetto pesi. |
| `diffusion_models\Flux\nunchaku\svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors` | 5.4 | CORREZIONE: docs/FUTURE_CHECKS.md #3 documenta esplicitamente che la wheel nunchaku (stabile e nightly) non ha ancora le classi FLUX.2, quindi questo file e' tenuto in attesa di supporto upstream ('si integra quando upstream le pubblica'), non e' uno scarto ComfyUI abbandonato. |
| `diffusion_models\Z-Image\nunchaku\svdq-fp4_r128-z-image-turbo.safetensors` | 3.91 | CORREZIONE: inference_server/backends/image_flux.py ha gia' il codice per il motore nunchaku Z-Image (NunchakuZImageTransformer2DModel), ma docs/MODEL_REVIEW_2026-07.md segnala bug noto 'Z-Image svdq: KO su wheel stabile Windows' (assert rotary_emb), tracciato in docs/FUTURE_CHECKS.md #3. Non e' uno scarto ComfyUI: e' una fea… |
| `LoRA\krea2_identity_edit_v1_2.safetensors` | 1.7 | GIA' INTEGRATA E ATTIVA: workers/krea2_infer.py::_find_identity_lora() cerca con glob 'krea2_identity_edit*.safetensors' sotto E:\AI\Models\Lora e la applica automaticamente a peso 1.0 in ogni editing Krea2 per preservare l'identita' del soggetto. Nessuna azione richiesta. |
| `LoRA\SDXL\gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | 0.59 | NON è SDXL: è una LoRA del text encoder Gemma 3 12B (chiavi text_encoders.*, dim 3840) che lo rende «abliterated» (senza rifiuti) — il TE di LTX-2.3. Per usarla serve il supporto LoRA sul text encoder (oggi l'app applica le LoRA solo al transformer): da integrare. |
| `Upscale\2x_Text2HD_v.1-RealPLKSR.pth` | 0.03 | inference_server/backends/image_upscale.py carica QUALSIASI modello della cartella Upscale/ con spandrel (il docstring cita esplicitamente 'RealESRGAN, UltraSharp, ecc.'); file piccolo ma dimensione coerente con un modello RealPLKSR 2x leggero, non risulta troncato. |
| `Upscale\4x-AnimeSharp.safetensors` | 0.06 | Upscaler ESRGAN generico caricabile da image_upscale.py via spandrel; complementare a UltraSharp-v2 (ottimizzato anime invece che foto generiche), non un doppione. |
| `Upscale\4x_NMKD-Siax_200k.pth` | 0.06 | Upscaler ESRGAN generico, caricabile da image_upscale.py via spandrel come qualsiasi file in Upscale/. |
| `Upscale\4x_RealisticRescaler_100000_G.pth` | 0.12 | Upscaler ESRGAN generico, caricabile da image_upscale.py via spandrel. |
| `Upscale\4xFaceUpDAT.pth` | 0.14 | Upscaler DAT specializzato volti, caricabile da image_upscale.py via spandrel (supporta ESRGAN/DAT/HAT); complementare a CodeFormer/GFPGAN (quelli fanno restauro, questo super-risoluzione). |
| `Upscale\4xNomos2_hq_dat2.pth` | 0.13 | Upscaler DAT generico, caricabile da image_upscale.py via spandrel; diverso da nomos8k-schat-l a catalogo (Nomos2 vs Nomos8k, famiglie di training diverse), non doppione. |
| `Upscale\4xNomos2_hq_drct-l.safetensors` | 0.23 | Upscaler ESRGAN/DRCT generico, caricabile da image_upscale.py via spandrel; variante diversa dal precedente (drct-l vs dat2), non doppione. |
| `Upscale\4xNomos8kDAT.pth` | 0.29 | Upscaler DAT generico, caricabile da image_upscale.py via spandrel; stesso dataset Nomos8k ma architettura DAT invece di ESRGAN del cataloga nomos8k-schat-l, non un doppione byte-identico. |
| `Upscale\4xRealWebPhoto_v4_drct-l.pth` | 0.23 | Upscaler generico per foto, caricabile da image_upscale.py via spandrel. |
| `Upscale\HAT-L_SRx4_ImageNet-pretrain.safetensors` | 0.15 | Upscaler transformer HAT, architettura diversa da ESRGAN ma supportata da spandrel (inference_server/backends/image_upscale.py carica qualunque file via ModelLoader().load_from_file). |
| `Upscale\RealESRGAN_x2plus.pth` | 0.06 | ERRORE analista junior: RealESRGAN non è 'architettura SDXL abbandonata', è un upscaler ESRGAN indipendente dal modello di diffusione. inference_server/backends/image_upscale.py lo cita esplicitamente nel docstring ('RealESRGAN, UltraSharp, ecc.') tra i formati supportati da spandrel. |
| `Upscale\RealESRGAN_x4plus.safetensors` | 0.06 | ERRORE analista junior (vedi x2plus): upscaler ESRGAN generico supportato da image_upscale.py/spandrel, formato safetensors preferibile al .pth gemello. |

### LoRA da integrare (architettura supportata) — 57 file, 30.8 GB

Regola tua: tutte le LoRA si tengono; vanno solo nella cartella/arch giusta perché il pannello LoRA le mostri.

| File | GB | Motivazione |
|---|---|---|
| `Checkpoints\DreamO\diffusion_pytorch_model.safetensors` | 0.65 | È FLUX.1-Turbo-Alpha (alimama), LoRA di accelerazione FLUX.1-dev: 694082424 B, chiavi transformer.single_transformer_blocks.*.lora_A/B (formato diffusers). Utilizzabile come LoRA turbo 8 passi con FLUX.1. |
| `LoRA\deepthroat_epoch_80_(LoRAs).safetensors` | 0.29 | LoRA Wan 2.1 14B (rank 32, dimensione 5120, 40 blocchi — NON LTX come stimato da un analista): SPOSTATA da LoRA/ a LoRA/Wan/, la cartella che MultiTalk/Krea Realtime scansionano. |
| `LoRA\fingering_i2v_e248_(LoRAs).safetensors` | 0.33 | LoRA Wan 2.1 14B (rank 32, dimensione 5120, 40 blocchi — NON LTX come stimato da un analista): SPOSTATA da LoRA/ a LoRA/Wan/, la cartella che MultiTalk/Krea Realtime scansionano. |
| `LoRA\Flux\almuFLUX.safetensors` | 0.57 | almu FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\aphoFLUX.safetensors` | 0.57 | apho FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\blleFLUXReality Softcore.safetensors` | 0.57 | blle FLUX Reality Softcore LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\dvelFLUX.safetensors` | 0.57 | dvel FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\est3tic_lingerie_v1_FLUX-DEV_(SwarmUI).safetensors` | 0.04 | est3tic Lingerie v1 FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\hourglassv32_FLUX.safetensors` | 0.29 | Hourglass v3.2 FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\jpavFLUXReality.safetensors` | 0.57 | jpav FLUX Reality LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\lmilFLUX.safetensors` | 0.57 | lmil FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\mawiFLUX.safetensors` | 0.57 | mawi FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\micro-thong-panties-V2.safetensors` | 0.04 | Micro Thong Panties v2 LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\mnraFLUX.safetensors` | 0.57 | mnra FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\mohrFLUX.safetensors` | 0.57 | mohr FLUX LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\Moody-000019.safetensors` | 0.29 | Moody LoRA v1.0: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\MysticXXX-V3.safetensors` | 0.15 | Mystic XXX v3 LoRA: formato kohya-ss standard FLUX.1 (lora_unet_single/double_blocks), verificato in inventory.json; cartella E:\AI\Models\LoRA\Flux gia' scansionata dal codice (cls 'C cartella'). Architettura supportata, regola 1: le LoRA si tengono sempre. |
| `LoRA\Flux\OiledSkin_FluxDev_(SwarmUI).safetensors` | 0.32 | LoRA FLUX.1 NSFW, formato diffusers nativo (transformer.single_transformer_blocks.*.attn.to_k/q/v.lora_A/B, 988 tensori F16) identico a RealUnderboob_FL gia' confermata FLUX.1. Architettura supportata dall'app. |
| `LoRA\Flux\PoV_Fist.safetensors` | 0.29 | Header con pattern lora_unet_single_blocks_N_linearN/double_blocks (912 tensori BF16) identico a Sexy_Festival_Goers.safetensors e vmorFLUX.safetensors nella stessa cartella, entrambe FLUX.1 confermate: e' FLUX.1 formato Kohya, non 'architettura sconosciuta' come ipotizzato da Haiku. |
| `LoRA\Flux\RealUnderboob_FL.safetensors` | 0.16 | LoRA FLUX.1 NSFW, formato diffusers nativo (988 tensori F16, transformer.single_transformer_blocks.*). Cartella LoRA\Flux usata dall'app (core/training_service.py, model_importer.py). |
| `LoRA\Flux\riccFLUX.safetensors` | 0.57 | Stesso pattern lora_unet_single_blocks/double_blocks (912 tensori BF16) di Sexy_Festival_Goers, FLUX.1 confermata: non e' 'architettura sconosciuta'. |
| `LoRA\Flux\Sexy_Festival_Goers.safetensors` | 0.57 | LoRA FLUX.1 NSFW con metadati ss_base_model_version=flux1 corretti, formato Kohya standard. |
| `LoRA\Flux\sgmkFLUXReality.safetensors` | 0.57 | LoRA FLUX.1 NSFW stile 'Reality', stesso formato Kohya delle altre LoRA Flux confermate nella cartella. |
| `LoRA\Flux\SteepSlopeTiddies_v1-2_rank16_bf16.safetensors` | 0.06 | LoRA FLUX.1 NSFW compatta rank16 (0.06GB), metadati flux_dev.safetensors coerenti. |
| `LoRA\Flux\tan-lines-01.safetensors` | 0.5 | LoRA FLUX.1-dev NSFW, modelspec.architecture=Flux.1-dev/lora esplicito nei metadati. |
| `LoRA\Flux\underland-alice-000012.safetensors` | 0.04 | LoRA FLUX.1-dev NSFW compatta (personaggio), modelspec.architecture=Flux.1-dev/lora. |
| `LoRA\Flux\vmorFLUX.safetensors` | 0.57 | LoRA FLUX.1 NSFW, modelspec.implementation=GitHub FLUX, formato Kohya standard. |
| `LoRA\Flux\yanjFLUXReality Hardcore.safetensors` | 0.57 | LoRA FLUX.1 NSFW 'Hardcore', stesso formato Kohya delle altre LoRA Flux Reality confermate. |
| `LoRA\Flux\yanjFLUXReality Softcore.safetensors` | 0.57 | LoRA FLUX.1 NSFW 'Softcore', variante meno esplicita della Hardcore, stesso formato. |
| `LoRA\Flux\yschFLUXReality Softcore v11.safetensors` | 0.57 | LoRA FLUX.1 NSFW v1.1, training 4590 step, stesso formato Kohya. |
| `LoRA\grinding_pussyjob_e130_(LoRAs).safetensors` | 0.29 | LoRA Wan 2.1 14B (rank 32, dimensione 5120, 40 blocchi — NON LTX come stimato da un analista): SPOSTATA da LoRA/ a LoRA/Wan/, la cartella che MultiTalk/Krea Realtime scansionano. |
| `LoRA\LongCat\cfg_step_lora.safetensors` | 2.35 | LoRA ufficiale di LongCat-Video (motore LongCat Avatar attivo: core/longcat_service.py, workers/longcat_avatar_infer.py), ma la voce lcav15-core ammette solo lora/dmd_lora.safetensors e nessun codice la cita per nome. |
| `LoRA\LongCat\refinement_lora.safetensors` | 2.99 | LoRA ufficiale di LongCat-Video (motore LongCat Avatar attivo: core/longcat_service.py, workers/longcat_avatar_infer.py), ma la voce lcav15-core ammette solo lora/dmd_lora.safetensors e nessun codice la cita per nome. |
| `LoRA\MagicQuill\color_lora.safetensors` | 0.28 | Stesso formato delle altre LoRA MagicQuill nella cartella (380 tensori BF16, single_transformer_blocks.N.attn.processor.q/k/v_loras.N.down/up.weight): e' il formato FLUX.1-Kontext usato da core/kontext_pipeline.py, lo stesso di MagicQuill\puzzle_lora.safetensors che la GUI gia' carica come 'aux_lora' in ui/tabs/frame1_tab.py … |
| `LoRA\MagicQuill\edge_lora.safetensors` | 0.28 | Stesso formato delle altre LoRA MagicQuill nella cartella (380 tensori BF16, single_transformer_blocks.N.attn.processor.q/k/v_loras.N.down/up.weight): e' il formato FLUX.1-Kontext usato da core/kontext_pipeline.py, lo stesso di MagicQuill\puzzle_lora.safetensors che la GUI gia' carica come 'aux_lora' in ui/tabs/frame1_tab.py … |
| `LoRA\MagicQuill\local_lora.safetensors` | 0.28 | Stesso formato delle altre LoRA MagicQuill nella cartella (380 tensori BF16, single_transformer_blocks.N.attn.processor.q/k/v_loras.N.down/up.weight): e' il formato FLUX.1-Kontext usato da core/kontext_pipeline.py, lo stesso di MagicQuill\puzzle_lora.safetensors che la GUI gia' carica come 'aux_lora' in ui/tabs/frame1_tab.py … |
| `LoRA\MagicQuill\removal_lora.safetensors` | 0.28 | Stesso formato delle altre LoRA MagicQuill nella cartella (380 tensori BF16, single_transformer_blocks.N.attn.processor.q/k/v_loras.N.down/up.weight): e' il formato FLUX.1-Kontext usato da core/kontext_pipeline.py, lo stesso di MagicQuill\puzzle_lora.safetensors che la GUI gia' carica come 'aux_lora' in ui/tabs/frame1_tab.py … |
| `LoRA\T2V-POV_BJ-14B_(LoRAs).safetensors` | 0.29 | LoRA Wan 2.1 14B (rank 32, dimensione 5120, 40 blocchi — NON LTX come stimato da un analista): SPOSTATA da LoRA/ a LoRA/Wan/, la cartella che MultiTalk/Krea Realtime scansionano. |
| `LoRA\Wan2.2\DR34MJOB_I2V_14b_LowNoise.safetensors` | 0.29 | Cartella LoRA\Wan2.2 registrata esplicitamente in workers/wan_multitalk.py (add('loras', .../'LoRA\\Wan2.2')): architettura Wan 2.2 (basata su Wan 2.1, gia' supportata da MultiTalk/Krea Realtime) attivamente utilizzabile dall'app. |
| `LoRA\Wan2.2\FusionX_FaceNaturalizer.safetensors` | 0.57 | Cartella LoRA\Wan2.2 registrata esplicitamente in workers/wan_multitalk.py (add('loras', .../'LoRA\\Wan2.2')): architettura Wan 2.2 (basata su Wan 2.1, gia' supportata da MultiTalk/Krea Realtime) attivamente utilizzabile dall'app. |
| `LoRA\Wan2.2\wan2.2-i2v-high-cumshot-v1.0.safetensors` | 0.29 | Cartella LoRA\Wan2.2 registrata esplicitamente in workers/wan_multitalk.py (add('loras', .../'LoRA\\Wan2.2')): architettura Wan 2.2 (basata su Wan 2.1, gia' supportata da MultiTalk/Krea Realtime) attivamente utilizzabile dall'app. |
| `LoRA\Wan2.2\wan2.2-i2v-low-cumshot-v1.0.safetensors` | 0.29 | Cartella LoRA\Wan2.2 registrata esplicitamente in workers/wan_multitalk.py (add('loras', .../'LoRA\\Wan2.2')): architettura Wan 2.2 (basata su Wan 2.1, gia' supportata da MultiTalk/Krea Realtime) attivamente utilizzabile dall'app. |
| `LoRA\Wan2.2\wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.14 | Cartella LoRA\Wan2.2 registrata esplicitamente in workers/wan_multitalk.py (add('loras', .../'LoRA\\Wan2.2')): architettura Wan 2.2 (basata su Wan 2.1, gia' supportata da MultiTalk/Krea Realtime) attivamente utilizzabile dall'app. |
| `LoRA\Wan2.2\wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.14 | Cartella LoRA\Wan2.2 registrata esplicitamente in workers/wan_multitalk.py (add('loras', .../'LoRA\\Wan2.2')): architettura Wan 2.2 (basata su Wan 2.1, gia' supportata da MultiTalk/Krea Realtime) attivamente utilizzabile dall'app. |
| `LoRA\Wan\detailz-wan.safetensors` | 0.29 | LoRA Wan 2.1 detail-enhancement, cartella LoRA\Wan registrata da workers/wan_multitalk.py. Copia primaria mantenuta (vedi doppione detailz-wan_(SwarmUI)). |
| `LoRA\Wan\masturbation_cumshot_wanI2V480p_v1.safetensors` | 0.29 | LoRA NSFW Wan 2.1 14B I2V (modelspec.architecture=wan-2_1-text2video-14b/lora), copia con metadati completi mantenuta come primaria. |
| `LoRA\Wan\nippleSuckWan_e40.safetensors` | 0.29 | LoRA NSFW Wan 2.1 14B T2V/I2V (modelspec.architecture=wan-2_1-text2video-14b/lora), copia con metadati completi mantenuta come primaria. |
| `LoRA\Wan\Phantom_Wan_14B_FusionX_LoRA.safetensors` | 0.3 | LoRA FusionX per Wan 2.1 14B (non NSFW, stile/qualita'), cartella LoRA\Wan registrata da workers/wan_multitalk.py. |
| `LoRA\Wan\WAN-facefuck-epoch110 fast speed full insertion man has sex with womans mouth.safetensors` | 0.33 | LoRA NSFW Wan 2.1 T2V/I2V, cartella LoRA\Wan registrata da workers/wan_multitalk.py. |
| `LoRA\Wan\wan-nsfw-e14-fixed.safetensors` | 0.57 | LoRA NSFW Wan 2.1 (800 tensori BF16, modelspec.architecture=wan-2_1-text2video-14b/lora, trigger 'nsfwsks'). Arch supportata (Krea Realtime/MultiTalk). Tenere per regola utente. |
| `LoRA\Wan\wan-t2v-i2v-anal-combined-e50.safetensors` | 0.29 | LoRA NSFW Wan 2.1 T2V/I2V (800 tensori BF16, hash_sha256 0xdb32af93..., arch dichiarata wan-2_1-text2video-14b/lora). Arch supportata, tenere per regola utente. |
| `LoRA\Wan\Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High.safetensors` | 1.14 | LoRA turbo Wan 2.2 I2V A14B 4-step rank64 (hash 0x3ef073b7...). workers/wan_multitalk.py registra esplicitamente la cartella LoRA\Wan2.2 come percorso LoRA valido per ComfyUI-WanVideoWrapper (riga 46), quindi Wan 2.2 e' un'arch LoRA supportata dal worker MultiTalk anche senza un checkpoint Wan2.2 base dedicato nel catalogo. |
| `LoRA\Wan\Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low.safetensors` | 1.14 | Variante 'Low' (qualita'/peso diverso) dello stesso LoRA turbo Wan2.2 I2V, hash diverso (0xce39cb5c...) quindi non e' un doppione ma una variante distinta da lasciare scegliere all'utente. Stessa motivazione arch di cui sopra (LoRA\Wan2.2 registrata da workers/wan_multitalk.py). |
| `LoRA\Wan\Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_High.safetensors` | 1.14 | LoRA turbo Wan 2.2 T2V A14B 4-step rank64 (hash 0x595346e0...). Stessa motivazione: LoRA\Wan2.2 e' registrata come cartella LoRA valida in workers/wan_multitalk.py. |
| `LoRA\Wan\Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_Low.safetensors` | 1.14 | Variante 'Low' dello stesso LoRA turbo Wan2.2 T2V, hash diverso (0xb36ebbbf...), non doppione. Stessa motivazione arch Wan2.2 supportata da workers/wan_multitalk.py. |
| `LoRA\Wan\wan_cumshot_i2v.safetensors` | 0.33 | LoRA NSFW Wan 2.1 I2V (960 tensori BF16, hash 0xc1cd62d5..., titolo 'Wan Cumshot'). Arch supportata, tenere per regola utente. |
| `LoRA\Wan\WAN_dr34mj0b.safetensors` | 0.17 | LoRA NSFW Wan 2.1 I2V (960 tensori BF16, hash 0x197ea48a..., titolo 'WAN DR34MJOB'). Arch supportata, tenere per regola utente. |

### Usato dall'app ma fuori catalogo (da mettere a catalogo) — 22 file, 111.5 GB

Il codice li usa già (per cartella o per nome): vanno solo registrati nel catalogo, così il downloader e la matrice li conoscono.

| File | GB | Motivazione |
|---|---|---|
| `_gguf_companion\ltx2.5-uncensored-v1.1-distilled-Q8_0.companion.safetensors` | 3.76 | Companion generato automaticamente da workers/ltx2_infer.py:_gguf_companion() per il GGUF ltx2.5-uncensored-v1.1 Q8_0 (connettori/VAE/proiezioni che il GGUF non contiene); senza, il GGUF a catalogo non parte o va rigenerato. |
| `checkpoint\Wan13B\Wan2.1-T2V-1.3B\diffusion_pytorch_model.safetensors` | 5.29 | Transformer della base Wan2.1-T2V-1.3B formato nativo (la cartella contiene anche Wan2.1_VAE.pth e umt5): è la base wan21_13b_original che workers/self_forcing_infer.py:_resolve_paths cerca. Sta però in E:\AI\Models\checkpoint\Wan13B, mentre il catalogo la cerca in E:\AI\Models\Checkpoints\Wan13B, quindi l'app la vede come «n… |
| `diffusers\_te\clip_l\model.safetensors` | 0.23 | CLIP-L text encoder caricato dal motore FLUX.1 nativo: image_flux.py:739 CLIPTextModel.from_pretrained(TE_DIR\clip_l) con TE_DIR = E:\AI\Models\diffusers\_te. L'inventario lo dava «D nessuno» perché il percorso è costruito con os.path.join. Non è una LoRA. |
| `diffusion_models\Flux\fuxCapacityNSFWPornFlux_20_(SwarmUI).safetensors` | 11.08 | Checkpoint FLUX.1 NSFW fp8 E5M2 usato dall'app: image_flux.py:519 lo cita come «DEFAULT NSFW», con upcast E5M2 gestito a image_flux.py:765, e i tab lo filtrano per nome (frame1_tab.py:52, server/api.py:40). |
| `diffusion_models\ltx-2-3-22b-dev-Q6_K.gguf` | 16.55 | workers/gguf_native.py include il dequant per quantizzazione Q6_K (_q6_k) ed e' rilevato in model_detect.json (family ltxv, variant base, precision gguf). Il catalogo non ha alcuna voce per il checkpoint DEV di LTX-2.3 in GGUF (ha solo il distilled Q4_K_M): non e' una 'versione migliore' del distilled catalogato (sono variant… |
| `diffusion_models\Z-Image\Z_Image_BF16.safetensors` | 11.46 | inference_server/backends/image_flux.py carica Z-Image single-file con _from_single_file_robust(ZImageTransformer2DModel,...) quando il path non e' un file 'svdq': questo bf16 single-file e' direttamente caricabile. Il catalogo ha solo 'zimage-turbo' come snapshot diffusers 'non scaricato'. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00001-of-00015.safetensors` | 3.7 | Shard 1/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00002-of-00015.safetensors` | 3.65 | Shard 2/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00003-of-00015.safetensors` | 3.65 | Shard 3/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00004-of-00015.safetensors` | 3.65 | Shard 4/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00005-of-00015.safetensors` | 3.65 | Shard 5/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00006-of-00015.safetensors` | 3.63 | Shard 6/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00007-of-00015.safetensors` | 3.72 | Shard 7/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00008-of-00015.safetensors` | 3.61 | Shard 8/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00009-of-00015.safetensors` | 3.65 | Shard 9/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00010-of-00015.safetensors` | 3.65 | Shard 10/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00011-of-00015.safetensors` | 3.65 | Shard 11/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00012-of-00015.safetensors` | 3.65 | Shard 12/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00013-of-00015.safetensors` | 3.72 | Shard 13/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00014-of-00015.safetensors` | 3.67 | Shard 14/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `text_encoder\Qwen\Qwen3.6-27B\model-00015-of-00015.safetensors` | 0.47 | Shard 15/15 di Qwen3.6-27B (download completo: 15 shard + config/tokenizer), base del rewriter prompt di LingBot (workers/lingbot_rewriter.py, core/prompt_rewrite_lingbot.py) = voce catalogo lingbot_rewriter_base, che però lo cerca in TextEncoders\Qwen. |
| `TextEncoders\Flux\mistral_3_small_flux2_fp4_mixed.safetensors` | 11.43 | È la copia che hai scelto di tenere: il catalogo e 8 file di codice vengono ridiretti qui. Nonostante i nomi, entrambe le copie sono fp4-mixed (pesi impacchettati + fp8). |

### Variante utile (formato/precisione diverso) — 6 file, 54.1 GB

Stesso modello in un altro formato utile (GGUF, int8, bf16) o per un'altra variante.

| File | GB | Motivazione |
|---|---|---|
| `Checkpoints\Unknown\ernie-image-turbo-BF16.gguf` | 14.96 | ERNIE-Image Turbo (distillato, pochi passi) in GGUF BF16: modello diverso dalla base ernie-image.safetensors che image_flux.py:59 carica. Oggi non c'è un loader GGUF per Ernie. |
| `Depth\Wan2.1\depth_anything_v2_vitl.pth` | 1.25 | DepthAnything V2 ViT-L nel formato .pth originale: stesso modello della voce depth-anything-v2 (non scaricata), ma core/controlnet_preprocess.py lo vuole in formato HF transformers in Segmentation\hf\depth-anything-v2-large. Motore depth attivo. |
| `diffusion_models\Flux\nunchaku\svdq-fp4_r32-flux.1-dev.safetensors` | 6.56 | CORREZIONE: non e' uno scarto ComfyUI. inference_server/backends/image_flux.py ha un motore nativo nunchaku dedicato (_load_nunchaku_flux_tr, funzione _is_svdq) che carica proprio i file 'svdq-fp4_*' per FLUX.1 (~7GB VRAM, qualita' quasi-16bit); il file e' anche rilevato in data/model_detect.json (family flux1, variant dev). |
| `diffusion_models\Flux\nunchaku\svdq-fp4_r32-flux.1-kontext-dev.safetensors` | 6.56 | CORREZIONE: stesso motore nativo nunchaku di image_flux.py (_is_svdq/_load_nunchaku_flux_tr) gestisce genericamente qualunque file 'svdq-fp4_*' nella cartella Flux\nunchaku; rilevato in data/model_detect.json come famiglia flux1/dev. Non e' un residuo ComfyUI. |
| `diffusion_models\LTXVideo\LTX-2\nvfp4\ltx-2.3-22b-dev-nvfp4.safetensors` | 20.21 | Rilevato in model_detect.json (family ltxv, precision fp4). Per nota di memoria del progetto, LTX nvfp4 viene oggi dequantizzato in fp8_cast da ltx-core (supporto reale ma il kernel FP4 vero e proprio e' ancora da fare): variante sperimentale accanto al checkpoint dev-fp8 gia' presente e validato in catalogo (ltx23-dev-fp8). |
| `TextEncoders\t5xxl_enconly.safetensors` | 4.56 | T5-XXL solo encoder in fp8 (4.56 GB contro 9.12 GB del t5xxl_fp16 a catalogo): il motore FLUX.1 nativo carica però il T5 dalla cartella diffusers\_te\t5xxl (image_flux.py:743), quindi oggi nessun codice lo usa. |

### Versione migliore di quella a catalogo — 2 file, 36.5 GB

Stesso modello di una voce a catalogo, ma più leggero/veloce o di qualità superiore.

| File | GB | Motivazione |
|---|---|---|
| `diffusion_models\Flux\nvfp4\flux2-dev-nvfp4-mixed.safetensors` | 21.21 | FLUX.2 dev in NVFP4 (22,8 GB): entra nei 32 GB della 5090 e gira coi kernel FP4; l'fp8 a catalogo (35,5 GB) non entra e va in offload. Da mettere a catalogo come variante consigliata. |
| `diffusion_models\LTXVideo\LTX-2\unsloth\distilled-1.1\ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf` | 15.26 | core/ltx2_service.py (auto-calibrazione VRAM) cerca genericamente '*distilled*Q4_K_M.gguf' sotto diffusion_models: questo file unsloth (dynamic quant 'UD', 15.26GB) corrisponde al pattern e di fatto soddisfa la voce di catalogo 'ltx23-gguf-q4-distilled' (ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf), segnata 'non scaricato' — qui e'… |

## 4. DECIDI TU — 49 file, 90.2 GB

### Incerto — 35 file, 84.1 GB

Non sono riuscito a stabilirne l'uso con certezza: nel dubbio non si cancella.

| File | GB | Motivazione |
|---|---|---|
| `Audio\Levo\model_2.safetensors` | 4.48 | Probabile LeVo (Tencent SongGeneration), generazione musica; nessun riferimento nel codice (grep «levo» trova solo una parola italiana in self_forcing_infer.py). L'app ha già HeartMuLa/ACE-Step; se si decide di tenere LeVo basta la variante _fixed. |
| `Audio\Levo\model_2_fixed.safetensors` | 4.38 | Variante «fixed» dello stesso modello LeVo (1284 vs 993 tensori); nessun motore nell'app la usa. Se l'utente vuole LeVo, tenere questa e cancellare model_2. |
| `Audio\Wan2.1\pytorch_model.bin` | 0.35 | Pesi wav2vec2 «base» (chiavi wav2vec2.encoder.*, 380 MB) senza config.json: compatibili con chinese-wav2vec2-base di MultiTalk, ma workers/wan_multitalk.py:212 lo scarica per repo id (TencentGameMate/chinese-wav2vec2-base) e non legge questo file. Non verificabile se sia la versione cinese o l'inglese. |
| `Checkpoints\SD\eroticVision_v4.safetensors` | 1.99 | Checkpoint NSFW (regola 2: non si cancella) ma SD1.5, architettura che l'app non ha più: nessun motore può caricarlo. Decide l'utente se tenerlo fuori dall'app. |
| `ControlNet\Flux\alimama_flux_inpainting.safetensors` | 3.99 | ControlNet inpainting FLUX.1 (alimama): l'architettura è supportata ma il codice non ha alcun loader ControlNet FLUX (grep FluxControlNet: nessun risultato), e l'inpainting FLUX è già coperto da FLUX.1 Fill (flux1-fill-single, presente). |
| `Depth\lotus-depth-d-v1-1.safetensors` | 1.62 | Lotus Depth (stima profondità diffusion-based, UNet stile SD2): nessun riferimento nel codice; il motore depth usa DepthAnything V2 (core/controlnet_preprocess.py:depth). |
| `diffusers\ltx2\gemma-3-12b-it-qat-q4_0-unquantized\model-00001-of-00005.safetensors` | 4.64 | Gemma-3 12B it QAT q4_0-unquantized (TE raccomandato da Lightricks), shard 1/5. Stesse 1065 chiavi e forme di TextEncoders\gemma-3-ltx (ltx23-gemma-te, usato da ltx2_infer.py:895) ma i byte dei tensori sono DIVERSI: non è un doppione. Nessun codice legge questa cartella. |
| `diffusers\ltx2\gemma-3-12b-it-qat-q4_0-unquantized\model-00002-of-00005.safetensors` | 4.59 | Shard 2/5 dello stesso Gemma-3 12B QAT-unquantized: vale la decisione dello shard 1 (tenere o cancellare tutti e 5 insieme). |
| `diffusers\ltx2\gemma-3-12b-it-qat-q4_0-unquantized\model-00003-of-00005.safetensors` | 4.59 | Shard 3/5 dello stesso Gemma-3 12B QAT-unquantized: vale la decisione dello shard 1. |
| `diffusers\ltx2\gemma-3-12b-it-qat-q4_0-unquantized\model-00004-of-00005.safetensors` | 4.59 | Shard 4/5 dello stesso Gemma-3 12B QAT-unquantized: vale la decisione dello shard 1. |
| `diffusers\ltx2\gemma-3-12b-it-qat-q4_0-unquantized\model-00005-of-00005.safetensors` | 4.29 | Shard 5/5 dello stesso Gemma-3 12B QAT-unquantized: vale la decisione dello shard 1. |
| `diffusion_models\Z-Image\zimageTurboNSFWBy_2602NSFWBF16.safetensors.normalized.safetensors` | 11.46 | Copia 'normalizzata' (prefisso model.diffusion_model. rimosso, stessi 453 tensori BF16) generata dall'app; oggi il codice crea queste cache solo per FLUX.1 (image_flux.py:1087). Il checkpoint NSFW resta comunque nell'originale; cancellabile se il loader Z-Image accetta il prefisso. |
| `LoRA\LTX-2\ltx23_edit_anything_global_rank128_v1_6000steps_prodigy.safetensors` | 1.22 | LoRA (regola 1: non si cancella in automatico), ma è un checkpoint di training precedente: la versione finale ltx23_edit_anything_global_rank128_v1_9000steps_adamw è presente e citata per nome da ui/tabs/video_edit_ai_tab.py. |
| `LoRA\LTX-2\ltx23_inpaint_masked_r2v_rank32_v1_3000steps.safetensors` | 0.3 | LoRA LTX-2.3 per inpaint Reference-to-Video: ui/tabs/video_edit_ai_tab.py cabla per nome SOLO la variante T2V ('ltx23_inpaint_masked_t2v_rank128_v1_10000steps.safetensors'); questa variante R2V rank32 non e' referenziata da nessun file del repo e il commento nel tab segnala che l'inpaint e' 'backend in completamento' (feature… |
| `LoRA\LTX-2\ltx23_inpaint_masked_t2v_rank128_v1_02500steps.safetensors` | 1.22 | LoRA (regola 1: non si cancella in automatico), ma è un checkpoint a 2500 step: la versione a 10000 step è presente e citata per nome da ui/tabs/video_edit_ai_tab.py. |
| `LoRA\Wan\wan-nsfw-e14-fixed_(SwarmUI).safetensors` | 0.57 | Proposto come doppione dall'analisi, ma il confronto byte per byte con la copia senza suffisso dice che il contenuto è DIVERSO (altra versione/epoca o altro export): non lo cancello. |
| `LoRA\Wan\wan-t2v-i2v-anal-combined-e50_(LoRAs).safetensors` | 0.29 | Proposto come doppione dall'analisi, ma il confronto byte per byte con la copia senza suffisso dice che il contenuto è DIVERSO (altra versione/epoca o altro export): non lo cancello. |
| `LoRA\Wan\wan_cumshot_i2v_(Framepack).safetensors` | 0.33 | Proposto come doppione dall'analisi, ma il confronto byte per byte con la copia senza suffisso dice che il contenuto è DIVERSO (altra versione/epoca o altro export): non lo cancello. |
| `LoRA\Wan\WAN_dr34mj0b_(Framepack).safetensors` | 0.17 | Proposto come doppione dall'analisi, ma il confronto byte per byte con la copia senza suffisso dice che il contenuto è DIVERSO (altra versione/epoca o altro export): non lo cancello. |
| `Other\encoders\encode-s12k.pt` | 3.69 | Nome generico, nessun metadata/header leggibile (file .pt opaco) e nessuna occorrenza di 'encode-s12k' o riferimento diretto nel codice. Non essendoci prova ne' di uso ne' di abbandono (regola: si cancella solo con prova), la decisione va lasciata all'utente. |
| `Other\htdemucs.pth` | 0.16 | core/vocal_separation.py e workers/wan_multitalk.py usano demucs via CLI (-n htdemucs), che scarica e legge il modello dalla propria cache torch hub, non da questo file; utile solo come copia offline. |
| `Other\pytorch_model.bin` | 0.35 | File generico 0.35GB senza metadata/header leggibile, nessun riferimento di percorso nel codice (classificazione inventory 'D nessuno'). Non c'e' prova sufficiente per identificarlo ne' per cancellarlo; lasciare la decisione all'utente dopo ispezione dei tensori. |
| `Segmentation\sam2\sam2_hiera_large.safetensors` | 0.84 | SAM2 nel formato Kijai (ComfyUI-segment-anything-2): i backend nativi usano la snapshot HF sam2-hiera-large (a catalogo, presente); serve solo al percorso ComfyUI legacy di ui/main_window.py. |
| `TextEncoders\ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors` | 0.3 | CLIP-L fine-tune community (hiT-GmP di zer0int); FLUX.1 usa CLIP-L+T5-XXL come encoder ma nessun riferimento di codice a questo file specifico lo lega esplicitamente alla pipeline. |
| `TextEncoders\Wan2.1\models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors` | 2.22 | CLIP di Wan2.1 I2V nel formato originale: MultiTalk usa clip_vision_h.safetensors (formato ComfyUI) e Self-Forcing/Krea Realtime sono T2V; nessun codice lo cita. |
| `TextEncoders\Wan2.1\models_t5_umt5-xxl-enc-quanto_int8.safetensors` | 6.27 | UMT5-XXL quantizzato con Quanto: nessun loader dell'app usa Quanto; Self-Forcing usa models_t5_umt5-xxl-enc-bf16.pth (workers/self_forcing_infer.py:9) e MultiTalk umt5_xxl_fp8_e4m3fn_scaled. |
| `VAE\autoencoder_music_1320k.ckpt` | 0.63 | Audio VAE in formato .ckpt deprecato (stile Stable-Audio-Open), nessun riferimento nel codice; i motori musica presenti (HeartMuLa/MMAudio/ACE-Step) hanno VAE propri cataloghi. |
| `VAE\diffusion_pytorch_model.safetensors` | 2.28 | Nome file generico (standard diffusers AutoencoderKL), nessuna cartella arch, nessun riferimento di codice. Pattern chiavi 'resnets.N.convN.conv' simile ma non identico a LTX2_video_vae ('res_blocks') né a Wan2.1 VAE ('residual/gamma'): non identificabile con certezza senza hash/metadata. |
| `VAE\LTX\LTX2_audio_vae.safetensors` | 0.2 | Audio VAE di LTX-2 (296 tensori, diverso dai 1329 di LTX-2.3): nessun codice lo cita per nome; LTX-2.3/2.5 usano i propri audio VAE. |
| `VAE\LTX\LTX2_video_vae.safetensors` | 2.28 | VAE video di LTX-2 (187 tensori, 2.28 GB, diverso dal VAE LTX-2.3 a 170 tensori): nessun codice lo cita per nome; LTX-2.3 usa LTX23_video_vae_bf16 e LTX-2.5 il proprio VAE a catalogo. |
| `VAE\LTX\LTX_VAE_13B_Dev_BF16.safetensors` | 2.32 | VAE della vecchia LTX-Video 13B 'dev' (generazione precedente a LTX-2.3/2.5, non negli engine attuali dell'app); nessun riferimento di codice lo lega a una pipeline attiva, ma non c'è prova sufficiente per classificarlo come progetto abbandonato con certezza. |
| `VAE\Qwen\qwen_image_vae_(SwarmUI).safetensors` | 0.24 | Proposto come doppione dall'analisi, ma il confronto byte per byte con la copia senza suffisso dice che il contenuto è DIVERSO (altra versione/epoca o altro export): non lo cancello. |
| `VAE\Wan2.2\wan2.2_vae.safetensors` | 1.31 | Il brief dice 'Wan non usato da Krea Realtime/MultiTalk/Self-Forcing' per le versioni diverse da 2.1, MA core/model_detect.py, core/model_importer.py (cartella 'Wan22') e ui/tabs/video_tab.py riconoscono esplicitamente la famiglia 'wan22' con VAE/TE propri: evidenza contraddittoria, non è chiaro se sia solo rilevamento morto … |
| `VAE\Wan2.2\Wan2.2_VAE_(Wan2GP).safetensors` | 2.63 | Stesso motivo del file gemello (evidenza contraddittoria su Wan 2.2 nel codice); possibile anche doppione in precisione diversa (f32 vs f16) dello stesso modello Wan 2.2 VAE. |
| `VisionEncoders\model.safetensors` | 3.27 | È un SigLIP completo testo+immagine fp32 (chiave logit_bias), non il CLIP-H: MultiTalk carica per nome clip_vision_h.safetensors (workers/wan_multitalk.py:211) dalla stessa cartella. Nessun loader cita questo file. |

### LoRA di architetture non supportate — 14 file, 6.2 GB

Il tuo 2.1 (cancella SD1.5/SDXL/…) e il 2.2 (tieni tutte le LoRA) qui si contraddicono: decidi tu. Per ora restano.

| File | GB | Motivazione |
|---|---|---|
| `Caption\adapter_model.safetensors` | 0.63 | E' una LoRA (adapter LLM) di JoyCaption alpha-two: core/caption_service.py usa ormai solo JoyCaption beta-one nf4 (joycaption-nf4, presente), quindi alpha-two e' superato e non ha un motore che la carichi. Per rispettare alla lettera la regola utente 'tutte le LoRA si tengono', anche se per un'architettura (captioner) ormai n… |
| `diffusion_models\DreamO\dreamo.safetensors` | 0.45 | Adattatori DreamO su FLUX.1 con processor custom (chiavi dreamo_*lora.*_processor.proj_lora*): richiedono la pipeline DreamO, assente nell'app (core/model_detect.py la riconosce solo per nome). |
| `diffusion_models\DreamO\dreamo_cfg_distill.safetensors` | 0.45 | Adattatori DreamO su FLUX.1 con processor custom (chiavi dreamo_*lora.*_processor.proj_lora*): richiedono la pipeline DreamO, assente nell'app (core/model_detect.py la riconosce solo per nome). |
| `LoRA\DreamO\dreamo_quality_lora_neg.safetensors` | 0.45 | Adattatori DreamO su FLUX.1 con processor custom (chiavi dreamo_*lora.*_processor.proj_lora*): richiedono la pipeline DreamO, assente nell'app (core/model_detect.py la riconosce solo per nome). |
| `LoRA\DreamO\dreamo_quality_lora_pos.safetensors` | 0.45 | Adattatori DreamO su FLUX.1 con processor custom (chiavi dreamo_*lora.*_processor.proj_lora*): richiedono la pipeline DreamO, assente nell'app (core/model_detect.py la riconosce solo per nome). |
| `LoRA\Flux\figures_TTP_Migration.safetensors` | 0.32 | CORREZIONE (errore rilevato, regola 1): Haiku proponeva di cancellarla come 'versione inferiore/superata'. I metadati del file dichiarano esplicitamente 'ss_base_model_version': 'sd_1.5' — e' una LoRA SD1.5, architettura non supportata dall'app, ma le LoRA non si cancellano MAI: va lasciata decidere all'utente. |
| `LoRA\Flux\Migration_Lora_cloth.safetensors` | 0.16 | CORREZIONE (errore rilevato, regola 1): stessa situazione di figures_TTP_Migration — metadati 'ss_base_model_version': 'sd_1.5'. LoRA SD1.5 non supportata, ma mai da cancellare per regola utente. |
| `LoRA\Hunyuan\Titfuck_v2_Release.safetensors` | 0.15 | LoRA HunyuanVideo: nessun motore Hunyuan nell'app (il nome compare solo come etichetta in core/auto_select.py e ui/tabs/video_tab.py). La copia (SwarmUI) è un doppione identico. |
| `LoRA\Qwen\Qwen-Image-Lightning-8steps-V1.1.safetensors` | 1.58 | Acceleratore 8 passi per Qwen-Image 1.x (60 blocchi, 3072): incompatibile con Qwen-Image 2.1, che ha già i suoi turbo 4/6 passi a catalogo. |
| `LoRA\Qwen\Qwen_Real_Nud3s.safetensors` | 0.27 | LoRA NSFW per Qwen-Image 1.x (60 blocchi, dimensione 3072): la nostra Qwen-Image 2.1 ha 32 blocchi e dimensione 4096, non si applica. Servirebbe Qwen-Image 1.x (non installato). |
| `LoRA\SD1.5\blindbox_v1_mix.safetensors` | 0.14 | Header con pattern lora_te_text_model_encoder + lora_unet_up_blocks (formato SD1.5 Kohya standard, 792 tensori), confermato SD1.5 genuino. SD1.5 e' tra i progetti abbandonati (nessun motore immagine dell'app usa SD1.5). Per regola utente una LoRA di architettura non supportata va comunque TENUTA e marcata per decisione dell'u… |
| `LoRA\SD1.5\CineStyle5.safetensors` | 0.14 | Header con pattern lora_te_text_model_encoder + lora_unet_up_blocks (formato SD1.5 Kohya standard, 792 tensori), confermato SD1.5 genuino. SD1.5 e' tra i progetti abbandonati (nessun motore immagine dell'app usa SD1.5). Per regola utente una LoRA di architettura non supportata va comunque TENUTA e marcata per decisione dell'u… |
| `LoRA\SD1.5\MoXinV1.safetensors` | 0.14 | Header con pattern lora_te_text_model_encoder + lora_unet_up_blocks (formato SD1.5 Kohya standard, 792 tensori), confermato SD1.5 genuino. SD1.5 e' tra i progetti abbandonati (nessun motore immagine dell'app usa SD1.5). Per regola utente una LoRA di architettura non supportata va comunque TENUTA e marcata per decisione dell'u… |
| `LoRA\SDXL\NSFW_ZI_V01_r1_Epoch31.safetensors` | 0.85 | Header confermato SDXL genuino (lora_teN_text_model_encoder + lora_unet_output_blocks, 2958 tensori BF16, doppio text-encoder tipico SDXL), nonostante il nome 'ZI' (Z-Image e' un motore diverso e non usa questo formato). SDXL non e' un'architettura immagine attiva nell'app. Essendo una LoRA (non un checkpoint), per regola ute… |

### La mia raccomandazione — 1 file, 35.5 GB

| File | GB | Motivazione |
|---|---|---|
| `diffusion_models\Flux\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors` | 35.5 | FLUX.2 dev fp8-mixed (è la voce di catalogo `flux2-dev-fp8mixed`, non un doppione). Esiste anche lo stesso modello in NVFP4 (22,8 GB) che entra nei 32 GB e usa i kernel FP4; l'fp8 non entra e va in offload (lento). Consiglio: NVFP4 come default e cancellare l'fp8. Se lo tieni, va solo collegato al catalogo (oggi risulta «da scaricare» perché ha un nome diverso). |

## 5. Spostamenti già fatti (reversibili)

- 4 LoRA Wan 2.1 14B dalla radice `LoRA` a `LoRA/Wan` (T2V-POV_BJ-14B, deepthroat_epoch_80, fingering_i2v_e248, grinding_pussyjob_e130), con i loro .metadata.json.
- Rimosse le cartelle vuote `LoRA/Unknown` e `LoRA/FramePack`.
- `krea2_identity_edit_v1_2` resta in `LoRA`: il worker Krea la cerca in tutte le sottocartelle.

## 6. Note sulle correzioni rispetto agli analisti

- nunchaku (svdq FLUX.1 / Kontext / Z-Image) NON è obsoleto: lo usano image_flux.py e WorldGen → tenuto. Solo il Klein 9B nunchaku è orfano (il loader Klein non lo supporta).
- RealESRGAN e gli upscaler DAT/HAT/DRCT non sono SDXL: li carica il loader generico spandrel → tenuti.
- Qwen3.6-27B è il riscrittore di LingBot (`lingbot_rewriter_base`), presente: tenuto.
- `diffusers\_te\t5xxl` sembrava inutilizzato ma image_flux.py lo carica con os.path.join (T5 di FLUX.1): hardlink, non cancellazione.
- mistral: l'analista voleva tenere l'altra copia; vale la tua scelta (resta `_fp4_mixed`, codice e catalogo ridiretti).
- FLUX.2 dev: `FLUX_2_Dev_FP8_Mixed_Scaled` (35,5 GB) È il FLUX.2 dev fp8 del catalogo; non c'è un doppione. La variante NVFP4 (22,8 GB) è lo stesso modello a 4 bit che ENTRA nella VRAM: consiglio di tenerla come default e decidere se cancellare l'fp8 (sezione 4).



## 7. Seconda parte, approvata il 14/10: file legati a modelli abbandonati

Dalla sezione 4: SD1.5 (anche il checkpoint NSFW eroticVision), SDXL, HunyuanVideo, DreamO, Qwen-Image 1.x, JoyCaption alpha-two, VAE di LTX-2 prima generazione e LTX-Video 13B, Wan 2.2; più FLUX.2 dev fp8 (resta la NVFP4, che entra nella VRAM). Gli altri «incerti» restano.

```
CANCELLA   1.99 GB  [SD1.5]  E:\AI\Models\Checkpoints\SD\eroticVision_v4.safetensors
CANCELLA   0.14 GB  [SD1.5]  E:\AI\Models\LoRA\SD1.5\CineStyle5.safetensors
CANCELLA   0.14 GB  [SD1.5]  E:\AI\Models\LoRA\SD1.5\MoXinV1.safetensors
CANCELLA   0.14 GB  [SD1.5]  E:\AI\Models\LoRA\SD1.5\blindbox_v1_mix.safetensors
CANCELLA   0.16 GB  [SD1.5]  E:\AI\Models\LoRA\Flux\Migration_Lora_cloth.safetensors
CANCELLA   0.32 GB  [SD1.5]  E:\AI\Models\LoRA\Flux\figures_TTP_Migration.safetensors
CANCELLA   0.85 GB  [SDXL]  E:\AI\Models\LoRA\SDXL\NSFW_ZI_V01_r1_Epoch31.safetensors
CANCELLA   0.15 GB  [HunyuanVideo]  E:\AI\Models\LoRA\Hunyuan\Titfuck_v2_Release.safetensors
CANCELLA   0.45 GB  [DreamO]  E:\AI\Models\LoRA\DreamO\dreamo_quality_lora_neg.safetensors
CANCELLA   0.45 GB  [DreamO]  E:\AI\Models\LoRA\DreamO\dreamo_quality_lora_pos.safetensors
CANCELLA   0.45 GB  [DreamO]  E:\AI\Models\diffusion_models\DreamO\dreamo.safetensors
CANCELLA   0.45 GB  [DreamO]  E:\AI\Models\diffusion_models\DreamO\dreamo_cfg_distill.safetensors
CANCELLA   1.58 GB  [Qwen-Image 1.x]  E:\AI\Models\LoRA\Qwen\Qwen-Image-Lightning-8steps-V1.1.safetensors
CANCELLA   0.27 GB  [Qwen-Image 1.x]  E:\AI\Models\LoRA\Qwen\Qwen_Real_Nud3s.safetensors
CANCELLA   0.63 GB  [JoyCaption alpha-two]  E:\AI\Models\Caption\adapter_model.safetensors
CANCELLA   0.20 GB  [LTX-2 / LTX-Video 13B]  E:\AI\Models\VAE\LTX\LTX2_audio_vae.safetensors
CANCELLA   2.28 GB  [LTX-2 / LTX-Video 13B]  E:\AI\Models\VAE\LTX\LTX2_video_vae.safetensors
CANCELLA   2.32 GB  [LTX-2 / LTX-Video 13B]  E:\AI\Models\VAE\LTX\LTX_VAE_13B_Dev_BF16.safetensors
CANCELLA   2.63 GB  [Wan 2.2]  E:\AI\Models\VAE\Wan2.2\Wan2.2_VAE_(Wan2GP).safetensors
CANCELLA   1.31 GB  [Wan 2.2]  E:\AI\Models\VAE\Wan2.2\wan2.2_vae.safetensors
CANCELLA  33.02 GB  [FLUX.2 dev fp8 (resta la NVFP4)]  E:\AI\Models\diffusion_models\Flux\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors
RIMUOVO cartella E:\AI\Models\LoRA\SD1.5
TENGO cartella (contiene altri pesi): E:\AI\Models\LoRA\SDXL → ['DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors', 'gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors']
TENGO cartella (contiene altri pesi): E:\AI\Models\LoRA\Hunyuan → ['Titfuck_v2_Release_(SwarmUI).safetensors']
RIMUOVO cartella E:\AI\Models\LoRA\DreamO
RIMUOVO cartella E:\AI\Models\diffusion_models\DreamO
RIMUOVO cartella E:\AI\Models\LoRA\Qwen
RIMUOVO cartella E:\AI\Models\VAE\Wan2.2
TENGO cartella (contiene altri pesi): E:\AI\Models\Checkpoints\SD → ['dreamshaper_8.safetensors']
spazio liberato ≈ 49.9 GB
```

## 8. Disposizioni dell'utente sulle sezioni 3 e 4 (14/10) — si integra PRIMA di cancellare

**Sezione 3 (TENERE) — approvata, con queste disposizioni:**
- Checkpoint NSFW da integrare → integrarli (voce di catalogo 18+ e menu giusti, solo in NSFW).
- Funzionalità aggiuntiva → integrarle (prima quelle già caricabili; per quelle che richiedono codice nuovo, piano e ordine).
- LoRA da integrare (architettura supportata) → integrarle nel pannello dell'architettura giusta.
- Usato dall'app ma fuori catalogo → metterli a catalogo.
- Variante utile (formato/precisione diverso) → integrarle come varianti selezionabili.
- Versione migliore di quella a catalogo → sostituirle alle peggiori (la peggiore passa al giro di cancellazione).

**Sezione 4 (DECIDI TU) — disposizioni:**
- Architetture abbandonate: cancellare anche le LoRA. Wan 2.1 SI usa (MultiTalk, Krea Realtime, Self-Forcing,
  FlashVSR) → resta; Wan 2.2 non ha motore → pesi e LoRA in elenco dopo verifica.
- Versioni precedenti di ciò che abbiamo già: cancellare.
- `ltx23_inpaint_masked_r2v_rank32_v1_3000steps`: analizzare se conviene (l'editing immagine/video è carente).
- `ltx23_inpaint_masked_t2v_rank128_v1_02500steps`: test dal vivo contro la versione a 10000 step prima di decidere.
- LoRA di architetture non supportate: cancellare.
- Eliminare ciò che non serve al progetto e occupa spazio (ogni «incerto» verificato con prova).

**Ordine di esecuzione:** 1) integrazioni (catalogo, menu, LoRA, varianti, sostituzioni); 2) poi le cancellazioni
(sezioni 1-2 e 7 già pronte negli script; il terzo giro, dalla sezione 4, sarà preparato con prove e sottoposto a te).
