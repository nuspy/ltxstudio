"""Motore video LTX-2.3 NATIVO che applica il MEMORY-MANAGEMENT di ComfyUI (NON l'app ComfyUI).

PERCHE': il pacchetto ufficiale Lightricks (ltx-core) non sta in 32GB (fp8-native XOR offload;
floor ~31GB). ComfyUI, con gli STESSI pesi, fa LTX-2.3 in 8GB grazie a:
  - UNETLoader → comfy.sd.load_diffusion_model + comfy.model_management (offload per-blocco con
    buffer GPU riusabile/DynamicVram, fp8-native matmul SENZA upcast bf16);
  - DualCLIPLoader (Gemma fp8 + text_projection, type 'ltxv');
ricavati dal workflow reale `E:\AI\workflows\Pluto_AllInOne.json`.

STRATEGIA DISTRIBUZIONE: si impacchettano SOLO i moduli di memory-management di ComfyUI
(comfy/model_management.py, comfy/ops.py, comfy/sd.py + le dipendenze del loader e i nodi
LTXV necessari) come libreria del progetto — NON il server/UI di ComfyUI.

FILE GIUSTI (dal workflow, nodi mode=0 attivi) — TUTTI piu' piccoli di quelli che avevo usato:
  TRANSFORMER  E:\AI\Models\diffusion_models\LTXVideo\v2\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors  (23GB, transformer-only, formato model.diffusion_model.*)
  GEMMA fp8    E:\AI\Models\diffusers\_te\gemma3_ltx_fp8  (cartella, ~12GB) — alias ComfyUI gemma_3_12B_it_fp8_scaled
  PROJECTION   E:\AI\Models\...\ltx-2.3_text_projection_bf16.safetensors   (DualCLIPLoader type 'ltxv')
  VAE video    LTX23_video_vae_bf16.safetensors
  VAE audio    LTX23_audio_vae_bf16.safetensors
  UPSAMPLER    ltx-2.3-spatial-upscaler-x2-1.1.safetensors
  (alternativa ancora piu' leggera: GGUF ltx-2-3-22b-dev-Q6_K.gguf 16GB / Q3_K_M 10GB via GGUFLoaderKJ)

PIPELINE (dal workflow Pluto_AllInOne):
  1. UNETLoader(transformer, weight_dtype='default')        → model (con offload comfy)
  2. DualCLIPLoader(gemma_fp8, projection, type='ltxv')     → clip
  3. TextGenerateLTX2Prompt / PromptRelayEncodeTimeline     → conditioning testo
  4. LTXVConditioning(cond, frame_rate)                     → cond+/cond-
  5. RandomNoise + sampler (euler_ancestral_cfg_pp, LTXVScheduler shift, 32 step, cfg 1.0)
  6. LTXVLatentUpsampler (stage 2, x2)                      → latent HR
  7. VAELoader + LTXVTiledVAEDecode                         → frame video
  8. (audio) LTXVEmptyLatentAudio + LTXVAudioVAEDecode + MelBandRoFormer + vocoder

TODO (prossimo step, da fare nel venv ComfyUI `E:\AI\Comfy_Nexus\ComfyUI\venv` che ha comfy + il
custom node ComfyUI-LTXVideo):
  - sys.path: ComfyUI root + custom_nodes/ComfyUI-LTXVideo
  - import comfy.sd, comfy.model_management, comfy.samplers, nodes (UNETLoader, DualCLIPLoader, VAELoader)
  - import dei nodi LTXV dal custom node
  - replicare i passi 1-8 chiamando le classi-nodo (instanziarle e chiamare i loro metodi FUNCTION)
  - encode_video output → mp4 (riusare la logica di workers/ltx2_infer.encode o ffmpeg)
Vedi memoria native-video-engine per i dettagli completi.
"""

# Scaffold: implementazione nel prossimo step (richiede l'API di comfy nel venv ComfyUI).
TRANSFORMER = r"E:\AI\Models\diffusion_models\LTXVideo\v2\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
GEMMA_FP8 = r"E:\AI\Models\diffusers\_te\gemma3_ltx_fp8"
TEXT_PROJECTION = r"ltx-2.3_text_projection_bf16.safetensors"
VAE_VIDEO = r"LTX23_video_vae_bf16.safetensors"
VAE_AUDIO = r"LTX23_audio_vae_bf16.safetensors"
UPSAMPLER = r"ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
COMFY_ROOT = r"E:\AI\Comfy_Nexus\ComfyUI"
