# Tips & Tricks

*A skimmable, cross‑module guide to getting the best results — the settings that matter, and the traps to avoid.*

This chapter consolidates the practical advice scattered through the app's tooltips, presets, and internal notes. Each section is a domain you can jump to. For per‑feature detail, follow the cross‑links to [Images](02-images.md), [Quality & restore](04-quality-restore.md), [Audio & voice](05-audio-voice.md), [Music](06-music.md), [3D](08-3d.md), and [Advanced settings](12-advanced.md).

---

## Images

- **Choose the scheduler deliberately — it changes image quality a lot.** For FLUX images, the flow‑aware schedulers **bong_tangent / kl_optimal / beta57** are often the best (high quality). **karras** gives classic detail, **beta** suits distilled models, and **exponential** is softer.
- **Sampler:** euler and dpmpp are reliable. **res_2s** gives more detail on FLUX images — but never use it for LTX video, where it breaks (see Video below).
- **Face identity from a photo:** with PuLID, keep the strength around **0.8** (recommended range 0.6–0.9). If faces look plastic, use a **base bf16** model rather than a distilled/quantized one, and push the reference/image strength higher.
- **Generate at full resolution.** Leave any half‑resolution / downscale option **off** for maximum native detail; it only hides the VAE decoder grid at the cost of softness.
- **Guidance:** FLUX text‑to‑image sits comfortably around guidance 3.5; inpaint/fill wants much higher guidance (~30) to fill cleanly.

## Video

- **Use the verified sampler.** The default **`euler_ancestral_cfg_pp`** with **`LTXVScheduler (shift)`** and **CFG 1.0** is the tuned, verified LTX‑2.3 pipeline — the guidance comes from NAG, not from a high CFG.
- **Avoid RES samplers on video.** `res_2s`, `res_multistep`, and similar look great on paper for flow but produce **green noise** on LTX‑2.3 (verified A/B) — they were removed for exactly this reason. If you see cyan/green artifacts, you are on the wrong sampler.
- **The base shift is the #1 video quality lever.** In LTXVScheduler, a **higher base shift** (default 0.95) spends more steps on structure and motion (temporal coherence, less flicker); a **lower** shift gives finer per‑frame detail but risks inconsistency between frames.
- **Image‑to‑video "image strength" controls motion feel.** Set the soft‑reference **image strength to 0.45–0.60** for natural motion; **0.80+** sticks tightly to the first frame but makes the motion stiff.
- **Use the built‑in presets as starting points:** **⚡ FAST** (8 steps, reduced res — a draft to validate the idea, ~1 min), **🎯 PRO** (24 steps, full 1280×704 — best balance), **💎 HQ** (44 steps, no cache — maximum quality, slowest).
- **Long videos are chunked automatically.** Clips over ~150 frames are segmented into memory‑bounded chunks and stitched, so you can request long durations without hitting out‑of‑memory.
- **TeaCache caveat.** With the raw L1 metric (no LTX‑2.3 coefficients) the aggressive thresholds skip almost everything — fast but destructive (PSNR collapses). Presets ship it at **0.0**; only nudge the manual slider up if you have validated a low threshold on your own content.

## Audio & Voice

- **Match the TTS engine to the job.** Use a lightweight engine (edge/kokoro) for fast narration and a clone‑capable engine (f5tts/xtts2/fish/vibevoice/cosyvoice2) when you supply a voice sample. Provide the `sample_transcript` for the best clone fidelity.
- **Separate vocals before lip‑sync** when your source audio has music under the speech — cleaner vocals give tighter lip motion.
- **Lip‑sync strength:** aim for **1.5–3.0** on the sync scale. 1.0 is weak; above 3 the mouth motion becomes exaggerated.

## Music

- **Sampler matters for lyric/style adherence.** For the music engine, **pingpong (SDE)** gives the best coherence and style/lyric alignment (recommended when you need the lyrics honored); **euler** is fast and standard; **heun** is more precise but slower.
- **Guidance:** the ACE‑Step engine defaults to guidance ~15.0 — a strong pull toward your tags and lyrics.
- **Write descriptive tags.** The style tags (genre, instruments, mood) drive the result as much as the lyrics do.

## 3D

- **Feed a clean, well‑lit single subject** for image→3D; busy backgrounds and occlusions hurt mesh quality.
- **Tune the mesh trade‑offs:** more `ss_steps`/`slat_steps` (default 12) improves quality at the cost of time; `simplify` (default 0.95) and `texture_size` (default 1024) trade detail for file size.
- Rigging expects a clean mesh; convert to FBX/OBJ afterward via the built‑in Blender step if your target tool needs it.

## Film Studio

- Keep character **assets consistent** by reusing the same face reference (PuLID) and LoRA across shots.
- Generate **asset images first, then shots** — the shot video step can take keyframes and LoRA to keep continuity.
- Heavy project orchestration (scheduler, editing, dubbing, Auto mode) is stateful and lives in the GUI; use it rather than scripting each shot by hand.

## Performance / VRAM

- **Pick a weight format for your GPU.** On 32 GB, **NVFP4 mixed** is fastest (~14 GB, ~2× faster than BF16, almost no visible difference). **FP8 Scaled** is the best universal compromise. For smaller cards, GGUF quants scale down: **Q8_0** (16 GB, 98–99% quality), **Q6_K** (12 GB, ~95%), **Q5_K_M** (8 GB sweet spot, ~90%), **Q4_K_M** (6–8 GB, 75–85%).
- **VRAM savings stack multiplicatively:** text‑encoder offload after encoding, VAE offload except during decode, transformer block‑swap, and unloading models between generations.
- **Use /free when a video job risks OOM** — it releases held VRAM before the next run.
- **Slow first run is normal.** The first time you open a heavy module the app downloads its on‑demand weights; subsequent runs are fast.
- **Use the prompt enhancer** to expand terse prompts into richer descriptions before generating, especially for video where detailed motion cues help coherence.

> See the [Appendix](17-appendix.md) for the troubleshooting table and the full capability → edition matrix.
