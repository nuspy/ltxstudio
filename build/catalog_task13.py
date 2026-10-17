"""Voci di catalogo del task 13 (18/10/2026): FLUX.3 Image Klein (immagini: crea, modifica, multi-riferimento, riquadri).

Opzione A scelta dall'utente (18/10): DiT distillato `variants/fp8r` (6,27 GB) + text encoder Qwen3-VL-4B bf16 (8,88 GB) +
prompt upsampler; il VAE è GIÀ sul disco (`flux2-vae`, sha256 identico a quello di FLUX.3: d64f3a68…c4b5).
Repo e dimensioni verificati il 18/10/2026 (HF API/hf_fs, mirror ml-resources per i json); sha256 del DiT dal manifest ufficiale
(variants/fp8r/manifest.json). Licenza FLUX Kommunity v1.0: l'app NON distribuisce i pesi; li scarica l'utente dal tab «Pesi
modelli» con avviso e conferma (`license_ack`, core/license_ack.py). I repo BFL sono «gated: auto»: serve accettare la licenza su
Hugging Face con l'account che scarica (senza, il download risponde 401/403).
Celle della feature matrix VUOTE: le spunta l'utente.
"""

REFRESH_IDS: tuple = ()          # voci nuove: nessun refresh di voci già nel json

LIC = ("Licenza FLUX Kommunity v1.0 (BFL): uso commerciale degli output solo se Qualifying User (ricavi annui sotto 5 milioni di "
       "dollari) con filtro/revisione dei contenuti e dichiarazione AI; i pesi li scarica l'utente dopo aver accettato la licenza "
       "su Hugging Face")
REPO = "black-forest-labs/flux-3-image-klein"
PUP_REPO = "webmp3/Sakura-FLUX.3-Image-Klein-PromptUpsampler-Uncensored-GGUF"
PUP_Q8 = "Sakura-FLUX.3-Image-Klein-PromptUpsampler-Uncensored-Q8_0-8.87GiB.gguf"


def entries(N):
    out = [
        N("flux3-dit-fp8r", "Immagini", "FLUX.3 Image Klein DiT distillato fp8 per riga (6.3GB, RTX 40xx/50xx)",
          category="image", arch="flux3", repo=REPO, gated=True, remote="variants/fp8r/model.safetensors",
          filename="flux3_klein_distilled_fp8r.safetensors", type="diffusion_models", family="Flux3", gb=6.27,
          sha256="30054053f630725b488aead9a81b422793cee18027856a16bd17b902ad612018",
          used_by=["cap.image_flux3"],
          note="DiT 5,55B (flow map: 16 passi, senza CFG né negative), fp8 E4M3 con scala per riga (kernel fuso, serve cc ≥ 8.9). "
               "Crea, modifica per istruzione, multi-riferimento fino a 10, riquadri (layout/edit). " + LIC),
        N("flux3-te", "Immagini", "FLUX.3 Klein text encoder Qwen3-VL-4B Instruct bf16 (8.9GB) + tokenizer",
          category="image", arch="flux3", kind="snapshot", repo=REPO, gated=True, type="TextEncoders", family="Flux3",
          gb=8.88, allow=["text_encoder/*", "LICENSE.md", "NOTICE.md"], used_by=["cap.image_flux3"],
          note="8 strati (4,8,…,32) impilati → token a 20480 canali; Apache-2.0 (text_encoder/LICENSE). La licenza del DiT è "
               "quella FLUX Kommunity: " + LIC),
        N("flux3-pup-configs", "Immagini", "FLUX.3 Klein upsampler: config + tokenizer + system prompt (json, ~20MB)",
          category="image", arch="flux3", kind="snapshot", repo="black-forest-labs/flux-3-image-klein-pup", type="configs",
          gb=0.02, allow=["*.json", "*.jinja", "LICENSE*"], used_by=["cap.flux3_upsampler"],
          note="solo json/chat template (NON i pesi dell'upsampler ufficiale): config Qwen3.5-9B, tokenizer e system_prompts.json "
               "(t2i, i2i, t2i_bbox, i2i_bbox). Apache-2.0"),
        N("flux3-pup-sakura-q8", "Immagini", "FLUX.3 Klein prompt upsampler abliterato Sakura Q8_0 (GGUF, 9.5GB)",
          category="image", arch="flux3", repo=PUP_REPO, remote=PUP_Q8, filename=PUP_Q8, type="TextEncoders",
          family="Flux3", gb=9.53, sha256="4fbde007612af89cae6be93a59b7a6ead980560504be29d7a0645fc36e704b41",
          deps=["flux3-pup-configs"], used_by=["cap.flux3_upsampler"],
          note="Qwen3.5-9B (fine-tune BFL) con abliterazione di terzi (webmp3/Sakura, Apache-2.0, not-for-all-audiences): rifiuta "
               "meno dell'ufficiale. GGUF Q8_0 SOLO linguaggio (il proiettore visivo è un file a parte): letto da "
               "workers/flux3_gguf.py. DA CONFERMARE con l'utente (task 13, DECISIONE RICHIESTA)."),
    ]
    for e in out:
        if e["id"] in ("flux3-dit-fp8r", "flux3-te"):    # mai in silenzio: dialog con la licenza + conferma (core/license_ack)
            e["license"] = "FLUX Kommunity License v1.0"
            e["license_ack"] = "flux_kommunity"
    return out


# id → (tag d'uso, deps) per catalog_use_deps. Catena: DiT → text encoder + VAE (flux2, già su disco) + upsampler.
USE = {
    "flux3-dit-fp8r": (["ckpt_image", "fp8", "distilled"], ["flux3-te", "flux2-vae", "flux3-pup-sakura-q8"]),
    "flux3-te": (["te"], []),
    "flux3-pup-configs": ([], []),
    "flux3-pup-sakura-q8": (["prompt_upsampler"], []),
}
NEW_USE_TAGS = {
    "prompt_upsampler": "Upsampler di prompt (LLM): trasforma la richiesta breve nel caption completo che FLUX.3 vuole; si carica prima del DiT.",
}
