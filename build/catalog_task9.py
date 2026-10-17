"""Voci di catalogo del task 9 (17/10/2026): editing immagine fase 3.

ControlNet FLUX.1 (Shakker Union Pro 2.0, AlimamaCreative Inpainting Beta), ControlNet Z-Image Union 2.1, Redux + SigLIP,
OpenPose (corpo), 5 LoRA MagicQuill V2. Repo e nomi file VERIFICATI su Hugging Face (API, 17/10/2026): dimensioni e
licenze nelle note. Tutti i pesi sono GIA' sul disco dell'utente (cartelle reali): `scan_names` li fa trovare.
"""

REFRESH_IDS: tuple = ()          # voci nuove: nessun refresh di voci già nel json

NC = "Licenza FLUX.1 [dev] non commerciale: i pesi li scarica l'utente"


def entries(N):
    out = [
        N("flux-cn-union-pro2", "Immagini", "FLUX.1 ControlNet Union Pro 2.0 (Shakker, canny/depth/pose/gray, 4.3GB)",
          category="image", arch="flux1", repo="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
          remote="diffusion_pytorch_model.safetensors", filename="shakker_union_pro_v2.safetensors",
          type="ControlNet", family="Flux", gb=4.28, used_by=["cap.image_controlnet"],
          note="6 blocchi doppi, senza mode embedding; ControlNet unico per canny, soft edge, depth, pose, gray. " + NC),
        N("flux-cn-inpaint-alimama", "Immagini", "FLUX.1 ControlNet Inpainting Beta (AlimamaCreative, 4.3GB)",
          category="image", arch="flux1", repo="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
          remote="diffusion_pytorch_model.safetensors", filename="alimama_flux_inpainting.safetensors",
          type="ControlNet", family="Flux", gb=4.28, used_by=["cap.image_cn_inpaint"],
          note="inpainting su FLUX.1 dev normale (68 canali: latenti dell'immagine mascherata + maschera). " + NC),
        N("zimage-cn-union21", "Immagini", "Z-Image Turbo ControlNet Union 2.1 8 step (alibaba-pai, 6.7GB)",
          category="image", arch="zimage", repo="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
          remote="Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
          filename="Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
          type="ControlNet", family="Z-Image", gb=6.71, used_by=["cap.image_controlnet"],
          note="canny/depth/pose/HED/gray con Z-Image Turbo (8 passi, guidance 0). Licenza Apache-2.0"),
        N("flux-redux", "Immagini", "FLUX.1 Redux dev (variazioni da immagine, 0.13GB)",
          category="image", arch="flux1", repo="black-forest-labs/FLUX.1-Redux-dev", gated=True,
          remote="flux1-redux-dev.safetensors", filename="FLUX_DEV_Redux.safetensors",
          type="Checkpoints", family="Flux", gb=0.129, deps=["siglip-so400m-384"], used_by=["cap.image_redux"],
          note="proiettore SigLIP→token di FLUX: l'immagine di riferimento guida stile e contenuto. " + NC),
        N("siglip-so400m-384", "Immagini", "SigLIP so400m-patch14-384 (encoder visivo di Redux, 3.5GB)",
          category="image", arch="flux1", repo="google/siglip-so400m-patch14-384",
          remote="model.safetensors", filename="model.safetensors", type="VisionEncoders", family="", gb=3.51,
          scan=["siglip-so400m"], used_by=["cap.image_redux"],
          note="file completo (testo+visione): a Redux serve solo la torre visiva (caricata da sola). Apache-2.0"),
        N("openpose-body", "ControlNet", "OpenPose corpo (CMU, preprocessore di posa, 0.21GB)",
          category="aux", arch="openpose", repo="lllyasviel/Annotators", remote="body_pose_model.pth",
          filename="body_pose_model.pth", type="ControlNet", family="Openpose", gb=0.209,
          used_by=["cap.image_controlnet"], note="mappa di posa (scheletro a 18 punti) per i ControlNet"),
        # CODICE derivato di MagicQuill V2 (CC BY-NC 4.0): componente opzionale, mai nella build; lo scarica l'utente. Hosting
        # DA DECIDERE: finche' manca la fonte e' `no_remote` (scripts/package_magicquill_code.py crea lo zip da pubblicare).
        N("mq2-code", "Immagini", "MagicQuill V2 · componente codice (CC BY-NC 4.0, ~0.1MB)",
          category="image", arch="flux1", kind="snapshot", tool_dest=r"E:\AI\Models\LoRA\MagicQuill\_code", gb=0.001,
          no_remote=True, used_by=["cap.image_magicquill"],
          note="codice derivato di MagicQuill V2 (opera derivata, CC BY-NC 4.0, uso NON commerciale): non e' incluso nell'app "
               "ne' nell'installer. Fonte in attesa di hosting: pubblica lo zip creato da scripts/package_magicquill_code.py "
               "(es. repo HF pubblico) e imposta repo/togli no_remote in build/catalog_task9.py"),
    ] + [
        N(f"mq2-{m}-lora", "Immagini", f"MagicQuill V2 · {lbl} (LoRA di controllo FLUX.1 Kontext, 0.3GB)",
          category="image", arch="flux1", repo="LiuZichen/MagicQuillV2-models", remote=f"v2_ckpt/{m}_lora.safetensors",
          filename=f"{m}_lora.safetensors", type="LoRA", family="MagicQuill", gb=gb, used_by=["cap.image_magicquill"],
          note=f"{desc} Licenza CC BY-NC-SA 4.0 (non commerciale): i pesi li scarica l'utente")
        for m, lbl, gb, desc in (
            ("local", "modifica locale", 0.299, "area disegnata + prompt (Kontext + condizione «local»)."),
            ("removal", "rimozione oggetto", 0.299, "toglie l'oggetto mascherato riempiendo lo sfondo."),
            ("edge", "bordi guidati", 0.299, "segue i contorni (originali + tratti aggiunti/rimossi)."),
            ("color", "colore guidato", 0.299, "ricolora seguendo blocchi di colore."),
            ("puzzle", "integrazione (foreground)", 0.345, "fonde un oggetto incollato nella scena (LoRA PEFT normale)."))
    ]
    for e in out:
        if e["id"] == "mq2-code":
            e["license"] = "CC BY-NC 4.0"
            e["license_ack"] = "cc_by_nc"
        elif e["id"].startswith("mq2-"):                # mai in silenzio: dialog con avviso di licenza + conferma (core/license_ack)
            e["license"] = "CC BY-NC-SA 4.0"
            e["license_ack"] = "cc_by_nc_sa"
    return out


# id → (tag d'uso, deps) per catalog_use_deps
USE = {
    "flux-cn-union-pro2": (["controlnet_image"], ["flux-vae-ae", "t5xxl-fp16|t5xxl-enconly-fp8"]),
    "flux-cn-inpaint-alimama": (["controlnet_image"], ["flux-vae-ae", "t5xxl-fp16|t5xxl-enconly-fp8"]),
    "zimage-cn-union21": (["controlnet_image"], []),
    "flux-redux": (["image_adapter"], []),
    "siglip-so400m-384": (["image_adapter"], []),
    "openpose-body": (["depth_ctrl", "controlnet_image"], []),
    "mq2-local-lora": (["lora_style", "control_lora"], []),
    "mq2-removal-lora": (["lora_style", "control_lora"], []),
    "mq2-edge-lora": (["lora_style", "control_lora"], []),
    "mq2-color-lora": (["lora_style", "control_lora"], []),
    "mq2-puzzle-lora": (["lora_style", "control_lora"], []),
    "mq2-code": (["control_lora"], []),
}
NEW_USE_TAGS = {
    "controlnet_image": "ControlNet immagini: la struttura (bordi, profondità, posa) di una foto guida decide la composizione.",
    "image_adapter": "Adattatore immagine (Redux): una foto di riferimento guida stile e contenuto della nuova immagine.",
    "control_lora": "LoRA di controllo MagicQuill: modifica guidata da maschere, bordi e colori (usata solo da MagicQuill).",
}
