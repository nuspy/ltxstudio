"""Krea 2 Edit — editing immagine per istruzione (in-context), port NATIVO di `comfyui-krea2edit`.

Krea 2 è della famiglia Qwen-Image (VAE Qwen-Image + text-encoder Qwen3-VL). L'edit replica i due canali
del pack ComfyUI, senza ComfyUI:

  1) CANALE ASPETTO — la/e foto sorgente vengono VAE-encodate e i loro token latenti (PULITI) sono
     concatenati alla sequenza immagine con `position_ids` all'asse 0 (frame) = 1..N (single: frame 1;
     multi: scena=1, soggetto=2), mentre il target sta al frame 0. È esattamente `Krea2EditModelPatch`
     (prepend di clean in-context tokens ai frame RoPE). Il transformer, addestrato così, tratta i token
     a frame>0 come riferimento pulito indipendentemente dal timestep.

  2) CANALE SEMANTICA — Qwen3-VL codifica prompt + immagine INSIEME (grounded encode): la foto (ridotta a
     lato lungo `grounding_px`) è inserita nel template con il blocco `<|vision_start|>…<|vision_end|>`
     (template a 2 blocchi per il multi). `grounding_px` è la manopola edit↔identità (basso = edit più
     forte, alto = preserva identità). È `Krea2EditGroundedEncode`.

Riferimenti verbatim: `diffusers.pipelines.krea2.pipeline_krea2.Krea2Pipeline` (pack/position_ids/loop) e
`diffusers.pipelines.qwenimage.pipeline_qwenimage_edit._get_qwen_prompt_embeds` (immagine→text-encoder).

Nota: lo snapshot Krea 2 porta `tokenizer` ma NON un image-processor: il processor Qwen3-VL e' costruito
localmente (config in `workers/qwen3vl_processor/`, niente rete); se manca l'edit si ferma con un errore
(`KREA2_EDIT_ALLOW_TEXT_ONLY=1` ripristina il vecchio degrado a solo-testo, che produce una scena NUOVA).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from diffusers.pipelines.krea2.pipeline_krea2 import (
    Krea2Pipeline,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.pipelines.krea2.pipeline_output import Krea2PipelineOutput


# Template grounded: sistema Krea 2 + blocco/i visione + istruzione. Modellato sul template Qwen-Image-Edit
# ma con la descrizione Krea 2. Il numero di token-prefisso da scartare (drop) è ricalcolato a runtime.
_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
_GROUND_SYS = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency with the original "
    "input where appropriate.<|im_end|>\n<|im_start|>user\n"
)
_GROUND_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


class Krea2EditPipeline(Krea2Pipeline):
    """`Krea2Pipeline` esteso con l'edit in-context (single & multi immagine)."""

    # processor Qwen-VL costruito pigramente (lo snapshot non lo include)
    _edit_processor: Any = None
    _edit_processor_tried: bool = False

    # ---------------------------------------------------------------- processor Qwen-VL (grounded)
    def _get_processor(self):
        """Processor Qwen3-VL per dare l'immagine al text-encoder, costruito SENZA rete (il worker gira con
        HF_HUB_OFFLINE=1): tokenizer dello snapshot + image/video processor dalle due config in
        `workers/qwen3vl_processor/` (Qwen3-VL: patch 16, merge 2 - le stesse del text-encoder dello snapshot).
        Prima del task 16b si tentava `AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')` dall'hub:
        offline falliva e l'edit degradava IN SILENZIO a solo-testo (= scena nuova al posto della modifica).
        Ritorna None solo se tutto fallisce; `grounded_text_hidden_states` allora solleva un errore chiaro."""
        if self._edit_processor_tried:
            return self._edit_processor
        self._edit_processor_tried = True
        errs: list[str] = []
        try:
            from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen3VLProcessor
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3vl_processor")
            for d in (os.environ.get("KREA2_PROCESSOR_DIR", ""), here):
                if not d or not os.path.isfile(os.path.join(d, "preprocessor_config.json")):
                    continue
                try:
                    self._edit_processor = Qwen3VLProcessor(
                        image_processor=AutoImageProcessor.from_pretrained(d),
                        video_processor=AutoVideoProcessor.from_pretrained(d), tokenizer=self.tokenizer)
                    print(f"[krea2-edit] processor Qwen3-VL locale: {d}", flush=True)
                    return self._edit_processor
                except Exception as ex:                          # noqa: BLE001
                    errs.append(f"{d}: {ex}")
        except Exception as ex:                                  # noqa: BLE001
            errs.append(str(ex))
        self._edit_processor_error = "; ".join(errs) or "config del processor non trovate"
        self._edit_processor = None
        return None

    # ---------------------------------------------------------------- canale semantica (grounded encode)
    def grounded_text_hidden_states(
        self,
        prompt: str | list[str],
        images: list | None,
        grounding_px: int = 768,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Come `get_text_hidden_states` ma con l'immagine (o due) data a Qwen3-VL. Fallback text-only se il
        processor non c'è. Ritorna `(hidden_states (B,seq,layers,dim), mask (B,seq) bool)`."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        proc = self._get_processor() if images else None

        if images and proc is None:
            # MAI degradare in silenzio a solo-testo: senza il canale immagine il modello inventa una scena nuova
            # dal testo (visto nel test Q4 del 18/10: due donne in studio invece della foto modificata).
            if os.environ.get("KREA2_EDIT_ALLOW_TEXT_ONLY") == "1":
                print("[krea2-edit] ATTENZIONE: processor Qwen3-VL assente -> encode SOLO TESTO", flush=True)
                return self.get_text_hidden_states(prompt, max_sequence_length, device)
            raise RuntimeError("Krea 2 Edit: processor Qwen3-VL non disponibile ("
                               + str(getattr(self, "_edit_processor_error", "?")) + ")")
        if proc is None:
            return self.get_text_hidden_states(prompt, max_sequence_length, device)

        # ridimensiona le immagini a lato lungo = grounding_px (area) come nel pack ComfyUI
        pil_imgs = [self._resize_grounding(im, grounding_px) for im in images]
        n_vision = len(pil_imgs)
        vision = _VISION_BLOCK * n_vision
        texts = [_GROUND_SYS + vision + e + _GROUND_SUFFIX for e in prompt]

        model_inputs = proc(
            text=texts,
            images=pil_imgs if n_vision else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs.get("pixel_values"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            # transformers >= 5: senza mm_token_type_ids il Qwen3-VL rifiuta l'input multimodale (mRoPE)
            **({"mm_token_type_ids": model_inputs["mm_token_type_ids"]} if "mm_token_type_ids" in model_inputs else {}),
            output_hidden_states=True,
        )
        # Krea 2 impila PIÙ layer (num_text_layers) → mantieni la stessa forma del t2i
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2
        )
        attention_mask = model_inputs["attention_mask"].bool()
        # scarta il prefisso di sistema (i token immagine restano: sono parte del grounding)
        # Il prefisso da scartare e' quello del TEMPLATE USATO qui (_GROUND_SYS = 64 token), non i 34 del t2i
        # (prompt_template_encode_start_idx): con 34 restavano 30 token di istruzione di sistema nel condizionamento.
        drop = self._ground_drop()
        hidden_states = hidden_states[:, drop:]
        attention_mask = attention_mask[:, drop:]
        return hidden_states, attention_mask

    def _ground_drop(self) -> int:
        """Numero di token del prefisso di sistema del template grounded (misurato col tokenizer dello snapshot)."""
        d = getattr(self, "_ground_drop_n", None)
        if d is None:
            d = len(self.tokenizer(_GROUND_SYS).input_ids)
            self._ground_drop_n = d
        return d

    @staticmethod
    def _resize_grounding(image, grounding_px: int):
        """PIL: ridimensiona così che il lato lungo = grounding_px (mantiene le proporzioni)."""
        from PIL import Image
        if not isinstance(image, Image.Image):
            return image
        w, h = image.size
        long = max(w, h)
        if long <= 0:
            return image
        s = float(grounding_px) / float(long)
        return image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

    # ---------------------------------------------------------------- canale aspetto (ref latents)
    def encode_ref_image(self, image, height, width, device, generator=None) -> tuple[torch.Tensor, int, int]:
        """VAE-encode di UNA foto sorgente → token latenti packed (spazio-modello) + grid (h,w) latente.
        Normalizzazione = inversa del decode di Krea2Pipeline: `model = (vae - mean) / std`."""
        # preprocess alla risoluzione target (stesse proporzioni: vincolo del modello)
        px = self.image_processor.preprocess(image, height=height, width=width)
        px = px.to(device=device, dtype=self.vae.dtype)
        if px.dim() == 4:                                        # (B,C,H,W) → (B,C,1,H,W): il VAE Qwen è 3D
            px = px.unsqueeze(2)
        enc = self.vae.encode(px).latent_dist.sample(generator)  # (B, z, 1, h, w)
        z = self.vae.config.z_dim
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=enc.dtype).view(1, z, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device, dtype=enc.dtype).view(1, z, 1, 1, 1)
        lat = (enc - mean) / std
        lat = lat[:, :, 0]                                       # (B, z, h, w)
        b, c, lh, lw = lat.shape
        tokens = self._pack_latents(lat, b, c, lh, lw)           # (B, (lh/p)*(lw/p), z*p*p)
        gh, gw = lh // self.patch_size, lw // self.patch_size
        # usa il `device` passato (cuda): dopo l'offload del text-encoder, self._execution_device
        # potrebbe ritornare CPU e mandare i ref-token sul device sbagliato.
        return tokens.to(device), gh, gw

    @staticmethod
    def _frame_image_ids(grid_h: int, grid_w: int, frame: int, device) -> torch.Tensor:
        """`(grid_h*grid_w, 3)` con asse 0 (frame) = `frame`, assi 1/2 = griglia latente."""
        ids = torch.zeros(grid_h, grid_w, 3, device=device)
        ids[..., 0] = float(frame)
        ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        return ids.reshape(grid_h * grid_w, 3)

    # ---------------------------------------------------------------- edit (flusso completo)
    @torch.no_grad()
    def edit(
        self,
        images: list,
        prompt: str,
        negative_prompt: str = "",
        grounding_px: int = 768,
        num_inference_steps: int = 8,
        guidance_scale: float = 0.0,
        height: int | None = None,
        width: int | None = None,
        sigmas: list[float] | None = None,
        generator: torch.Generator | None = None,
        max_sequence_length: int = 512,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        """Edit per istruzione. `images`: 1 (single) o 2 (multi: scena, soggetto). Le proporzioni target
        seguono la 1ª immagine se height/width non sono dati (vincolo stesse-proporzioni del modello)."""
        assert images, "krea2 edit: serve almeno un'immagine sorgente"
        device = self._execution_device

        # risoluzione target dalla 1ª immagine (arrotondata al multiplo richiesto, ≤ ~2MP)
        from PIL import Image
        base = images[0]
        if height is None or width is None:
            if isinstance(base, Image.Image):
                width, height = base.size
            else:
                height, width = 1024, 1024
        height, width = self._round_to_multiple(height, width)

        # 1) canale semantica: prompt (+ neg) grounded con l'immagine/e
        prompt_embeds, prompt_embeds_mask = self.grounded_text_hidden_states(
            prompt, images, grounding_px, max_sequence_length, device
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        do_cfg = guidance_scale > 0
        if do_cfg:
            neg_imgs = images  # a CFG>1 il negativo usa le STESSE immagini + prompt vuoto (come da README)
            negative_prompt_embeds, negative_prompt_embeds_mask = self.grounded_text_hidden_states(
                negative_prompt or "", neg_imgs, grounding_px, max_sequence_length, device
            )
        # Il text-encoder (Qwen3-VL ~9GB) NON serve più nel loop di denoise: scaricalo in CPU così le
        # attivazioni della sequenza in-context (target+ref, lunga) non sbordano in shared-VRAM (→ 100× più lento).
        try:
            if not getattr(self, "_ls_offloaded", False):        # non toccare se c'è già model_cpu_offload
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()
        except Exception:                                        # noqa: BLE001
            pass

        # 2) canale aspetto: VAE-encode delle sorgenti → token puliti + position_ids frame 1..N.
        # Ogni riferimento è codificato alle SUE proporzioni (il position_ids per-frame gestisce griglie
        # diverse): la 1ª immagine = target (stessa dimensione, per il single edit same-size); i riferimenti
        # AGGIUNTIVI (soggetto nel multi) mantengono il LORO aspect così il volto non viene deformato
        # (era la causa della bassa fedeltà del viso nel multi).
        ref_tokens_list, ref_ids_list = [], []
        for k, im in enumerate(images):
            if k == 0:
                rh, rw = height, width
            elif isinstance(im, Image.Image):
                rw0, rh0 = im.size
                rh, rw = self._round_to_multiple(rh0, rw0)
            else:
                rh, rw = height, width
            toks, gh, gw = self.encode_ref_image(im, rh, rw, device, generator)
            ref_tokens_list.append(toks.to(device=device, dtype=prompt_embeds.dtype))
            ref_ids_list.append(self._frame_image_ids(gh, gw, frame=k + 1, device=device))
        ref_tokens = torch.cat(ref_tokens_list, dim=1) if ref_tokens_list else None

        # 3) target: latente rumoroso + position_ids (text @0, target @frame0, ref @frame1..N)
        num_channels_latents = self.transformer.config.in_channels // (self.patch_size**2)
        latents = self.prepare_latents(
            latents=None, batch_size=1, num_channels_latents=num_channels_latents,
            height=height, width=width, dtype=prompt_embeds.dtype, device=device, generator=generator,
        )
        gh_t = height // (self.vae_scale_factor * self.patch_size)
        gw_t = width // (self.vae_scale_factor * self.patch_size)
        text_seq = prompt_embeds.shape[1]
        pos_ids = [self.prepare_position_ids(text_seq, gh_t, gw_t, device)]  # text + target(frame0)
        pos_ids.extend(ref_ids_list)                                          # + ref(frame1..N)
        position_ids = torch.cat(pos_ids, dim=0)
        target_seq = latents.shape[1]

        # 4) timesteps (schedule Krea 2)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if self.config.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                target_seq, self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5), self.scheduler.config.get("max_shift", 1.15),
            )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self.scheduler.set_begin_index(0)

        # 5) loop: concat ref (puliti) → forward → slice target → step
        with self.progress_bar(total=num_inference_steps) as bar:
            for t in timesteps:
                self._current_timestep = t
                timestep = (t / self.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(latents.dtype)
                hs = torch.cat([latents, ref_tokens], dim=1) if ref_tokens is not None else latents

                noise_pred = self.transformer(
                    hidden_states=hs, encoder_hidden_states=prompt_embeds, timestep=timestep,
                    position_ids=position_ids, encoder_attention_mask=prompt_embeds_mask,
                    attention_kwargs=self._attention_kwargs, return_dict=False,
                )[0][:, :target_seq]

                if do_cfg:
                    neg_pred = self.transformer(
                        hidden_states=hs, encoder_hidden_states=negative_prompt_embeds, timestep=timestep,
                        position_ids=position_ids, encoder_attention_mask=negative_prompt_embeds_mask,
                        attention_kwargs=self._attention_kwargs, return_dict=False,
                    )[0][:, :target_seq]
                    noise_pred = noise_pred + guidance_scale * (noise_pred - neg_pred)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                bar.update()
        self._current_timestep = None

        # 6) decode (identico a Krea2Pipeline)
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width).to(self.vae.dtype)
            z = self.vae.config.z_dim
            mean = torch.tensor(self.vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
            std_inv = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / std_inv + mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Krea2PipelineOutput(images=image)

    def _round_to_multiple(self, height: int, width: int) -> tuple[int, int]:
        m = self.vae_scale_factor * self.patch_size
        h = ((int(height) + m - 1) // m) * m
        w = ((int(width) + m - 1) // m) * m
        # cap a ~2MP (vincolo del modello: oltre → bleed/duplicazioni)
        max_px = 2_000_000
        if h * w > max_px:
            s = (max_px / (h * w)) ** 0.5
            h = max(m, (int(h * s) // m) * m)
            w = max(m, (int(w * s) // m) * m)
        return h, w
