"""MagicQuill V2 NATIVO su FLUX.1 Kontext (diffusers 0.38): modifica locale / rimozione / bordi / colore / integrazione.

LICENZA: il codice di riferimento MagicQuill V2 (https://github.com/zliucz/MagicQuillV2, (c) Zichen Liu et al.) è CC BY-NC 4.0 =
USO NON COMMERCIALE (il README upstream porta anche un badge CC BY-SA 4.0; il catalogo pesi dice CC BY-NC-SA 4.0): questo
modulo e i pesi `*_lora.safetensors` NON vanno distribuiti in un'edizione commerciale senza una decisione di licenza
dell'utente. Dettagli e attribuzione: `inference_server/backends/mq2/__init__.py`.

Handler `image_mq` (job_runner / persistent_runner): `run(params, mm, output_dir)` -> {"status","output":[png]} | {"status":"error","error"}.
Gira SOLO nel processo figlio del server (importa torch): mai dal GUI.

Contratto dei parametri:
    mode            "local" | "removal" | "edge" | "color" | "foreground"
    model           path assoluto del checkpoint FLUX.1 Kontext single-file (nessuna scelta automatica)
    image_path      immagine base RGB (foreground: la composizione con l'oggetto già incollato)
    mask_path       bianco = area da modificare (local/removal/foreground; color_rgb: area da colorare).
                    foreground: se manca `prop_mask_path`, e' la maschera dell'OGGETTO incollato
    prepared        (opz., foreground, default False) True = `image_path` e' GIA' l'immagine preparata come upstream
                    (composito con l'oggetto + alone BIANCO nella zona di edit): usata cosi' com'e' per Kontext, senza
                    aggiungere alone; `mask_path` = zona di edit e serve SOLO alla ricomposizione (grow 25 come
                    final_mask upstream, se output_blend). `prop_mask_path` e `grow` sono ignorati.
    prop_mask_path  (opz., foreground) maschera dell'oggetto; allora `mask_path` = area aggiuntiva da riempire
    edge_path / erase_path (edge, opz.) tratti AGGIUNTI / bordi da TOGLIERE (bianco = tratto)
    edge_source     "canny" (default, nessun peso) | "hed" (richiede ControlNetHED.pth: altrimenti errore chiaro)
    color_path      (color) immagine uguale alla base ma con le zone ricolorate;  oppure color_rgb [r,g,b] + mask_path
    prompt, negative (ignorato), steps (20), guidance (3.5), seed, strength_edge (0.6), strength_color (1.5),
    strength_local (1.0), grow (10; foreground 50), fix_perspective, loras [{'name','strength'}], output_blend (True),
    max_side (1536), lora_dir (opz.), placement_mode (opz., come image_flux)
"""
from __future__ import annotations

import os
import random
import time
import traceback
from pathlib import Path

LORA_DIR_DEFAULT = r"E:\AI\Models\LoRA\MagicQuill"
CONTROL_MODES = ("local", "removal", "edge", "color")
ARCH = "kontext"


def _lora_path(mode: str, params: dict) -> str:
    from core.model_paths import p as _cat_p
    base = params.get("lora_dir") or LORA_DIR_DEFAULT
    fallback = os.path.join(base, f"{mode}_lora.safetensors")
    if params.get("lora_dir"):
        return fallback
    return _cat_p(f"mq2-{mode}-lora", fallback)


def _gamma(params: dict, mode: str) -> float:
    key, dflt = {"local": ("strength_local", 1.0), "removal": ("strength_local", 1.0),
                 "edge": ("strength_edge", 0.6), "color": ("strength_color", 1.5)}[mode]
    v = params.get(key)
    return float(dflt if v is None else v)


def prepare_inputs(params: dict):
    """Validazione + preprocessing (tutto numpy/PIL, nessun modello): ritorna un dict con l'immagine base per
    Kontext, i controlli (condizione PIL + gamma + file LoRA), il prompt, e la maschera di ricomposizione.
    Solleva MQError con messaggi utente. Separata da `run` per i test CPU."""
    from PIL import Image
    from .mq2 import preprocess as pp
    mode = str(params.get("mode") or "").lower()
    if mode not in pp.MODES:
        raise pp.MQError(f"mode '{mode}' non valido (ammessi: {', '.join(pp.MODES)})")
    img_p = params.get("image_path")
    if not img_p or not Path(str(img_p)).is_file():
        raise pp.MQError("image_path mancante o inesistente")
    orig = Image.open(img_p).convert("RGB")
    iw, ih = orig.size
    W, H = pp.gen_size(iw, ih, int(params.get("max_side") or 1536))
    size = (W, H)
    img = pp.load_rgb(img_p, size)

    def _need_mask(key, what="maschera"):
        pth = params.get(key)
        if not pth or not Path(str(pth)).is_file():
            raise pp.MQError(f"{key} mancante o inesistente (serve per mode='{mode}': {what})")
        m = pp.load_mask(str(pth), size)
        if not m.any():
            raise pp.MQError(f"{key}: la maschera è vuota (nessun pixel bianco)")
        return m

    def _opt_mask(key):
        pth = params.get(key)
        if not pth:
            return None
        if not Path(str(pth)).is_file():
            raise pp.MQError(f"{key} inesistente: {pth}")
        m = pp.load_mask(str(pth), size)
        return m if m.any() else None

    grow = params.get("grow")
    out: dict = {"mode": mode, "size": size, "orig_size": (iw, ih), "orig": orig, "controls": [],
                 "blend_mask": None, "base_image": None,
                 "prompt": pp.build_prompt(mode, str(params.get("prompt") or ""), bool(params.get("fix_perspective"))),
                 "max_sequence_length": pp.max_sequence_length(mode)}
    if mode in ("local", "removal"):
        m = _need_mask("mask_path", "area da modificare" if mode == "local" else "oggetto da rimuovere")
        g = pp.grow_mask(m, int(10 if grow is None else grow))
        out["base_image"] = Image.fromarray(img)
        out["controls"].append({"mode": mode, "image": pp.black_masked(img, g), "gamma": _gamma(params, mode)})
        out["blend_mask"] = g
    elif mode == "foreground" and params.get("prepared"):
        m = _need_mask("mask_path", "zona di edit (solo ricomposizione)")
        out["base_image"] = Image.fromarray(img)             # immagine preparata: nessun alone aggiunto
        out["blend_mask"] = pp.grow_mask(m, 25)
    elif mode == "foreground":
        prop = _opt_mask("prop_mask_path")
        if prop is not None:
            fill = _need_mask("mask_path", "area da riempire")
            add_prop = prop
        else:
            fill = None
            add_prop = _need_mask("mask_path", "maschera dell'oggetto incollato")
        pil, edit = pp.foreground_image(img, add_prop, fill, int(50 if grow is None else grow))
        out["base_image"] = pil
        out["blend_mask"] = pp.grow_mask(edit, 25)
    elif mode == "edge":
        add = _opt_mask("edge_path")
        erase = _opt_mask("erase_path")
        out["base_image"] = Image.fromarray(img)
        src = str(params.get("edge_source") or "canny").lower()
        out["controls"].append({"mode": "edge", "image": pp.edge_control_image(img, src, add, erase),
                                "gamma": _gamma(params, "edge")})
    else:  # color
        out["base_image"] = Image.fromarray(img)
        cp = params.get("color_path")
        if cp:
            if not Path(str(cp)).is_file():
                raise pp.MQError(f"color_path inesistente: {cp}")
            colored = pp.load_rgb(str(cp), size)
            import numpy as np
            if np.array_equal(colored, img):
                raise pp.MQError("color_path è identico all'immagine base: nessuna zona ricolorata")
        elif params.get("color_rgb") is not None:
            rgb = params["color_rgb"]
            if len(rgb) != 3:
                raise pp.MQError("color_rgb deve essere [r,g,b]")
            m = _need_mask("mask_path", "area da colorare con color_rgb")
            colored = pp.fill_color_region(img, pp.grow_mask(m, int(10 if grow is None else grow)), rgb)
        else:
            raise pp.MQError("mode='color' richiede color_path oppure color_rgb + mask_path")
        out["controls"].append({"mode": "color", "image": pp.color_block(colored), "gamma": _gamma(params, "color")})
    for c in out["controls"]:
        c["lora_path"] = _lora_path(c["mode"], params)
        if not os.path.isfile(c["lora_path"]):
            raise pp.MQError(f"LoRA di controllo mancante: {c['lora_path']} (catalogo: mq2-{c['mode']}-lora)")
    if mode == "foreground":
        out["puzzle_path"] = _lora_path("puzzle", params)
        if not os.path.isfile(out["puzzle_path"]):
            raise pp.MQError(f"LoRA puzzle mancante: {out['puzzle_path']} (catalogo: mq2-puzzle-lora)")
    return out


def _real_loras(loras):
    """Stesso filtro di ModelManager.apply_loras: scarta i segnaposto della GUI."""
    def _real(lo):
        n = (lo.get("name") if isinstance(lo, dict) else lo) or ""
        return str(n).strip() and str(n).strip().lower() not in ("none", "nessuna", "nessuno", "-")
    return [lo for lo in (loras or []) if _real(lo)]


def _run(params: dict, mm, output_dir: str) -> dict:
    import torch
    from PIL import Image
    from inference_server import preview_bus
    from inference_server.backends import image_flux as imf
    from .mq2 import pipeline as mqp, puzzle as mqz, preprocess as pp
    from .mq2.control_lora import ControlLoRA, ControlSpec
    t0 = time.perf_counter()
    imf._step_reset(params.get("_t_button"))
    imf._step("magicquill_v2.run() START")

    model = params.get("model")
    if not model:
        raise pp.MQError("model mancante: serve il checkpoint FLUX.1 Kontext (nessuna scelta automatica)")
    model_path = imf._resolve_model_path(str(model))
    if not Path(model_path).is_file():
        raise pp.MQError(f"checkpoint non trovato: {model_path}")
    if imf._is_svdq(model_path):
        raise pp.MQError("MagicQuill V2 non supporta i checkpoint nunchaku/SVDQ: usa FLUX.1 Kontext fp8/bf16/GGUF")
    ich = imf._flux_in_channels(model_path)
    if ich not in (None, 64):
        raise pp.MQError(f"il checkpoint ha {ich} canali in ingresso: MagicQuill V2 richiede FLUX.1 Kontext (64)")
    job = prepare_inputs(params)                          # fallisce PRIMA di toccare la VRAM
    mode, (W, H) = job["mode"], job["size"]
    controls_job = job["controls"]
    loras_user = _real_loras(params.get("loras"))

    # --- stessa sequenza di image_flux.run per il modello Kontext (stessa chiave => pipeline calda riusata) ---
    key = imf.register(mm, str(model), ARCH)
    req_ident = imf._weights_ident(ARCH, imf._resolve_model_path(str(model)), "")
    preview_bus.set_message("Caricamento modello…")
    if os.environ.get("NATIVE_AUTOEVICT", "1") == "1" and key not in mm._cache:
        try:
            from inference_server.component_store import STORE
            if mm.evict_others(key, keep_ident=req_ident):
                tot = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else 0
                STORE.evict_unused(tot)
                mm.flush()
                import gc
                gc.collect()
        except Exception as e:                            # noqa: BLE001
            print(f"[mq2] pre-evict cambio modello saltato ({e})", flush=True)
    pipe = mm.ensure(key)
    pipe._ls_want_sticky = str(params.get("placement_mode", "")).lower() == "sticky"
    if os.environ.get("NATIVE_AUTOEVICT", "1") == "1":
        try:
            from inference_server.component_store import STORE
            free_mb = (torch.cuda.mem_get_info()[0] // 2**20) if torch.cuda.is_available() else 0
            need_mb = imf._measure_pipe_mb(pipe) + int(getattr(pipe, "_ls_working_mb", 0) or 3200)
            need_mb = max(0, need_mb - int(STORE.resident_of(getattr(pipe, "_ls_component_keys", ()) or ())))
            if free_mb < need_mb and mm.evict_others(key, keep_ident=req_ident):
                STORE.evict_unused(max(1, need_mb - free_mb))
                mm.flush()
        except Exception as e:                            # noqa: BLE001
            print(f"[mq2] auto-evict saltato ({e})", flush=True)
    where = imf._ensure_placement(pipe, mm, key)
    imf._set_attention(pipe, ARCH)
    print(f"[mq2] job mode={mode} placement={where} size={W}x{H}", flush=True)

    # --- LoRA normali dell'utente (+ puzzle SOLO in foreground) ---
    puzzle_on = False
    try:
        mm.apply_loras(pipe, loras_user)
        if mode == "foreground":
            try:
                mqz.attach(pipe, mqz.load_state_dict(job["puzzle_path"]),
                           [f"a{i}" for i in range(len(loras_user))],
                           [float(lo.get("strength", 1.0)) if isinstance(lo, dict) else 1.0 for lo in loras_user])
                puzzle_on = True
            except Exception as e:                        # noqa: BLE001
                hint = (" (base con storage fp8 layerwise: riavvia il server con NATIVE_FLUX1_FP8=0)"
                        if getattr(pipe.transformer, "_ls_fp8_storage", False) else "")
                raise pp.MQError(f"LoRA puzzle non caricabile: {type(e).__name__}: {e}{hint}") from e

        seed = int(params.get("seed", 0))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        steps = int(params.get("steps") or 20)
        kw = dict(prompt=job["prompt"], image=job["base_image"], height=H, width=W,
                  num_inference_steps=steps, guidance_scale=float(params.get("guidance") if params.get("guidance")
                                                                  is not None else 3.5),
                  generator=gen, max_sequence_length=job["max_sequence_length"],
                  max_area=W * H, _auto_resize=False)
        if params.get("negative"):
            print("[mq2] negative IGNORATO (Kontext guidance-distilled)", flush=True)
        if os.environ.get("NATIVE_NO_PREVIEW_CB") != "1":
            try:
                kw["callback_on_step_end"] = imf._make_preview_cb(ARCH, W, H, steps)
                kw["callback_on_step_end_tensor_inputs"] = ["latents"]
            except Exception:                             # noqa: BLE001
                pass
        preview_bus.set_message("Generazione…")
        imf._step(f"GEN START mode={mode}")
        if controls_job:
            specs = [ControlSpec(lora=ControlLoRA.from_file(c["lora_path"]), image=c["image"], gamma=c["gamma"],
                                 lora_weight=1.0, cond_size=512) for c in controls_job]
            with mqp.control_session(pipe, specs):
                res = pipe(**kw).images[0]
        else:
            res = pipe(**kw).images[0]
        imf._step("GEN FATTA")
    finally:
        if puzzle_on:
            mqz.detach(pipe)

    # --- ricomposizione: dimensioni originali + (local/removal/foreground) pixel originali fuori maschera ---
    orig = job["orig"]
    if res.size != orig.size:
        res = res.resize(orig.size, Image.LANCZOS)
    if job["blend_mask"] is not None and params.get("output_blend", True):
        import cv2
        import numpy as np
        mk = cv2.resize(job["blend_mask"], orig.size, interpolation=cv2.INTER_NEAREST)
        s = max(1.0, orig.size[0] / W)
        res = pp.blend_original(res, orig, mk, ksize=int(round(10 * s)), sigma=10.0 * s)
    os.makedirs(output_dir, exist_ok=True)
    path = str(Path(output_dir) / f"mq2_{mode}_{seed}_{int(time.time() * 1000) % 10**9}.png")
    res.save(path)
    try:                                                  # firma invisibile di provenienza (come image_flux)
        from core import stego
        if stego.enabled():
            stego.sign_file(path)
    except Exception:                                     # noqa: BLE001
        pass
    try:                                                  # watermark obbligatorio (gated feature_matrix)
        from core import watermark
        path = watermark.apply_to_file(path)
    except RuntimeError:
        return {"status": "error", "error": "watermark obbligatorio non applicabile"}
    except Exception:                                     # noqa: BLE001
        pass
    print(f"[mq2] fatto in {time.perf_counter() - t0:.1f}s -> {path}", flush=True)
    return {"status": "success", "output": [path]}


def run(params: dict, mm, output_dir: str) -> dict:
    from .mq2.attention import MQUnsupported
    from .mq2.preprocess import MQError
    try:
        return _run(params, mm, output_dir)
    except (MQError, MQUnsupported) as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:                                # noqa: BLE001
        print(f"[mq2] ERRORE: {traceback.format_exc()[-1500:]}", flush=True)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
