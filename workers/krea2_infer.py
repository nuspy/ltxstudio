"""Worker inference Krea 2 — architettura IMMAGINI alternativa (accanto a FLUX/Z-Image/Ernie).

Eseguito nel venv `E:\\AI\\Tools\\lingbot_venv311` (py3.11 + torch 2.11 cu130 + diffusers 0.39 + Qwen3-VL +
FlashAttention-2 sm120 — stesse deps di LingBot) via subprocess da `core/img_gen_job.py`.

Krea 2 è diffusers (`Krea2Pipeline`): text-encoder Qwen3-VL, VAE Qwen-Image, transformer ~13B → su 32GB
serve **fp8** (layerwise casting, come FLUX). Due varianti: Turbo (8 step, cfg 0, mu 1.15) e Raw (52 step,
cfg 3.5). Prompt in TESTO semplice (niente JSON). Legge params JSON (argv[1]); scrive un PNG; stampa OK:<path>.

⚠️ LIVE-TEST PENDENTE: firma esatta di `Krea2Pipeline.__call__` (mu/guidance) e fp8 da validare al 1° run.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_grad_enabled(False)

_VARIANT_DEFAULTS = {                                        # step/cfg/mu per variante (dai docs Krea 2)
    "turbo": {"steps": 8, "guidance": 0.0, "mu": 1.15},
    "raw":   {"steps": 52, "guidance": 3.5, "mu": None},
}


def _resolve_model_dir(p: dict) -> tuple[str, str]:
    """(model_dir, variant). variant ∈ {turbo, raw}."""
    variant = str(p.get("krea2_variant", "turbo")).lower()
    m = p.get("model") or ""
    if m and os.path.isdir(m):
        return m, ("raw" if "raw" in m.lower() else "turbo")
    try:
        from core import weights_catalog as wc
        entries = {e["id"]: e for e in wc.load()}
        eid = "krea2_raw" if variant == "raw" else "krea2_turbo"
        e = entries.get(eid)
        if e and wc.is_present(e):
            return str(wc.dest_for(e)), variant
        if e:
            return str(wc.dest_for(e)), variant
    except Exception as ex:                                   # noqa: BLE001
        print(f"[krea2] risoluzione catalogo fallita: {ex}", flush=True)
    raise RuntimeError("Modello Krea 2 non trovato. Scaricalo dal wizard 'Pesi' (modulo Immagini).")


def _free_vram_mb() -> int:
    try:
        from core import gpu_mem
        return int(gpu_mem.free_vram_mb())
    except Exception:                                        # noqa: BLE001
        return 0


_PIPE_CACHE: dict = {}


def _evict_pipes() -> None:
    """Scarica tutti i pipe in cache e libera la VRAM (single-slot per il server persistente)."""
    import gc
    n = len(_PIPE_CACHE)
    try:                                                       # F3: ledger live
        from core import vram_ledger
        for _k in _PIPE_CACHE:
            vram_ledger.unregister(f"krea2:{_k[0]}")
    except Exception:                                          # noqa: BLE001
        pass
    _PIPE_CACHE.clear()
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                          # noqa: BLE001
        pass
    if n:
        print(f"[krea2] single-slot: sfrattati {n} pipe per liberare la VRAM prima del nuovo load", flush=True)


def _demote_pipes() -> float:
    """M-park (unified-model-manager): parcheggia i pipe in cache su CPU-RAM invece di distruggerli
    (la pipeline successiva è di un'altra categoria, o ignota: se si torna a Krea2 il pipe rientra
    dalla RAM senza rileggere l'SSD). I pipe con cpu_offload hanno già i pesi in RAM → no-op.
    Ritorna i MB (stimati) lasciati parcheggiati."""
    import gc
    mb = 0.0
    for key, pipe in list(_PIPE_CACHE.items()):
        try:
            if getattr(pipe, "_hf_hook", None) or getattr(getattr(pipe, "transformer", None), "_hf_hook", None):
                continue                                       # cpu_offload: pesi già in RAM, hook fragili
            pipe.to("cpu")
            pipe._ls_parked_cpu = True                         # ri-promosso al prossimo cache-hit
            for _m in ("transformer", "vae", "text_encoder"):
                _mod = getattr(pipe, _m, None)
                if _mod is not None:
                    try:
                        mb += sum(p.numel() * p.element_size() for p in _mod.parameters()) / 2**20
                    except Exception:                          # noqa: BLE001
                        pass
            try:                                               # F3: ledger live
                from core import vram_ledger
                vram_ledger.set_state(f"krea2:{key[0]}", "ram_parked")
            except Exception:                                  # noqa: BLE001
                pass
        except Exception as ex:                                # noqa: BLE001 — pipe non spostabile → distruggi
            print(f"[krea2] demote fallita ({ex}) → destroy", flush=True)
            _PIPE_CACHE.pop(key, None)
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                          # noqa: BLE001
        pass
    if mb:
        print(f"[krea2] parcheggiati in RAM ~{mb:.0f}MB di pipe (VRAM libera)", flush=True)
    return mb


def _resolve_dit(p: dict, key: str = "krea2_dit", what: str = "Transformer") -> str:
    """File singolo scelto dall'utente (`krea2_dit` transformer / `krea2_te` text encoder: id di catalogo o path),
    "" = quello dello snapshot. Mancante su disco → errore chiaro, MAI ripiego silenzioso sull'ufficiale."""
    d = str(p.get(key) or "").strip()
    if not d:
        return ""
    if os.path.isfile(d):
        return d
    from core import weights_catalog as wc
    e = {x["id"]: x for x in wc.load()}.get(d)
    if e and e.get("filename"):
        path = os.path.join(str(wc.dest_for(e)), e["filename"])
        if os.path.isfile(path):
            return path
    raise RuntimeError(f"{what} Krea 2 «{d}» non trovato su disco: scaricalo dal tab «Pesi modelli».")


def _te_kwargs(model_dir: str, te: str) -> dict:
    """`text_encoder=` per from_pretrained quando l'utente ha scelto un TE single-file (es. uncensored fp8)."""
    if not te:
        return {}
    from workers import krea2_custom as K
    print(f"[krea2] text encoder single-file: {os.path.basename(te)}", flush=True)
    return {"text_encoder": K.build_text_encoder(te, os.path.join(model_dir, "text_encoder"))}


def _load_pipeline_single_file(model_dir: str, variant: str, loras: tuple, dit: str, te: str = ""):
    """Transformer da file singolo nativo (int8 ConvRot / fp8: già compresso, niente layerwise casting) su
    text encoder, VAE e scheduler dello snapshot `model_dir`. LoRA agganciate ai Linear quantizzati."""
    from diffusers import DiffusionPipeline
    from workers import krea2_custom as K
    print(f"[krea2] transformer single-file: {os.path.basename(dit)} (base {os.path.basename(model_dir)})", flush=True)
    tr = K.build_transformer(dit, os.path.join(model_dir, "transformer"))
    if loras:
        K.apply_loras(tr, list(loras), log=lambda m: print(m, flush=True))
    pipe = DiffusionPipeline.from_pretrained(model_dir, transformer=tr, torch_dtype=torch.bfloat16,
                                             **_te_kwargs(model_dir, te))
    free = _free_vram_mb()
    if free < 26000:                       # TE (8.8GB bf16) + VAE + attivazioni accanto al transformer int8: su 32GB → offload sequenziale (verificato 14/10)
        pipe.enable_model_cpu_offload()
        print(f"[krea2] model_cpu_offload (VRAM libera {free}MB)", flush=True)
    else:
        pipe = pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:                                          # noqa: BLE001
        pass
    return pipe


def _load_pipeline(model_dir: str, variant: str, loras: tuple = (), dit: str = "", te: str = ""):
    key = (model_dir, variant, tuple(loras), dit, te)
    if key in _PIPE_CACHE:
        pipe = _PIPE_CACHE[key]
        if getattr(pipe, "_ls_parked_cpu", False):             # M-park: rientro VRAM dalla RAM (no SSD)
            print("[krea2] pipe parcheggiato in RAM → rientro su GPU", flush=True)
            pipe = pipe.to("cuda")
            pipe._ls_parked_cpu = False
            _PIPE_CACHE[key] = pipe
            try:                                               # F3: ledger live
                from core import vram_ledger
                vram_ledger.set_state(f"krea2:{key[0]}", "vram")
            except Exception:                                  # noqa: BLE001
                pass
        return pipe
    # SINGLE-SLOT: prima di caricare un pipe con una chiave NUOVA, sfratta gli altri e libera la VRAM.
    # Nel server PERSISTENTE i job alternano t2i (no LoRA) ed edit (LoRA identità): tenere 2 pipe fp8
    # (~13GB l'uno) satura i 32GB → il 2° carica con VRAM stretta → `model_cpu_offload`, che rompe
    # l'iniezione dei token di riferimento dell'edit (device/offload) → l'identità si perde (drift anime).
    # Tenendo UN pipe per volta ogni load ha la VRAM piena → niente offload → edit coerente. Nel subprocess
    # one-shot c'è comunque una sola chiave, quindi è un no-op.
    if _PIPE_CACHE:
        _evict_pipes()
    if dit:
        _PIPE_CACHE[key] = _load_pipeline_single_file(model_dir, variant, tuple(loras), dit, te)
        return _PIPE_CACHE[key]
    from diffusers import DiffusionPipeline
    print(f"[krea2] carico Krea2Pipeline: {model_dir} (variant={variant})", flush=True)
    _te = _te_kwargs(model_dir, te)
    try:
        from krea2 import Krea2Pipeline as _Pipe                # se il pacchetto è installato
        pipe = _Pipe.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **_te)
    except Exception:                                          # noqa: BLE001 — fallback custom-code HF
        pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 **_te)

    # LoRA PRIMA del fp8: si caricano e si FONDONO nei pesi bf16, poi si casta a fp8 (altrimenti la matmul
    # della LoRA su pesi Float8_e4m3fn non è implementata).
    _apply_loras(pipe, list(loras))

    # ── PLACEMENT PLANNER-DRIVEN (unificazione, commit 3): decide dal footprint MISURATO nel DB
    # canonico (core/vram_estimate, popolato al primo load) invece delle soglie fisse. Prima run
    # (DB muto) = 'legacy' → soglie storiche INVARIATE; dalla seconda il piano è esatto:
    # bf16 pieno se entra (qualità massima), fp8 layerwise se entra solo così, offload se niente.
    free = _free_vram_mb()
    _plan = "legacy"
    try:
        from core import vram_calibration as _vc, vram_estimate as _ve
        _plan = _vc.plan_krea2_placement(free, _ve.cache_get(model_dir),
                                         _ve.cache_get(model_dir, "fp8"))
    except Exception:                                          # noqa: BLE001
        _plan = "legacy"
    _fp8_applied = False
    try:
        _want_fp8 = (_plan in ("cuda_fp8", "offload_fp8")) if _plan != "legacy" else (free < 40000)
        if _want_fp8 and hasattr(pipe, "transformer"):
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            _fp8_applied = True
            print(f"[krea2] transformer fp8 layerwise (plan={_plan}, VRAM libera {free}MB)", flush=True)
    except Exception as ex:                                    # noqa: BLE001
        print(f"[krea2] fp8 non applicato ({ex})", flush=True)

    try:
        _offload = (_plan == "offload_fp8") if _plan != "legacy" else (free < 30000)
        if _offload:
            pipe.enable_model_cpu_offload()
            print(f"[krea2] model_cpu_offload (plan={_plan})", flush=True)
        else:
            pipe = pipe.to("cuda")
            if _plan != "legacy":
                print(f"[krea2] residente su GPU (plan={_plan})", flush=True)
    except Exception:                                          # noqa: BLE001
        pipe = pipe.to("cuda")
    try:
        if hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()
    except Exception:                                          # noqa: BLE001
        pass
    try:                                                       # F3: ledger live — nuova pipe caricata
        from core import vram_ledger
        _mb = 0.0
        for _m in ("transformer", "vae", "text_encoder"):
            _mod = getattr(pipe, _m, None)
            if _mod is not None and hasattr(_mod, "parameters"):
                _mb += sum(q.numel() * q.element_size() for q in _mod.parameters()) / 2**20
        vram_ledger.register(f"krea2:{model_dir}", _mb, category="image", state="vram", path=model_dir)
        # commit 3: PERSISTI il footprint misurato nel DB canonico (con layerwise casting lo storage
        # è float8 → element_size()==1 → _mb È il footprint fp8; senza casting è il bf16). Dalla
        # prossima run il planner ha il dato esatto.
        if _mb > 0:
            from core import vram_estimate as _ve2
            _ve2.cache_put(model_dir, int(_mb), variant=("fp8" if _fp8_applied else ""),
                           source="measured", category="image", arch="krea2",
                           dtype=("fp8" if _fp8_applied else "bf16"))
    except Exception:                                          # noqa: BLE001
        pass
    # NB: il backend di attenzione lo imposta il CHIAMANTE — solo l'EDIT (sequenza in-context lunga con
    # attn_mask di padding) vuole flash_varlen (vedi _set_attention). Il t2i resta sul default del modello:
    # forzare flash_varlen anche sul t2i introduceva artefatti (macchie/griglie colorate seed-dipendenti).
    _PIPE_CACHE[key] = pipe
    return pipe


def _set_attention(pipe, backend: str) -> None:
    try:
        pipe.transformer.set_attention_backend(backend)
        print(f"[krea2] attention backend = {backend}", flush=True)
    except Exception as ex:                                    # noqa: BLE001
        print(f"[krea2] set_attention_backend('{backend}') fallito ({ex}) → default", flush=True)


def _guidance_of(p: dict, default: float) -> float:
    """La GUI emette `cfg`; il worker t2i storicamente leggeva `guidance` → default sempre. Accetta entrambe."""
    for k in ("guidance", "cfg"):
        if p.get(k) is not None:
            return float(p[k])
    return float(default)


def _find_identity_lora() -> str | None:
    """La LoRA `krea2_identity_edit` (raccomandata per preservare l'identità nell'edit) non è indicizzata
    pubblicamente → la si cerca per nome nelle cartelle LoRA locali (l'utente la scarica da Patreon/Civitai
    e la deposita lì). Ritorna il path (preferendo la versione più recente) o None."""
    import glob
    roots = [r"E:\AI\Models\Lora", r"E:\AI\Models\lora", r"E:\AI\Models\loras"]
    hits: list[str] = []
    for r in roots:
        hits += glob.glob(os.path.join(r, "**", "krea2_identity_edit*.safetensors"), recursive=True)
        hits += glob.glob(os.path.join(r, "**", "krea2*identity*edit*.safetensors"), recursive=True)
    hits = sorted(set(hits))
    return hits[-1] if hits else None                             # l'ordinamento mette v1_2 dopo v1


def _resolve_lora_paths(p: dict, edit: bool) -> list[tuple[str, float]]:
    """(path, weight) delle LoRA da applicare. In edit include, se presente, `krea2_identity_edit` @1.0
    (raccomandata per l'identità), più le LoRA scelte dall'utente in `p['loras']`."""
    out: list[tuple[str, float]] = []
    if edit:
        idp = _find_identity_lora()
        if idp:
            out.append((idp, 1.0))
            print(f"[krea2] LoRA identity_edit trovata: {os.path.basename(idp)}", flush=True)
        else:
            print("[krea2] LoRA krea2_identity_edit NON trovata (edit ok, identità meno preservata)", flush=True)
    for lo in (p.get("loras") or []):
        path = lo.get("path") or lo.get("name")
        if path and os.path.isfile(path):
            out.append((path, float(lo.get("strength", 1.0))))
    return out


def _convert_comfy_krea2_lora(sd: dict) -> dict:
    """Converte le chiavi di una LoRA Krea 2 in formato ComfyUI (es. krea2_identity_edit) al naming del
    `Krea2Transformer2DModel` di diffusers. Verificato sui moduli reali:
      diffusion_model.  → transformer.          | .blocks.  → .transformer_blocks.
      attn.wq/wk/wv/wo  → attn.to_q/to_k/to_v/to_out.0   | attn.gate → attn.to_gate
      mlp.gate/up/down  → ff.gate/up/down
      txtfusion.        → text_fusion.   (LoRA NSFW KNP/Mystic/SNOFS: senza, 64 tensori scartati in silenzio)
    """
    reps = [
        ("diffusion_model.", "transformer."), ("transformer.txtfusion.", "transformer.text_fusion."),
        (".blocks.", ".transformer_blocks."),
        (".attn.wq.", ".attn.to_q."), (".attn.wk.", ".attn.to_k."), (".attn.wv.", ".attn.to_v."),
        (".attn.wo.", ".attn.to_out.0."), (".attn.gate.", ".attn.to_gate."),
        (".mlp.gate.", ".ff.gate."), (".mlp.up.", ".ff.up."), (".mlp.down.", ".ff.down."),
    ]
    out = {}
    for k, v in sd.items():
        nk = k
        for a, b in reps:
            nk = nk.replace(a, b)
        out[nk] = v
    return out


def _apply_loras(pipe, loras: list[tuple[str, float]]) -> None:
    if not loras:
        return
    from safetensors.torch import load_file
    names, weights = [], []
    for i, (path, w) in enumerate(loras):
        try:
            an = f"krea2_lora_{i}"
            sd = load_file(path)
            if any(k.startswith("diffusion_model.") for k in sd):
                sd = _convert_comfy_krea2_lora(sd)               # ComfyUI → naming diffusers
            pipe.load_lora_weights(sd, adapter_name=an)
            names.append(an); weights.append(w)
            print(f"[krea2] LoRA caricata: {os.path.basename(path)} @{w} ({len(sd)} tensori)", flush=True)
        except Exception as ex:                                   # noqa: BLE001
            print(f"[krea2] LoRA fallita {os.path.basename(path)}: {ex}", flush=True)
    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            pipe.fuse_lora()                                      # fonde nei pesi bf16 (prima del cast fp8)
            pipe.unload_lora_weights()
            print("[krea2] LoRA fuse+unload OK", flush=True)
        except Exception as ex:                                   # noqa: BLE001
            print(f"[krea2] fuse_lora fallito ({ex})", flush=True)


def _load_images(p: dict):
    """Immagini sorgente per l'edit: `images[]` (multi) o `image_path` (single). PIL RGB."""
    from PIL import Image
    paths = p.get("images") or ([p["image_path"]] if p.get("image_path") else [])
    imgs = []
    for pth in paths[:2]:                                        # il modello gestisce fino a 2 riferimenti
        if pth and os.path.isfile(pth):
            imgs.append(Image.open(pth).convert("RGB"))
    return imgs


def run_edit(p: dict) -> str:
    """Editing per istruzione (single/multi) via Krea2EditPipeline (port nativo di comfyui-krea2edit)."""
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imgs = _load_images(p)
    if not imgs:
        raise RuntimeError("Krea 2 Edit: nessuna immagine sorgente (image_path/images).")
    model_dir, variant = _resolve_model_dir(p)
    pipe = _load_pipeline(model_dir, variant, tuple(_resolve_lora_paths(p, edit=True)), _resolve_dit(p),
                          _resolve_dit(p, "krea2_te", "Text encoder"))
    # trasforma la pipeline t2i caricata in quella di edit (stessi componenti, aggiunge edit())
    from workers.krea2_edit_pipeline import Krea2EditPipeline
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline as _Base
    if isinstance(pipe, _Base) and not isinstance(pipe, Krea2EditPipeline):
        pipe.__class__ = Krea2EditPipeline
    _set_attention(pipe, os.environ.get("KREA2_ATTN", "flash_varlen"))   # edit: sequenza lunga → flash_varlen
    d = _VARIANT_DEFAULTS.get(variant, _VARIANT_DEFAULTS["turbo"])

    result = pipe.edit(
        images=imgs,
        prompt=p.get("prompt", ""),
        negative_prompt=(p.get("negative") or ""),
        grounding_px=int(p.get("grounding_px", 768)),
        num_inference_steps=int(p.get("steps") or d["steps"]),
        guidance_scale=_guidance_of(p, d["guidance"]),
        generator=torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(int(p.get("seed", 0))),
    )
    img = result.images[0] if hasattr(result, "images") else result[0]
    img.save(out)
    _sign_and_watermark(out)
    return out


def _sign_and_watermark(out: str) -> str:
    try:                                                          # #107 stego + #106 watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                      # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                      # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model_dir, variant = _resolve_model_dir(p)
    pipe = _load_pipeline(model_dir, variant, tuple(_resolve_lora_paths(p, edit=False)),  # t2i ora applica le LoRA
                          _resolve_dit(p), _resolve_dit(p, "krea2_te", "Text encoder"))
    # ATTENZIONE T2I DETERMINISTICA: forza 'native' (SDPA). Il t2i NON ha attn_mask di padding, quindi
    # flash_varlen/flash/sage danno griglie/strisce colorate (macchie seed-dipendenti). Sul server
    # PERSISTENTE il pipe è condiviso: un run_edit precedente lo lascia in flash_varlen → il t2i successivo
    # erediterebbe quel backend e artefatterebbe. Resettando a 'native' ogni t2i è pulito e ripetibile,
    # indipendentemente da edit precedenti o da backend globali ereditati dall'ambiente. (bug KAIRA 18/07)
    _set_attention(pipe, os.environ.get("KREA2_ATTN_T2I", "native"))
    d = _VARIANT_DEFAULTS.get(variant, _VARIANT_DEFAULTS["turbo"])

    def _m16(v, default):
        v = int(p.get(v, default))
        return max(512, (v // 16) * 16)
    kwargs = dict(
        prompt=p.get("prompt", ""),
        num_inference_steps=int(p.get("steps") or d["steps"]),
        guidance_scale=_guidance_of(p, d["guidance"]),
        width=_m16("width", 1024), height=_m16("height", 1024),
        generator=torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(int(p.get("seed", 0))),
    )
    neg = p.get("negative") or ""
    if neg:
        kwargs["negative_prompt"] = neg
    # NB (validato 17/07): Krea2Pipeline.__call__ NON accetta `mu` — la firma espone `sigmas`.
    # Lo shift-timestep del Turbo (mu 1.15) è già codificato nel FlowMatchEulerDiscreteScheduler del
    # model_index (is_distilled=true). Quindi non passiamo mu: lo scheduler applica lo shift corretto.

    result = pipe(**kwargs)
    img = result.images[0] if hasattr(result, "images") else (result[0] if isinstance(result, (list, tuple)) else result)
    # NaN/immagine nera-degenere? affidiamo la validità all'utente; salviamo il PNG.
    img.save(out)
    out = _sign_and_watermark(out)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    is_edit = str(p.get("arch", "")).lower() == "krea2_edit" or str(p.get("mode", "")).lower() == "edit"
    print("OK:" + (run_edit(p) if is_edit else run_generation(p)), flush=True)


if __name__ == "__main__":
    main()
