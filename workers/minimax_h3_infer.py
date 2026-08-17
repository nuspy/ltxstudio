"""Worker MiniMax H3 — 3ª architettura video (omni-modale: video + audio stereo insieme).

H3 è un unico transformer denso da 33.1B che denoisa NELLA STESSA PASSATA i latenti video e
quelli audio (stereo 32 kHz), condizionato da Qwen3-VL-32B. Tre modalità:
  · T2VA / FL2VA  → testo, o primo/ultimo fotogramma, verso video+audio   (checkpoint `fl2va`)
  · Ref2VA        → riferimenti multipli verso video+audio                (checkpoint `ref2va`)
I riferimenti (≤9 immagini, ≤3 video, ≤3 audio, max 12 file) NON hanno posizione temporale e
vanno citati nel prompt coi token `<Picture 1>`, `<Video 1>`, `<Audio 1>` — il pannello
«Riferimenti multipli» della GUI li assegna e li inserisce.

PESI: config+tokenizer ufficiali (~25MB) + single-file quantizzato, stesso schema di FLUX.2
(`init_empty_weights` + `from_config` + set_module_tensor_to_device coi Linear quantizzati):
evita i 124GB dello snapshot bf16 ufficiale. Il nvfp4 (12.5GB) riusa `Nvfp4Linear`, già
validato su FLUX.2-dev-NVFP4.

⚠ La chiamata di generazione (MiniMaxH3ModularPipeline) è scritta sulla documentazione
ufficiale ma NON è ancora stata eseguita su GPU: serve la prima generazione live coi pesi
scaricati. Finché non è validata, ogni errore risale alla GUI col traceback completo.

Uso:  python minimax_h3_infer.py params.json     → stampa "OK:<path mp4>"
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # repo root

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(f"[h3] {msg}", flush=True)


# ── risoluzione pesi dal REGISTRO UNICO (mai path hardcoded) ──────────────────
_CKPT_IDS = {                       # (modalità, preferenza) → id di catalogo, in ordine di scelta
    "fl2va": ["mmh3-fl2va-nvfp4", "mmh3-fl2va-int8", "mmh3-fl2va-fp8",
              "mmh3-fl2va-gguf-q4", "mmh3-fl2va-gguf-q3", "mmh3-fl2va-bf16"],
    "ref2va": ["mmh3-ref2va-nvfp4", "mmh3-ref2va-int8", "mmh3-ref2va-fp8",
               "mmh3-ref2va-gguf-q4"],
}
_TE_IDS = ["mmh3-te-nvfp4", "mmh3-te-int8", "mmh3-te-gguf-q4"]


def _first_present(ids: list[str]) -> str | None:
    """Primo peso PRESENTE su disco fra quelli elencati (nessuna sostituzione silenziosa:
    l'ordine è una preferenza dichiarata, non una scelta automatica di modello)."""
    from core import model_paths as mp
    from core import weights_catalog as wc
    for i in ids:
        e = mp.entry(i)
        if e and wc.is_present(e):
            return mp.p(i, "")
    return None


def resolve_weights(params: dict) -> dict:
    """Path dei componenti richiesti dalla modalità. Solleva con gli ID di catalogo mancanti."""
    from core import model_paths as mp
    mode = "ref2va" if params.get("references") else "fl2va"
    explicit = params.get("model") or ""
    ckpt = explicit if explicit and os.path.exists(explicit) else _first_present(_CKPT_IDS[mode])
    te = _first_present(_TE_IDS)
    cfg = mp.p("mmh3-configs", "")
    vae = _first_present(["mmh3-video-vae"])
    avae = _first_present(["mmh3-audio-vae"])
    missing = [name for name, val in (
        (f"checkpoint {mode} ({'/'.join(_CKPT_IDS[mode][:2])})", ckpt),
        ("text encoder (mmh3-te-nvfp4)", te),
        ("config+tokenizer (mmh3-configs)", cfg and os.path.isdir(cfg) and cfg),
        ("video VAE (mmh3-video-vae)", vae),
        ("audio VAE (mmh3-audio-vae)", avae),
    ) if not val]
    if missing:
        raise RuntimeError(
            "Pesi MiniMax H3 mancanti: " + "; ".join(missing) +
            ". Scaricali dal tab «Pesi modelli» (la catena tira encoder, VAE e config).")
    return {"mode": mode, "ckpt": ckpt, "te": te, "config_dir": cfg, "vae": vae, "audio_vae": avae}


# ── costruzione del transformer dal single-file quantizzato ───────────────────
def load_transformer(ckpt_path: str, config_dir: str, mode: str):
    """MiniMaxH3Transformer3DModel dai config ufficiali + pesi dal single-file quantizzato.
    Stesso schema di inference_server/backends/flux2_transformer.load_flux2_transformer: i Linear
    quantizzati restano compressi in RAM e si dequantizzano un layer alla volta nel forward."""
    import torch
    from torch import nn
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open

    from inference_server.backends.flux2_transformer import Fp8Linear, Nvfp4Linear

    sub = "FL2VA" if mode == "fl2va" else "Ref2VA"
    cfg_path = os.path.join(config_dir, sub, "transformer", "config.json")
    if not os.path.exists(cfg_path):                     # snapshot parziale/rinominato
        raise RuntimeError(f"config del transformer non trovato: {cfg_path} "
                           f"(riscarica la voce di catalogo 'mmh3-configs')")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        tr = MiniMaxH3Transformer3DModel.from_config(cfg)

    with safe_open(ckpt_path, framework="pt") as h:
        keys = set(h.keys())
        # nvfp4 = weight U8 + weight_scale a blocchi + weight_scale_2 globale; fp8/int8 = solo scala
        nvfp4 = {k[:-len(".weight")] for k in keys
                 if k.endswith(".weight") and (k[:-len(".weight")] + ".weight_scale_2") in keys}
        fp8 = {k[:-len(".weight")] for k in keys
               if k.endswith(".weight") and (k[:-len(".weight")] + ".weight_scale") in keys} - nvfp4
        _log(f"quantizzazione: {len(nvfp4)} moduli nvfp4, {len(fp8)} fp8/int8 "
             f"su {len(keys)} tensori")
        # i nomi del repack sono già quelli di diffusers (nessuna conversione tipo FLUX.2 BFL)
        modules = dict(tr.named_modules())
        for names, cls in ((nvfp4, Nvfp4Linear), (fp8, Fp8Linear)):
            for mod_name in names:
                old = modules.get(mod_name)
                if isinstance(old, nn.Linear):
                    parent_name, child = mod_name.rsplit(".", 1)
                    setattr(modules.get(parent_name, tr), child,
                            cls(old.in_features, old.out_features, bias=old.bias is not None))
        modules = dict(tr.named_modules())               # rinfresca dopo le sostituzioni
        targets = {n for n, _ in tr.named_parameters()} | {n for n, _ in tr.named_buffers()}

        assigned = skipped = 0
        for k in sorted(keys):
            if k.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                continue                                  # assegnati insieme al loro weight
            if k not in targets:
                skipped += 1
                continue
            mod = k.rsplit(".", 1)[0]
            t = h.get_tensor(k)
            if mod in nvfp4:                              # U8 packed: NON castare
                set_module_tensor_to_device(tr, k, "cpu", value=t.contiguous())
                set_module_tensor_to_device(tr, mod + ".weight_scale", "cpu",
                                            value=h.get_tensor(mod + ".weight_scale"))
                set_module_tensor_to_device(tr, mod + ".weight_scale_2", "cpu",
                                            value=h.get_tensor(mod + ".weight_scale_2").float())
            elif mod in fp8:
                set_module_tensor_to_device(tr, k, "cpu", value=t.contiguous())
                set_module_tensor_to_device(tr, mod + ".weight_scale", "cpu",
                                            value=h.get_tensor(mod + ".weight_scale").float())
            else:
                set_module_tensor_to_device(tr, k, "cpu", value=t.to(torch.bfloat16),
                                            dtype=torch.bfloat16)
            assigned += 1
    if skipped:
        _log(f"⚠ {skipped} tensori del file non hanno un corrispettivo nel modello "
             f"(formato del repack diverso da quello atteso?)")
    _log(f"transformer costruito ({assigned} tensori assegnati)")
    return tr


# ── generazione ───────────────────────────────────────────────────────────────
def generate(params: dict) -> str:
    import torch
    from diffusers import ModularPipeline

    w = resolve_weights(params)
    _log(f"modalità {w['mode']} · checkpoint {os.path.basename(w['ckpt'])}")

    # COORDINAMENTO VRAM: il TE Qwen3-VL (15.7GB) e il transformer (12.5-21GB) non stanno
    # insieme in 32GB. Encode del prompt PRIMA, poi il TE viene parcheggiato in RAM e si carica
    # il transformer — è il meccanismo di handoff già validato per LTX.
    try:
        from core import mem_coordinator as mc
        mc.free_and_wait("video", need_mb=int(float(params.get("need_mb") or 20000)))
    except Exception as e:                                   # noqa: BLE001
        _log(f"coordinamento VRAM non disponibile ({e}) — proseguo")

    sub = "FL2VA" if w["mode"] == "fl2va" else "Ref2VA"
    pipe = ModularPipeline.from_pretrained(os.path.join(w["config_dir"], sub))
    pipe.load_components(names=["text_encoder", "tokenizer", "processor", "scheduler",
                                "audio_scheduler"])
    pipe.update_components(transformer=load_transformer(w["ckpt"], w["config_dir"], w["mode"]))

    kw = {
        "prompt": params.get("prompt", ""),
        "negative_prompt": params.get("negative") or None,
        "height": int(params.get("height", 768)),
        "width": int(params.get("width", 1344)),
        "num_frames": int(params.get("num_frames", 121)),
        "num_inference_steps": int(params.get("steps", 20)),
        "guidance_scale": float(params.get("guidance", 4.0)),
        "generator": torch.Generator("cpu").manual_seed(int(params.get("seed", 0))),
    }
    # RIFERIMENTI: liste separate per tipo, nell'ordine dei token <Picture N>/<Video N>/<Audio N>
    refs = params.get("references") or []
    if refs:
        kw["reference_images"] = [r["path"] for r in refs if r["kind"] == "image"] or None
        kw["reference_videos"] = [r["path"] for r in refs if r["kind"] == "video"] or None
        kw["reference_audios"] = [r["path"] for r in refs if r["kind"] == "audio"] or None
        _log(f"riferimenti: {sum(1 for r in refs if r['kind'] == 'image')} immagini, "
             f"{sum(1 for r in refs if r['kind'] == 'video')} video, "
             f"{sum(1 for r in refs if r['kind'] == 'audio')} audio")
    elif params.get("image_path"):                           # i2v / primo fotogramma
        kw["image"] = params["image_path"]
    if params.get("last_image_path"):                        # FL2V: ultimo fotogramma
        kw["last_image"] = params["last_image_path"]
    kw = {k: v for k, v in kw.items() if v is not None}

    _log(f"genero {kw['width']}x{kw['height']} · {kw['num_frames']} frame · "
         f"{kw['num_inference_steps']} step")
    out = pipe(**kw)
    return save_output(out, params)


def save_output(out, params: dict) -> str:
    """Salva video+audio in un mp4 unico (l'audio H3 è già sincronizzato: nessun allineamento)."""
    from diffusers.utils import export_to_video
    dest = params.get("output") or str(REPO_ROOT / "output" / "minimax_h3.mp4")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    frames = out.frames[0] if hasattr(out, "frames") else out["frames"][0]
    audio = getattr(out, "audios", None) or (out.get("audios") if isinstance(out, dict) else None)
    fps = int(params.get("fps", 24))
    if audio is None:
        export_to_video(frames, dest, fps=fps)
        _log("nessuna traccia audio nell'output (modalità senza audio?)")
        return dest
    import tempfile
    import subprocess
    import soundfile as sf
    tmp_v = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    tmp_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    export_to_video(frames, tmp_v, fps=fps)
    wav = audio[0] if isinstance(audio, (list, tuple)) else audio
    sf.write(tmp_a, wav.T if hasattr(wav, "T") else wav, 32000)      # 32 kHz stereo nativo
    subprocess.run(["ffmpeg", "-y", "-i", tmp_v, "-i", tmp_a, "-c:v", "copy", "-c:a", "aac",
                    "-shortest", dest], capture_output=True)
    for t in (tmp_v, tmp_a):
        try:
            os.unlink(t)
        except OSError:
            pass
    return dest


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("uso: minimax_h3_infer.py params.json", file=sys.stderr)
        return 2
    params = json.loads(Path(argv[1]).read_text("utf-8"))
    try:
        dest = generate(params)
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        return 1
    if not os.path.exists(dest):
        print(f"[h3] output non creato: {dest}", file=sys.stderr)
        return 1
    print(f"OK:{dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
