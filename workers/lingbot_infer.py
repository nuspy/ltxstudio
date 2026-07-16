"""Worker inference LingBot-Video — 2ª architettura video (accanto a LTX-2).

Eseguito nel venv dedicato `E:\\AI\\Tools\\lingbot_venv311` (Python 3.11 + torch 2.11 cu130 + FlashAttention-2
sm120) via subprocess da `core/ltx_video_job.py`.

RICETTA VALIDATA (17/07, vedi memoria [[lingbot-video-2nd-arch]]): LingBot genera correttamente SOLO col
**runner ufficiale** + **--batch_cfg** (ramo attention "packed" che usa flash_attn_varlen) + **caption JSON
RICCA** (la prosa dà garbage) + VAE fp32. La `pipe()` cruda (path non-packed) NON funziona. Perciò questo
worker DELEGA al runner ufficiale `lingbot_video.runner`, poi applica stego (#107) + watermark (#106).

Requisiti d'ambiente (creati da scripts/install_lingbot.py):
  - flash-attn 2.x (wheel sm120) nel venv
  - patch in lingbot_video/transformer_lingbot_video.py: flash_attn_varlen_func da `flash_attn` (FA2)
    invece che da `flash_attn_interface` (FA3/Hopper, non su Blackwell)

Protocollo (identico a workers/ltx2_infer.py): legge params JSON (argv[1]); `output` = mp4; stampa `OK:<path>`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root repo (import core.*)

NEG_DEFAULT = ("worst quality, inconsistent motion, blurry, jittery, distorted, "
               "low quality, deformed, static")


# ── risoluzione modello / variante ───────────────────────────────────────────
def _resolve_model_dir(p: dict) -> tuple[str, str]:
    """Ritorna (model_dir, variant). variant ∈ {dense, moe}."""
    variant = str(p.get("lingbot_variant", "dense")).lower()
    m = p.get("model") or ""
    if m and os.path.isdir(m):
        low = m.lower()
        if "moe" in low or "30b" in low:
            variant = "moe"
        elif "dense" in low or "1.3b" in low:
            variant = "dense"
        return m, variant
    try:
        from core import weights_catalog as wc
        entries = {e["id"]: e for e in wc.load()}
        order = (["lingbot_moe_uint4", "lingbot_moe"] if variant == "moe"
                 else ["lingbot_dense", "lingbot_dense_fp8"])
        for eid in order:
            e = entries.get(eid)
            if e and wc.is_present(e):
                return str(wc.dest_for(e)), variant
        e = entries.get(order[0])
        if e:
            return str(wc.dest_for(e)), variant
    except Exception as ex:                                   # noqa: BLE001
        print(f"[lingbot] risoluzione catalogo fallita: {ex}", flush=True)
    raise RuntimeError(
        "Modello LingBot non trovato su disco. Scaricalo dal wizard 'Pesi' (modulo Video).")


# ── caption JSON ricca (obbligatoria: la prosa dà garbage) ───────────────────
def _rich_caption(p: dict) -> dict:
    """Ritorna il dict caption per il runner (--prompt_json).

    Se i params contengono già `prompt_json` (prodotto a monte dal rewriter/composer nel processo GUI),
    lo si usa. Altrimenti si incapsula il prompt testuale nello schema minimo VALIDO (meglio della prosa
    nuda, che il DiT rifiuta) — il composer ricco vero sta in core/prompt_struct.to_lingbot_json."""
    pj = p.get("prompt_json")
    if pj:
        if isinstance(pj, str):
            try:
                return json.loads(pj)
            except Exception:                                # noqa: BLE001
                pass
        elif isinstance(pj, dict):
            return pj
    prompt = (p.get("prompt") or "").strip()
    dur = max(1.0, p.get("num_frames", 121) / float(p.get("fps", 24)))
    return {
        "caption": {
            "comprehensive_description": {
                "scene_content_description": prompt,
                "camera_info": {},
                "prominent_elements": [],
                "world_knowledge": [],
            }
        },
        "duration": round(dur, 2),
    }


def _mode(p: dict) -> str:
    if int(p.get("num_frames", 121)) <= 1:
        return "t2i"
    if p.get("image_path") and os.path.exists(str(p.get("image_path"))):
        return "ti2v"
    return "t2v"


def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model_dir, variant = _resolve_model_dir(p)
    mode = _mode(p)
    # scrivi la caption JSON ricca in un file temporaneo
    cap = _rich_caption(p)
    cf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(cap, cf, ensure_ascii=False)
    cf.close()

    steps = int(p.get("steps", 40))
    args = [
        sys.executable, "-m", "lingbot_video.runner",
        "--backend", "diffusers", "--model_dir", model_dir, "--mode", mode,
        "--prompt_json", cf.name, "--output", out,
        "--height", str(int(p.get("height", 480))), "--width", str(int(p.get("width", 832))),
        "--num_frames", str(int(p.get("num_frames", 121))),
        "--steps", str(steps), "--guidance_scale", str(float(p.get("guidance", 3.0))),
        "--shift", str(float(p.get("shift", 3.0))), "--seed", str(int(p.get("seed", 42))),
        "--fps", str(int(p.get("fps", 24))),
        "--transformer_dtype", "bf16", "--text_encoder_dtype", "bf16", "--vae_dtype", "fp32",
        "--batch_cfg",                                        # OBBLIGATORIO (ramo packed = corretto)
    ]
    if mode == "ti2v":
        args += ["--image", str(p["image_path"])]
    if variant == "moe" and p.get("run_refiner"):
        args += ["--run_refiner", "--refiner_steps", str(int(p.get("refiner_steps", 8)))]

    env = dict(os.environ)
    env["LINGBOT_QWEN_ATTN_IMPLEMENTATION"] = "sdpa"          # text-encoder VLM: evita il probe FA3
    env["LINGBOT_QUIET_PROGRESS"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    print(f"[lingbot] runner ufficiale: mode={mode} variant={variant} "
          f"{p.get('width')}x{p.get('height')} {p.get('num_frames')}f steps={steps}", flush=True)
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    try:
        os.remove(cf.name)
    except OSError:
        pass
    if not os.path.exists(out):
        tail = ((r.stdout or "")[-800:] + (r.stderr or "")[-800:]).strip()
        raise RuntimeError(f"runner LingBot non ha prodotto l'output:\n{tail}")

    # #107 firma invisibile + #106 watermark sul file finale
    try:
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                  # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                  # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
