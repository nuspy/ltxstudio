"""Auto-captioning di un dataset di immagini per il training LoRA (router multi-motore).

Genera, per ogni immagine, un file .txt omonimo con la caption. Due motori:
- **florence2** (default legacy): Florence-2-large `<DETAILED_CAPTION>` — leggero (~1.5GB), descrizioni generiche,
  ⚠️ tende a evitare/annacquare i contenuti espliciti.
- **joycaption**: JoyCaption Beta One (siglip2 + Llama-3.1-8B, LLaVA) — nato per dati di training diffusion,
  NON censurato, modalità multiple (descrittiva / tag booru / prompt SD). Pesi in E:\\AI\\Models\\JoyCaption
  (nf4). ⚠️ SCAFFOLD: percorso JoyCaption da VALIDARE live (GPU occupata) — vedi [VALIDA].

Uso:
  python scripts/caption_dataset.py --dir <cartella> [--engine florence2|joycaption]
        [--mode descriptive|tags|sd-prompt] [--trigger ohwx] [--overwrite]
Il motore joycaption gira nel venv15 (transformers + bitsandbytes); florence2 nel venv di training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
FLORENCE_ID = "multimodalart/Florence-2-large-no-flash-attn"
JOYCAPTION_DIR = r"E:\AI\Models\JoyCaption"

# istruzione JoyCaption per modalità (prompt utente nel chat template)
_JOY_PROMPTS = {
    "descriptive": "Write a descriptive caption for this image in a formal tone.",
    "tags": "Write a list of Booru tags for this image.",
    "sd-prompt": "Write a stable diffusion prompt for this image.",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _todo(folder: Path, ext: str, overwrite: bool):
    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
    todo = [img for img in images if overwrite or not img.with_suffix("." + ext).exists()]
    return images, todo


# ── Florence-2 ────────────────────────────────────────────────────────────────
def caption_florence2(todo, trigger, ext):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"[caption] carico {FLORENCE_ID} su {device}…")
    model = AutoModelForCausalLM.from_pretrained(FLORENCE_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
    proc = AutoProcessor.from_pretrained(FLORENCE_ID, trust_remote_code=True)
    prompt = "<DETAILED_CAPTION>"
    done = 0
    for i, img in enumerate(todo, 1):
        try:
            image = Image.open(img).convert("RGB")
            inputs = proc(text=prompt, images=image, return_tensors="pt").to(device, dtype)
            ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 max_new_tokens=256, num_beams=3, do_sample=False)
            text = proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = proc.post_process_generation(text, task=prompt, image_size=(image.width, image.height))
            cap = parsed.get(prompt, "").replace("The image shows ", "").strip()
            _write(img, cap, trigger, ext); done += 1
            log(f"[caption] {i}/{len(todo)} {img.name}: {cap[:80]}")
        except Exception as e:                               # noqa: BLE001
            log(f"[caption] {i}/{len(todo)} {img.name}: ERRORE {e}")
    return done


# ── JoyCaption (SCAFFOLD — da validare live) ──────────────────────────────────
def caption_joycaption(todo, trigger, ext, mode, model_dir):
    import torch
    from PIL import Image
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[caption] carico JoyCaption (nf4) da {model_dir} su {device}…")
    # [VALIDA] nf4 pre-quantizzato: quantization_config nel config del modello (bitsandbytes 4-bit)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = LlavaForConditionalGeneration.from_pretrained(model_dir, device_map=device)
    model.eval()
    user_prompt = _JOY_PROMPTS.get(mode, _JOY_PROMPTS["descriptive"])
    done = 0
    for i, img in enumerate(todo, 1):
        try:
            image = Image.open(img).convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": user_prompt},
            ]
            convo_string = proc.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[convo_string], images=[image], return_tensors="pt").to(device)
            if "pixel_values" in inputs:                     # [VALIDA] dtype pixel_values per nf4
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=300, do_sample=True,
                                     temperature=0.6, top_p=0.9, use_cache=True)
            gen = gen[:, inputs["input_ids"].shape[1]:]
            cap = proc.tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            _write(img, cap, trigger, ext); done += 1
            log(f"[caption] {i}/{len(todo)} {img.name}: {cap[:80]}")
        except Exception as e:                               # noqa: BLE001
            log(f"[caption] {i}/{len(todo)} {img.name}: ERRORE {e}")
    return done


def _write(img: Path, caption: str, trigger: str, ext: str):
    if trigger:
        caption = f"{trigger}, {caption}"
    img.with_suffix("." + ext).write_text(caption, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ext", default="txt")
    ap.add_argument("--engine", default="florence2", choices=["florence2", "joycaption"])
    ap.add_argument("--mode", default="descriptive", choices=["descriptive", "tags", "sd-prompt"])
    ap.add_argument("--joycaption-dir", dest="joycaption_dir", default=JOYCAPTION_DIR)
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        log(f"[caption] ERRORE: cartella inesistente: {folder}"); return 2
    images, todo = _todo(folder, args.ext, args.overwrite)
    if not images:
        log("[caption] ERRORE: nessuna immagine nella cartella."); return 2
    log(f"[caption] {len(todo)}/{len(images)} immagini da descrivere (motore: {args.engine}).")
    if not todo:
        log("[caption] niente da fare (usa --overwrite per rigenerare)."); return 0

    trig = (args.trigger or "").strip()
    try:
        if args.engine == "joycaption":
            done = caption_joycaption(todo, trig, args.ext, args.mode, args.joycaption_dir)
        else:
            done = caption_florence2(todo, trig, args.ext)
    except Exception as e:                                   # noqa: BLE001
        log(f"[caption] ERRORE motore {args.engine}: {e}"); return 3
    log(f"[caption] FINE: {done}/{len(todo)} caption scritte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
