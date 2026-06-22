"""Auto-captioning di un dataset di immagini per il training LoRA.

Genera, per ogni immagine, un file .txt omonimo con la descrizione (caption) prodotta da
**Florence-2-large** (stesso modello usato dalla UI di ai-toolkit): leggero (~1.5GB), no gating,
descrive il contenuto visivo in modo dettagliato — adatto ai dataset di training.

Gira nel VENV di training (ha già torch + transformers). Eseguito come subprocess dalla tab
«Allena LoRA»; stampa il progresso su stdout in modo parsabile.

Uso:
  python scripts/caption_dataset.py --dir <cartella> [--trigger ohwx] [--overwrite]

Note:
- Il modello si scarica da HuggingFace al primo uso (no token necessario).
- Con --trigger il token viene anteposto alla caption ("ohwx, a woman ...").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MODEL_ID = "multimodalart/Florence-2-large-no-flash-attn"   # stessa scelta di ai-toolkit UI


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="cartella del dataset")
    ap.add_argument("--trigger", default="", help="trigger word da anteporre")
    ap.add_argument("--overwrite", action="store_true", help="rigenera anche le caption esistenti")
    ap.add_argument("--ext", default="txt", help="estensione file caption")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        log(f"[caption] ERRORE: cartella inesistente: {folder}")
        return 2

    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
    if not images:
        log("[caption] ERRORE: nessuna immagine nella cartella.")
        return 2

    # filtra quelle già fatte (se non overwrite)
    todo = []
    for img in images:
        cap = img.with_suffix("." + args.ext)
        if cap.exists() and not args.overwrite:
            continue
        todo.append(img)
    log(f"[caption] {len(todo)}/{len(images)} immagini da descrivere (modello: Florence-2).")
    if not todo:
        log("[caption] niente da fare (usa --overwrite per rigenerare).")
        return 0

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as e:  # noqa: BLE001
        log(f"[caption] ERRORE import dipendenze ({e}). Installa: transformers, timm, einops, pillow.")
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"[caption] carico {MODEL_ID} su {device}…")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        log(f"[caption] ERRORE caricamento modello: {e}")
        return 3

    trigger = (args.trigger or "").strip()
    prompt = "<DETAILED_CAPTION>"
    done = 0
    for i, img in enumerate(todo, 1):
        try:
            image = Image.open(img).convert("RGB")
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
            ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=256, num_beams=3, do_sample=False)
            text = processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(text, task=prompt, image_size=(image.width, image.height))
            caption = parsed.get(prompt, "").replace("The image shows ", "").strip()
            if trigger:
                caption = f"{trigger}, {caption}"
            img.with_suffix("." + args.ext).write_text(caption, encoding="utf-8")
            done += 1
            log(f"[caption] {i}/{len(todo)} {img.name}: {caption[:80]}")
        except Exception as e:  # noqa: BLE001
            log(f"[caption] {i}/{len(todo)} {img.name}: ERRORE {e}")

    log(f"[caption] FINE: {done}/{len(todo)} caption scritte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
