"""Wrapper di training LoRA LTX-2 (gira nel venv ltx2, che ha PyYAML + ltx_trainer).
Carica un config template YAML, applica override "a punti" (es. lora.rank), scrive il config finale
e lancia `scripts/train.py <config>`. Al termine recupera la LoRA prodotta e la copia in --out_lora.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

TRAINER = r"E:\AI\Tools\LTX-2\packages\ltx-trainer"


def set_dotted(d: dict, dotted: str, value):
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--overrides", required=True)   # json {dotted_key: value}
    ap.add_argument("--out_lora", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.template, encoding="utf-8"))
    for k, v in json.load(open(args.overrides, encoding="utf-8")).items():
        set_dotted(cfg, k, v)
    out_dir = cfg.get("output_dir") or os.path.dirname(args.out_lora) or "."
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, "_train_config.yaml")
    yaml.safe_dump(cfg, open(final, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    print(f"[train] config → {final}", flush=True)

    cmd = [sys.executable, "-u", os.path.join(TRAINER, "scripts", "train.py"), final]
    r = subprocess.run(cmd, cwd=TRAINER)
    if r.returncode != 0:
        print(f"[train] train.py exit {r.returncode}", flush=True)

    # recupera l'ultima LoRA prodotta (.safetensors) sotto output_dir
    cands = sorted(glob.glob(os.path.join(out_dir, "**", "*.safetensors"), recursive=True),
                   key=os.path.getmtime)
    if cands:
        shutil.copyfile(cands[-1], args.out_lora)
        print(f"[train] LoRA → {args.out_lora} (da {cands[-1]})", flush=True)
    else:
        print("[train] nessuna LoRA .safetensors trovata in output_dir", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
