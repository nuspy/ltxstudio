"""Wrapper training LoRA ACE-Step (venv ace-step): converte il dataset in formato HF poi lancia il trainer.

Sequenza:
  1) convert2hf_dataset.py  (data_dir con name.ext + name_prompt.txt + name_lyrics.txt → dataset HF)
  2) trainer.py             (allena la LoRA sul dataset)
L'output live dei due processi figli fluisce nello stdout di questo wrapper (catturato dalla GUI).
"""
import argparse, os, subprocess, sys
from pathlib import Path

ACE_DIR = Path(r"E:\AI\Tools\ace-step\ACE-Step")
PY = sys.executable


def _run(cmd, cwd):
    print(f"\n[train] $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    if r.returncode != 0:
        print(f"[train] passo fallito (codice {r.returncode})", flush=True)
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", dest="data_dir", required=True)
    ap.add_argument("--dataset-path", dest="dataset_path", required=True)   # dove salvare il dataset HF
    ap.add_argument("--repeat", type=int, default=2000)
    ap.add_argument("--exp-name", dest="exp_name", default="music_lora")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=2000)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lora-config", dest="lora_config", default=str(ACE_DIR / "config" / "zh_rap_lora_config.json"))
    ap.add_argument("--logger-dir", dest="logger_dir", required=True)
    ap.add_argument("--every-n", dest="every_n", type=int, default=500)
    a = ap.parse_args()

    # 1) conversione dataset → HF
    _run([PY, "convert2hf_dataset.py", "--data_dir", a.data_dir,
          "--repeat_count", a.repeat, "--output_name", a.dataset_path], ACE_DIR)

    # 2) training LoRA
    Path(a.logger_dir).mkdir(parents=True, exist_ok=True)
    _run([PY, "trainer.py",
          "--dataset_path", a.dataset_path, "--exp_name", a.exp_name,
          "--learning_rate", a.lr, "--max_steps", a.max_steps,
          "--precision", a.precision, "--accumulate_grad_batches", a.batch,
          "--lora_config_path", a.lora_config, "--logger_dir", a.logger_dir,
          "--every_n_train_steps", a.every_n, "--devices", 1], ACE_DIR)
    print(f"OK:{a.logger_dir}", flush=True)


if __name__ == "__main__":
    main()
