"""Download HuggingFace robusto con RESUME automatico.

snapshot_download è resumable (riprende dai file .incomplete in .cache/.../download/).
Qui lo avvolgiamo in un loop di retry: se la connessione cade, riprova all'infinito con
backoff, riprendendo da dove era. Pensato per reti instabili / file grandi (TE Mistral).

Uso:
  python scripts/resume_download.py <repo_id> <local_dir> [allow_patterns_csv]
Es:
  python scripts/resume_download.py baidu/ERNIE-Image E:\\AI\\Models\\diffusers\\Ernie
  python scripts/resume_download.py black-forest-labs/FLUX.2-dev E:\\AI\\Models\\diffusers\\flux2 "text_encoder/*,tokenizer/*,vae/config.json,scheduler/*,model_index.json"
"""
from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

MAX_ATTEMPTS = 1000          # praticamente infinito; si esce al completamento
BACKOFF_S = 15


def main() -> int:
    if len(sys.argv) < 3:
        print("uso: python scripts/resume_download.py <repo_id> <local_dir> [allow_patterns_csv]")
        return 2
    repo = sys.argv[1]
    local = sys.argv[2]
    patterns = None
    if len(sys.argv) > 3 and sys.argv[3].strip():
        patterns = [p.strip() for p in sys.argv[3].split(",") if p.strip()]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[resume] tentativo {attempt} → {repo}", flush=True)
            snapshot_download(repo, local_dir=local, allow_patterns=patterns,
                              max_workers=4, etag_timeout=60)
            print(f"[resume] ✔ COMPLETO {repo}", flush=True)
            return 0
        except KeyboardInterrupt:
            print("[resume] interrotto manualmente", flush=True)
            return 1
        except Exception as e:                          # noqa: BLE001
            print(f"[resume] caduto ({type(e).__name__}: {str(e)[:140]}) → riprendo tra {BACKOFF_S}s", flush=True)
            time.sleep(BACKOFF_S)
    print("[resume] esauriti i tentativi", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
