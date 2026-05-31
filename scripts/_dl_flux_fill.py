import os
from huggingface_hub import hf_hub_download

# Regola utente: scaricare SEMPRE in E:\AI\inbox, poi lanciare ingest.py
dst = r"E:\AI\inbox"
os.makedirs(dst, exist_ok=True)
print("[DL] FluxUnchained Fill Hyp8 Q5_K_S -> inbox ...", flush=True)
p = hf_hub_download(repo_id="mhnakif/fluxunchained-fill-hyp8-gguf-k",
                    filename="fluxunchained-fill-hyp8-Q5_K_S.gguf",
                    local_dir=dst)
print("[OK]", p, flush=True)
print("=== FLUX FILL DOWNLOAD DONE ===", flush=True)
