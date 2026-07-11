"""Runner face-swap eseguito nell'inference_server_venv (che ha insightface).

Il processo chiamante (app GUI / REST API su venv Comfy) NON ha insightface → core.faceswap delega qui.
FACESWAP_INPROC=1 evita la ri-delega ricorsiva (qui si esegue in-process).

Uso:  python scripts/faceswap_run.py <image|video> <target/video> <source_face> <out>
Stampa `OK:<path>` a fine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root repo
os.environ["FACESWAP_INPROC"] = "1"                     # esegui in-process, non ri-delegare


def main() -> None:
    mode, target, source_face, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    from core import faceswap
    if mode == "video":
        res = faceswap.swap_video(target, source_face, out)
    else:
        res = faceswap.swap_image(target, source_face, out)
    print("OK:" + str(res), flush=True)


if __name__ == "__main__":
    main()
