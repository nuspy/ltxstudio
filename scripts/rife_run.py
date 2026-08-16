"""Runner RIFE nel venv inference_server (pattern faceswap_run.py) — FUORI dal processo GUI/API.

Perché: core/rife girava IN-PROCESS nel processo app → contesto CUDA + modello in _CACHE mai
liberati nella GUI (violazione della regola no-CUDA-in-app). Qui si esegue nello stesso venv che
già serve SAM2/face-swap; il processo esce a fine job e la VRAM torna all'OS.

Uso:  python scripts/rife_run.py <video_in> <video_out> [factor] [target_fps]
Output: righe "PROG:<i>/<n>" per il progress del chiamante, poi "OK:<path>" a successo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RIFE_INPROC"] = "1"                        # nel runner il ramo in-process È quello giusto


def main() -> int:
    if len(sys.argv) < 3:
        print("uso: rife_run.py <video_in> <video_out> [factor] [target_fps]", flush=True)
        return 2
    inp, outp = sys.argv[1], sys.argv[2]
    factor = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    tfps = float(sys.argv[4]) if len(sys.argv) > 4 and float(sys.argv[4]) > 0 else None
    from core import rife
    out = rife.interpolate_video(inp, outp, factor=factor, target_fps=tfps,
                                 progress=lambda i, n: print(f"PROG:{i}/{n}", flush=True))
    print("OK:" + out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
