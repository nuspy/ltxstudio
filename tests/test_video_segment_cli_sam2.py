"""Ramo SAM2 VERO del tracking video, lanciato come lo lancia l'app (`python video_segment.py <video> <testo> <guide> <dilate>` nel venv dell'inference
server), su CPU con un video minuscolo: GroundingDINO trova l'oggetto, SAM2 lo traccia, escono le maschere e il guide. Giro C del 03/11: la CLI moriva con
ModuleNotFoundError ('core') e i test con funzioni finte non lo vedevano. Serve il venv, i pesi SAM2 + GroundingDINO e ffmpeg: altrimenti si salta.
Nessuna GPU: LTX_SEG_DEVICE=cpu e CUDA nascosto da un preambolo (torch non crea contesti CUDA)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

VENV_PY = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")
SEG = Path(__file__).resolve().parents[1] / "inference_server" / "backends" / "video_segment.py"
SCENE = Path(r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\fixtures\scena_1152x832.png")
SEG_ROOT = Path(r"E:\AI\Models\Segmentation")

PREAMBLE = (
    "import os, sys, runpy; os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['TORCHDYNAMO_DISABLE'] = '1';"
    "import torch; import torch.utils._triton as t; t.has_triton = lambda: False;"
    "torch.cuda.get_device_capability = lambda *a, **k: (0, 0); torch.cuda.is_available = lambda: False;"
    "sys.argv = ['video_segment.py'] + sys.argv[1:]; runpy.run_path(r'%s', run_name='__main__')" % SEG)


def _have_weights():
    try:
        sys.path.insert(0, str(SEG.parents[2]))
        from core import model_paths as mp
        return bool(mp.path_of("sam2-hiera-large")) and bool(mp.path_of("grounding-dino-base"))
    except Exception:                                    # noqa: BLE001
        return (SEG_ROOT / "hf" / "sam2-hiera-large").is_dir() and (SEG_ROOT / "hf" / "grounding-dino-base").is_dir()


@pytest.mark.skipif(not VENV_PY.exists() or not SCENE.exists() or not _have_weights(), reason="servono il venv dell'inference server, i pesi SAM2/DINO e la scena di prova")
def test_cli_tracking_runs_the_real_sam2_branch_on_cpu(tmp_path):
    from PIL import Image
    frames = tmp_path / "f"
    frames.mkdir()
    im = Image.open(SCENE).convert("RGB")
    for i in range(4):                                                   # 4 frame 256x192: l'astronauta e a sinistra, la camera si sposta di poco
        im.crop((i * 4, 0, i * 4 + 576, 416)).resize((256, 184)).save(frames / f"{i:03d}.png")
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "4", "-i", str(frames / "%03d.png"), "-pix_fmt", "yuv420p", str(video)], check=True)
    guide = tmp_path / "guide.mp4"
    env = dict(os.environ, LTX_SEG_DEVICE="cpu", PYTHONIOENCODING="utf-8")
    r = subprocess.run([str(VENV_PY), "-c", PREAMBLE, str(video), "astronaut", str(guide), "0"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=str(tmp_path), timeout=900)
    assert "GUIDE:" in r.stdout, r.stdout[-600:] + r.stderr[-1500:]
    masks = sorted((tmp_path / "v_track" / "masks").glob("*.png"))
    assert len(masks) == 4, masks
    import numpy as np
    m = np.asarray(Image.open(masks[0])) > 127
    assert 0.02 < m.mean() < 0.9, f"maschera vuota o enorme: {m.mean():.3f}"       # un oggetto, non il nulla ne tutta la scena
    assert guide.is_file() and guide.stat().st_size > 1000
