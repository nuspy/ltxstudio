"""SAM 3.1 (Meta) VIDEO con tracker Object Multiplex: testo sul fotogramma 0 -> maschere per OGNI oggetto e per ogni fotogramma. Gira SOLO tramite il
lanciatore E:\\AI\\Tools\\sam3_run.sh (diffusers_next_venv, -P -s, PYTHONPATH su sam3_src revisionato e sam3_deps). Inferenza senza rete: il checkpoint e locale.
Il checkpoint si legge con torch.load(weights_only=True) o con safetensors (copia Comfy-Org, fp16 -> float32) tramite il patch di torch.load qui sotto: nel codice di
Meta resta un torch.load senza weights_only in sam3_multiplex_base.py (non raggiunto da questo percorso e da non usare mai su file di terzi).

  bash E:/AI/Tools/sam3_run.sh sam31_video_text.py --video clip.mp4 --text person --out-dir DIR [--checkpoint FILE] [--max-frames N] [--device cpu|cuda]

Scrive DIR/obj_<id>/<fotogramma:05d>.png (0/255) e stampa `SAM31V:{json}` con id degli oggetti, numero di fotogrammi, tempi e il device.
Copia di lavoro: E:\\AI\\Tools\\sam3_jobs (la fonte nel repo e workers/sam3_jobs).
"""
import argparse
import json
import os
import sys
import time

DEFAULT_CKPT = r"E:\AI\Models\Segmentation\sam3.1\sam3.1_multiplex.pt"


def patch_torch_load():
    """torch.load(<file .safetensors>) -> dizionario float32; ogni altra chiamata passa all'originale (gia con weights_only=True nel codice di Meta)."""
    import torch
    orig = torch.load

    def _load(f, *args, **kw):
        if isinstance(f, (str, os.PathLike)) and str(f).lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            return {k: (v.float() if v.is_floating_point() else v) for k, v in load_file(str(f)).items()}
        return orig(f, *args, **kw)
    torch.load = _load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file video o cartella di fotogrammi JPEG")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tutti")
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    patch_torch_load()
    t0 = time.time()
    # use_fa3=False: niente kernel FlashAttention 3 (Windows / CPU); use_rope_real=False: RoPE complessa, come nel checkpoint
    pred = build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint, use_fa3=False, use_rope_real=False, compile=False, warm_up=False,
                                                async_loading_frames=False)
    if dev == "cpu":
        pred.model = pred.model.cpu()
    t_load = time.time() - t0
    sid = pred.start_session(a.video, offload_video_to_cpu=(dev == "cpu"), offload_state_to_cpu=(dev == "cpu"))["session_id"]
    r = pred.add_prompt(session_id=sid, frame_idx=0, text=a.text.strip().rstrip("."))
    os.makedirs(a.out_dir, exist_ok=True)
    seen, n = {}, 0

    def dump(frame_index, outputs):
        ids = [int(i) for i in outputs["out_obj_ids"]]
        for oid, m in zip(ids, outputs["out_binary_masks"]):
            d = os.path.join(a.out_dir, f"obj_{oid}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(np.asarray(m).astype("uint8") * 255).save(os.path.join(d, f"{frame_index:05d}.png"))
            seen[oid] = seen.get(oid, 0) + 1
    dump(r["frame_index"], r["outputs"])
    for item in pred.propagate_in_video(session_id=sid, propagation_direction="forward", start_frame_idx=0,
                                        max_frame_num_to_track=(a.max_frames or None)):
        dump(item["frame_index"], item["outputs"])
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    pred.close_session(session_id=sid) if hasattr(pred, "close_session") else None
    print("SAM31V:" + json.dumps({"objects": sorted(seen), "frames_per_object": seen, "frames": n + 1, "device": dev,
                                 "t_load_s": round(t_load, 1), "t_total_s": round(time.time() - t0, 1), "out_dir": a.out_dir}), flush=True)


if __name__ == "__main__":
    main()
