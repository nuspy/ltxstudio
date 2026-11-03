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

if any(a == "cpu" for a in sys.argv[1:]) and "--device" in sys.argv:        # --device cpu: nessuna GPU, nemmeno un contesto CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def patch_fp32_cpu():
    """Su CPU il codice di Meta forza matmul bfloat16 (perflib.fused.addmm_act: kernel pensati per GPU, sotto autocast bf16) e con pesi float32 va in errore di dtype.
    Qui si sostituisce SOLO quella funzione con la versione float32 equivalente (linear + attivazione): stessi numeri, niente bf16 lento sulla CPU."""
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def addmm_act(activation, linear, mat1):
        y = linear(mat1)
        if isinstance(activation, type) and issubclass(activation, nn.Module):
            return activation()(y)
        if activation is F.relu:
            return F.relu(y)
        if activation is F.gelu:
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.startswith("sam3") and getattr(mod, "addmm_act", None) is not None:
            mod.addmm_act = addmm_act
            n += 1
    # Il codice di Meta scrive device="cuda" fisso in molti punti (decoder, posizioni, ecc.): in modalita CPU ogni richiesta di device cuda diventa cpu.
    def _cpu_kw(fn):
        def w(*a, **k):
            d = k.get("device")
            if d is not None and str(d).startswith("cuda"):
                k["device"] = "cpu"
            if k.get("dtype") in (torch.bfloat16, torch.float16):          # CPU: tutto in float32 (il codice di Meta presume l'autocast bf16 della GPU)
                k["dtype"] = torch.float32
            return fn(*a, **k)
        return w
    for nm in ("arange", "zeros", "ones", "empty", "full", "tensor", "as_tensor", "linspace", "rand", "randn", "zeros_like", "ones_like", "eye", "randperm", "empty_like"):
        if hasattr(torch, nm):
            setattr(torch, nm, _cpu_kw(getattr(torch, nm)))
    import types
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(major=0, minor=0, name="cpu", total_memory=0, multi_processor_count=1)   # lettura al momento dell'import
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    torch.Tensor.bfloat16 = lambda self, *a, **k: self.float()
    torch.Tensor.half = lambda self, *a, **k: self.float()
    nn.Module.cuda = lambda self, *a, **k: self
    _to = torch.Tensor.to

    def _tensor_to(self, *a, **k):
        a = tuple("cpu" if isinstance(x, (str, torch.device)) and str(x).startswith("cuda") else (torch.float32 if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16) else x) for x in a)
        if k.get("dtype") in (torch.bfloat16, torch.float16):
            k["dtype"] = torch.float32
        if str(k.get("device", "")).startswith("cuda"):
            k["device"] = "cpu"
        return _to(self, *a, **k)
    torch.Tensor.to = _tensor_to
    # La cache delle posizioni del builder viene precalcolata con device="cuda" scritto fisso (position_encoding.py:55): su CPU si rimanda al primo uso.
    import sam3.model_builder as mb
    from sam3.model.position_encoding import PositionEmbeddingSine
    mb._create_position_encoding = lambda precompute_resolution=None: PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000,
                                                                                            precompute_resolution=None)
    return n


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
    if dev == "cpu":
        print(f"[sam31] CPU: addmm_act float32 in {patch_fp32_cpu()} moduli", file=sys.stderr, flush=True)
    t0 = time.time()
    # use_fa3=False: niente kernel FlashAttention 3 (Windows / CPU); use_rope_real=False: RoPE complessa, come nel checkpoint
    pred = build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint, use_fa3=False, use_rope_real=False, compile=False, warm_up=False,
                                                async_loading_frames=False)
    if dev == "cpu":
        pred.model = pred.model.cpu()
    t_load = time.time() - t0
    # Incongruenza del codice di Meta: Sam3BasePredictor.start_session passa offload_state_to_cpu / video_loader_type, ma init_state del tracker multiplex non li accetta.
    _init = pred.model.init_state

    def _init_state(**kw):
        kw.pop("offload_state_to_cpu", None)
        if kw.pop("video_loader_type", "cv2") == "cv2":
            kw["use_cv2"] = True
        return _init(**kw)
    pred.model.init_state = _init_state
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
