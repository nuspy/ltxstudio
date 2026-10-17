"""Sonda eseguita nel venv LTX (ltx_core/ltx_pipelines disponibili, niente pytest): verifica la LoRA Gemma abliterata
senza GPU e senza caricare i pesi veri. Stampa un JSON {check: bool|str}. Lanciata da tests/test_gemma_lora.py.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
res = {}
try:
    import torch
    from safetensors.torch import save_file
    from ltx_core.loader import fuse_loras
    from ltx_core.loader.primitives import StateDict
    from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
    from core import gemma_lora as gl

    fuse_loras._get_device = lambda: torch.device("cpu")           # niente CUDA nel test

    torch.manual_seed(0)
    r, H, V = 4, 16, 40
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "lora.safetensors")
    P = "text_encoders.transformer."
    t = {}
    for k, (o, i) in {"model.layers.0.mlp.down_proj": (H, 32), "model.layers.1.self_attn.q_proj": (24, H),
                      "model.embed_tokens": (V, H), "vision_model.encoder.layers.0.mlp.fc1": (8, 8)}.items():
        t[f"{P}{k}.lora_down.weight"] = torch.randn(r, i, dtype=torch.bfloat16)
        t[f"{P}{k}.lora_up.weight"] = torch.randn(o, r, dtype=torch.bfloat16)
    save_file(t, p)
    sd = SafetensorsStateDictLoader().load([p], gl.lora_sd_ops(), torch.device("cpu"))
    keys = set(sd.sd)
    want = {"model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
            "model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
            "model.model.language_model.layers.1.self_attn.q_proj.lora_A.weight",
            "model.model.language_model.layers.1.self_attn.q_proj.lora_B.weight",
            "model.model.language_model.embed_tokens.lora_A.weight",
            "model.model.language_model.embed_tokens.lora_B.weight",
            "model.lm_head.lora_A.weight", "model.lm_head.lora_B.weight"}
    res["mappa_chiavi"] = keys == want or sorted(keys ^ want)
    res["vision_scartato"] = not any("vision" in k for k in keys)

    # fusione: W + B@A su tutti i pesi toccati (anche lm_head, copia dell'embedding)
    base = {"model.model.language_model.layers.0.mlp.down_proj.weight": torch.randn(H, 32, dtype=torch.bfloat16),
            "model.model.language_model.layers.1.self_attn.q_proj.weight": torch.randn(24, H, dtype=torch.bfloat16),
            "model.model.language_model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16)}
    base["model.lm_head.weight"] = base["model.model.language_model.embed_tokens.weight"].clone()
    from ltx_core.loader.primitives import LoraStateDictWithStrength
    msd = StateDict({k: v.clone() for k, v in base.items()}, torch.device("cpu"), 0, {torch.bfloat16})
    out = dict(fuse_loras.fuse_lora_weights(msd, [LoraStateDictWithStrength(sd, 1.0)]))
    ok = set(out) == set(base)
    for k, w in base.items():
        pre = k[:-len(".weight")]
        delta = (sd.sd[pre + ".lora_B.weight"].float() @ sd.sd[pre + ".lora_A.weight"].float())
        ok = ok and torch.allclose(out[k].float(), w.float() + delta, atol=0.25, rtol=0.02) \
            and not torch.equal(out[k], w)
    res["fusione_W_piu_BA"] = bool(ok)

    # attach sul PromptEncoder reale (solo header del Gemma, nessun peso caricato)
    root = r"E:\AI\Models\TextEncoders\gemma-3-ltx"
    from core import model_paths as _mp
    ckpt = _mp.p("ltx23-dev-fp8", r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors")
    if os.path.isdir(root) and os.path.isfile(ckpt):
        import types
        from ltx_pipelines.utils.blocks import PromptEncoder
        from ltx_pipelines.utils.types import OffloadMode
        pe = PromptEncoder(checkpoint_path=ckpt, gemma_root=root, dtype=torch.bfloat16,
                           device=torch.device("cpu"), offload_mode=OffloadMode.CPU
                           if hasattr(OffloadMode, "CPU") else OffloadMode.NONE)
        pipe = types.SimpleNamespace(prompt_encoder=pe)
        n = gl.attach(pipe, p)
        n2 = gl.attach(pipe, p)                                     # idempotente
        b = pe._text_encoder_builder
        res["attach_n"] = n == 1 and n2 == 1
        res["builder_lora"] = len(b.loras) == 1 and b.loras[0].path == p
        sb = pe._streaming_text_encoder_builder
        res["streaming_lora"] = sb is None or len(sb.loras) == 1
        from core import emb_cache as ec
        frag = gl.key_fragment(b, ec.file_fp)
        res["key_fragment"] = bool(frag) and gl.key_fragment(types.SimpleNamespace(loras=()), ec.file_fp) is None
    else:
        res["attach_n"] = res["builder_lora"] = res["streaming_lora"] = res["key_fragment"] = "SKIP"
except Exception as e:                                              # noqa: BLE001
    import traceback
    res["errore"] = traceback.format_exc()[-1500:]
print("PROBE_JSON=" + json.dumps(res))
