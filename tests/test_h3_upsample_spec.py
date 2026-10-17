"""H3 Video Upsampler: spec, catalogo pesi, API, worker (solo costruzione/parsing, nessun modello, nessuna GPU)."""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import h3_upsample_spec as U  # noqa: E402
from core import minimax_h3_spec as S  # noqa: E402
from core import native_convert as nc  # noqa: E402
from core import weights_catalog as wc  # noqa: E402


def test_pesi_dal_profilo_esplicito_e_ref2va():
    ids = U.weight_ids({"h3_profile": "int8"})
    assert ids["dit"] == "mmh3-ref2va-int8"                       # la LoRA di guida è addestrata sul Ref2VA
    assert ids["upscaler"] == U.UPSCALER_ID and ids["guide_lora"] == U.GUIDE_LORA_ID
    assert ids["te"] == S.PROFILES["int8"][2] and ids["vae"] == "mmh3-video-vae" and ids["audio_vae"] == "mmh3-audio-vae"
    assert "guide_lora" not in U.weight_ids({"h3_profile": "nvfp4", "guide": False})
    assert U.weight_ids({"h3_profile": "fp8", "h3_te": "mmh3-te-int8"})["te"] == "mmh3-te-int8"


def test_nessuna_scelta_automatica_del_modello():
    with pytest.raises(ValueError, match="scegli il modello"):
        U.weight_ids({})
    with pytest.raises(ValueError):
        U.weight_ids({"h3_profile": "bf16"})
    with pytest.raises(ValueError):
        U.weight_ids({"h3_profile": "int8", "h3_te": "mmh3-te-boh"})
    assert any("modello H3" in e for e in U.validate(U.build_params({"video": "x.mp4"})))


def test_build_params_default_del_nodo_studiato():
    p = U.build_params({"video": "a.mp4", "h3_profile": "nvfp4"})
    assert p["megapixels"] == 2.0 and p["denoise"] == 0.1 and p["guide"] is True
    assert p["lora_strength"] == 1.0 and p["tile_tokens"] == 70000
    p = U.build_params({"video": "a.mp4", "h3_profile": "nvfp4", "denoise": 0.0, "megapixels": 1.5})
    assert p["denoise"] == 0.0 and p["megapixels"] == 1.5          # 0 è un valore valido (solo ingrandimento)


def test_validate_intervalli():
    ok = U.build_params({"video": "a.mp4", "h3_profile": "nvfp4"})
    assert U.validate(ok) == []
    assert U.validate({**ok, "megapixels": 99}) and U.validate({**ok, "denoise": 1.5}) and U.validate({**ok, "tile_tokens": 10})
    assert U.validate({**ok, "video": ""})


def test_catalogo_arch_funzione_deps_e_conversione():
    by = {e["id"]: e for e in wc.load()}
    up, lo = by[U.UPSCALER_ID], by[U.GUIDE_LORA_ID]
    for e in (up, lo):
        assert e["arch"] == "minimax_h3" and e["category"] == "video" and e["use"] == ["upscale_video"]
        assert e["used_by"] == [U.CAP] and e["kind"] == "file" and e["repo_id"] and e["remote_filename"]
    assert up["type"] == "latent_upscale_models" and up["filename"].endswith("_fp16.safetensors")
    assert abs(up["approx_gb"] - 0.69) < 0.01 and abs(lo["approx_gb"] - 0.62) < 0.01       # 690.592.672 B / 620.285.992 B
    assert lo["convert"] == "mmh3_lora_guide" and "mmh3_lora_guide" in nc.CONVERTERS
    assert lo["filename"].endswith("-native.safetensors") and lo["type"] == "LoRA" and lo["family"] == "MiniMax-H3"
    # catena: upscaler → LoRA di guida → un DiT Ref2VA (alternative), mai un FL2VA
    assert up["deps"] == [U.GUIDE_LORA_ID]
    alts = wc.dep_alternatives(lo["deps"][0])
    assert alts and all(a in by and "ref2va" in a for a in alts) and not any("fl2va" in a for a in alts)
    for e in (up, lo):
        for d in e.get("deps", []):
            assert all(a in by for a in wc.dep_alternatives(d))


def test_matrix_gate_e_api_conoscono_la_cap():
    from api import gate
    assert U.CAP in gate.KNOWN_CAPS
    src = (ROOT / "api" / "handlers" / "video.py").read_text("utf-8")
    assert "h3_upsampler" in src and "cap.video_h3_upsampler" in src


def test_api_enhance_request_accetta_il_motore():
    pytest.importorskip("fastapi")
    from api.handlers.video import EnhanceReq
    r = EnhanceReq(video_path="v.mp4", engine="h3_upsampler", h3_profile="int8")
    assert r.megapixels == 2.0 and r.denoise == 0.1 and r.guide is True and r.tile_tokens == 70000
    assert r.engine == U.ENGINE


def test_worker_e_servizio_sintatticamente_validi_e_senza_default_nascosti():
    for rel in ("workers/h3_upsample_infer.py", "workers/h3_upsample_core.py", "core/h3_upsample_service.py"):
        ast.parse((ROOT / rel).read_text("utf-8"))
    w = (ROOT / "workers" / "h3_upsample_infer.py").read_text("utf-8")
    assert "ComfyUI" in w and "import folder_paths" not in w and "import comfy" not in w   # nativo: nessun modulo comfy


def test_preflight_pesi_mancanti_nessun_download(tmp_path, monkeypatch):
    """Con una radice modelli vuota il preflight elenca i pesi mancanti (id e GB) e non scarica nulla."""
    from core import h3_upsample_service as hs
    monkeypatch.setenv("AI_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(wc, "get_base_dir", lambda: str(tmp_path))      # ew.missing legge la cartella base del tab Pesi
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    p = U.build_params({"video": str(v), "h3_profile": "nvfp4"})
    errs = hs.preflight(p)
    txt = " ".join(errs)
    assert U.UPSCALER_ID in txt and U.GUIDE_LORA_ID in txt and "GB" in txt and "Pesi modelli" in txt, errs
    miss = {e["id"] for e in U.missing_weights(p)}
    assert {U.UPSCALER_ID, U.GUIDE_LORA_ID, "mmh3-ref2va-nvfp4", "mmh3-video-vae"} <= miss


def test_lora_di_guida_fuori_dal_pannello_lora(tmp_path, monkeypatch):
    from core import lora_catalog as lc
    by = {e["id"]: e for e in wc.load()}
    e = by[U.GUIDE_LORA_ID]
    monkeypatch.setenv("AI_MODELS_DIR", str(tmp_path))
    dest = Path(wc.dest_for(e, str(tmp_path)))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / e["filename"]).write_bytes(b"x")
    (dest / "altra_lora_h3.safetensors").write_bytes(b"x")
    # una LoRA H3 normale presente ma NON a catalogo non c'entra; qui si controlla solo la regola «funzione»
    names = [x.filename for x in lc._registry_loras("MiniMax-H3")]
    assert e["filename"] not in names
