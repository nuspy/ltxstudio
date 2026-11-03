"""Guida di latenti allineata H3 (03/11): registro delle LoRA di guida, trigger nel prompt, aritmetica dei fotogrammi, params dell'edit e dell'«edit a parole»,
chiave dei checkpoint invariata senza guida, ingresso del worker. La parte sul modello (coordinate, denoise reale con DiT minuscolo) gira nel venv diffusers_next
con tests/h3_guide_diffusers_check.py: qui la lancia pytest, se il venv esiste."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
from core import h3_guide as CG
from core import video_edit_h3 as V

ALL = lambda item: True                       # noqa: E731
VENV_PY = Path(r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe")
H3_KEYS = ["transformer_blocks.0.attn.to_qkv.lora_A.weight", "transformer_blocks.0.attn.to_qkv.lora_B.weight"]


@pytest.fixture(autouse=True)
def _no_disk_defect(monkeypatch):
    monkeypatch.setattr(V, "turbo_lora_defect", lambda key: "")


@pytest.fixture()
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    p.write_bytes(b"x")
    return str(p)


def _lora(path: Path, native=True):
    hdr = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in H3_KEYS}
    if native:
        hdr["__metadata__"] = {"ltx_native": "ltx-native-1"}
    raw = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
    return str(path)


def _cfg(clip, **kw):
    c = {"video": clip, "mode": "change_background", "engine": "h3_edit", "object": "person", "prompt": "a rainy neon street", "h3_profile": "nvfp4"}
    c.update(kw)
    return c


# ── registro delle LoRA di guida ────────────────────────────────────────────────────────────────
def test_guide_info_recognizes_the_alissonerdx_loras_and_not_the_others():
    want = {"minimax_h3_vfx_edit_v1.0_r128": ("vfx_edit", "vfx_edit:"), "minimax_h3_vfx_edit_v1.0_r128_ffp": ("vfx_edit", "vfx_edit:"),
            "minimax_h3_style_transfer_v1.0_r64": ("style_transfer", "style_transfer:"), "head_swap_v1.0_r32": ("head_swap", "head_swap:"),
            "minimax_h3_motion_transfer_v1.0_ref2va": ("motion_transfer", "motion_transfer:"),
            "minimax_h3_motion_transfer_ffp_v1.0_ref2va": ("motion_transfer_ffp", "vfx_edit:"),     # FFP: trigger vfx_edit (README)
            "minimax_h3_lms_v1.0_r64": ("lms", "")}
    for nm, (gid, trig) in want.items():
        gi = CG.guide_info(nm + ".safetensors")
        assert gi and gi["id"] == gid and gi["trigger"] == trig, nm
    assert CG.guide_info("minimax_h3_lms_v1.0_r64.safetensors")["caption"] == CG.LMS_CAPTION
    for nm in ("h3_character_swap_v1.5", "H3-Person-Remover-V1", "minimax_h3_character_swap_v1.1_ref2va_fro", "turbo_v4", "minimax_h3_motion_transfer_tara_v1.0_ref2va",
               "solarrenderer_control_comfy"):
        assert CG.guide_info(nm) is None, nm
    assert CG.is_tara("minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors") and not CG.is_tara("minimax_h3_motion_transfer_v1.0_ref2va")
    # la cartella davanti non conta
    assert CG.guide_info("MiniMax-H3/vfx_edit_x.safetensors")["id"] == "vfx_edit"


def test_trigger_is_added_once_and_without_guide_the_prompt_is_untouched():
    gi = CG.guide_info("minimax_h3_vfx_edit_v1.0_r128")
    assert CG.apply_trigger("Make the coat red.", gi) == "vfx_edit: Make the coat red."
    assert CG.apply_trigger("vfx_edit: Make the coat red.", gi) == "vfx_edit: Make the coat red."          # scritto dall'utente: non raddoppia
    assert CG.apply_trigger("VFX_EDIT: x", gi) == "vfx_edit: x"
    assert CG.apply_trigger("anything", None) == "anything"                                                   # nessuna LoRA di guida: identico
    lms = CG.guide_info("minimax_h3_lms_v1.0_r64")
    assert CG.apply_trigger("whatever the user wrote", lms) == CG.LMS_CAPTION                                # caption FISSA
    with pytest.raises(ValueError, match="Due LoRA di guida"):
        CG.guide_of_loras([{"name": "minimax_h3_vfx_edit_v1.0_r128"}, {"name": "minimax_h3_style_transfer_v1.0_r64"}])
    assert CG.guide_of_loras([{"name": "turbo"}, {"name": "minimax_h3_vfx_edit_v1.0_r128"}, {"name": "minimax_h3_vfx_edit_v1.0_r128_ffp"}])["id"] == "vfx_edit"   # stessa famiglia: ok


# ── fotogrammi ──────────────────────────────────────────────────────────────────────────────────
def test_frame_arithmetic():
    assert [CG.grid_floor(n) for n in (4, 5, 21, 22, 288)] == [0, 5, 5, 22, 277]
    assert [CG.latent_steps(n) for n in (1, 5, 22, 39, 277)] == [1, 2, 7, 12, 82]
    assert CG.latent_start_frames(40)[:8] == [0, 1, 5, 9, 13, 17, 18, 22]
    plan = CG.plan_guide(288)                                                  # 12 s a 24 fps
    assert plan == {"guide_frames": 277, "frames": 277, "frame_idx": 0, "steps": 82}
    assert CG.plan_guide(288, max_frames=144)["frames"] == 141                # tetto 6 s -> 17n+5
    p5 = CG.plan_guide(300, frame_idx=5, target_frames=277)
    assert p5["frame_idx"] == 5 and p5["guide_frames"] == CG.grid_floor(277 - 5) == 260     # la guida non esce dal video
    assert CG.plan_guide(124, frame_idx=-107, target_frames=124)["frame_idx"] == 17            # negativo = dalla fine, come AddGuide
    for kw, frag in (({"source_frames": 3}, "almeno 5"), ({"source_frames": 124, "frame_idx": 2}, "frame_idx"),
                     ({"source_frames": 124, "target_frames": 120}, "forma 17n")):
        with pytest.raises(ValueError, match=frag):
            CG.plan_guide(**kw)
    assert CG.guide_rows(82, 480, 864) == 82 * (480 // 32) * (864 // 32)


# ── edit con maschera: params e chiave ───────────────────────────────────────────────────────────
def test_edit_without_guide_has_the_params_of_always_and_the_key_is_unchanged(clip):
    p = V.build_params(_cfg(clip), "o.mp4", present=ALL)
    assert "guide" not in p["h3_edit"] and "guide_lora" not in p["h3_edit"] and "h3_guide" not in p
    import h3_edit_run as ER
    assert ER.guide_key(p["h3_edit"]) == () and ER.opts_key(p) == ()                # i checkpoint dei clip gia fatti restano validi
    assert ER.guide_key({"guide": True}) == (("guide_v1", 0),)


def test_guide_lora_turns_the_guide_on_and_h3_guide_flag_alone_too(clip, tmp_path):
    g = _lora(tmp_path / "minimax_h3_vfx_edit_v1.0_r128-native.safetensors")
    p = V.build_params(_cfg(clip, loras=[{"path": g, "strength": 0.8}]), "o.mp4", present=ALL)
    assert p["h3_edit"]["guide"] is True and p["h3_edit"]["guide_lora"] == "vfx_edit" and p["loras"] == [{"name": g, "strength": 0.8}]
    p2 = V.build_params(_cfg(clip, h3_guide=True), "o.mp4", present=ALL)             # prova «rifugio con guida», senza LoRA
    assert p2["h3_edit"]["guide"] is True and "guide_lora" not in p2["h3_edit"] and "loras" not in p2
    with pytest.raises(V.H3EditError, match="Veda"):                                    # Veda + guida: errore subito, non dopo minuti di clip
        V.build_params(_cfg(clip, h3_guide=True, h3_veda=True), "o.mp4", present=ALL)
    tara = _lora(tmp_path / "minimax_h3_motion_transfer_tara_v1.0_ref2va-native.safetensors")
    with pytest.raises(V.H3EditError, match="TARA"):
        V.build_params(_cfg(clip, loras=[{"path": tara, "strength": 1.0}]), "o.mp4", present=ALL)
    hs = _lora(tmp_path / "minimax_h3_head_swap_v1.0_r32-native.safetensors")
    with pytest.raises(V.H3EditError, match="foto"):                                   # Head Swap vuole la foto del volto
        V.build_params(_cfg(clip, mode="change_background", loras=[{"path": hs, "strength": 1.0}]), "o.mp4", present=ALL)
    v2 = _lora(tmp_path / "minimax_h3_style_transfer_v1.0_r64-native.safetensors")
    with pytest.raises(V.H3EditError, match="Due LoRA di guida"):
        V.build_params(_cfg(clip, loras=[{"path": g, "strength": 1.0}, {"path": v2, "strength": 1.0}]), "o.mp4", present=ALL)


def test_edit_run_adds_the_guide_to_meta_only_when_chosen():
    src = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert 'if e.get("guide"):' in src and 'meta["guide"]' in src                      # niente chiave `guide` in meta senza guida: la chiave del clip e quella di sempre
    assert "*opts_key(p), *guide_key(e))" in src


# ── «edit a parole» (proposta): params del worker ────────────────────────────────────────────────────
def test_build_guide_params_for_the_words_only_edit(clip, tmp_path):
    g = _lora(tmp_path / "minimax_h3_vfx_edit_v1.0_r128-native.safetensors")
    p = V.build_guide_params({"video": clip, "prompt": "Change the coat to red while preserving everything else.", "h3_profile": "nvfp4",
                              "loras": [{"path": g, "strength": 1.0}], "max_s": 6}, "o.mp4", present=ALL)
    assert p["h3_guide"] == {"video": clip, "frame_idx": 0, "keep_audio": True, "max_s": 6.0}
    assert p["prompt"].startswith("vfx_edit: Change the coat") and p["h3_guide_lora"] == "vfx_edit" and p["loras"][0]["strength"] == 1.0
    assert p["h3_turbo"] == "v4" and p["h3_video_shift"] == 12.0 and "references" not in p and "h3_edit" not in p
    import minimax_h3_infer as W
    from core import minimax_h3_spec as S
    assert S.mode_for(p) == "ref2va"                                                   # i DiT Ref2VA anche senza riferimenti nativi
    assert S.weight_ids(p)["dit"] == "mmh3-ref2va-nvfp4" and S.weight_ids(p)["turbo"] == "mmh3-turbo-v4-ref2v"
    assert W.build_prompt(p, 5.0) == p["prompt"]                                       # prompt della LoRA: una frase, trigger davanti, nessuna struttura inventata
    # senza LoRA di guida / senza richiesta / LMS (caption fissa, nessuna richiesta)
    with pytest.raises(V.H3EditError, match="LoRA di guida"):
        V.build_guide_params({"video": clip, "prompt": "x", "h3_profile": "nvfp4"}, "o.mp4", present=ALL)
    with pytest.raises(V.H3EditError, match="richiesta"):
        V.build_guide_params({"video": clip, "prompt": " ", "h3_profile": "nvfp4", "loras": [{"path": g}]}, "o.mp4", present=ALL)
    lms = _lora(tmp_path / "minimax_h3_lms_v1.0_r64-native.safetensors")
    pl = V.build_guide_params({"video": clip, "prompt": "", "h3_profile": "nvfp4", "loras": [{"path": lms, "strength": 1.0}]}, "o.mp4", present=ALL)
    assert pl["prompt"] == CG.LMS_CAPTION
    st = _lora(tmp_path / "minimax_h3_style_transfer_v1.0_r64-native.safetensors")
    pic = tmp_path / "style.png"
    pic.write_bytes(b"x")
    ps = V.build_guide_params({"video": clip, "prompt": "Re-render this video with the style of <Picture 1>.", "h3_profile": "nvfp4", "ref_image": str(pic),
                               "loras": [{"path": st, "strength": 0.7}]}, "o.mp4", present=ALL)
    assert ps["references"][0]["kind"] == "image" and ps["prompt"].startswith("style_transfer: ")
    assert W.build_prompt(ps, 5.0) == ps["prompt"]                                     # con foto e LoRA di guida: ancora la frase della LoRA, non la struttura Ref2VA


def test_worker_prompt_is_unchanged_without_guide_and_gets_only_the_trigger_with_a_structured_edit_prompt():
    import minimax_h3_infer as W
    from core import h3_prompt
    base = {"prompt": "A cat walks.", "h3_profile": "nvfp4"}
    assert W.build_prompt(base, 5.0) == h3_prompt.compose(base, 5.0)                   # senza guida: invariato
    structured = "subject_definitions:\n<Subject 1> is the person.\n\nsummary: x"
    p = {"prompt": structured, "references": [{"kind": "image"}], "_h3_guide_latents": object(),
         "loras": [{"name": "minimax_h3_vfx_edit_v1.0_r128"}]}
    assert W.build_prompt(p, 5.0) == "vfx_edit: " + structured                          # edit con maschera + LoRA di guida: solo il trigger davanti
    p2 = {"prompt": structured, "references": [{"kind": "image"}], "_h3_guide_latents": object()}
    assert W.build_prompt(p2, 5.0) == h3_prompt.compose(p2, 5.0) == structured          # guida senza LoRA: prompt invariato


def test_worker_guide_setup_refuses_veda_and_missing_video(tmp_path):
    import minimax_h3_infer as W
    assert W._guide_setup({"prompt": "x"}, None) is None
    with pytest.raises(RuntimeError, match="Veda"):
        W._guide_setup({"h3_guide": {"video": "x"}, "h3_veda": True}, None)
    with pytest.raises(RuntimeError, match="non trovato"):
        W._guide_setup({"h3_guide": {"video": str(tmp_path / "manca.mp4")}}, None)


# ── nel venv di diffusers: coordinate e denoise reale ───────────────────────────────────────────────
@pytest.mark.skipif(not VENV_PY.is_file(), reason="venv diffusers_next assente")
def test_guide_layout_and_denoise_on_the_real_diffusers_pipeline():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    r = subprocess.run([str(VENV_PY), str(ROOT / "tests" / "h3_guide_diffusers_check.py")], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env, timeout=1200)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out[-4000:]
    assert "verifiche passate: 8" in out
