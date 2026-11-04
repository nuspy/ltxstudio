"""Opzioni di velocita e modello dell'edit H3 (Modifica Video, brief 03/11): LoRA turbo/passi, LoRA utente, Veda, LightVAE, menu «Modello H3».
Default invariati; ogni scelta arriva ai params del worker, al figlio del clip e alla chiave dei checkpoint."""
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
from core import minimax_h3_spec as S
from core import video_edit_engines as E
from core import video_edit_h3 as V

ALL = lambda item: True                       # noqa: E731
_REAL_DEFECT = V.turbo_lora_defect


@pytest.fixture(autouse=True)
def _no_disk_defect(monkeypatch):
    """La turbo 4 passi v0.1 sul disco di oggi e monca (0 adattatori AdaLN): i test che non parlano del difetto non devono dipendere dal disco."""
    monkeypatch.setattr(V, "turbo_lora_defect", lambda key: "")


@pytest.fixture()
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    p.write_bytes(b"x")
    return str(p)


def _cfg(clip, **kw):
    c = {"video": clip, "mode": "change_background", "engine": "h3_edit", "object": "person", "prompt": "a rainy neon street", "h3_profile": "nvfp4"}
    c.update(kw)
    return c


def _lora(path: Path, keys, native=True, source=None):
    hdr = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
    if native:
        hdr["__metadata__"] = {"ltx_native": "ltx-native-1", **({"source": source} if source else {})}
    raw = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
    return str(path)


H3_KEYS = ["transformer_blocks.0.attn.to_qkv.lora_A.weight", "transformer_blocks.0.attn.to_qkv.lora_B.weight"]


def test_defaults_are_unchanged_and_the_ignored_steps_parameter_is_gone(clip):
    p = V.build_params(_cfg(clip), "o.mp4", present=ALL)
    assert p["h3_turbo"] == "v4" and p["h3_profile"] == "nvfp4"
    for k in ("h3_dit", "h3_veda", "h3_lightvae", "loras", "steps", "engine_steps"):
        assert k not in p, k                                                   # `steps` (6) era IGNORATO dal worker: tolto, i passi li decide la turbo
    assert "steps" not in E.ENGINES["h3_edit"].params
    assert V.turbo_info("v4") == (8, 12.0) and V.turbo_info("4") == (4, 12.0)   # passi e shift li dice la tabella TURBO
    import h3_edit_run as ER
    assert ER.opts_key(p) == ()                                                # chiave dei checkpoint di sempre


def test_turbo_choice_reaches_params_and_sets_steps_and_shift(clip):
    p = V.build_params(_cfg(clip, h3_turbo="4"), "o.mp4", present=ALL)
    assert p["h3_turbo"] == "4"
    ids = S.weight_ids({**p, "references": [{"kind": "image"}]})
    assert ids["turbo"] == "mmh3-turbo-lora-ref2v" and S.TURBO["4"][1] == 4 and S.TURBO["4"][2] == 12.0
    ch = {t["key"]: t for t in V.turbo_choices()}
    assert ch["v4"]["ref2va"] and ch["4"]["ref2va"] and not ch["8"]["ref2va"] and not ch["dmad"]["ref2va"]
    for bad in ("8", "dmad", "nonesiste"):                                    # le turbo solo FL2VA non si possono scegliere nell'edit (lavora in Ref2VA)
        with pytest.raises(V.H3EditError, match="Ref2VA"):
            V.build_params(_cfg(clip, h3_turbo=bad), "o.mp4", present=ALL)


def test_turbo_fused_dit_uses_its_own_steps_and_shift():
    assert V.turbo_info("v4", "mmh3-dit-10erosmax-b7") == (8, 12.0)


def test_custom_dit_is_a_checkpoint_the_user_chooses_and_must_be_installed(clip):
    assert set(V.CUSTOM_MODELS) == {"mmh3-dit-pinkcherry18", "mmh3-dit-10erosmax-b7", "mmh3-dit-dasiwa-v4", "mmh3-dit-rockerboo-nvfp4"}
    p = V.build_params(_cfg(clip, h3_profile="mmh3-dit-dasiwa-v4"), "o.mp4", present=ALL)
    assert p["h3_dit"] == "mmh3-dit-dasiwa-v4" and p["h3_profile"] == V.CUSTOM_TE_PROFILE
    ids = S.weight_ids({**p, "references": [{"kind": "image"}]})
    assert ids["dit"] == "mmh3-dit-dasiwa-v4" and ids["te"] == "mmh3-te-nvfp4"
    with pytest.raises(V.H3EditError, match="non e installato"):
        V.build_params(_cfg(clip, h3_profile="mmh3-dit-dasiwa-v4"), "o.mp4", present=lambda i: i != "mmh3-dit-dasiwa-v4")
    with pytest.raises(V.H3EditError, match="Scegli il profilo H3"):
        V.build_params(_cfg(clip, h3_profile="gguf-q4"), "o.mp4", present=ALL)
    # menu: tutte le voci, anche quelle non sul disco (la GUI le segna), con il DiT del catalogo pesi
    ch = {c["id"]: c for c in V.model_choices(lambda i: i == "mmh3-ref2va-nvfp4")}
    assert list(ch) == ["int8", "fp8", "nvfp4", "int4", "mmh3-dit-pinkcherry18", "mmh3-dit-10erosmax-b7", "mmh3-dit-dasiwa-v4", "mmh3-dit-rockerboo-nvfp4"]
    assert ch["nvfp4"]["installed"] and not ch["int8"]["installed"] and ch["int8"]["dit"] == "mmh3-ref2va-int8" and ch["nvfp4"]["kind"] == "profile"
    assert ch["mmh3-dit-pinkcherry18"]["kind"] == "custom" and ch["mmh3-dit-pinkcherry18"]["dit"] == "mmh3-dit-pinkcherry18"


def test_veda_and_lightvae_reach_params_and_weight_ids(clip):
    p = V.build_params(_cfg(clip, h3_veda=True, h3_lightvae="dec"), "o.mp4", present=ALL)
    assert p["h3_veda"] is True and p["h3_lightvae"] == "dec"
    ids = S.weight_ids({**p, "references": [{"kind": "image"}]})
    assert ids["veda"] == S.VEDA_R2VA and ids["vae_dec_lite"] == "mmh3-lightvae-dec" and "vae_enc_lite" not in ids
    with pytest.raises(V.H3EditError, match="Veda"):
        V.build_params(_cfg(clip, h3_veda=True), "o.mp4", present=lambda i: i != S.VEDA_R2VA)
    with pytest.raises(V.H3EditError, match="LightVAE"):
        V.build_params(_cfg(clip, h3_lightvae="dec"), "o.mp4", present=lambda i: i != "mmh3-lightvae-dec")
    with pytest.raises(V.H3EditError, match="sconosciuto"):
        V.build_params(_cfg(clip, h3_lightvae="xyz"), "o.mp4", present=ALL)


def test_each_option_changes_the_checkpoint_key():
    import h3_edit_run as ER
    base = {}
    keys = {ER.opts_key({"h3_veda": True}), ER.opts_key({"h3_lightvae": "dec"}), ER.opts_key({"h3_dit": "mmh3-dit-dasiwa-v4"}),
            ER.opts_key({"loras": [{"name": "MiniMax-H3\\a.safetensors", "strength": 1.0}]}),
            ER.opts_key({"loras": [{"name": "MiniMax-H3\\a.safetensors", "strength": 0.5}]})}
    assert len(keys) == 5 and () not in keys and ER.opts_key(base) == ()


def test_user_loras_are_checked_by_header_keys_and_declared_mode(clip, tmp_path):
    good = _lora(tmp_path / "style_h3-native.safetensors", H3_KEYS)
    assert V.lora_edit_compat(good) == (True, "")
    other = _lora(tmp_path / "flux_style.safetensors", ["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"])
    ok, why = V.lora_edit_compat(other)
    assert not ok and "chiavi" in why                                         # architettura dalle CHIAVI dell'header, mai dai metadati
    fl = _lora(tmp_path / "motion_fl2va-native.safetensors", H3_KEYS)
    ok, why = V.lora_edit_compat(fl)
    assert not ok and "solo FL2VA" in why
    assert V.lora_edit_compat(_lora(tmp_path / "x_ref2va.safetensors", H3_KEYS)) == (True, "")
    assert V.lora_edit_compat(str(tmp_path / "manca.safetensors"))[0] is False
    # arrivano ai params col nome e la forza scelti; una LoRA non compatibile ferma il lancio con il motivo
    p = V.build_params(_cfg(clip, loras=[{"path": good, "strength": 0.7}]), "o.mp4", present=ALL)
    assert p["loras"] == [{"name": good, "strength": 0.7}]
    for bad in (other, fl):
        with pytest.raises(V.H3EditError, match="non usabile nell'edit H3"):
            V.build_params(_cfg(clip, loras=[{"path": bad, "strength": 1.0}]), "o.mp4", present=ALL)


def test_options_flow_to_the_clip_child_params():
    """h3_edit_run copia TUTTI i params nel `base` di ogni clip (base = dict(p)): il figlio risolve pesi e LoRA con gli stessi params."""
    import h3_edit_run as ER
    src = Path(ER.__file__).read_text(encoding="utf-8")
    assert "base = dict(p)" in src and 'base.pop("h3_edit", None)' in src
    p = {"h3_profile": "nvfp4", "h3_turbo": "4", "h3_veda": True, "h3_lightvae": "dec", "h3_dit": "mmh3-dit-pinkcherry18",
         "loras": [{"name": "MiniMax-H3\\a.safetensors", "strength": 0.6}]}
    ids = S.weight_ids({**p, "references": [{"kind": "image"}]})                  # come W.resolve({**base, "references": [...]}) nel figlio
    assert ids["dit"] == "mmh3-dit-pinkcherry18" and ids["veda"] == S.VEDA_R2VA and ids["vae_dec_lite"] == "mmh3-lightvae-dec" and ids["turbo"] == "mmh3-turbo-lora-ref2v"


def test_loras_not_native_are_disabled_with_a_reason_and_guide_loras_are_now_enabled(tmp_path):
    nonnative = _lora(tmp_path / "style_h3.safetensors", H3_KEYS, native=False)
    assert V.lora_edit_check(nonnative) == (False, "convert")                       # l'edit usa solo LoRA native: «convertila dal tab Pesi»
    for nm in ("minimax_h3_vfx_edit_v1.0_r128-native.safetensors", "minimax_h3_style_transfer_v1.0_r64-native.safetensors",
               "minimax_h3_motion_transfer_v1.0_ref2va-native.safetensors", "minimax_h3_head_swap_v1.0_r32-native.safetensors"):
        assert V.lora_edit_check(_lora(tmp_path / nm, H3_KEYS)) == (True, ""), nm               # Alissonerdx: dal 03/11 la guida allineata c'e (workers/h3_guide.py)
    assert V.lora_edit_check(_lora(tmp_path / "minimax_h3_motion_transfer_tara_v1.0_ref2va-native.safetensors", H3_KEYS)) == (False, "tara")   # altra tecnica: patch del modello
    for nm in ("h3_character_swap_v1.5-native.safetensors", "H3-Person-Remover-V1-native.safetensors",
               "minimax_h3_character_swap_v1.1_ref2va_fro-native.safetensors"):
        assert V.lora_edit_check(_lora(tmp_path / nm, H3_KEYS)) == (True, ""), nm               # akatz: <Video 1> nativo, e il nostro caso
    assert "TARA" in V.LORA_REASONS["tara"] and "nativo" in V.LORA_REASONS["convert"]


def test_turbo_lora_without_adaln_adapters_is_refused(clip, tmp_path, monkeypatch):
    """4 passi Ref2V v0.1 (conversione ComfyUI sui pruned): 0 chiavi transformer_blocks.N.adaln_proj, la v4 ne ha 100 (ModelTC/Minimax-H3-Turbo#7)."""
    from core import model_paths as mp
    bad = _lora(tmp_path / "turbo4.safetensors", H3_KEYS, source="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")     # da ComfyUI, nessuna adaln_proj
    ok = _lora(tmp_path / "turbov4.safetensors", H3_KEYS + [f"transformer_blocks.{i}.adaln_proj.lora_A.weight" for i in range(3)])
    diffusers = _lora(tmp_path / "turbo_r4v10.safetensors", H3_KEYS, source="minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors")    # lightx2v diffusers: senza AdaLN per costruzione
    monkeypatch.setattr(mp, "p", lambda i, sub="": bad if i == "mmh3-turbo-lora-ref2v" else (diffusers if i == "mmh3-turbo-ref2v-4-v10" else ok))
    monkeypatch.setattr(V, "turbo_lora_defect", _REAL_DEFECT)
    V._TURBO_DEFECT_CACHE.clear()
    assert V.turbo_lora_defect("4") == "adaln" and V.turbo_lora_defect("v4") == "" and V.turbo_lora_defect("8") == ""
    assert V.turbo_lora_defect("r4v10") == ""                                       # la versione diffusers NON e segnata (si giudica dal giro dal vivo)
    ch = {t["key"]: t for t in V.turbo_choices()}
    assert ch["4"]["defect"] == "adaln" and ch["v4"]["defect"] == ""
    with pytest.raises(V.H3EditError, match="AdaLN"):
        V.build_params(_cfg(clip, h3_turbo="4"), "o.mp4", present=ALL)
    assert V.build_params(_cfg(clip, h3_turbo="v4"), "o.mp4", present=ALL)["h3_turbo"] == "v4"
    V._TURBO_DEFECT_CACHE.clear()


def test_text_encoder_is_explicit_when_chosen_and_must_be_installed(clip):
    p = V.build_params(_cfg(clip, h3_profile="int8", h3_te="mmh3-te-nvfp4"), "o.mp4", present=ALL)
    assert p["h3_te"] == "mmh3-te-nvfp4" and S.weight_ids({**p, "references": [{"kind": "image"}]})["te"] == "mmh3-te-nvfp4"   # stesso encoder per tutti i DiT
    assert "h3_te" not in V.build_params(_cfg(clip), "o.mp4", present=ALL)                                                        # default: quello del profilo
    no_te_int8 = lambda i: i != "mmh3-te-int8"          # noqa: E731
    with pytest.raises(V.H3EditError, match="Encoder testo"):
        V.build_params(_cfg(clip, h3_profile="int8"), "o.mp4", present=no_te_int8)         # il profilo int8 vorrebbe il TE int8, che non c'e: scegli un altro, mai in silenzio
    with pytest.raises(V.H3EditError, match="sconosciuto"):
        V.build_params(_cfg(clip, h3_te="xyz"), "o.mp4", present=ALL)
    import h3_edit_run as ER
    assert ER.opts_key({"h3_te": "mmh3-te-nvfp4"}) != () and ER.opts_key({}) == ()


def test_rockerboo_nvfp4_is_a_custom_ref2va_model_in_the_menu(clip):
    assert "mmh3-dit-rockerboo-nvfp4" in V.CUSTOM_MODELS and S.CUSTOM_DITS["mmh3-dit-rockerboo-nvfp4"]["modes"] == ("ref2va",)
    p = V.build_params(_cfg(clip, h3_profile="mmh3-dit-rockerboo-nvfp4"), "o.mp4", present=ALL)
    assert p["h3_dit"] == "mmh3-dit-rockerboo-nvfp4" and S.weight_ids({**p, "references": [{"kind": "image"}]})["dit"] == "mmh3-dit-rockerboo-nvfp4"
    from core import model_paths as mp
    assert mp.entry("mmh3-dit-rockerboo-nvfp4")["filename"].startswith("minimax_h3_ref2va_pruned_nvfp4_convrot_int8")


def test_new_lightx2v_ref2va_turbos_are_registered_with_their_steps_and_catalog_entries(clip):
    assert S.TURBO["r4v10"][1:] == (4, 12.0) and S.TURBO["r8v11"][1:] == (8, 6.0)
    ids = {k: S.weight_ids({"h3_profile": "nvfp4", "h3_turbo": k, "references": [{"kind": "image"}]})["turbo"] for k in ("r4v10", "r8v11")}
    assert ids == {"r4v10": "mmh3-turbo-ref2v-4-v10", "r8v11": "mmh3-turbo-ref2v-8-v11"}
    from core import model_paths as mp
    for cid, fn in (("mmh3-turbo-ref2v-4-v10", "minimax_h3_ref2v_turbo_4step_v1.0_768p-native.safetensors"),
                    ("mmh3-turbo-ref2v-8-v11", "minimax_h3_ref2v_turbo_8step_v1.1_768p-native.safetensors")):
        e = mp.entry(cid)
        assert e["filename"] == fn and e["convert"] == "mmh3_lora_diffusers" and e["type"] == "LoRA"
    for k in ("r4v10", "r8v11"):
        assert V.build_params(_cfg(clip, h3_turbo=k), "o.mp4", present=ALL)["h3_turbo"] == k
        assert V.turbo_info(k) == S.TURBO[k][1:]


def test_original_lora_with_a_native_twin_is_hidden_behind_the_native_one(tmp_path):
    orig = _lora(tmp_path / "h3_character_swap_v1.5.safetensors", H3_KEYS, native=False)
    assert V.lora_edit_check(orig) == (False, "convert")
    _lora(tmp_path / "h3_character_swap_v1.5-native.safetensors", H3_KEYS)
    assert V.lora_edit_check(orig) == (False, "twin")
    assert V.lora_edit_check(str(tmp_path / "h3_character_swap_v1.5-native.safetensors")) == (True, "")


def test_text_info_reaches_params_only_when_the_text_was_processed(clip):
    base = V.build_params(_cfg(clip), "o.mp4", present=ALL)
    assert "text_info" not in base["h3_edit"]                                                  # default: params di sempre
    eng = V.build_params(_cfg(clip, h3_text_info={"status": "english", "original": {}, "sent": {}}), "o.mp4", present=ALL)
    assert "text_info" not in eng["h3_edit"]                                                   # testo gia inglese: niente di nuovo
    ti = {"status": "ok", "seconds": 4.2, "original": {"prompt": "un rifugio"}, "sent": {"prompt": "A refuge."}}
    p = V.build_params(_cfg(clip, h3_text_info=ti), "o.mp4", present=ALL)
    assert p["h3_edit"]["text_info"]["original"] == {"prompt": "un rifugio"} and p["h3_edit"]["text_info"]["sent"] == {"prompt": "A refuge."}
