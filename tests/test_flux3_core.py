"""FLUX.3 Image Klein (task 13): parte PURA — specifica, riquadri/caption, token dei riferimenti, rilevamento da header,
famiglia nei filtri, LoRA dalle chiavi, API (validazione, 409/422). Nessuna GPU, nessun peso: gli header sono quelli VERI
dei file ufficiali (letti dal mirror ml-resources il 18/10/2026, `tests/data/flux3_header_*.json`)."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from core import arch_filter as af
from core import flux3_caption as C
from core import flux3_spec as S
from core import image_refs as R
from core import model_detect as md

DATA = Path(__file__).parent / "data"


def _hdr(name: str) -> dict:
    return json.loads((DATA / f"flux3_header_{name}.json").read_text("utf-8"))["tensors"]


def _write_st(path: Path, tensors: dict, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [0, 0]} for k, v in tensors.items()}
    if meta:
        head["__metadata__"] = meta
    raw = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


# ── specifica ────────────────────────────────────────────────────────────────────────────────────────────
def test_clamp_size_griglia_16_e_limiti():
    assert S.clamp_size(1010, 1010) == (1008, 1008)
    assert S.clamp_size(10, 99999) == (64, 4096)
    w, h = S.clamp_size(1920, 1080)
    assert w % 16 == 0 and h % 16 == 0


def test_stima_vram_cresce_con_la_risoluzione_e_con_i_riferimenti():
    a = S.estimate_vram_gb(1024, 1024)
    b = S.estimate_vram_gb(4096, 4096)
    c = S.estimate_vram_gb(1024, 1024, n_refs=4)
    assert b["dit"] > a["dit"] and c["dit"] > a["dit"]
    assert S.estimate_vram_gb(1024, 1024, with_pup=True)["pup"] > 18 and a["pup"] == 0
    assert a["peak"] == max(a["pup"], a["te"], a["dit"], a["vae"])


def test_distillato_senza_cfg_e_16_passi():
    assert S.DITS[S.DEFAULT_DIT]["cfg"] is None and S.default_steps() == 16
    assert S.MIN_STEPS <= S.DEFAULT_STEPS <= S.MAX_STEPS


# ── riquadri ─────────────────────────────────────────────────────────────────────────────────────────────
LAYOUT = [{"id": "balloon_1", "desc": "a red balloon", "bbox": [0.1, 0.2, 0.5, 0.6]},
          {"id": "En_Text_1", "desc": "the word SALE", "bbox": [0.6, 0.1, 0.8, 0.9]}]


def test_validate_layout_ok_e_errori():
    assert C.validate(LAYOUT) == []
    bad = [{"id": "1abc", "desc": "x", "bbox": [0.1, 0.1, 0.5, 0.5]},                 # id non valido
           {"id": "ok", "desc": "", "bbox": [0.1, 0.1, 0.5, 0.5]},                    # senza descrizione
           {"id": "ok", "desc": "y", "bbox": [0.5, 0.1, 0.1, 0.5]}]                   # y_max < y_min, id duplicato
    errs = " | ".join(C.validate(bad))
    assert "1abc" in errs and "descrizione" in errs and "duplicato" in errs and "coordinate" in errs
    assert C.validate([{"id": "a", "desc": "x", "bbox": [0, 0, 1.2, 1]}])               # fuori 0-1
    assert C.validate([{"id": f"o{i}", "desc": "x", "bbox": [0, 0, .5, .5]} for i in range(S.MAX_BOXES + 1)])


def test_to_rows_griglia_1000():
    rows = C.to_rows(LAYOUT)
    assert rows[0] == {"id": "balloon_1", "bbox": [100, 200, 500, 600], "desc": "a red balloon"}
    assert rows[1]["bbox"] == [600, 100, 800, 900]


EDIT = [{"id": "hat_1", "desc": "a straw hat", "mode": "new", "tgt_bbox": [0.0, 0.3, 0.2, 0.6]},
        {"id": "dog_1", "desc": "the dog", "mode": "move", "ref": 1, "src_bbox": [0.5, 0.0, 1.0, 0.4],
         "tgt_bbox": [0.5, 0.6, 1.0, 1.0]},
        {"id": "tree_1", "desc": "the tree", "mode": "anchor", "ref": 2, "src_bbox": [0.1, 0.1, 0.6, 0.4]},
        {"id": "sign_1", "desc": "", "mode": "remove", "ref": 1, "src_bbox": [0.2, 0.5, 0.4, 0.7]}]


def test_edit_quattro_modalita_formato_ufficiale():
    assert C.validate(EDIT, n_refs=2) == []
    rows = {r["id"]: r for r in C.to_rows(EDIT)}
    assert rows["hat_1"] == {"id": "hat_1", "from": "target", "src_bbox": None, "tgt_bbox": [0, 300, 200, 600],
                             "desc": "a straw hat"}                                  # NUOVO: solo destinazione
    assert rows["dog_1"]["from"] == "ref_image_0" and rows["dog_1"]["src_bbox"] != rows["dog_1"]["tgt_bbox"]   # SPOSTA
    assert rows["tree_1"]["from"] == "ref_image_1" and rows["tree_1"]["src_bbox"] == rows["tree_1"]["tgt_bbox"]  # ANCORA
    assert rows["sign_1"]["tgt_bbox"] is None and rows["sign_1"]["src_bbox"] == [200, 500, 400, 700]      # RIMUOVI


def test_edit_errori():
    assert C.validate([{"id": "a", "desc": "x", "mode": "move", "ref": 3, "src_bbox": [0, 0, .5, .5],
                        "tgt_bbox": [.5, .5, 1, 1]}], n_refs=2)                      # riferimento 3 su 2
    assert C.validate([{"id": "a", "desc": "x", "mode": "move", "ref": 1, "src_bbox": [0, 0, .5, .5],
                        "tgt_bbox": [0, 0, .5, .5]}], n_refs=1)                      # sposta con src = tgt
    assert C.validate([{"id": "a", "desc": "x", "mode": "boh", "tgt_bbox": [0, 0, .5, .5]}], n_refs=1)
    assert C.validate([{"id": "a", "desc": "x", "mode": "new"}], n_refs=1)             # nuovo senza destinazione


def test_ref_name_e_ordinali():
    assert C.ref_name(1) == "ref_image_0" and C.ref_name(10) == "ref_image_9"
    assert [C.ordinal(i) for i in (1, 2, 10)] == ["first", "second", "tenth"]


def test_plan_chiavi_del_system_prompt():
    rows = C.to_rows(LAYOUT)
    assert C.plan("a poster", None, 0)[0] == "t2i"
    assert C.plan("a poster", None, 2)[0] == "i2i"
    k, text, locked = C.plan("a poster", rows, 0)
    assert k == "t2i_bbox" and locked == rows and "LOCKED BOUNDING-BOX ANNOTATIONS" in text and "balloon_1" in text
    k, text = C.user_message("add a hat", C.to_rows(EDIT), 2)
    assert k == "i2i_bbox" and "ref_image_0" in text and "annotation record" in text
    k, text = C.user_message("make it night", None, 1)
    assert k == "i2i" and "the first image" in text and "make it night" in text


def test_split_detect_finished():
    caption = "A poster with <balloon_1> and the text <En_Text_1>."
    rows = C.to_rows(LAYOUT)
    full = f"{caption} {json.dumps(rows)}"
    assert C.split(full)[0] == caption and C.split(full)[1] == rows
    assert C.is_finished(full)
    assert not C.is_finished(f"A poster. {json.dumps(rows)}")                          # box non citati: è una richiesta
    assert not C.is_finished("A caption ending in [SALE]")                              # lista che non è una tabella
    assert C.caption_without_upsampler("hello", None) == "hello"
    assert C.caption_without_upsampler("hello", rows).endswith(json.dumps(rows, ensure_ascii=False))


def test_reconcile_riquadri_dell_utente_restano_com_erano():
    locked = C.to_rows([{"id": "obj_1", "desc": "balloon", "bbox": [0.1, 0.1, 0.5, 0.5]}])
    # l'upsampler ripianifica la stessa zona con un id suo (IoU = 1) e cambia la desc: l'id bloccato vince
    answer = ('A sky with <balloon_1> floating. [{"id": "balloon_1", "bbox": [100, 100, 500, 500], '
              '"desc": "a big red balloon"}]')
    out = C.reconcile(answer, locked)
    cap, rows = C.split(out)
    assert "<obj_1>" in cap and "<balloon_1>" not in cap
    assert rows == [{"id": "obj_1", "bbox": [100, 100, 500, 500], "desc": "a big red balloon"}]
    # una riga aggiunta ben formata e citata resta; una non citata o malformata cade
    answer2 = ('Two things <obj_1> and <bird_1>. [{"id": "obj_1", "bbox": [100,100,500,500], "desc": "x"}, '
               '{"id": "bird_1", "bbox": [600,600,800,800], "desc": "a bird"}, '
               '{"id": "ghost", "bbox": [0,0,10,10], "desc": "never cited"}]')
    ids = [r["id"] for r in C.split(C.reconcile(answer2, locked))[1]]
    assert ids == ["obj_1", "bird_1"]


def test_dimensioni_riferimenti_come_il_codice_ufficiale():
    fit, canvas = C.reference_canvas(8000, 6000)                                       # > 4 MiPixel: si riduce
    assert canvas[0] * canvas[1] <= S.FIRST_REF_MAX_PIXELS and max(canvas) <= S.MAX_SIDE
    assert all(c % 16 == 0 for c in canvas)
    assert C.reference_canvas(512, 512) == ((512, 512), (512, 512))
    w, h = C.multi_reference_size(4000, 3000)
    assert w * h <= S.MULTI_REF_MAX_PIXELS
    assert C.multi_reference_size(800, 600) == (800, 600)
    w, h = C.pup_reference_size(4000, 3000, 1)
    assert w % 32 == 0 and h % 32 == 0 and w * h <= S.PUP_MAX_PIXELS[1]
    w, h = C.pup_reference_size(4000, 3000, 6)
    assert w * h <= 512 * 512
    with pytest.raises(ValueError):
        C.pup_reference_size(5000, 10, 1)                                              # rapporto > 200:1


# ── token @Foto_N → «the first image» ────────────────────────────────────────────────────────────────────
def test_token_foto_tradotti_per_ordinale():
    out = R.translate("metti @Foto_2 accanto a @Foto_1", {"image": [1, 2]}, "flux3")
    assert out == "metti the second image accanto a the first image"
    out = R.translate("@Foto_3 su @Foto_1", {"image": [1, 3]}, "flux3")                  # Foto_3 è la 2ª passata
    assert out == "the second image su the first image"
    assert R.MODEL_SYNTAX["flux3"]["max_images"] == 10 and R.MODEL_SYNTAX["flux3"]["verified"] is False
    assert R.translate("@Foto_1", {"image": [1]}, "qwen21") == "image 1"                 # gli altri stili non cambiano


# ── rilevamento dall'HEADER ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["bf16", "fp8r"])
def test_header_ufficiale_riconosciuto_flux3_distillato(name):
    t = _hdr(name)
    assert md._from_keys(t, "model.safetensors") == ("flux3", "distilled", "t2i", "image")


def test_header_senza_ref_time_in_e_base():
    t = {k: v for k, v in _hdr("bf16").items() if not k.startswith("ref_time_in.")}
    assert md._from_keys(t, "x.safetensors") == ("flux3", "base", "t2i", "image")


def test_precisione_fp8_dall_header():
    t = _hdr("fp8r")
    assert md.precision_from_header("x.safetensors", t) == "fp8"
    assert md.precision_from_header("x.safetensors", _hdr("bf16")) == "fp16"


def test_flux3_non_confuso_con_flux1_flux2(tmp_path, monkeypatch):
    """Un FLUX.3 con «klein» nel nome NON finisce fra i FLUX.2 (il nome contiene «klein» come i FLUX.2 Klein): decide l'header."""
    root = tmp_path / "models"
    _write_st(root / "diffusion_models" / "Flux" / "flux2_klein_named_but_flux3.safetensors", _hdr("fp8r"))
    _write_st(root / "diffusion_models" / "anonimo.safetensors", _hdr("bf16"))
    from core import offline_inventory
    monkeypatch.setattr(offline_inventory, "MODELS_ROOT", str(root))
    monkeypatch.setattr(md, "CACHE_PATH", tmp_path / "md.json")
    md._MEM.update({"cache": None, "mtime": None})
    af.clear_memo()
    for rel in ("Flux\\flux2_klein_named_but_flux3.safetensors", "anonimo.safetensors"):
        c = af.classify(rel)
        assert c["family"] == "flux3" and c["source"] == "header", (rel, c)
        assert not af.is_flux2(rel)
        assert not af.is_t2i_for_arch("FLUX", rel) and af.is_t2i_for_arch("FLUX.3 Klein", rel)
        assert not af.is_t2i_for_arch("Z-Image", rel) and not af.is_t2i_for_arch("Ernie", rel)
    md._MEM.update({"cache": None, "mtime": None})
    af.clear_memo()


def test_nome_ultima_spiaggia_flux3_prima_di_flux2():
    assert af._from_name("flux3_klein_distilled_fp8r.safetensors") == "flux3"
    assert af._from_name("FLUX.3-klein.safetensors") == "flux3"
    assert af._from_name("flux2-klein-9b.safetensors") == "flux2"
    assert af._from_name("PornFlux_20.safetensors") == "flux1"
    assert not af.is_flux2_name("flux3_klein")


def test_componenti_per_arch_flux3():
    assert af.image_arch_key("FLUX.3 Klein") == "flux3"
    assert af.compatible_text_encoder("flux3", "Flux3\\qwen3_vl_4b_instruct.safetensors")
    assert not af.compatible_text_encoder("flux3", "mistral_3_small_flux2_fp4_mixed.safetensors")
    assert not af.compatible_text_encoder("flux3", "qwen_3_8b_fp8mixed.safetensors")
    assert af.compatible_vae("flux3", "Flux\\flux2-vae.safetensors") and not af.compatible_vae("flux3", "Flux\\ae.safetensors")
    comp = md.compatible_components({"family": "flux3"})
    assert comp["vae"].endswith("flux2-vae.safetensors")


def test_funzioni_dichiarate_flux3():
    from core import model_catalog as mc
    caps = mc.image_caps("flux3")
    assert caps["t2i"] and caps["edit"] and caps["multiref"] == 10 and caps["boxes"]
    assert caps["inpaint"] and caps["inpaint_native"] is False          # inpaint = composito nostro, dichiarato
    assert not caps["controlnet"] and not caps["pulid"] and not caps["rgba"]
    assert mc.ARCH_DEFAULTS["FLUX.3 Klein"]["steps"] == 16 and mc.ARCH_DEFAULTS["FLUX.3 Klein"]["cfg"] == 1.0
    # le altre arch non dichiarano riquadri
    assert "boxes" not in mc.image_caps("qwen_image21") and "boxes" not in mc.image_caps("krea2")


def test_controlnet_redux_non_per_flux3():
    assert af.image_function_backend("controlnet", "FLUX.3 Klein") == ""
    assert af.image_function_backend("redux", "FLUX.3 Klein") == ""


# ── LoRA dalle chiavi ────────────────────────────────────────────────────────────────────────────────────
def _lora(path: Path, keys):
    _write_st(path, {k: {"dtype": "F16", "shape": [4, 4]} for k in keys})


def test_lora_flux3_dalle_chiavi_non_dal_nome(tmp_path):
    from core import lora_catalog as lc
    native = tmp_path / "L" / "senza_nome_utile.safetensors"
    _lora(native, ["diffusion_model.content_mode_blocks.image.0.q_proj.lora_A.weight",
                   "diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight"])
    diff = tmp_path / "L" / "diffusers_style.safetensors"
    _lora(diff, ["transformer.mode_blocks.image.0.attn.to_q.lora_A.weight", "transformer.single_transformer_blocks.3.ff.linear_in.lora_B.weight"])
    txt = tmp_path / "L" / "txt.safetensors"
    _lora(txt, ["diffusion_model.txt_mode_blocks.2.mlp_out.lora_A.weight"])
    flux1 = tmp_path / "L" / "flux3_nel_nome_ma_flux1.safetensors"                        # il nome mente: chiavi FLUX.1
    _lora(flux1, ["transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
                  "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"])
    assert lc.lora_family_from_keys(str(native)) == "flux3"
    assert lc.lora_family_from_keys(str(diff)) == "flux3"
    assert lc.lora_family_from_keys(str(txt)) == "flux3"
    assert lc.lora_family_from_keys(str(flux1)) == ""
    assert lc.lora_family_from_keys(str(tmp_path / "manca.safetensors")) == ""


def test_lora_catalog_separa_flux3_da_flux(tmp_path, monkeypatch):
    from core import lora_catalog as lc
    root = tmp_path / "models"
    _lora(root / "LoRA" / "Flux3" / "mia_f3.safetensors", ["diffusion_model.content_mode_blocks.image.1.k_proj.lora_A.weight"])
    _lora(root / "LoRA" / "Flux" / "mia_f1.safetensors", ["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"])
    _lora(root / "LoRA" / "Flux" / "f3_in_cartella_flux.safetensors", ["diffusion_model.txt_mode_blocks.0.q_proj.lora_A.weight"])
    monkeypatch.setattr(lc, "_models_root", lambda: root)
    monkeypatch.setattr(lc, "_registry_loras", lambda arch: [])
    cat = tmp_path / "dict.json"
    cat.write_text("{}", "utf-8")
    f3 = {e.filename for e in lc.LoraCatalog(path=cat, compatible_arch="Flux3").loras()}
    f1 = {e.filename for e in lc.LoraCatalog(path=cat, compatible_arch="Flux").loras()}
    assert f3 == {"mia_f3.safetensors", "f3_in_cartella_flux.safetensors"}
    assert f1 == {"mia_f1.safetensors"}


# ── API ──────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def api(monkeypatch):
    from api.handlers import images
    monkeypatch.setattr(images.gate, "require", lambda *a, **k: None)
    monkeypatch.setattr(images, "_weights_409", lambda *a, **k: None)
    return images


def test_api_flux3_token_e_riferimenti(api):
    p = api._flux3({"prompt": "@Foto_2 sul tavolo di @Foto_1", "arch": "flux3", "ref_images": ["a.png", "b.png"]})
    assert p["prompt"] == "the second image sul tavolo di the first image"
    assert p["flux3_dit"] == S.DEFAULT_DIT and p["steps"] == 16 and "guidance" not in p and "negative" not in p
    with pytest.raises(api.HTTPException) as e:
        api._flux3({"prompt": "@Foto_3", "arch": "flux3", "ref_images": ["a.png"]})
    assert e.value.status_code == 422
    with pytest.raises(api.HTTPException):
        api._flux3({"prompt": "x", "arch": "flux3", "ref_images": [f"{i}.png" for i in range(11)]})


def test_api_flux3_riquadri_validati(api):
    p = api._flux3({"prompt": "a poster", "arch": "flux3", "boxes": LAYOUT})
    assert p["boxes"] == LAYOUT
    with pytest.raises(api.HTTPException) as e:
        api._flux3({"prompt": "a poster", "arch": "flux3", "boxes": [{"id": "9x", "desc": "", "bbox": [0, 0, 2, 2]}]})
    assert e.value.status_code == 422 and "riquadri" in str(e.value.detail)
    p = api._flux3({"prompt": "add", "arch": "flux3", "ref_images": ["a.png"], "boxes": EDIT[:1]})
    assert p["boxes"][0]["mode"] == "new"


def test_api_model_flux3_vale_arch_flux3(api):
    p = {"model": "flux3", "arch": "flux"}
    api._flux3_arch(p)
    assert p["arch"] == "flux3"
    q = {"model": "altro.safetensors", "arch": "flux"}
    api._flux3_arch(q)
    assert q["arch"] == "flux"
    api._need_model({"arch": "flux3"})                                                   # nessun file-modello richiesto


def test_api_passi_default_dell_endpoint_non_valgono(api):
    req = api.GenerateReq(prompt="x", arch="flux3")
    p = req.model_dump(exclude_none=True)
    out = api._flux3(p, req)
    assert out["steps"] == 16                                                            # il 20 dell'endpoint (FLUX.1) non passa
    req = api.GenerateReq(prompt="x", arch="flux3", steps=8)
    assert api._flux3(req.model_dump(exclude_none=True), req)["steps"] == 8
    req = api.GenerateReq(prompt="x", arch="flux3", steps=40)
    assert api._flux3(req.model_dump(exclude_none=True), req)["steps"] == 16             # tetto del distillato


def test_api_pesi_mancanti_409(monkeypatch):
    from api.handlers import images
    from core import ensure_weights as ew
    monkeypatch.setattr(images.gate, "require", lambda *a, **k: None)
    monkeypatch.setattr(ew, "missing", lambda ids: [{"id": i, "label": i, "approx_gb": 1} for i in ids])
    with pytest.raises(images.HTTPException) as e:
        images._flux3({"prompt": "x", "arch": "flux3"})
    assert e.value.status_code == 409


def test_api_outpaint_flux3_rifiutato():
    from api.handlers import images
    assert "flux3" in images.GenerateReq.model_fields["arch"].description
    assert "boxes" in images.EditReq.model_fields and "flux3_pup" in images.InpaintReq.model_fields


# ── catalogo ─────────────────────────────────────────────────────────────────────────────────────────────
def test_catalogo_flux3_voci_e_catena():
    from core import weights_catalog as wc
    by = {e["id"]: e for e in wc.load()}
    dit = by["flux3-dit-fp8r"]
    assert dit["type"] == "diffusion_models" and dit["family"] == "Flux3" and dit["arch"] == "flux3"
    assert dit["sha256"].startswith("30054053") and dit["approx_gb"] == 6.27
    assert dit["license_ack"] == "flux_kommunity" and dit["gated"] is True
    assert set(dit["deps"]) >= {"flux3-te", "flux2-vae", "flux3-pup-sakura-q8"}
    te = by["flux3-te"]
    assert te["kind"] == "snapshot" and te["license_ack"] == "flux_kommunity" and "text_encoder/*" in te["allow_patterns"]
    pup = by["flux3-pup-sakura-q8"]
    assert pup["filename"].endswith("Q8_0-8.87GiB.gguf") and pup["deps"] == ["flux3-pup-configs"]
    assert pup["sha256"].startswith("4fbde007") and "license_ack" not in pup                # Apache-2.0: nessun avviso
    cfg = by["flux3-pup-configs"]
    assert not any(p.endswith((".safetensors", ".gguf")) for p in cfg["allow_patterns"])    # NON i pesi dell'ufficiale
    for i in ("flux3-dit-fp8r", "flux3-te", "flux3-pup-sakura-q8", "flux3-pup-configs"):
        assert by[i]["used_by"], i
    # destinazioni coerenti col catalogo (diffusion_models\Flux3, TextEncoders\Flux3)
    root = "E:\\AI\\Models"
    assert str(wc.dest_for(dit, root)).replace("/", "\\").endswith("diffusion_models\\Flux3")
    assert "TextEncoders\\Flux3" in str(wc.dest_for(te, root)).replace("/", "\\")


def test_licenza_ack_richiesta_per_dit_e_te(monkeypatch):
    from core import license_ack as la
    from core import weights_catalog as wc
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: False)
    need = {e["id"] for e in la.needing_ack(["flux3-dit-fp8r"])}
    assert {"flux3-dit-fp8r", "flux3-te"} <= need and "flux3-pup-sakura-q8" not in need
