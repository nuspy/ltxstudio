"""Modifica Video, motore H3: la guida di latenti allineata (widget VERI della tab) arriva al cfg dell'edit e ai params del worker; le LoRA di guida la accendono da sole
e dicono quale trigger entra nel prompt; i default restano quelli di sempre."""
import json
import os
import struct

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

H3_KEYS = ["transformer_blocks.0.attn.to_qkv.lora_A.weight", "transformer_blocks.0.attn.to_qkv.lora_B.weight"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(qapp, tmp_path, monkeypatch):
    from core import video_edit_engines as veng
    monkeypatch.setattr(veng, "_present", lambda item: True)
    from core import video_edit_h3 as vh3
    monkeypatch.setattr(vh3, "turbo_lora_defect", lambda key: "")
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    t = VideoEditAITab()
    t.set_nsfw_getter(lambda: True)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    t._video = str(clip)
    t.select_mode("change_background")
    t.obj.setText("person")
    t.prompt.setPlainText("a mountain refuge at night")
    t._cfgs = []
    t.request_edit.connect(t._cfgs.append)
    return t


def _fake_lora(tmp_path, name):
    hdr = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in H3_KEYS}
    hdr["__metadata__"] = {"ltx_native": "ltx-native-1"}
    raw = json.dumps(hdr).encode()
    p = tmp_path / name
    p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
    return str(p)


def _go(t):
    t._cfgs.clear()
    t._go()
    return t._cfgs[-1] if t._cfgs else None


def test_guide_is_off_by_default_and_the_checkbox_alone_turns_it_on(tab):
    assert tab.h3_guide.isEnabled() and not tab.h3_guide.isChecked() and not tab.h3_guide_info.isVisibleTo(tab)
    assert tab.h3_guide.isVisibleTo(tab) and tab.h3_guide.minimumHeight() >= 32                 # bersaglio non minuscolo
    cfg = _go(tab)
    assert "h3_guide" not in cfg and "loras" not in cfg                                          # default = oggi
    tab.h3_guide.setChecked(True)
    cfg = _go(tab)
    assert cfg["h3_guide"] is True
    from core import video_edit_h3 as vh3
    p = vh3.build_params(dict(cfg), "o.mp4", present=lambda i: True)
    assert p["h3_edit"]["guide"] is True and "guide_lora" not in p["h3_edit"]                    # la prova «rifugio con guida», senza LoRA


def test_a_guide_lora_turns_the_guide_on_by_itself_and_says_the_trigger(tab, tmp_path, monkeypatch):
    name = "minimax_h3_vfx_edit_v1.0_r128-native.safetensors"
    path = _fake_lora(tmp_path, name)
    entries = []

    class _E:                                                    # voce LoRA finta: solo cio che serve a selected_loras()/ai params
        def __init__(self, nm):
            self.comfy_name = nm
            self.path = path
            self.name = nm
    monkeypatch.setattr(tab.h3_lora, "selected_loras", lambda: [{"name": path, "strength": 0.9, "_entry": _E(path)}])
    tab._sync_h3_guide()
    assert tab.h3_guide.isChecked() and not tab.h3_guide.isEnabled()                              # accesa e bloccata
    assert tab.h3_guide_info.isVisibleTo(tab) and "vfx_edit:" in tab.h3_guide_info.text() and "VFX Edit" in tab.h3_guide_info.text()
    cfg = _go(tab)
    assert cfg["h3_guide"] is True and cfg["loras"] == [{"name": path, "strength": 0.9}]
    from core import video_edit_h3 as vh3
    p = vh3.build_params(dict(cfg), "o.mp4", present=lambda i: True)
    assert p["h3_edit"]["guide"] is True and p["h3_edit"]["guide_lora"] == "vfx_edit"
    # tolta la LoRA la casella torna a quella scelta a mano (spenta)
    monkeypatch.setattr(tab.h3_lora, "selected_loras", lambda: [])
    tab._sync_h3_guide()
    assert not tab.h3_guide.isChecked() and tab.h3_guide.isEnabled() and not tab.h3_guide_info.isVisibleTo(tab)
    # due LoRA di guida diverse: avviso, nessuna guida forzata
    p2 = _fake_lora(tmp_path, "minimax_h3_style_transfer_v1.0_r64-native.safetensors")
    monkeypatch.setattr(tab.h3_lora, "selected_loras", lambda: [{"name": path, "strength": 1.0, "_entry": _E(path)}, {"name": p2, "strength": 1.0, "_entry": _E(p2)}])
    tab._sync_h3_guide()
    assert tab.h3_guide_info.isVisibleTo(tab) and tab.h3_guide_info.text().startswith("⛔") and tab.h3_guide.isEnabled()


def test_the_new_texts_exist_in_all_15_languages():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "ui" / "i18n"
    keys = ("vedit.h3.guide", "vedit.h3.guide.tt", "vedit.h3.guide.lora_info", "vedit.h3.guide.lora_info_caption", "vedit.h3.lora.why.tara", "h3.guide.kind.lms",
            "h3.guide.kind.vfx_edit", "h3.guide.kind.style_transfer", "h3.guide.kind.head_swap", "h3.guide.kind.motion_transfer", "h3.guide.kind.motion_transfer_ffp")
    files = sorted(root.glob("*.json"))
    assert len(files) == 15
    for f in files:
        d = json.loads(f.read_text("utf-8"))
        for k in keys:
            assert d.get(k), (f.name, k)
        assert "{kind}" in d["vedit.h3.guide.lora_info"] and "{trigger}" in d["vedit.h3.guide.lora_info"] and "{kind}" in d["vedit.h3.guide.lora_info_caption"], f.name
        assert "vedit.h3.lora.why.guide" not in d


def test_registry_lists_the_guide_loras_in_the_h3_panel_when_the_files_are_on_disk(tab):
    """Le LoRA di guida sono nel registro pesi (build/catalog_v2.py): compaiono nel pannello LoRA H3 SOLO se il file e sul disco (l'LMS non scaricato resta fuori)."""
    from core import model_paths as mp
    from core import weights_catalog as wc
    ids = ("mmh3-lora-vfx-edit-r128", "mmh3-lora-vfx-edit-r128-ffp", "mmh3-lora-style-transfer-r64", "mmh3-lora-motion-transfer", "mmh3-lora-lms-r64")
    ents = {e["id"]: e for e in wc.load() if e["id"] in ids}
    assert set(ents) == set(ids), "voci mancanti nel registro pesi"
    for e in ents.values():
        assert e["type"] == "LoRA" and e["family"] == "MiniMax-H3" and e["filename"].endswith("-native.safetensors") and e.get("repo_id") == "Alissonerdx/Minimax-H3-ComfyUI"
        assert "mmh3-ref2va-nvfp4|mmh3-ref2va-int4|mmh3-ref2va-int8|mmh3-ref2va-fp8" in (e.get("deps") or [])
    tab.h3_lora.refresh()
    shown = {r["entry"].filename for r in tab.h3_lora._rows.values()}
    root = str(mp.models_root())
    for e in ents.values():
        assert (e["filename"] in shown) == bool(wc.is_present(e, root)), e["id"]
