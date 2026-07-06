"""Test importer modelli (#17) — offline, nessuna rete (Civitai mockato)."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import civitai_client as C          # noqa: E402
from core import model_importer as MI         # noqa: E402


def test_hash_autov2():
    assert len(C.autov2("a" * 64)) == 10
    assert C.autov2("ABCDEF0000" + "0" * 54) == "abcdef0000"


def test_parse_url():
    assert C.parse_url("12345") == {"model_id": 12345, "version_id": None}
    assert C.parse_url("https://civitai.com/models/7")["model_id"] == 7
    r = C.parse_url("https://civitai.red/models/1/x?modelVersionId=2")
    assert r["model_id"] == 1 and r["version_id"] == 2
    assert C.parse_url("https://civitai.com/api/v1/model-versions/99")["version_id"] == 99
    try:
        C.parse_url("https://example.com/nope"); assert False
    except ValueError:
        pass


def test_classify_nsfw_cascade():
    assert MI.classify_nsfw({"model": {"nsfw": True}, "nsfwLevel": 16}, "a", [])[0] is True
    assert MI.classify_nsfw({"model": {"nsfw": False}, "nsfwLevel": 1}, "a", [])[0] is False
    assert MI.classify_nsfw(None, "super_nsfw_lora", [])[0] is True
    assert MI.classify_nsfw(None, "landscape_lora", [])[0] is True   # incerto → NSFW (conservativo)


def test_gate(monkeypatch):
    monkeypatch.setattr(MI, "adult_enabled", lambda: False)
    assert MI.gate_ok(False) is True and MI.gate_ok(True) is False
    monkeypatch.setattr(MI, "adult_enabled", lambda: True)
    assert MI.gate_ok(True) is True


def test_analyze_lora(tmp_path):
    N = 2_000_000
    hdr = {"lora_unet_double_blocks_0.lora_down.weight":
           {"dtype": "F16", "shape": [1000, 1000], "data_offsets": [0, N]},
           "__metadata__": {"modelspec.architecture": "flux-1-dev"}}
    b = json.dumps(hdr).encode()
    p = tmp_path / "l.safetensors"
    p.write_bytes(struct.pack("<Q", len(b)) + b + b"\x00" * N)
    it = MI.analyze(str(p), fetch_civitai=False)
    assert it.type == "LoRA" and it.family == "Flux" and it.dest_subfolder == "LoRA/Flux"


def test_resolve_civitai(monkeypatch):
    monkeypatch.setattr(C, "version", lambda vid, api_key=None: {
        "id": vid, "baseModel": "Pony", "trainedWords": ["score_9"],
        "model": {"name": "X", "type": "LORA", "nsfw": True},
        "nsfwLevel": 8, "images": [{"url": "http://x/p.png"}],
        "files": [{"name": "x.safetensors", "primary": True, "downloadUrl": "http://x/dl", "sizeKB": 100}]})
    r = MI.resolve_civitai("https://civitai.red/models/1?modelVersionId=42")
    assert r["type"] == "LORA" and r["base_model"] == "Pony" and r["nsfw"] is True
    assert r["download_url"] == "http://x/dl"
