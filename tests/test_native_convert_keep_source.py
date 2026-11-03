"""keep_source nel catalogo: ensure_entry non cancella il sorgente ComfyUI (le cancellazioni le decide chi ne ha l'autorita), di default si cancella come sempre."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import native_convert as nc  # noqa: E402


def _run(tmp_path, monkeypatch, extra):
    seen = {}
    monkeypatch.setattr(nc, "is_complete", lambda p: True)
    monkeypatch.setattr(nc, "convert", lambda conv, src, dst, **kw: seen.update(kw) or Path(dst))
    entry = {"id": "x", "kind": "file", "convert": "mmh3_dit", "type": "diffusion_models", "filename": "a-native.safetensors",
             "remote_filename": "a.safetensors", **extra}
    monkeypatch.setattr(nc, "source_path", lambda e, d: Path(d) / "a.safetensors")
    (tmp_path / "a.safetensors").write_bytes(b"x")
    nc.ensure_entry(entry, tmp_path, log=lambda m: None)
    return seen


def test_default_deletes_source(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, {})["delete_source"] is True


def test_keep_source_keeps_it(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, {"keep_source": True})["delete_source"] is False


def test_rockerboo_entry_has_keep_source():
    import json
    d = json.load(open(ROOT / "data" / "weights_catalog.json", encoding="utf-8"))
    d = d["weights"]
    e = next(x for x in d if x.get("id") == "mmh3-dit-rockerboo-nvfp4")
    assert e["keep_source"] is True
