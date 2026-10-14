"""build/runtime_bundle.stage: i componenti delle feature spente non vengono spediti, file singoli e cartelle intere."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build import runtime_bundle as rb  # noqa: E402


def test_esclude_file_e_cartelle(tmp_path):
    app = tmp_path / "app"
    for rel in ("workers/veda/core/engine.py", "workers/veda/__init__.py", "workers/veda_h3.py",
                "workers/_qlinear.py", "core/x.py"):
        f = app / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# x\n", "utf-8")
    out = tmp_path / "stage"
    rb.stage(app, out, exclude=["workers/veda", "workers/veda_h3.py"])
    assert not (out / "workers" / "veda").exists()
    assert not (out / "workers" / "veda_h3.py").exists()
    assert (out / "workers" / "_qlinear.py").is_file() and (out / "core" / "x.py").is_file()
