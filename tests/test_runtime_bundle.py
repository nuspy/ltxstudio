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


def test_codice_noncommerciale_mai_nello_stage(tmp_path):
    """MagicQuill V2 (derivato CC BY-NC) e' escluso SEMPRE, anche se un'edizione lo includerebbe."""
    app = tmp_path / "app"
    for rel in ("optional/magicquill/magicquill_v2.py", "optional/magicquill/mq2/attention.py",
                "inference_server/backends/magicquill_v2.py", "inference_server/backends/mq2/__init__.py",
                "inference_server/backends/image_control.py", "core/optional_components.py"):
        f = app / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# x\n", "utf-8")
    out = tmp_path / "stage"
    rb.stage(app, out, exclude=[])                       # nessuna esclusione d'edizione
    assert rb.forbidden_in(out) == []
    assert (out / "inference_server" / "backends" / "image_control.py").is_file()
    assert (out / "core" / "optional_components.py").is_file()


def test_repo_reale_nessun_codice_derivato_nelle_cartelle_runtime():
    """Le cartelle spedite (workers/scripts/inference_server/core/assets) non contengono il codice MagicQuill: sta in optional/."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = []
    for d in rb.RUNTIME_DIRS:
        base = os.path.join(root, d)
        for dp, dn, fn in os.walk(base):
            for n in list(dn) + list(fn):
                if n in ("mq2", "magicquill_v2.py"):
                    found.append(os.path.relpath(os.path.join(dp, n), root))
    assert not found, found
    assert os.path.isdir(os.path.join(root, "optional", "magicquill"))
