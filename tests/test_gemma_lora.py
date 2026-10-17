"""LoRA Gemma-3 abliterata sull'encoder LTX 2.3 (core/gemma_lora + worker): mappa chiavi, fusione, aggancio ai builder
(sonda nel venv LTX, niente GPU) e risoluzione del file dal catalogo."""
import json
import os
import subprocess

import pytest

from core import gemma_lora as gl

LTX_PY = r"E:\AI\Tools\ltx2_venv\Scripts\python.exe"
PROBE = os.path.join(os.path.dirname(__file__), "_gemma_lora_probe.py")


@pytest.mark.skipif(not os.path.isfile(LTX_PY), reason="serve il venv LTX")
def test_sonda_venv_ltx():
    r = subprocess.run([LTX_PY, "-I", PROBE], capture_output=True, text=True, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON=")), "")
    assert line, r.stdout[-800:] + r.stderr[-800:]
    res = json.loads(line[len("PROBE_JSON="):])
    assert "errore" not in res, res.get("errore")
    for k in ("mappa_chiavi", "vision_scartato", "fusione_W_piu_BA"):
        assert res[k] is True, res
    for k in ("attach_n", "builder_lora", "streaming_lora", "key_fragment"):
        assert res[k] in (True, "SKIP"), res


def test_solo_release_23():
    with pytest.raises(RuntimeError, match="2.3"):
        gl.resolve_lora_path(gl.LORA_CATALOG_ID, "2.5")


def test_id_sconosciuto_e_file_assente(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="sconosciuta"):
        gl.resolve_lora_path("nonesiste", "2.3")
    from core import weights_catalog as wc
    monkeypatch.setattr(wc, "dest_for", lambda e, base=None: tmp_path)
    with pytest.raises(RuntimeError, match="non trovata su disco"):
        gl.resolve_lora_path(gl.LORA_CATALOG_ID, "2.3")
    (tmp_path / gl.LORA_FILENAME).write_bytes(b"x")
    assert gl.resolve_lora_path(gl.LORA_CATALOG_ID, "2.3").endswith(gl.LORA_FILENAME)


def test_catalogo_voce_lora():
    from core import weights_catalog as wc
    e = {x["id"]: x for x in wc.load()}[gl.LORA_CATALOG_ID]
    assert e["arch"] == "ltx23" and e["filename"] == gl.LORA_FILENAME and e["type"] == "TextEncoders"
    assert e.get("sensitivity") == "adult" and "ltx23-gemma-te" in (e.get("deps") or [])
