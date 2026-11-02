"""workers/gguf_native.gguf_offload_decision — il log non deve piu dire «offload attivo» dove la libreria non lo supporta (LTX 2.5, ltx-pipelines 1.2)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gguf_native_t", ROOT / "workers" / "gguf_native.py")
try:
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
except Exception:                                            # noqa: BLE001  (gguf/torch/ltx_core non importabili nel python della GUI)
    G = None

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(G is None, reason="gguf_native non importabile qui (usare il venv ltx)")


class OldStage:                                              # ltx-pipelines 1.1.6: l'attributo _offload_mode vale
    pass


class NewStage:                                              # ltx-pipelines >= 1.2: lo streaming lo decide il tipo del builder
    @property
    def _is_streaming(self):
        return False


def test_old_library_big_gguf_offloads_to_cpu():
    mode, msg = G.gguf_offload_decision(OldStage(), 19.0, "auto")
    assert mode == "cpu" and "offload a blocchi su CPU attivo" in msg


def test_new_library_big_gguf_stays_resident_and_says_so():
    mode, msg = G.gguf_offload_decision(NewStage(), 21.0, "auto")
    assert mode == "none" and "RESIDENTE" in msg and "INT8" in msg and "attivo" not in msg.split("RESIDENTE")[0].split(":")[-1]
    mode2, msg2 = G.gguf_offload_decision(NewStage(), 5.0, "cpu")          # forzato ma impossibile: stessa onesta
    assert mode2 == "none" and "RESIDENTE" in msg2


def test_small_gguf_unchanged():
    assert G.gguf_offload_decision(NewStage(), 7.0, "auto") == ("none", "")
    mode, msg = G.gguf_offload_decision(OldStage(), 13.0, "auto")
    assert mode == "none" and "residente" in msg
