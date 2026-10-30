"""Task 23 — memoria dei job LTX (campagna 29/10): nessuna GPU, nessun peso.

1. Parcheggio RAM: con l'offload a blocchi la partizione `*__blocks` NON si parcheggia (sarebbe una seconda
   copia dei buffer pinned); le altre partizioni si'.
2. Retry su OOM: gira FUORI dall'except (il traceback non tiene la pipeline) e non si ripete se il primo giro
   era gia' in offload=cpu; un GGUF che non entra riceve un messaggio chiaro.
3. Menu Video: file di servizio nascosti, voci fallback per 10Eros v1.4 / LTX-2.5 INT8 / GGUF Q8 (solo NSFW).

Il blocco 2 importa il worker (ltx_core): si esegue col venv LTX (E:\\AI\\Tools\\ltx2_venv); altrimenti e' saltato."""
from __future__ import annotations

import gc
import os
import sys
import weakref

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── 1. politica di parcheggio ────────────────────────────────────────────────────────────────────────────
def test_blocchi_in_streaming_non_si_parcheggiano(monkeypatch):
    from core import mem_budget as M
    monkeypatch.delenv("LTX_PARK_BLOCKS", raising=False)
    assert M.skip_park_streaming_blocks("LTXV_MODEL_COMFY_RENAMING_MAP__blocks", True) is True
    # partizione non-block, VAE, vocoder: restano parcheggiabili
    assert M.skip_park_streaming_blocks("LTXV_MODEL_COMFY_RENAMING_MAP__non_block", True) is False
    assert M.skip_park_streaming_blocks("VAE_ENCODER_COMFY_KEYS_FILTER", True) is False
    # senza streaming nessun cambiamento di comportamento
    assert M.skip_park_streaming_blocks("X__blocks", False) is False
    assert M.skip_park_streaming_blocks(None, True) is False
    assert M.skip_park_streaming_blocks("", True) is False


def test_park_blocks_env_ripristina(monkeypatch):
    from core import mem_budget as M
    monkeypatch.setenv("LTX_PARK_BLOCKS", "1")
    assert M.skip_park_streaming_blocks("X__blocks", True) is False


# ── 2. worker (venv LTX) ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def W():
    pytest.importorskip("torch")
    pytest.importorskip("ltx_core")
    import workers.ltx2_infer as mod
    import torch
    torch.cuda.is_available = lambda: False          # mai un contesto CUDA nei test
    return mod


class _Sd:
    def __init__(self, n=4):
        import torch
        self.sd = {f"k{i}": torch.zeros(1024) for i in range(n)}
        self.device, self.size, self.dtype = torch.device("cpu"), 0, {torch.float32}


class _Ops:
    def __init__(self, name):
        self.name = name


def test_registry_salta_i_blocchi_in_streaming(W, monkeypatch):
    monkeypatch.delenv("LTX_PARK_BLOCKS", raising=False)
    monkeypatch.setattr(W, "_STREAMING", True)
    reg = W._CpuParkRegistry()
    sd = _Sd()
    out = reg.add(["a.safetensors"], _Ops("n__blocks"), sd)
    assert out is sd and not reg._sd, "i blocchi in streaming non vanno parcheggiati"
    monkeypatch.setattr(W, "_STREAMING", False)
    # senza streaming la logica storica decide (qui puo' anche saltare per commit/RAM: basta che non esploda)
    assert reg.add(["a.safetensors"], _Ops("n__blocks"), _Sd()) is not None


def test_get_pipe_imposta_il_flag_streaming(W, monkeypatch):
    sentinel = object()
    key_holder = {}

    def fake_uses(_ckpt):                            # ferma _get_pipe subito dopo il flag
        raise KeyError("stop")
    monkeypatch.setattr(W, "_uses_aux_bundle", fake_uses)
    for off, atteso in (("cpu", True), ("disk", True), ("none", False), ("", False)):
        W._STREAMING = not atteso
        with pytest.raises(KeyError):
            W._get_pipe("x.safetensors", "fp8_cast", off, [])
        assert W._STREAMING is atteso, off
    del sentinel, key_holder


class _Big:
    """Stand-in di una pipeline/tensore su GPU: se il traceback la tiene viva lo vediamo col weakref."""


def test_retry_gira_fuori_dall_except(W, monkeypatch):
    riferimenti = []
    chiamate = []

    def fake_run(p):
        chiamate.append(dict(p))
        if len(chiamate) == 1:
            big = _Big()                              # vive nel frame del primo giro
            riferimenti.append(weakref.ref(big))
            p["_offload_used"] = "none"
            raise RuntimeError("CUDA out of memory. Tried to allocate 68.00 MiB")
        # secondo giro: la "pipeline" del primo e' stata liberata?
        gc.collect()
        assert riferimenti[0]() is None, "il traceback del primo giro tiene ancora vivi i suoi frame"
        return "OK.mp4"

    monkeypatch.setattr(W, "run_generation", fake_run)
    assert W.run_generation_with_oom_retry({"prompt": "x"}) == "OK.mp4"
    assert chiamate[1]["offload"] == "cpu" and chiamate[1]["_dequant_retry"] is True


def test_niente_retry_se_gia_cpu(W, monkeypatch):
    chiamate = []

    def fake_run(p):
        chiamate.append(1)
        p["_offload_used"] = "cpu"                    # scelto da _auto_offload, non dall'utente
        raise RuntimeError("CUDA out of memory.")

    monkeypatch.setattr(W, "run_generation", fake_run)
    with pytest.raises(RuntimeError, match="out of memory"):
        W.run_generation_with_oom_retry({"prompt": "x"})
    assert len(chiamate) == 1, "un secondo giro identico non serve"


def test_gguf_oom_messaggio_chiaro(W, monkeypatch):
    def fake_run(p):
        p["_offload_used"] = "cpu"
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")

    monkeypatch.setattr(W, "run_generation", fake_run)
    with pytest.raises(RuntimeError, match="GGUF .*Q8_0.gguf.* residente") as ei:
        W.run_generation_with_oom_retry({"gguf_transformer": r"E:\m\ltx-Q8_0.gguf"})
    assert "out of memory" in str(ei.value).lower()


def test_errore_non_oom_passa_intatto(W, monkeypatch):
    def fake_run(p):
        raise ValueError("altro")

    monkeypatch.setattr(W, "run_generation", fake_run)
    with pytest.raises(ValueError, match="altro"):
        W.run_generation_with_oom_retry({})


# ── 3. menu Video ────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def tab():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from ui.tabs.video_tab import VideoTab
    # niente detection sugli header reali: scriverebbe data/model_detect.json (cache) nel repo
    VideoTab._detected_family = staticmethod(lambda name: "")
    from core import model_detect
    model_detect.get = lambda *a, **k: {}
    t = VideoTab()
    yield t
    del app


def _items(t):
    return [t.model.itemText(i) for i in range(t.model.count())]


def test_menu_fallback_nsfw_ha_le_voci(tab):
    tab.nsfw.setCurrentText("NSFW")
    tab._inventory_ready = False
    tab._diffusion_models, tab._checkpoint_models, tab._gguf_models = [], [], []
    tab._refilter_models()
    testo = " | ".join(_items(tab))
    for atteso in ("10EROS v1.4", "10EROS v1.5", "INT8"):
        assert atteso in testo, atteso
    tab.nsfw.setCurrentText("SFW")
    tab._refilter_models()
    testo = " | ".join(_items(tab))
    for vietato in ("10EROS", "INT8"):
        assert vietato not in testo, f"{vietato} visibile in SFW"


def test_menu_nasconde_file_di_servizio(tab):
    tab.nsfw.setCurrentText("NSFW")
    inv = [r"LTX\_ltx23_aux_components.safetensors", r"LTX\ltx10eros_v14_2989669.safetensors",
           r"LTX\ltx2.5-uncensored-v1.1-distilled-int8-native.safetensors"]
    tab.set_model_inventory(inv, [], [r"LTXltro-modello-Q6_K.gguf"])
    testo = " | ".join(_items(tab))
    assert "_ltx23_aux_components" not in testo
    assert "ltx10eros_v14" in testo and "int8-native" in testo and "Q6_K.gguf" in testo


# ── 4. classificazione per chiavi + niente doppioni GGUF ─────────────────────────────────────────────────
def _write_st(path, keys):
    import json
    import struct
    hdr = {k: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]} for k in keys}
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"\0")


def test_header_says_not_ltx(tmp_path):
    from core import ltx_release as R
    qwen = tmp_path / "noct.safetensors"
    _write_st(qwen, ["transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.img_mlp.out.weight",
                     "img_in.weight"])
    ltx = tmp_path / "ltx.safetensors"
    _write_st(ltx, ["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                    "model.diffusion_model.transformer_blocks.0.audio_attn1.to_q.weight"])
    altro = tmp_path / "vae.safetensors"
    _write_st(altro, ["decoder.conv_in.weight"])
    assert R.header_says_not_ltx(str(qwen)) is True
    assert R.header_says_not_ltx(str(ltx)) is False
    assert R.header_says_not_ltx(str(altro)) is False            # senza transformer_blocks: nessuna esclusione
    assert R.header_says_not_ltx(str(tmp_path / "manca.safetensors")) is False
    assert R.header_says_not_ltx(str(tmp_path / "x.gguf")) is False


def test_menu_esclude_qwen_e_non_duplica_gguf(tab, tmp_path, monkeypatch):
    qwen = tmp_path / "NoctQ_V4_int8_convrot-native.safetensors"
    _write_st(qwen, ["transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.img_mlp.out.weight"])
    # il catalogo esterno lo dichiara «LTX»: la classificazione deve vincere dalle chiavi
    monkeypatch.setattr(type(tab._mcat), "arch_of", lambda self, n: "LTX")
    monkeypatch.setattr(tab, "_resolve_model_path",
                        lambda n: str(qwen) if "NoctQ" in n else None)
    tab.nsfw.setCurrentText("NSFW")
    inv = [r"Qwen21\NoctQ_V4_int8_convrot-native.safetensors", "ltx-2-3-22b-dev-Q6_K.gguf",
           r"LTX\ltx10eros_v14_2989669.safetensors"]
    tab.set_model_inventory(inv, [], ["ltx-2-3-22b-dev-Q6_K.gguf"])
    testo = _items(tab)
    assert not any("NoctQ" in t for t in testo)
    assert sum("Q6_K.gguf" in t for t in testo) == 1, testo
    assert any("ltx10eros_v14" in t for t in testo)
    # senza lista GGUF il .gguf resta come «diffusion» (unica via per sceglierlo)
    tab.set_model_inventory(inv, [], [])
    assert sum("Q6_K.gguf" in t for t in _items(tab)) == 1
