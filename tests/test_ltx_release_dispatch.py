"""Dispatch per RELEASE LTX (2.3 vs 2.5) — CPU-only.

Regressione del 17/08: la GUI leggeva `_ARCH_RUNNER["ltx"]` (cablato sulla 2.3) invece di
`runner_for()`, quindi ogni generazione video finiva sul venv 2.3 e la 2.5 era irraggiungibile
dalla GUI (tutto il pack 2.5 scaricabile restava inerte). Qui si blinda il contratto.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ltx_release as lr  # noqa: E402
from core import ltx_video_job as lvj  # noqa: E402


def test_detect_release_dai_nomi_reali():
    """Nomi VERI del catalogo (safetensors del mirror ungate e GGUF community)."""
    assert lr.detect_release("ltx-2.5-22b-distilled-bf16.safetensors") == "2.5"
    assert lr.detect_release("LTX-2.5-Distilled-Q4_K_M.gguf") == "2.5"
    assert lr.detect_release(r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-distilled-fp8.safetensors") == "2.3"
    assert lr.detect_release("qualcosa-di-ignoto.safetensors") is None   # mai indovinare


def test_runner_per_release():
    """venv DIVERSI per le due release (2.5 richiede transformers>=5.8 per Gemma 4)."""
    py23, wk23 = lr.runner("2.3")
    py25, wk25 = lr.runner("2.5")
    assert py23 != py25, "le due release devono avere venv distinti"
    assert wk23 == wk25, "stesso worker, venv diverso"
    # runner_for instrada per release, non per arch
    assert lvj.runner_for({"arch": "ltx", "ltx_release": "2.5"}) == (py25, wk25)
    assert lvj.runner_for({"arch": "ltx", "ltx_release": "2.3"}) == (py23, wk23)
    assert lvj.runner_for({"arch": "ltx"}) == lr.runner(lr.DEFAULT_RELEASE)   # default invariato
    # le altre architetture NON passano dalle release
    assert lvj.runner_for({"arch": "lingbot"}) == lvj._ARCH_RUNNER["lingbot"]


def test_gui_usa_il_dispatch_centrale():
    """La GUI deve chiamare runner_for e NON importare _ARCH_RUNNER (era il bug).
    Si guarda il CODICE, non i commenti: la spiegazione del bug cita _ARCH_RUNNER apposta."""
    import inspect
    code = [ln.split("#", 1)[0] for ln in inspect.getsource(_native_worker_run()).splitlines()]
    src = "\n".join(code)
    assert "runner_for" in src, "il worker GUI non usa il dispatch centrale"
    assert "_ARCH_RUNNER" not in src, "il worker GUI legge ancora la tabella per-arch (bypassa la release)"


def test_gui_rileva_la_release_dal_checkpoint():
    """_generate_video_native deve impostare params['ltx_release'] dal checkpoint scelto."""
    import inspect
    import ui.main_window as mw
    src = inspect.getsource(mw.MainWindow._generate_video_native)
    assert "detect_release" in src and 'params["ltx_release"]' in src


def _native_worker_run():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import ui.main_window as mw
    return mw._NativeLtx2Worker.run
