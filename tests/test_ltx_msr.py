"""MSR — riferimenti multipli su LTX (LoRA Multiple-Subject-Reference) — CPU-only.

La ricetta (dal plugin ufficiale ComfyUI-LTX2.5-MSR): ogni riferimento è codificato col VAE, ci
si somma uno slot embedding Fourier-MLP estratto DALLA LoRA, e finisce a una posizione temporale
NEGATIVA. Qui si verifica tutta la matematica e la selezione della LoRA per release — l'unica
parte che richiede GPU e pesi è la generazione vera.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
import torch  # noqa: E402

from core import ltx_release as lr  # noqa: E402
from core import model_paths as mp  # noqa: E402
from core import weights_catalog as wc  # noqa: E402
from workers import ltx_msr  # noqa: E402


# ── la domanda posta esplicitamente: funziona anche per 2.3? ──────────────────
def test_entrambe_le_release_hanno_la_loro_lora():
    """Sì, MSR esiste per 2.3 E per 2.5, ma sono file DIVERSI: mai intercambiarli."""
    for rel_id, atteso in (("2.3", "LTX-2.3-Licon-MSR"), ("2.5", "LTX-2.5-Licon-MSR")):
        rel = lr.get(rel_id)
        assert rel.caps.get("multiref") == lr.CAP_YES, f"{rel_id}: multiref non dichiarata"
        assert atteso in os.path.basename(rel.msr_lora), \
            f"{rel_id} punta alla LoRA sbagliata: {rel.msr_lora}"
    assert lr.get("2.3").msr_lora != lr.get("2.5").msr_lora


def test_le_due_lora_sono_a_catalogo_e_scaricabili():
    for cid, arch in (("ltx23-lora-multisubject", "ltx23"), ("ltx25-lora-multisubject", "ltx25")):
        e = mp.entry(cid)
        assert e is not None, f"{cid} assente dal catalogo"
        assert wc.downloadable(e) and e["arch"] == arch
        assert "LiconStudio" in e["repo_id"]
    # la destinazione deve essere quella che la release si aspetta
    for rel_id, cid in (("2.3", "ltx23-lora-multisubject"), ("2.5", "ltx25-lora-multisubject")):
        assert os.path.basename(mp.p(cid, "")) == os.path.basename(lr.get(rel_id).msr_lora)


# ── ricetta MSR ───────────────────────────────────────────────────────────────
def test_posizioni_negative():
    """slot i → -(N-i): con 3 riferimenti [-3, -2, -1]. Fuori dalla finestra del video, così
    non rubano frame alla generazione, e ognuno ha la sua posizione distinta."""
    assert ltx_msr.frame_offsets(1) == [-1]
    assert ltx_msr.frame_offsets(3) == [-3, -2, -1]
    assert ltx_msr.frame_offsets(5) == [-5, -4, -3, -2, -1]
    for n in range(1, 6):
        offs = ltx_msr.frame_offsets(n)
        assert len(set(offs)) == n and max(offs) < 0


def test_slot_embedding_matematica():
    """Fourier features (scaled=/16, sin+cos) → Linear+SiLU → Linear. Slot diversi ⇒ embedding
    diversi (è ciò che permette al modello di distinguere i riferimenti)."""
    torch.manual_seed(0)
    dim, n_freq, hidden = 128, 8, 64
    state = {
        "frequencies": torch.randn(n_freq),
        "net.0.weight": torch.randn(hidden, 1 + 2 * n_freq),
        "net.0.bias": torch.randn(hidden),
        "net.2.weight": torch.randn(dim, hidden),
        "net.2.bias": torch.randn(dim),
    }
    e0 = ltx_msr.slot_embedding(state, 0, "cpu", torch.float32)
    e1 = ltx_msr.slot_embedding(state, 1, "cpu", torch.float32)
    assert e0.shape == (dim,) and e1.shape == (dim,)
    assert not torch.allclose(e0, e1), "slot diversi devono dare embedding diversi"
    assert torch.isfinite(e0).all()
    # deterministico e indipendente dal dtype richiesto
    assert torch.allclose(e0, ltx_msr.slot_embedding(state, 0, "cpu", torch.float32))
    assert ltx_msr.slot_embedding(state, 2, "cpu", torch.bfloat16).dtype == torch.bfloat16
    # senza bias (LoRA che non li porta) non deve esplodere
    slim = {k: v for k, v in state.items() if not k.endswith(".bias")}
    assert torch.isfinite(ltx_msr.slot_embedding(slim, 1, "cpu", torch.float32)).all()


def test_references_to_inputs(monkeypatch):
    """Il pannello dà path+kind: diventano ImageConditioningInput a frame negativi, solo immagini
    (la MSR di LTX non prende video/audio) e mai più di 5."""
    class _ICI(tuple):
        def __new__(cls, path, frame_idx, strength, crf=None):
            o = super().__new__(cls, (path, frame_idx, strength, crf))
            o.path, o.frame_idx, o.strength, o.crf = path, frame_idx, strength, crf
            return o

    mod = type(sys)("ltx_pipelines.utils.args")
    mod.ImageConditioningInput = _ICI
    monkeypatch.setitem(sys.modules, "ltx_pipelines", type(sys)("ltx_pipelines"))
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils", type(sys)("ltx_pipelines.utils"))
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils.args", mod)

    refs = [{"path": f"{i}.png", "kind": "image"} for i in range(3)]
    refs.append({"path": "clip.mp4", "kind": "video"})       # ignorato: MSR è solo immagini
    out = ltx_msr.references_to_inputs(refs, strength=0.9)
    assert [o.frame_idx for o in out] == [-3, -2, -1]
    assert [o.path for o in out] == ["0.png", "1.png", "2.png"]
    assert all(o.strength == 0.9 for o in out)
    troppi = [{"path": f"{i}.png", "kind": "image"} for i in range(9)]
    assert len(ltx_msr.references_to_inputs(troppi)) == ltx_msr.MAX_REFERENCES == 5


def test_lora_senza_slot_embedding_non_viene_applicata(tmp_path):
    """Una LoRA qualunque NON è una MSR: install() deve dirlo e rifiutarsi, non fingere."""
    from safetensors.torch import save_file
    p = tmp_path / "lora_normale.safetensors"
    save_file({"transformer.blocks.0.lora_A.weight": torch.zeros(4, 4)}, str(p))
    msgs = []
    assert ltx_msr.install(str(p), log=msgs.append) is False
    assert any("non è una LoRA MSR" in m for m in msgs), msgs


def test_slot_state_dict_estrae_col_prefisso(tmp_path):
    from safetensors.torch import save_file
    p = tmp_path / "msr.safetensors"
    save_file({
        "reference_slot_embedding.frequencies": torch.randn(8),
        "diffusion_model.reference_slot_embedding.net.0.weight": torch.randn(64, 17),
        "transformer.blocks.0.lora_A.weight": torch.zeros(4, 4),
    }, str(p))
    st = ltx_msr.slot_state_dict(str(p))
    assert set(st) == {"frequencies", "net.0.weight"}, st.keys()


# ── cablaggio nel worker e nella GUI ─────────────────────────────────────────
def test_worker_carica_la_lora_e_i_riferimenti():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "workers" / "ltx2_infer.py").read_text("utf-8")
    assert 'p.get("references")' in src and "ltx_msr.install" in src
    assert 'p["msr_lora"]' in src, "la LoRA MSR non viene aggiunta al modello"
    # il ramo i2v deve APPENDERE (assegnando cancellerebbe i riferimenti già inseriti)
    assert "images.append(ImageConditioningInput(p[\"image_path\"], 0, _ref_str, 33))" in src


def test_gui_sceglie_la_lora_della_release():
    import inspect
    import ui.main_window as mw
    src = inspect.getsource(mw.MainWindow._generate_video_native)
    assert 'params["msr_lora"]' in src and "ltx_release" in src
    assert 'vals.get("video_arch")' in src, "senza questo, con H3 cercherebbe la LoRA di LTX"


@pytest.mark.parametrize("arch,imgs,vids", [("ltx", 5, 0), ("minimax_h3", 9, 3)])
def test_limiti_pannello_per_arch(qtbot, arch, imgs, vids):
    from core import user_prefs
    from core.model_catalog import video_caps
    from ui.tabs.video_tab import VideoTab
    prev = user_prefs.get("video_arch", "ltx")
    try:
        t = VideoTab(); qtbot.addWidget(t)
        t._apply_reference_limits(arch, video_caps(arch))
        assert t.reference_panel._lim["image"] == imgs
        assert t.reference_panel._lim["video"] == vids
    finally:
        user_prefs.set("video_arch", prev)
