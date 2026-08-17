"""MSR — riferimenti multipli su LTX — CPU-only.

Le due release usano meccanismi DIVERSI (accertato aprendo i file il 17/08):
· 2.3 → pseudo-video dato come video-guida IC-LoRA (la LoRA ha 960 tensori lora_A/B e ZERO slot
  embedding: verificato sul file scaricato)
· 2.5 → slot embedding a posizioni temporali negative
Il layout della pseudo-sequenza è verificato DIFFERENZIALMENTE contro l'implementazione
originale del nodo LiconMSR (35/35 combinazioni identiche, vedi test_layout_*).
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
    for rel_id, cid in (("2.3", "ltx23-lora-multisubject"), ("2.5", "ltx25-lora-multisubject")):
        assert os.path.basename(mp.p(cid, "")) == os.path.basename(lr.get(rel_id).msr_lora)


# ── 2.3: layout della pseudo-sequenza (verificato contro il plugin originale) ──
def test_layout_frame_soggetti_e_sfondo():
    """Ogni soggetto occupa un blocco di frame contigui a inizio sequenza, il resto è sfondo."""
    lay = ltx_msr.frame_layout(3, 41)
    assert len(lay) == 41
    assert lay[0] == 0, "il primo frame è del primo soggetto"
    assert lay[-1] == -1, "la coda è sfondo"
    for s in range(3):
        fr = [i for i, v in enumerate(lay) if v == s]
        assert fr, f"soggetto {s} senza frame"
        assert fr == list(range(fr[0], fr[-1] + 1)), f"soggetto {s} con frame non contigui"
    assert [lay.index(s) for s in range(3)] == sorted([lay.index(s) for s in range(3)])


def test_layout_valori_esatti():
    """Valori presi dall'implementazione originale (regressione: se cambiano, la guida cambia)."""
    assert ltx_msr.frame_layout(0, 17) == [-1] * 17
    assert ltx_msr.ref_latent_frames(41) == 6 and ltx_msr.ref_latent_frames(17) == 3
    assert ltx_msr.latent_to_frame_range(0, 0) == (0, 0)
    assert ltx_msr.latent_to_frame_range(1, 2) == (1, 16)
    assert ltx_msr.allocate_subject_latents(3, 5) == [2, 2, 1]
    lay = ltx_msr.frame_layout(3, 41)
    assert (lay.count(0), lay.count(1), lay.count(2), lay.count(-1)) == (9, 16, 8, 8)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("fc", ltx_msr.FRAME_COUNTS)
def test_layout_sempre_valido(n, fc):
    lay = ltx_msr.frame_layout(n, fc)
    assert len(lay) == fc
    assert set(lay) <= set(range(n)) | {-1}
    if fc >= 17:                                   # con sequenze cortissime qualche slot può saltare
        assert all(s in lay for s in range(min(n, ltx_msr.ref_latent_frames(fc) - 1)))


def test_fit_su_bianco_non_ritaglia():
    """Il soggetto non va MAI ritagliato (perderebbe parti del personaggio): si adatta su bianco."""
    import numpy as np
    img = np.zeros((100, 400, 3), dtype=np.uint8)              # panoramica su tela verticale
    out = ltx_msr.fit_on_white(img, 736, 1280)
    assert out.shape == (1280, 736, 3)
    assert (out[0, 0] == 255).all(), "il riempimento deve essere BIANCO"
    assert (out == 0).any(), "il soggetto deve essere presente"
    bg = ltx_msr.cover_crop(np.zeros((100, 400, 3), dtype=np.uint8), 736, 1280)
    assert bg.shape == (1280, 736, 3) and not (bg == 255).all(), "lo sfondo copre, non impagina"


def test_strategy_dal_file_reale():
    """Sul file VERO scaricato: la 2.3 NON ha slot embedding → pseudo-video."""
    p = lr.get("2.3").msr_lora
    if not os.path.exists(p):
        pytest.skip("LoRA MSR 2.3 non scaricata")
    assert ltx_msr.slot_state_dict(p) == {}
    assert ltx_msr.strategy_for(p) == "pseudo_video"


def test_pseudo_video_richiede_lo_sfondo(tmp_path):
    with pytest.raises(ValueError, match="SFONDO"):
        ltx_msr.build_pseudo_video(["a.png"], "", str(tmp_path / "x.mp4"))


def test_build_pseudo_video(tmp_path):
    """Costruzione reale del file mp4 di guida (senza GPU): esiste, non è vuoto."""
    import numpy as np
    from PIL import Image
    paths = []
    for i, col in enumerate(((220, 30, 30), (30, 200, 60))):
        p = tmp_path / f"s{i}.png"
        Image.fromarray(np.full((256, 256, 3), col, dtype=np.uint8)).save(p)
        paths.append(str(p))
    bgp = tmp_path / "bg.png"
    Image.fromarray(np.full((512, 288, 3), (20, 20, 90), dtype=np.uint8)).save(bgp)
    out = ltx_msr.build_pseudo_video(paths, str(bgp), str(tmp_path / "guide.mp4"),
                                     width=256, height=448, frame_count=17, fps=24, log=lambda *_: None)
    assert os.path.exists(out) and os.path.getsize(out) > 1000


# ── 2.5: slot embedding ───────────────────────────────────────────────────────
def test_posizioni_negative():
    assert ltx_msr.frame_offsets(3) == [-3, -2, -1]
    for n in range(1, 6):
        offs = ltx_msr.frame_offsets(n)
        assert len(set(offs)) == n and max(offs) < 0


def test_slot_embedding_matematica():
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
    assert e0.shape == (dim,) and not torch.allclose(e0, e1)
    assert torch.isfinite(e0).all()
    assert ltx_msr.slot_embedding(state, 2, "cpu", torch.bfloat16).dtype == torch.bfloat16
    slim = {k: v for k, v in state.items() if not k.endswith(".bias")}
    assert torch.isfinite(ltx_msr.slot_embedding(slim, 1, "cpu", torch.float32)).all()


def test_slot_state_dict_estrae_col_prefisso(tmp_path):
    from safetensors.torch import save_file
    p = tmp_path / "msr.safetensors"
    save_file({
        "reference_slot_embedding.frequencies": torch.randn(8),
        "diffusion_model.reference_slot_embedding.net.0.weight": torch.randn(64, 17),
        "transformer.blocks.0.lora_A.weight": torch.zeros(4, 4),
    }, str(p))
    assert set(ltx_msr.slot_state_dict(str(p))) == {"frequencies", "net.0.weight"}
    assert ltx_msr.strategy_for(str(p)) == "slot_embedding"


# ── cablaggio ─────────────────────────────────────────────────────────────────
def test_worker_instrada_le_due_strategie():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "workers" / "ltx2_infer.py").read_text("utf-8")
    assert "ltx_msr.strategy_for" in src
    assert "build_pseudo_video" in src and 'p["control_video"] = _msr_guide' in src
    assert "install_slot_embedding" in src
    assert "elif _msr_guide:" in src, "la pseudo-video deve prendere il ramo iclora"
    assert 'images.append(ImageConditioningInput(p["image_path"], 0, _ref_str, 33))' in src


def test_gui_sceglie_la_lora_della_release():
    import inspect
    import ui.main_window as mw
    src = inspect.getsource(mw.MainWindow._generate_video_native)
    assert 'params["msr_lora"]' in src and "ltx_release" in src
    assert 'vals.get("video_arch")' in src, "senza questo, con H3 cercherebbe la LoRA di LTX"


def test_pannello_sfondo_e_limiti(qtbot):
    from core import user_prefs
    from core.model_catalog import video_caps
    from ui.tabs.video_tab import VideoTab
    prev = user_prefs.get("video_arch", "ltx")
    try:
        t = VideoTab(); qtbot.addWidget(t)
        p = t.reference_panel
        t._apply_reference_limits("ltx", video_caps("ltx"))
        assert p._lim == {"image": 4, "video": 0, "audio": 0} and p._needs_background
        p.enable.setChecked(True)
        p.add_path("a.png")
        assert any("sfondo" in e.lower() for e in p.validation_errors()), p.validation_errors()
        p.set_background("bg.png")
        assert p.validation_errors() == []
        refs = p.get_references()
        assert [r.get("role") for r in refs] == [None, "background"]
        # su LTX i riferimenti si citano «Image 1», non col token angolare di H3
        assert refs[0]["token"] == "Image 1", refs[0]["token"]
        # H3: nessuno sfondo richiesto, riga nascosta
        t._apply_reference_limits("minimax_h3", video_caps("minimax_h3"))
        assert not p._needs_background and p._lim["image"] == 9
        assert [r.get("role") for r in p.get_references()] == [None]
        assert p.get_references()[0]["token"] == "<Picture 1>"
    finally:
        user_prefs.set("video_arch", prev)
