"""Unit test tooltip pesi + catene di dipendenze (weights-unification T) — CPU-only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import model_paths as mp  # noqa: E402
from core import weights_catalog as wc  # noqa: E402
from core import weights_tooltips as wt  # noqa: E402


def test_use_tags_validi():
    """Ogni tag usato nel catalogo esiste nel vocabolario (niente tag orfani nei tooltip)."""
    for e in wc.load():
        for tag in e.get("use", []):
            assert tag in wt.USE_TAGS, f"{e['id']}: tag sconosciuto '{tag}'"


def test_tooltip_compone_le_parti():
    """Language-agnostic: in headless tr() può risolvere en/it — si verificano i token stabili."""
    e = mp.entry("ltx25-gguf-q4-distilled")
    tt = wt.tooltip_for(e)
    assert "checkpoint" in tt.lower()         # tag ckpt_video (frase it o en)
    assert "GGUF" in tt                       # tag gguf_small
    assert "14.6" in tt                       # dimensione
    assert "ltx25-te-gemma4-gguf" in tt       # riga deps (catena visibile)
    assert "E:" in tt or "Models" in tt       # riga destinazione
    # dev: menziona la LoRA di distillazione obbligatoria
    dev = wt.tooltip_for(mp.entry("ltx23-dev-fp8"))
    assert "distill" in dev.lower()


def test_catene_deps():
    """Le catene: i checkpoint tirano TE/VAE; plan_ids le espande deps-first."""
    from core import downloader as dl
    plan = [x["id"] for x in dl.plan_ids(["ltx25-gguf-q4-distilled"])]
    for dep in ("ltx25-te-gemma4-gguf", "ltx25-video-vae-conv", "ltx25-audio-vae"):
        assert dep in plan, f"catena 2.5 senza {dep}"
        assert plan.index(dep) < plan.index("ltx25-gguf-q4-distilled"), "dep dopo il checkpoint"
    plan23 = [x["id"] for x in dl.plan_ids(["ltx23-dev-fp8"])]
    assert "ltx23-distilled-lora-384" in plan23, "il dev DEVE tirare la distilled-LoRA"
    planf = [x["id"] for x in dl.plan_ids(["flux1-dev-single"])]
    assert "flux-vae-ae" in planf and "t5xxl-fp16" in planf


def test_deps_solo_scaricabili():
    """Mai una dep verso una voce no_remote (il download a catena morirebbe)."""
    by_id = {e["id"]: e for e in wc.load()}
    for e in wc.load():
        for d in e.get("deps", []):
            assert not by_id[d].get("no_remote"), f"{e['id']}: dep no_remote '{d}'"


def test_short_line():
    s = wt.short_line(mp.entry("ltx25-lora-multisubject"))
    assert "Multiple Subject" in s and "GB" in s
