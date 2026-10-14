"""Unit test del REGISTRO UNICO pesi (weights_catalog schema v2) — CPU-only, no rete.

Invarianti che il v1 violava (e che il CRUD tools/catalog_manager fa rispettare):
  - ogni voce è SCARICABILE (repo_id/url) oppure dichiarata no_remote — mai la terra di
    mezzo che finiva in KeyError silenziato nel downloader;
  - assi tassonomici presenti (category/arch), id unici, deps risolvibili;
  - niente profili utente hardcoded nei tool_dest;
  - le destinazioni COMBACIANO coi path che i motori leggono davvero (tabella attesa);
  - la lista base per l'installer esce dal registro (niente più manifesti paralleli).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import model_paths, weights_catalog as wc  # noqa: E402

BASE = r"E:\AI\Models"


def _all():
    ws = wc.load()
    assert len(ws) >= 100, f"catalogo sospettosamente piccolo: {len(ws)} voci"
    return ws


def test_ids_unici_e_schema():
    ws = _all()
    ids = [e["id"] for e in ws]
    assert len(ids) == len(set(ids)), "id duplicati nel catalogo"
    for e in ws:
        assert e.get("category") in wc.CATEGORIES, f"{e['id']}: category mancante/invalida"
        assert e.get("arch"), f"{e['id']}: arch mancante"
        assert e.get("kind") in ("file", "snapshot"), f"{e['id']}: kind invalido"
        if e["kind"] == "file":
            assert e.get("filename"), f"{e['id']}: kind=file senza filename"


def test_scaricabile_o_no_remote():
    """La classe di bug '9 voci non scaricabili' è morta: fonte remota o dichiarazione onesta."""
    for e in _all():
        assert wc.downloadable(e) or e.get("no_remote"), \
            f"{e['id']}: né fonte remota né no_remote (KeyError garantito nel downloader)"


def test_no_profili_utente_hardcoded():
    for e in _all():
        td = str(e.get("tool_dest", ""))
        assert "\\Users\\" not in td or td.startswith("~"), \
            f"{e['id']}: tool_dest col profilo utente hardcoded: {td}"
    e = model_paths.entry("acestep-3.5b")
    d = wc.dest_for(e)
    assert "~" not in str(d), "dest_for non espande ~ nel tool_dest"


def test_deps_risolvibili():
    ids = {e["id"] for e in _all()}
    for e in _all():
        for d in e.get("deps", []):
            for alt in d.split("|"):                     # «a|b» = dipendenze alternative
                assert alt in ids, f"{e['id']}: dep sconosciuta {alt}"


def test_deps_alternative():
    """«a|b»: basta una delle due; se nessuna è presente si scarica la prima (catena di download)."""
    from core import weights_catalog as wc
    by = {"a": {"id": "a", "kind": "file", "filename": "assente.bin"},
          "b": {"id": "b", "kind": "file", "filename": "assente2.bin"}}
    assert wc.dep_alternatives("a|b") == ["a", "b"] and wc.dep_alternatives("a") == ["a"]
    assert wc.resolve_dep("a|b", by, base=r"Z:\non_esiste") == "a"
    assert wc.resolve_dep("b", by) == "b"


def test_dest_allineati_ai_motori():
    """Le destinazioni del catalogo = i path che i MOTORI leggono (censimento 16/08)."""
    expected = {
        # LTX-2.3 (workers/ltx2_infer.py:764-789, core/ltx_release.py)
        "ltx23-dev-fp8": BASE + r"\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors",
        "ltx23-distilled-lora-384": BASE + r"\LoRA\LTX-2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "ltx23-spatial-upscaler": BASE + r"\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx23-gemma-te": BASE + r"\TextEncoders\gemma-3-ltx",
        "ltx-iclora-canny": BASE + r"\LoRA\LTX-2\ltx-2-19b-ic-lora-canny-control.safetensors",
        "ltx-iclora-lipdub": BASE + r"\LoRA\LTX-2\ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
        # LTX-2.5 (core/ltx_release.py:121-149)
        "ltx25-video-vae-conv": BASE + r"\vae\ltx-2.5-video-vae-conv-bf16.safetensors",
        "ltx25-audio-vae": BASE + r"\vae\ltx-2.5-audio-vae-bf16.safetensors",
        "ltx25-duration-head": BASE + r"\model_patches\ltx-2.5-duration-head-bf16.safetensors",
        "ltx25-distilled-lora-450": BASE + r"\LoRA\LTX-2.5\ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        # immagini (inference_server/backends/image_flux.py, flux2_te.py)
        "flux1-dev-single": BASE + r"\diffusion_models\Flux\FLUX_Dev.safetensors",
        "flux1-fill-single": BASE + r"\diffusion_models\Flux\FLUX_DEV_Fill.safetensors",
        "mistral-te-fp8-single": BASE + r"\TextEncoders\Flux\mistral_3_small_flux2_fp4_mixed.safetensors",
        # aux (engine constants)
        "gfpgan-14": BASE + r"\facerestore_models\GFPGANv1.4.pth",
        "pulid-flux": BASE + r"\pulid\pulid_flux_v0.9.1.safetensors",
        "inswapper-128": BASE + r"\insightface\inswapper_128.onnx",
        "seedvr2-3b-fp8": BASE + r"\SeedVR2\seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "flashvsr-v11": BASE + r"\FlashVSR\FlashVSR-v1.1",
        "sam2-hiera-large": BASE + r"\Segmentation\hf\sam2-hiera-large",
        "grounding-dino-base": BASE + r"\Segmentation\hf\grounding-dino-base",
        "depth-anything-v2": BASE + r"\Segmentation\hf\depth-anything-v2-large",
        "ultrasharp-v2": BASE + r"\Upscale\4x-UltraSharpV2.safetensors",
        "voice-cosyvoice2": BASE + r"\TTS\CosyVoice2-0.5B",
        "joycaption-nf4": BASE + r"\JoyCaption",
    }
    for eid, want in expected.items():
        e = model_paths.entry(eid)
        assert e is not None, f"voce mancante a catalogo: {eid}"
        got = e and (wc.dest_for(e, BASE) / e["filename"] if e.get("kind") == "file" and e.get("filename")
                     else wc.dest_for(e, BASE))
        assert str(got).lower() == want.lower(), f"{eid}: dest {got} ≠ atteso {want}"


def test_base_weights_per_installer():
    """install_role=base + cap attive = ciò che l'installer scarica (rimpiazza install_spec)."""
    base = wc.base_weights_for_edition(allows_adult=True)
    ids = {e["id"] for e in base}
    assert "ltx23-distilled-fp8" in ids, "il checkpoint video base manca dalla lista installer"
    assert "ltx23-gemma-te" in ids, "il TE video base manca dalla lista installer"
    for e in base:
        assert wc.downloadable(e)
    sfw = wc.base_weights_for_edition(allows_adult=False)
    assert all(e.get("sensitivity") != "adult" for e in sfw), "adult presente nella lista SFW"
    assert any(e.get("sensitivity") == "adult" for e in base), \
        "con allows_adult=True le voci adult base devono esserci (gemma-abliterated)"


def test_categorie_e_raggruppamento():
    groups = wc.by_category()
    assert set(groups) <= set(wc.CATEGORIES)
    assert "video" in groups and "ltx23" in groups["video"], "gruppo video/ltx23 assente"
    assert "ltx25" in groups["video"], "gruppo video/ltx25 assente"
    n = sum(len(v) for g in groups.values() for v in g.values())
    assert n == len(wc.load())


def test_model_paths_resolver():
    p = model_paths.path_of("ltx23-dev-fp8")
    assert p is not None and p.name == "ltx-2.3-22b-dev-fp8.safetensors"
    assert model_paths.path_of("id-inesistente") is None
    assert model_paths.models_root()                     # mai vuoto


def test_filtro_18_senza_keyword_nel_nome():
    """I finetune 18+ senza keyword nel nome (PinkCherry, Stubelius) restano fuori dalla modalità SFW:
    vale la sensitivity del registro pesi, sia per nome locale sia per percorso completo."""
    from core import nsfw_policy
    assert wc.is_adult_file("PinkCherry_FineTune_fp8scaled_v1_8_LTX23.safetensors")
    assert nsfw_policy.looks_nsfw(r"E:\AI\Models\diffusion_models\LTX\ltx2.5-Stubelius_remix_v1_distilled_int8_convrot.safetensors")
    assert not nsfw_policy.looks_nsfw("ltx-2.3-22b-dev-fp8.safetensors")
