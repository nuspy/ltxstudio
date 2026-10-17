"""Integrazione MiniMax H3 (4ª architettura video) — offscreen, senza pesi né GPU.

Si verifica tutto ciò che non richiede i ~36 GB di pesi: catalogo (file NATIVI, conversione dichiarata),
catene, dispatch verso il venv condiviso, specifiche (fotogrammi 17n+5, turbo, profili espliciti), capability,
sezione H3 del tab Video e l'errore CHIARO del worker quando i pesi mancano.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from core import downloader as dl  # noqa: E402
from core import ltx_video_job as lvj  # noqa: E402
from core import minimax_h3_spec as S  # noqa: E402
from core import model_paths as mp  # noqa: E402
from core import weights_catalog as wc  # noqa: E402
from core.model_catalog import video_caps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _worker():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mmh3w", os.path.join(ROOT, "workers", "minimax_h3_infer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_catalogo_h3_completo_e_nativo():
    ids = {e["id"]: e for e in wc.load() if e.get("arch") == "minimax_h3"}
    for need in ("mmh3-configs", "mmh3-fl2va-nvfp4", "mmh3-ref2va-nvfp4", "mmh3-te-nvfp4", "mmh3-te-fp8",
                 "mmh3-video-vae", "mmh3-audio-vae", "mmh3-turbo-lora-fl2v", "mmh3-turbo-lora-fl2v-8"):
        assert need in ids, f"manca la voce {need}"
    for e in ids.values():
        assert wc.downloadable(e), f"{e['id']} non scaricabile"
        # il predittore Veda ha una sua capability (attenzione sparsa opzionale), non quella base di H3
        # idem l'H3 Video Upsampler (upscaler latente + LoRA di guida): funzione propria, cap propria
        want = (["cap.video_h3_veda"] if e["id"] == "mmh3-veda-r2va"
                else ["cap.video_h3_upsampler"] if e["id"] in ("mmh3-latent-upscaler-v1", "mmh3-lora-h3upscale-x2")
                else ["cap.video_minimax_h3"])
        assert e.get("used_by") == want, f"{e['id']}: gating errato"
        assert "gguf" not in e["id"], "i GGUF H3 sono ritirati (nessun loader nativo)"
        if e.get("kind") == "file" and "Comfy" in (e.get("repo_id") or "") or e.get("convert"):
            # file ComfyUI → convertito al download: il nome locale è quello NATIVO
            assert e.get("convert"), f"{e['id']}: file ComfyUI senza conversione nativa"
            assert e["filename"].endswith("-native.safetensors"), e["filename"]
            assert e["filename"] != os.path.basename(e.get("remote_filename") or "")
    assert ids["mmh3-audio-vae"]["repo_id"] == "MiniMaxAI/MiniMax-H3", "audio VAE: file ufficiale diffusers"
    assert ids["mmh3-te-heretic-nvfp4"].get("sensitivity") == "adult"


def test_catene_per_modalita():
    for ckpt in ("mmh3-fl2va-nvfp4", "mmh3-ref2va-nvfp4"):
        plan = [x["id"] for x in dl.plan_ids([ckpt])]
        assert plan[-1] == ckpt, f"{ckpt} deve venire dopo le sue dipendenze"
        for dep in ("mmh3-configs", "mmh3-video-vae", "mmh3-audio-vae"):
            assert dep in plan, f"{ckpt}: manca {dep}"
        assert any(t in plan for t in ("mmh3-te-nvfp4", "mmh3-te-fp8", "mmh3-te-int8")), f"{ckpt}: manca un TE"


def test_ogni_dit_h3_ha_tutte_le_alternative_te():
    """Il TE e' indipendente dalla DiT (worker legge solo W['te']; spec accetta ogni TE_IDS): tutte le DiT H3 ->
    gruppo con nvfp4, fp8 e int8 (prova in build/catalog_use_deps.py)."""
    dits = [e for e in wc.load() if e.get("arch") == "minimax_h3" and e.get("type") == "diffusion_models"]
    assert len(dits) >= 12
    for e in dits:
        groups = [d for d in e["deps"] if d.startswith("mmh3-te-")]
        assert len(groups) == 1, (e["id"], e["deps"])
        assert set(wc.dep_alternatives(groups[0])) == {"mmh3-te-nvfp4", "mmh3-te-fp8", "mmh3-te-int8"}, e["id"]
    gb = sum(float(mp.entry(i).get("approx_gb", 0)) for i in [x["id"] for x in dl.plan_ids(["mmh3-fl2va-nvfp4"])])
    assert 30 < gb < 40, f"scarico minimo inatteso: {gb:.1f} GB"


def test_configs_senza_pesi():
    """I config ufficiali (layout diffusers) NON devono tirarsi dietro i pesi bf16 (124 GB)."""
    e = mp.entry("mmh3-configs")
    assert e["kind"] == "snapshot" and e["repo_id"] == "MiniMaxAI/MiniMax-H3"
    assert e["allow_patterns"], "senza allow_patterns scaricherebbe l'intero repo"
    assert not any(p.endswith((".safetensors", "*")) and "/" not in p for p in e["allow_patterns"])
    assert not any("safetensors" in p for p in e["allow_patterns"])
    assert float(e["approx_gb"]) < 0.1


def test_dispatch_arch_venv_condiviso():
    py, worker = lvj.runner_for({"arch": "minimax_h3"})
    assert worker.endswith("minimax_h3_infer.py") and os.path.exists(worker)
    assert "diffusers_next_venv" in py, py


def test_spec_fotogrammi_e_turbo():
    for s in (3, 5, 6.5, 10, 15, 30):
        n = S.snap_frames(s)
        assert (n - 5) % 17 == 0, n
        assert S.MIN_S - 1e-6 <= n / S.FPS <= S.MAX_S + 1e-6, (s, n)
    assert S.TURBO["4"][1] == 4 and S.TURBO["8"][1] == 8 and S.TURBO["8"][2] == 6.0
    assert S.dialogue_tag("Italian", " Ciao! ") == "<d>[Italian] Ciao!</d>"


def test_id_espliciti_nessun_ripiego():
    ids = S.weight_ids({"h3_profile": "fp8"})
    assert ids["dit"] == "mmh3-fl2va-fp8" and ids["te"] == "mmh3-te-fp8"
    ids = S.weight_ids({"h3_profile": "nvfp4", "references": [{"path": "a.png"}], "h3_turbo": "4"})
    assert ids["dit"] == "mmh3-ref2va-nvfp4" and ids["turbo"] == "mmh3-turbo-lora-ref2v"
    with pytest.raises(ValueError):                    # l'8 passi non esiste per Ref2VA: errore, non ripiego
        S.weight_ids({"references": [{"path": "a.png"}], "h3_turbo": "8"})
    with pytest.raises(ValueError):
        S.weight_ids({"h3_profile": "gguf"})


def test_capability():
    caps = video_caps("minimax_h3")
    assert caps["audio"] and caps["multiref"] and caps["t2v"] and caps["i2v"]
    assert not caps["controlnet"] and not caps["v2v"] and not caps["storyboard"]
    assert video_caps("lingbot")["multiref"] is False


def test_sezione_h3_tab_video(qtbot):
    """Con H3: profili nel dropdown modello, sezione H3 visibile, 24 fps fissi, durata 5–15 s, negative e
    CFG spenti; tornando a LTX tutto si ripristina. La preferenza di arch va ripristinata a fine test."""
    from core import user_prefs
    from ui.tabs.video_tab import VideoTab
    prev = user_prefs.get("video_arch", "ltx")
    try:
        t = VideoTab(); qtbot.addWidget(t)
        t.video_arch.setCurrentIndex(t.video_arch.findData("ltx"))
        t.length.setValue(30.0)
        t.video_arch.setCurrentIndex(t.video_arch.findData("minimax_h3"))
        assert not t.h3_box.isHidden()
        kinds = {(t.model.itemData(i) or {}).get("kind") for i in range(t.model.count())}
        assert kinds == {"minimax_h3"}, kinds
        assert not t.fps.isEnabled() and t.fps.value() == 24
        assert t.length.maximum() == 15.0 and t.length.value() == 15.0
        assert not t.neg.isEnabled() and not t.sampler_seed.cfg.isEnabled()
        assert t.reference_panel._lim == {"image": 9, "video": 3, "audio": 3}
        t.h3_turbo.setCurrentIndex(t.h3_turbo.findData("8"))
        assert t.sampler_seed.steps.value() == 8 and not t.sampler_seed.steps.isEnabled()
        t.h3_dialogue.add_row("Anna", "Italian", "Chi ha nascosto il telecomando nel frigo?")
        t.h3_soundscape.setText("frigorifero che ronza")
        v = t.get_values()
        assert v["video_arch"] == "minimax_h3" and v["h3_turbo"] == "8"
        assert v["h3_dialogue"][0]["text"].startswith("Chi ha nascosto")
        assert v["h3_profile"] in S.PROFILES
        heretic = t.h3_te.findData("mmh3-te-heretic-nvfp4")
        t.nsfw.setCurrentText("SFW")
        assert t.h3_te.view().isRowHidden(heretic)
        t.nsfw.setCurrentText("NSFW")
        assert not t.h3_te.view().isRowHidden(heretic)
        t.video_arch.setCurrentIndex(t.video_arch.findData("ltx"))
        assert t.h3_box.isHidden() and t.fps.isEnabled() and t.neg.isEnabled()
        assert t.length.value() == 30.0, "la durata LTX va ripristinata"
    finally:
        user_prefs.set("video_arch", prev)


def test_worker_errore_chiaro_senza_pesi(monkeypatch):
    """Senza pesi il worker dice QUALI id mancano e dove rimediare (nessun ripiego su altri profili)."""
    m = _worker()
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: False)
    with pytest.raises(RuntimeError) as ei:
        m.resolve({"prompt": "x", "h3_profile": "fp8"})
    msg = str(ei.value)
    assert "mancanti" in msg.lower() and "mmh3-fl2va-fp8" in msg and "mmh3-te-fp8" in msg, msg
    assert "Pesi modelli" in msg


def test_worker_prompt_formato_ufficiale():
    """I2VA: istruzione del keyframe + i tre campi; speaker (S1) con il modo di dire FUORI da <d>."""
    m = _worker()
    txt = m.build_prompt({"prompt": "Una cuoca litiga con un robot aspirapolvere", "image_path": "x.png",
                          "h3_dialogue": [{"speaker": "Cuoca", "language": "Italian",
                                           "text": "[emotion:angry]Fuori dalla mia cucina![/emotion]"}],
                          "h3_soundscape": "sfrigolio di padelle", "h3_music": ""}, 6.58)
    assert txt.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1>")
    assert "Cuoca (S1) says, angrily, <d>[Italian] Fuori dalla mia cucina!</d>" in txt
    assert "overall_soundscape: sfrigolio di padelle" in txt and "non_diegetic_music: N/A" in txt
    assert "[emotion" not in txt


def test_worker_prompt_ref2va_timbro():
    m = _worker()
    txt = m.build_prompt({"prompt": "Lei racconta un segreto",
                          "references": [{"kind": "image", "path": "a.png", "role": "the woman with red hair"},
                                         {"kind": "audio", "path": "v.wav"}],
                          "h3_dialogue": [{"language": "Italian", "text": "Non dirlo a [emph]nessuno[/emph]!"}]}, 8.0)
    for sec in ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:"):
        assert sec in txt, sec
    assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)." in txt
    assert "<d>[Italian] Non dirlo a nessuno!</d>" in txt


def test_prompt_gia_strutturato_passa_invariato():
    m = _worker()
    raw = "\n\n".join(["integrated_multimodal_description: [Shot 1] x", "overall_soundscape: y",
                       "non_diegetic_music: N/A"])
    assert m.build_prompt({"prompt": raw}, 5.0) == raw


def test_markup_voce():
    from core import voice_markup as V
    segs = V.parse("Ciao [pause:800ms] [whisper]segreto[/whisper] [accel:0.9-1.4]uno, due, tre![/accel]")
    kinds = [s.kind for s in segs]
    assert kinds == ["text", "pause", "text", "text"] and segs[1].ms == 800
    assert segs[2].style.emotion == "whisper" and V.effective(segs[2].style)[2] < -10
    ch = V.accel_chunks(segs[3])
    assert len(ch) == 3 and ch[0].style.speed < ch[-1].style.speed
    assert V.plain("[emotion:happy]Evviva[/emotion] [pause] , amici!") == "Evviva, amici!"
    p = V.parse('<prosody rate="+20%" pitch="+2st" volume="-3dB">x</prosody>')[0].style
    assert abs(p.speed - 1.2) < 1e-6 and p.pitch == 2.0 and p.volume_db == -3.0
    assert V.delivery_en("[speed:1.3][loud]Presto![/loud][/speed]") == "quickly, loudly"
