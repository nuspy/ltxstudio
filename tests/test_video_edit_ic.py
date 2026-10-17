"""core/video_edit_ic.py: build_job/finalize/checkpoint_info su video sintetici (nessuna GPU, nessun Qt)."""
import os
import shutil
import subprocess

import pytest

from core import video_edit_ic as vic
from core import video_edit_modes as vem
from core import video_ic_prep as P

_HAVE_FFMPEG = bool(shutil.which(P.ffmpeg_path()) or os.path.exists(P.ffmpeg_path()))
pytestmark = pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg non disponibile")


def _gen(out, n, size="320x192", audio=False):
    cmd = [P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-shortest"]
    cmd += ["-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


@pytest.fixture(autouse=True)
def _weights_present(monkeypatch):
    monkeypatch.setattr(vic, "missing_weight", lambda spec: "")


def test_checkpoint_info_states():
    assert vic.checkpoint_info(None)["state"] == "unknown"
    assert vic.checkpoint_info({"model_kind": "auto", "model": "auto"})["state"] == "auto"
    d = vic.checkpoint_info({"model_kind": "diffusion", "model": r"E:\m\ltx-2.3-22b-distilled-1.1.safetensors"})
    assert d["state"] == "distilled" and d["params"]["model"].endswith("distilled-1.1.safetensors")
    e = vic.checkpoint_info({"model_kind": "diffusion", "model": r"E:\m\ltx-2.3-22b-dev-fp8.safetensors"})
    assert e["state"] == "dev" and not e["distilled"]
    g = vic.checkpoint_info({"model_kind": "gguf", "model_gguf": r"E:\m\ltx-2.3-distilled-Q4_K_M.gguf"})
    assert g["state"] == "distilled" and "gguf_transformer" in g["params"] and "model" not in g["params"]
    assert vic.checkpoint_info({"model_kind": "minimax_h3", "model": ""})["state"] == "other_arch"


def test_checkpoint_info_dlora_flag(monkeypatch):
    """Col dev il worker aggiunge la distilled-LoRA: `dlora_ok` dice se il file e' su disco (mai download)."""
    dev = {"model_kind": "diffusion", "model": r"E:\m\ltx-2.3-22b-dev-fp8.safetensors"}
    monkeypatch.setattr(vic.os.path, "exists", lambda p: True)
    assert vic.checkpoint_info(dev)["dlora_ok"] is True
    monkeypatch.setattr(vic.os.path, "exists", lambda p: False)
    i = vic.checkpoint_info(dev)
    assert i["dlora_ok"] is False and i["dlora_path"].endswith(".safetensors") and "distilled-lora" in i["dlora_path"]
    assert vic.checkpoint_info(dev)["state"] == "dev", "il checkpoint scelto non cambia"


def test_restore_job_no_copy_and_audio(tmp_path):
    src = _gen(tmp_path / "s.mp4", 50, audio=True)
    ij = vic.build_job({"mode": "refocus", "video": src, "prompt": ""}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert p["control_video"] == src, "il restauro usa il sorgente senza copiarlo"
    assert p["control_type"] == "refocus" and p["edit_lora"] == "" and p["_strict_ic"] is True
    assert p["num_frames"] == 49 and (p["width"], p["height"]) == (320, 192)
    assert "chunk_frames" not in p and ij.mux_audio_from == src


def test_checkpoint_passed_as_chosen(tmp_path):
    src = _gen(tmp_path / "s.mp4", 17)
    ij = vic.build_job({"mode": "uncompress", "video": src, "model": "X/ltx-2.3-22b-dev-fp8.safetensors",
                        "ltx_release": "2.3"}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    assert ij.params["model"] == "X/ltx-2.3-22b-dev-fp8.safetensors" and ij.params["ltx_release"] == "2.3"
    ij2 = vic.build_job({"mode": "uncompress", "video": src}, str(tmp_path), 2, str(tmp_path / "o2.mp4"))
    assert "model" not in ij2.params                      # nessuna scelta automatica


def test_long_clip_segments_and_cap(tmp_path):
    src = _gen(tmp_path / "l.mp4", 200, size="1920x1088")
    ij = vic.build_job({"mode": "refocus", "video": src}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert max(p["width"], p["height"]) <= 1280 and p["width"] % 64 == 0 and p["height"] % 64 == 0
    assert p["chunk_frames"] == 97 and p["overlap_frames"] >= 8 and ij.segments >= 2
    assert any("ridotto" in n for n in ij.notes)


def test_outpaint_and_finalize(tmp_path):
    src = _gen(tmp_path / "v.mp4", 25, size="192x320")           # verticale
    out = str(tmp_path / "o.mp4")
    ij = vic.build_job({"mode": "outpaint", "video": src, "out_format": "16:9", "out_dark": True,
                        "out_paste": True, "keep_audio": True}, str(tmp_path), 1, out)
    lay = ij.layout
    assert (lay.canvas_w, lay.canvas_h) == (ij.params["width"], ij.params["height"])
    assert lay.canvas_w % 64 == 0 and lay.canvas_w > lay.canvas_h and lay.new_area > 0
    guide = ij.params["control_video"]
    gi = P.probe(guide, count_frames=True)
    assert (gi.width, gi.height, gi.nb_frames) == (lay.canvas_w, lay.canvas_h, 25)
    assert ij.gamma == 2.0 and ij.paste_original and ij.feather == P.DEFAULT_FEATHER      # feather di default attivo
    # il «risultato del worker» simulato = la guida stessa: finalize applica gamma inverso + originale incollato
    shutil_out = str(tmp_path / "gen.mp4")
    shutil.copy(guide, shutil_out)
    final = vic.finalize(ij, shutil_out, cleanup=False)
    fi = P.probe(final, count_frames=True)
    assert (fi.width, fi.height) == (lay.canvas_w, lay.canvas_h) and fi.nb_frames >= 24
    # custom margini senza area -> errore chiaro
    with pytest.raises(vic.IcJobError):
        vic.build_job({"mode": "outpaint", "video": src, "out_format": "custom", "out_margins": (0, 0, 0, 0)},
                      str(tmp_path), 3, out)


def test_restyle_short_vs_long(tmp_path):
    from PIL import Image
    img = str(tmp_path / "style.png")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(img)
    src = _gen(tmp_path / "s.mp4", 33)
    ij = vic.build_job({"mode": "restyle", "video": src, "prompt": "Make it Disney 2D Animation style.",
                        "style_image": img}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    assert ij.params["control_video"].endswith("guide_f0.mp4") and "ic_frame0_image" not in ij.params
    fr = list(P._read_frames(ij.params["control_video"], 320, 192))
    assert fr[0].mean(axis=(0, 1))[0] > 150                        # frame 0 = stile (rosso)
    long_src = _gen(tmp_path / "l.mp4", 200)
    ij2 = vic.build_job({"mode": "restyle", "video": long_src, "prompt": "x", "style_image": img},
                        str(tmp_path), 2, str(tmp_path / "o2.mp4"))
    assert ij2.params["ic_frame0_image"] == img and ij2.params["control_video"] == long_src
    with pytest.raises(vic.IcJobError):
        vic.build_job({"mode": "restyle", "video": src, "prompt": "x"}, str(tmp_path), 3, str(tmp_path / "o3.mp4"))


def test_cameraman_job(tmp_path):
    ref = _gen(tmp_path / "r.mp4", 49)
    img = str(tmp_path / "i.png")
    from PIL import Image
    Image.new("RGB", (64, 64)).save(img)
    ij = vic.build_job({"mode": "cameraman", "cam_ref": ref, "cam_img": img, "cam_img_strength": 0.6,
                        "width": 960, "height": 576, "cam_seconds": 1.0, "prompt": ""},
                       str(tmp_path), 1, str(tmp_path / "o.mp4"))
    p = ij.params
    assert p["control_video"] == ref and p["image_path"] == img and p["ref_strength"] == 0.6
    assert (p["width"], p["height"]) == (960, 576) and p["num_frames"] == 25 and ij.mux_audio_from == ""
    with pytest.raises(vic.IcJobError):
        vic.build_job({"mode": "cameraman"}, str(tmp_path), 2, str(tmp_path / "o2.mp4"))


def test_disabled_and_unknown_modes_rejected(tmp_path):
    src = _gen(tmp_path / "s.mp4", 17)
    for m in ("nope", "edit"):
        with pytest.raises(vic.IcJobError):
            vic.build_job({"mode": m, "video": src}, str(tmp_path), 1, str(tmp_path / "o.mp4"))


def test_outpaint_feather_cfg_and_min_bar(tmp_path):
    src = _gen(tmp_path / "v.mp4", 17, size="640x384")
    out = str(tmp_path / "o.mp4")
    base = {"mode": "outpaint", "video": src, "out_format": "4:3", "keep_audio": False}
    a = vic.build_job({**base, "out_feather": 40}, str(tmp_path), 1, out)
    assert a.feather == 40 and a.layout.scale == 1.0
    b = vic.build_job({**base, "out_feather": 0}, str(tmp_path), 2, out)
    assert b.feather == 0
    c = vic.build_job({**base, "out_feather": 5000, "out_paste": True}, str(tmp_path), 3, out)
    assert c.feather == P.MAX_FEATHER
    d = vic.build_job({**base, "out_paste": False}, str(tmp_path), 4, out)
    assert d.feather == 0 and not d.paste_original
    e = vic.build_job({**base, "out_min_bar": 0.15}, str(tmp_path), 5, out)
    assert e.layout.scale < 1.0 and not e.layout.narrow_bar and e.layout.new_area > a.layout.new_area
    assert a.layout.narrow_bar and any("stretta" in n for n in a.notes)


# ── lip dub (fase 1b) ────────────────────────────────────────────────────────────────────────────
def test_compose_lipdub_prompt():
    assert vic.compose_lipdub_prompt("Ciao a tutti!", "it", "a man in a kitchen") == \
        'a man in a kitchen, speaking in Italian saying: "Ciao a tutti!"'
    assert vic.compose_lipdub_prompt("Hola", "", "") == 'A person, speaking saying: "Hola"'
    assert vic.compose_lipdub_prompt('say "x"  now', "Klingon", "") == \
        "A person, speaking in Klingon saying: \"say 'x' now\""         # lingua libera passa; virgolette neutralizzate
    assert vic.compose_lipdub_prompt("   ", "it", "x") == ""


def test_lipdub_text_fit():
    assert vic.lipdub_text_fit("", 5)[0] == "unknown" and vic.lipdub_text_fit("a b", 0)[0] == "unknown"
    assert vic.lipdub_text_fit(" ".join(["parola"] * 12), 5)[0] == "ok"        # ~4.8 s su 5 s
    assert vic.lipdub_text_fit("ciao", 5)[0] == "short"
    assert vic.lipdub_text_fit(" ".join(["parola"] * 30), 5)[0] == "long"
    assert vic.estimate_speech_seconds("你好，欢迎大家来到这里") > 1.5           # CJK: caratteri/s


def test_lipdub_job(tmp_path):
    src = _gen(tmp_path / "a.mp4", 49, audio=True)
    out = str(tmp_path / "o.mp4")
    ij = vic.build_job({"mode": "lipdub", "video": src, "lipdub_line": "Buongiorno a tutti, benvenuti nel mio canale.",
                        "lipdub_lang": "it", "strength": 0.3, "keep_audio": True, "model": "E:/m/x-dev-fp8.safetensors"},
                       str(tmp_path), 1, out)
    p = ij.params
    assert p["control_type"] == "lipdub" and p["control_video"] == src and p["_strict_ic"] is True
    assert p["prompt"] == 'A person, speaking in Italian saying: "Buongiorno a tutti, benvenuti nel mio canale."'
    assert p["control_strength"] == 1.0                      # forza fissa (ufficiale): la cfg non la cambia
    assert ij.mux_audio_from == "" and p["model"] == "E:/m/x-dev-fp8.safetensors"   # voce NUOVA, checkpoint come scelto
    assert p["num_frames"] == 49 and (p["width"], p["height"]) == (320, 192) and ij.segments == 1
    assert any("durata dell'uscita" in n for n in ij.notes)
    # prompt libero vince su battuta/lingua/scena
    ij2 = vic.build_job({"mode": "lipdub", "video": src, "prompt": "custom", "lipdub_line": "ignorata"},
                        str(tmp_path), 2, out)
    assert ij2.params["prompt"] == "custom"
    # battuta lunga per 2 s di video: avviso
    ij3 = vic.build_job({"mode": "lipdub", "video": src, "lipdub_line": " ".join(["parola"] * 40)},
                        str(tmp_path), 3, out)
    assert any("battuta lunga" in n for n in ij3.notes)


def test_lipdub_rejects(tmp_path):
    mute = _gen(tmp_path / "m.mp4", 49)
    with pytest.raises(vic.IcJobError, match="audio"):
        vic.build_job({"mode": "lipdub", "video": mute, "lipdub_line": "x"}, str(tmp_path), 1, str(tmp_path / "o.mp4"))
    src = _gen(tmp_path / "a.mp4", 49, audio=True)
    with pytest.raises(vic.IcJobError, match="battuta"):
        vic.build_job({"mode": "lipdub", "video": src}, str(tmp_path), 2, str(tmp_path / "o.mp4"))
    longv = _gen(tmp_path / "l.mp4", 300, audio=True)                # 300 frame > 249: non si spezza
    with pytest.raises(vic.IcJobError, match="tetto"):
        vic.build_job({"mode": "lipdub", "video": longv, "lipdub_line": "x"}, str(tmp_path), 3, str(tmp_path / "o.mp4"))


# ── lip dub: lingue, variante da audio (B), traduzione (C) ───────────────────────────────────────
def test_spoken_languages_table():
    codes = [c for c, _n, _e in vic.LIPDUB_SPOKEN_LANGS]
    assert {"en", "es", "fr", "de", "ru", "zh", "it", "pt", "ja", "ko"} == set(codes) and tuple(codes) == vic.SPOKEN_CODES
    assert all(e for _c, _n, e in vic.LIPDUB_SPOKEN_LANGS), "ogni lingua ha la sua prova"
    assert vic.LIPDUB_LANGUAGES[0] == ("", "") and len(vic.LIPDUB_LANGUAGES) == len(codes) + 1


def test_compose_lipdub_audio_prompt():
    assert vic.compose_lipdub_audio_prompt() == "A person, speaking"
    assert vic.compose_lipdub_audio_prompt(scene="a singer on stage,") == "a singer on stage, speaking"
    assert vic.compose_lipdub_audio_prompt("la la", "it", "x") == 'x, speaking in Italian saying: "la la"'


def _gen_audio(out, seconds):
    subprocess.run([P.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=300:sample_rate=22050:duration={seconds}", str(out)], check=True, capture_output=True)
    return str(out)


def test_lipdub_audio_job_equal_and_policies(tmp_path):
    src = _gen(tmp_path / "v.mp4", 49, audio=True)                   # 1.96 s
    out = str(tmp_path / "o.mp4")
    a = _gen_audio(tmp_path / "a.wav", 2.0)
    ij = vic.build_job({"mode": "lipdub_audio", "video": src, "lipdub_audio": a, "lipdub_scene": "a singer"},
                       str(tmp_path), 1, out)
    p = ij.params
    assert p["control_type"] == "lipdub" and p["control_video"] == src and p["_strict_ic"] is True
    assert p["lipdub_audio"].endswith("ext_audio.wav") and p["lipdub_audio_ref"] == "external"
    assert p["prompt"] == "a singer, speaking" and p["num_frames"] == 49 and ij.mux_audio_from == ""
    assert abs(P.audio_duration(p["lipdub_audio"]) - 49 / 25.0) < 0.08, "audio preparato alla durata del video"
    assert any("SPERIMENTALE" in n for n in ij.notes)
    # audio PIU' CORTO del video (1 s su ~4 s): l'audio comanda, il video e' tagliato (8k+1) e la guida e' il taglio
    src2 = _gen(tmp_path / "v2.mp4", 97, audio=True)                 # 3.88 s
    a2 = _gen_audio(tmp_path / "a2.wav", 1.0)
    ij2 = vic.build_job({"mode": "lipdub_audio", "video": src2, "lipdub_audio": a2}, str(tmp_path), 2, out)
    assert ij2.params["num_frames"] == 25 and ij2.params["control_video"].endswith("guide_cut.mp4")
    assert P.probe(ij2.params["control_video"], count_frames=True).nb_frames == 25
    assert any("tagliato" in n for n in ij2.notes)
    # audio PIU' LUNGO (6 s su ~2 s): audio tagliato alla durata del video, il video non cambia
    a3 = _gen_audio(tmp_path / "a3.wav", 6.0)
    ij3 = vic.build_job({"mode": "lipdub_audio", "video": src, "lipdub_audio": a3, "lipdub_audio_ref": "original"},
                        str(tmp_path), 3, out)
    assert ij3.params["control_video"] == src and ij3.params["lipdub_audio_ref"] == "original"
    assert abs(P.audio_duration(ij3.params["lipdub_audio"]) - 49 / 25.0) < 0.08 and any("tagliato" in n for n in ij3.notes)


def test_lipdub_audio_rejects(tmp_path):
    src = _gen(tmp_path / "v.mp4", 49, audio=False)
    out = str(tmp_path / "o.mp4")
    with pytest.raises(vic.IcJobError, match="audio mancante"):
        vic.build_job({"mode": "lipdub_audio", "video": src}, str(tmp_path), 1, out)
    a = _gen_audio(tmp_path / "a.wav", 2.0)
    ij = vic.build_job({"mode": "lipdub_audio", "video": src, "lipdub_audio": a}, str(tmp_path), 2, out)   # video muto: ok (ref external)
    assert ij.params["lipdub_audio_ref"] == "external"
    with pytest.raises(vic.IcJobError, match="originale"):
        vic.build_job({"mode": "lipdub_audio", "video": src, "lipdub_audio": a, "lipdub_audio_ref": "original"},
                      str(tmp_path), 3, out)
    with pytest.raises(vic.IcJobError, match="traccia audio"):
        vic.build_job({"mode": "lipdub_audio", "video": src, "lipdub_audio": src}, str(tmp_path), 4, out)   # un video muto non e' audio


# C: traduzione automatica (mock di STT e LLM: nessuna GPU)
class _FakeLLM:
    def __init__(self, answers, up=True, provider="lmstudio"):
        self.answers, self.up, self.provider, self.calls, self.unloaded = list(answers), up, provider, [], 0

    def available(self):
        return self.up

    def get_config(self):
        return {"provider": self.provider, "lmstudio_port": 1234}

    def ensure_lmstudio(self):
        return False

    def complete_json(self, system, user, **kw):
        self.calls.append(user)
        return self.answers.pop(0)

    def unload(self):
        self.unloaded += 1


def _patch_llm(monkeypatch, fake):
    import core.llm_service as real
    for n in ("available", "get_config", "ensure_lmstudio", "complete_json", "unload"):
        monkeypatch.setattr(real, n, getattr(fake, n))


def test_translate_fit_retry_and_unload(monkeypatch):
    from core import video_dub_translate as vdt
    long_t = " ".join(["parola"] * 30)
    fake = _FakeLLM([{"source_language": "it", "translation": long_t},
                     {"source_language": "it", "translation": "Tonight our guest is a very special toaster"}])
    _patch_llm(monkeypatch, fake)
    r = vdt.translate("Stasera ospite d'eccezione: un tostapane!", "en", 4.0)
    assert len(fake.calls) == 2 and "HARD LIMIT" in fake.calls[1] and "English" in fake.calls[0]
    assert r["text"].startswith("Tonight") and r["source_language"] == "it" and r["warning"] == "" and fake.unloaded == 1
    # ancora troppo lungo dopo il secondo giro: avviso
    fake2 = _FakeLLM([{"translation": long_t}, {"translation": long_t}])
    _patch_llm(monkeypatch, fake2)
    assert vdt.translate("ciao a tutti", "es", 4.0)["warning"] == "long"


def test_translate_errors(monkeypatch):
    from core import video_dub_translate as vdt
    _patch_llm(monkeypatch, _FakeLLM([], up=False))
    with pytest.raises(vdt.DubTranslateError) as e:
        vdt.translate("ciao", "en", 3.0)
    assert e.value.key == "vedit.dub.err_llm" and "LM Studio" in str(e.value)
    _patch_llm(monkeypatch, _FakeLLM([{"translation": "  "}]))
    with pytest.raises(vdt.DubTranslateError) as e:
        vdt.translate("ciao", "en", 3.0)
    assert e.value.key == "vedit.dub.err_translate"
    with pytest.raises(vdt.DubTranslateError) as e:
        vdt.translate("ciao", "xx", 3.0)                              # lingua non tra quelle parlabili
    assert e.value.key == "vedit.dub.err_no_target"


def test_extract_speech_backends(monkeypatch, tmp_path):
    from core import video_dub_translate as vdt
    from core import handy_api
    src = _gen(tmp_path / "a.mp4", 25, audio=True)
    monkeypatch.setattr(handy_api, "available", lambda: True)
    monkeypatch.setattr(handy_api, "transcribe", lambda p, **k: "Stasera ospite d'eccezione")
    r = vdt.extract_speech(src)
    assert r["text"].startswith("Stasera") and r["backend"] in ("handy", "faster-whisper")
    if r["backend"] == "handy":
        monkeypatch.setattr(handy_api, "transcribe", lambda p, **k: "")
        with pytest.raises(vdt.DubTranslateError) as e:
            vdt.extract_speech(src)
        assert e.value.key == "vedit.dub.err_no_speech"
        monkeypatch.setattr(handy_api, "available", lambda: False)
        monkeypatch.setattr(handy_api, "ensure", lambda *a, **k: False)
        with pytest.raises(vdt.DubTranslateError) as e2:
            vdt.extract_speech(src)
        assert e2.value.key == "vedit.dub.err_stt" and "Handy-API" in str(e2.value)


def test_run_and_fill_cfg(monkeypatch, tmp_path):
    from core import video_dub_translate as vdt
    src = _gen(tmp_path / "a.mp4", 49, audio=True)
    _patch_llm(monkeypatch, _FakeLLM([{"source_language": "it", "translation": "Hello everyone"}]))
    r = vdt.run(src, "en", source_text="Ciao a tutti")                 # testo dato: niente STT
    assert r["text"] == "Hello everyone" and r["original_text"] == "Ciao a tutti"
    assert r["original_language"] == "it" and r["backend"] == "testo dato"
    _patch_llm(monkeypatch, _FakeLLM([{"source_language": "it", "translation": "Hello everyone"}]))
    cfg = {"video": src, "translate_to": "en", "source_text": "Ciao a tutti", "lipdub_line": "", "prompt": ""}
    assert vdt.fill_cfg(cfg)["text"] == "Hello everyone" and cfg["lipdub_line"] == "Hello everyone" and cfg["lipdub_lang"] == "en"
    cfg2 = {"video": src, "translate_to": "en", "lipdub_line": "gia' scritta"}
    assert vdt.fill_cfg(cfg2) == {} and cfg2["lipdub_line"] == "gia' scritta"          # battuta presente: niente traduzione
