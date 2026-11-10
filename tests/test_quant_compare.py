"""Confronto di quantizzazione (turno 41): parsing del log H3, tabella modello x (tempo, VRAM), pagina di revisione con i video affiancati."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app_live"))
import quant_compare as QC  # noqa: E402

SAMPLE_LOG = """[h3] text encoder pronto in 5s — VRAM 14.6 GB (picco 14.6)
[h3] testo codificato in 103s
[h3] condizionamenti VAE in 20s
[h3] transformer pronto in 4s — VRAM 26.1 GB (picco 28.8)
 12%|█▎        | 1/8 [00:33<03:55, 33.62s/it][h3] [mem] fine passo: allocata 21.1 GB, picco del passo 28.7, riservata 26.7
[h3] denoise 8 passi in 274s (34.2s/passo) — VRAM 20.4 GB (picco 23.1)
[h3] decode in 12s
[h3] edit finito in 469s -> E:\\x.mp4
"""


def test_parse_phase_log():
    phases = QC.parse_phase_log(SAMPLE_LOG)
    assert phases["te_load_s"] == 5 and phases["te_encode_s"] == 103 and phases["vae_cond_s"] == 20 and phases["dit_load_s"] == 4
    assert phases["denoise_s"] == 274 and phases["s_per_step"] == 34.2 and phases["decode_s"] == 12 and phases["total_s"] == 469
    assert phases["worker_peak_gb"] == 28.8
    assert QC.parse_phase_log("niente")["total_s"] is None


def test_newest_log_after_marker(tmp_path):
    marker = tmp_path / "marker"
    marker.write_text("x")
    old = tmp_path / "_h3edit_1.log"
    old.write_text("vecchio")
    os.utime(old, (time.time() - 500, time.time() - 500))
    os.utime(marker, (time.time() - 100, time.time() - 100))
    new = tmp_path / "_h3edit_2.log"
    new.write_text("nuovo")
    assert QC.newest_job_log(str(marker), (str(tmp_path),)) == str(new)
    assert QC.newest_job_log(str(tmp_path / "assente"), (str(tmp_path / "vuota"),)) == ""


def make_entry(case, model, tempo, vram_peak, video="", esito="ok", face=None):
    checks = {} if face is None else {"video_volto_ritratto": {"esito": "OK", "face_sim_max": face}}
    return {"case": case, "id": case + "_" + model, "model": model, "esito": esito, "motivo": "" if esito == "ok" else "job fallito", "tempo_s": tempo, "vram_peak_mb": vram_peak,
            "vram_base_mb": 3000, "ram_peak_mb": 30000, "video": video, "log": "", "checks": checks,
            "phases": {"dit_load_s": 4.0, "s_per_step": 34.2, "worker_peak_gb": 28.8}}


def test_table_has_every_model_and_case_and_the_mean():
    entries = [make_entry("q41_intera", "nvfp4", 470, 27000, face=0.41), make_entry("q41_intera", "int8", 640, 30000),
               make_entry("q41_parlante", "nvfp4", 500, 27500), make_entry("q41_parlante", "int8", 0, None, esito="fallito")]
    text = QC.table(entries)
    assert "q41_intera" in text and "q41_parlante" in text and "MEDIA PER MODELLO" in text
    assert "0.41" in text and "fallito" in text
    media = text.split("MEDIA PER MODELLO")[1]
    assert "nvfp4" in media and "2/2 ok" in media and "1/2 ok" in media                      # int8: un caso su due riuscito
    assert abs(QC.delta_vram_gb(entries[0]) - 23.4375) < 1e-6 and QC.delta_vram_gb(entries[3]) is None


def test_entry_from_results_and_review_page(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    video = tmp_path / "out.mp4"
    video.write_bytes(b"x")
    result = {"id": "q41_intera_int8", "esito": "ok", "tempo_s": 640.5, "vram_peak_mb": 30000, "vram_base_mb": 3000, "ram_peak_mb": 31000, "uscite": [str(video)],
              "autocheck": [{"nome": "video_volto_ritratto", "esito": "OK", "misure": {"face_sim_max": 0.55, "fotogrammi": 12}}]}
    (run / "results.jsonl").write_text(json.dumps({"id": "altro", "esito": "fallito"}) + "\n" + json.dumps(result) + "\n", encoding="utf-8")
    entry = QC.build_entry("q41_intera_int8", "int8", str(run), str(tmp_path / "nessun_marcatore"))
    assert entry["case"] == "q41_intera" and entry["tempo_s"] == 640.5 and entry["video"] == str(video) and entry["checks"]["video_volto_ritratto"]["face_sim_max"] == 0.55
    assert QC.last_result(str(run), "inesistente") == {}
    entries = [entry, make_entry("q41_intera", "nvfp4", 470, 27000, video=str(video))]
    index = QC.build_review(entries, str(tmp_path / "review"))
    page = Path(index).read_text(encoding="utf-8")
    data = json.loads((tmp_path / "review" / "data.json").read_text("utf-8"))
    card = data["cards"][0]
    results = [item for item in card["items"] if item["role"] == "result"]
    assert len(results) == 2 and "int8 ufficiale" in results[0]["label"] and "min" in results[0]["label"] and "Pubblica" in page      # i due modelli affiancati nella stessa scheda
    assert card["esito"] == "ok" and card["id"] == "q41_intera"


def test_key_log_lines_and_notes_in_the_table():
    text = ("[h3] GEMM int8 attivo: errore relativo 0.0088 contro il riferimento sul primo layer" + chr(10) + "rumore" + chr(10) +
            "[h3] video di riferimento (movimento): tela 512x288" + chr(10) + "[h3] decode in 12s")
    lines = QC.key_log_lines(text)
    assert lines == ["[h3] GEMM int8 attivo: errore relativo 0.0088 contro il riferimento sul primo layer", "[h3] video di riferimento (movimento): tela 512x288"]
    entry = make_entry("s42", "tutto_attivo", 500, 20000)
    entry["notes"] = lines
    assert "RIGHE DEL LOG PER PROVA" in QC.table([entry]) and "tela 512x288" in QC.table([entry])
    assert "tutto_attivo" in QC.MODEL_LABELS and "rif_pieno" in QC.MODEL_LABELS


def test_explicit_group_puts_all_variants_in_one_card(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text(json.dumps({"id": "s42_rif_pieno", "esito": "ok", "tempo_s": 100.0}) + chr(10), encoding="utf-8")
    entry = QC.build_entry("s42_rif_pieno", "rif_pieno", str(run), str(tmp_path / "nessun_marcatore"), "s42")
    assert entry["case"] == "s42" and entry["model"] == "rif_pieno"
    assert QC.build_entry("q41_intera_int8", "int8", str(run), str(tmp_path / "x"))["case"] == "q41_intera"


def test_a_case_redone_counts_only_with_its_latest_run(tmp_path):
    """Turno 43r: s43_pika6_nvfp4 compariva tre volte (assente, errore, ok) e s43_dasiwa_8 due: la tabella conta l'ultima esecuzione."""
    rows = [{"case": "s43", "id": "a", "model": "m1", "esito": "assente"}, {"case": "s43", "id": "b", "model": "m2", "esito": "ok"},
            {"case": "s43", "id": "a", "model": "m1", "esito": "errore"}, {"case": "s43", "id": "a", "model": "m1", "esito": "ok"},
            {"case": "altro", "id": "a", "model": "m1", "esito": "ok"}]
    path = tmp_path / "casi.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    entries = QC.read_entries(str(path))
    assert [(e["case"], e["id"], e["esito"]) for e in entries] == [("s43", "b", "ok"), ("s43", "a", "ok"), ("altro", "a", "ok")]
    assert QC.read_entries(str(tmp_path / "manca.jsonl")) == []
