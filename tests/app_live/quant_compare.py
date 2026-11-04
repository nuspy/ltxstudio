"""Confronto di quantizzazione del DiT H3 Ref2VA (turno 41): stesso caso, modelli diversi, stesso seme/risoluzione/passi.
Per ogni prova registra tempo totale e per fase (dal log del job H3), VRAM di picco (campionata da run_suite con core/gpu_mem, mai torch.cuda nella GUI),
RAM e controlli automatici; a fine turno scrive la tabella modello x (tempo, VRAM) e la pagina di revisione con i video dei modelli AFFIANCATI per caso.

    python quant_compare.py add    <casi.jsonl> <id caso> <modello> <cartella run> <marcatore> [gruppo]   # dopo ogni caso (gruppo: scheda di confronto)
    python quant_compare.py report <casi.jsonl> <misure.txt> [<cartella revisione>]               # a fine turno
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_LABELS = {
    "nvfp4": "nvfp4 Abiray (predefinito, 12,5 GB)",
    "int8": "int8 ufficiale (21,0 GB)",
    "mmh3-dit-rockerboo-nvfp4": "nvfp4+int8 rockerBOO (20,1 GB)",
    # turno 42: accelerazioni dall'originale (stesso caso, 736p, un fattore alla volta rispetto a «tutto attivo»)
    "tutto_attivo": "TUTTO ATTIVO (rif. 512 px, decode fp16; GEMM e attenzione int8 spenti)",
    "con_gemm_int8": "tutto attivo + GEMM int8 vero (W8A8)",
    "rif_pieno": "riferimento a tela piena (senza accelerazione 1)",
    "attenzione_int8": "tutto attivo + attenzione int8",
    "nvfp4_base": "DiT nvfp4 Abiray, tutto attivo (base del GEMM FP4)",
    "con_gemm_fp4": "DiT nvfp4 Abiray + GEMM FP4 vero (W4A4)",
    "decode_fp32": "decode VAE in fp32 (senza accelerazione 4)",
    # turno 42b: margine della maschera dell'originale (restringimento della persona a 1008 px), accanto al turno 40 e all'originale Openjutsu
    "originale_openjutsu": "ORIGINALE Openjutsu (ComfyUI, maschera 20 px, stesso rifugio; video intero)",
    "margine20_t40": "turno 40: margine 20 px (~26 px a 1312), int8 + DMAD 6 passi",
    "margine10": "margine 10 px (~13 px a 1312) + attenzione int8",
    "margine4": "margine 4 px (~5 px a 1312) + attenzione int8",
    # turno 43: confronto turbo sullo stesso caso del turno 41
    "pika6_int8": "int8 ufficiale + Pika Flex 6 passi (renoise)",
    "pika6_nvfp4": "nvfp4 Abiray + Pika Flex 6 passi (renoise)",
    "dasiwa_4": "DaSiWa Hybrid V4 Turbo int8, 4 passi",
    "dasiwa_8": "DaSiWa Hybrid V4 Turbo int8, 8 passi",
    "int8_v4_turno41": "RIFERIMENTO turno 41: int8 + turbo v4 8 passi",
    "nvfp4_v4_turno41": "RIFERIMENTO turno 41: nvfp4 Abiray + turbo v4 8 passi",
}
KEY_LINES = re.compile(r"\[h3\] (GEMM|attenzione int8|video di riferimento|pesi quantizzati|cache del testo|testo: ripreso|piano VRAM|memoria insufficiente)")
LOG_DIRS = (r"E:\ai\ltxstudio\output", r"E:\ai\ltxstudio\.claude\worktrees\prove\output")
PHASES = (
    ("te_load_s", r"text encoder pronto in (\d+)s"),
    ("te_encode_s", r"testo codificato in (\d+)s"),
    ("vae_cond_s", r"condizionamenti VAE in (\d+)s"),
    ("dit_load_s", r"transformer pronto in (\d+)s"),
    ("denoise_s", r"denoise \d+ passi in (\d+)s"),
    ("s_per_step", r"denoise \d+ passi in \d+s \(([\d.]+)s/passo\)"),
    ("decode_s", r"decode in (\d+)s"),
    ("total_s", r"edit finito in (\d+)s"),
)
WORKER_PEAK = re.compile(r"picco (?:del passo )?([\d.]+)")


def parse_phase_log(text: str) -> dict:
    """Tempi per fase e picco di VRAM visto dal worker (torch, nel processo figlio) dal log del job H3. Le fasi assenti restano None."""
    phases = {}
    for name, pattern in PHASES:
        found = re.findall(pattern, text)
        phases[name] = float(found[-1]) if found else None
    peaks = [float(value) for value in WORKER_PEAK.findall(text)]
    phases["worker_peak_gb"] = max(peaks) if peaks else None
    return phases


def newest_job_log(marker: str, directories=LOG_DIRS) -> str:
    """Il log `_h3edit_<stamp>.log` piu recente scritto dopo il marcatore ('' se non c'e)."""
    marker_time = os.path.getmtime(marker) if os.path.isfile(marker) else 0.0
    candidates = [path for folder in directories for path in glob.glob(os.path.join(folder, "_h3edit_*.log")) if os.path.getmtime(path) >= marker_time]
    return max(candidates, key=os.path.getmtime) if candidates else ""


def last_result(run_dir: str, case_id: str) -> dict:
    """Ultima riga di results.jsonl del caso ({} se manca)."""
    path = os.path.join(run_dir, "results.jsonl")
    if not os.path.isfile(path):
        return {}
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return next((row for row in reversed(rows) if row.get("id") == case_id), {})


def key_log_lines(text: str, limit: int = 8) -> list:
    """Righe del log del job che dicono quali accelerazioni hanno girato (GEMM, riferimento, piano VRAM, cache del testo)."""
    return [line.strip()[:200] for line in text.splitlines() if KEY_LINES.search(line)][:limit]


def checks_summary(result: dict) -> dict:
    """Misure dei controlli automatici (somiglianza del volto ArcFace, invariato) in forma piatta."""
    summary = {}
    for check in result.get("autocheck") or []:
        measures = check.get("misure") or {}
        summary[check.get("nome", "?")] = {"esito": check.get("esito"), **{k: v for k, v in measures.items() if isinstance(v, (int, float))}}
    return summary


def build_entry(case_id: str, model: str, run_dir: str, marker: str, group: str = "") -> dict:
    result = last_result(run_dir, case_id)
    log_path = newest_job_log(marker)
    log_text = open(log_path, encoding="utf-8", errors="replace").read() if log_path else ""
    phases = parse_phase_log(log_text) if log_text else {}
    outputs = [path for path in (result.get("uscite") or []) if isinstance(path, str) and os.path.isfile(path)]
    return {"case": group or case_id.rsplit("_", 1)[0], "id": case_id, "model": model, "esito": result.get("esito") or "assente", "motivo": result.get("motivo") or "",
            "tempo_s": result.get("tempo_s"), "vram_peak_mb": result.get("vram_peak_mb"), "vram_base_mb": result.get("vram_base_mb"),
            "ram_peak_mb": result.get("ram_peak_mb"), "video": outputs[0] if outputs else "", "log": log_path, "phases": phases, "checks": checks_summary(result), "notes": key_log_lines(log_text)}


def delta_vram_gb(entry: dict):
    peak, base = entry.get("vram_peak_mb"), entry.get("vram_base_mb")
    return None if peak is None or base is None else (peak - base) / 1024.0


def fmt(value, template="%.1f"):
    return "n/d" if value is None else template % value


def table(entries: list) -> str:
    """Tabella modello x (tempo totale, DiT, denoise per passo, VRAM di picco) per caso, piu la media per modello."""
    header = "%-34s %-9s %8s %8s %9s %10s %10s  %s" % ("caso / modello", "esito", "tempo s", "DiT s", "s/passo", "VRAM +GB", "worker GB", "volto ArcFace")
    lines = ["CONFRONTO DI QUANTIZZAZIONE (solo clip 1, stesso seme/risoluzione/passi). VRAM +GB = picco campionato meno la base del PC; worker GB = picco torch del processo H3.", "", header]
    for case in sorted({entry["case"] for entry in entries}):
        lines.append("-- " + case)
        for entry in (e for e in entries if e["case"] == case):
            ph, face = entry.get("phases") or {}, (entry.get("checks") or {}).get("video_volto_ritratto", {})
            lines.append("%-34s %-9s %8s %8s %9s %10s %10s  %s" % ("  " + entry["model"][:30], entry["esito"], fmt(entry.get("tempo_s"), "%.0f"), fmt(ph.get("dit_load_s"), "%.0f"),
                                                                   fmt(ph.get("s_per_step")), fmt(delta_vram_gb(entry)), fmt(ph.get("worker_peak_gb")), fmt(face.get("face_sim_max"), "%.2f")))
    lines += ["", "MEDIA PER MODELLO (sui casi riusciti)", header]
    for model in sorted({entry["model"] for entry in entries}):
        ok = [e for e in entries if e["model"] == model and e["esito"] == "ok"]
        mean = lambda values: (sum(values) / len(values)) if values else None          # noqa: E731
        lines.append("%-34s %-9s %8s %8s %9s %10s %10s" % (model[:34], "%d/%d ok" % (len(ok), sum(1 for e in entries if e["model"] == model)),
                     fmt(mean([e["tempo_s"] for e in ok if e.get("tempo_s") is not None]), "%.0f"), fmt(mean([e["phases"].get("dit_load_s") for e in ok if e["phases"].get("dit_load_s") is not None]), "%.0f"),
                     fmt(mean([e["phases"].get("s_per_step") for e in ok if e["phases"].get("s_per_step") is not None])),
                     fmt(mean([delta_vram_gb(e) for e in ok if delta_vram_gb(e) is not None])), fmt(mean([e["phases"].get("worker_peak_gb") for e in ok if e["phases"].get("worker_peak_gb") is not None]))))
    lines += ["", "RIGHE DEL LOG PER PROVA (cosa ha girato davvero)"]
    for entry in entries:
        lines += ["-- %s / %s" % (entry["case"], entry["model"])] + ["   " + note for note in entry.get("notes") or ["(nessuna riga)"]]
    return "\n".join(lines)


def result_label(entry: dict) -> str:
    face = (entry.get("checks") or {}).get("video_volto_ritratto", {}).get("face_sim_max")
    minutes = None if entry.get("tempo_s") is None else float(entry["tempo_s"]) / 60.0
    return "%s · %s min · VRAM +%s GB%s" % (MODEL_LABELS.get(entry["model"], entry["model"]), fmt(minutes), fmt(delta_vram_gb(entry)),
                                              "" if face is None else " · volto %.2f" % face)


def card_for_case(case: str, entries: list, review_dir: str, source_items: list) -> dict:
    import review
    items = list(source_items)
    media = []
    for entry in entries:
        thumb = review._thumb(entry["video"], review_dir, f"{case}_{entry['model'][:8]}") if entry["video"] else None
        big = review._thumb(entry["video"], review_dir, f"{case}_{entry['model'][:8]}", True) if entry["video"] else None
        items.append({"role": "result", "label": result_label(entry), "path": entry["video"], "thumb": thumb, "big": big, "video": True})
        media.append({"path": entry["video"], "thumb": thumb, "big": big, "video": True})
    bad = [e for e in entries if e["esito"] != "ok"]
    return {"items": items, "id": case, "fn": "ops", "esito": "ok" if not bad else "fallito", "motivo": "; ".join("%s: %s" % (e["model"], e["motivo"][:120]) for e in bad), "dry": False,
            "tempo": None, "vram": None, "ram": None, "commit": None, "handler": "h3_edit", "arch": "minimax_h3", "modello": ", ".join(e["model"] for e in entries),
            "prompt": "", "negativo": "", "params": {}, "loras": [], "media": media, "orig": [], "dialoghi": [], "iniz": None,
            "descr": ("Stesso caso, prove affiancate: stesso seme, solo clip 1. Turno 41: passi turbo v4 (8), tela 480p, encoder testo nvfp4. "
                      "Turno 42: 736p, int8 + DMAD a 6 passi, maschera dell'originale."),
            "verifica": "Quale modello rende meglio? Volto uguale al ritratto, gesti e labbra come l'originale, nitidezza, artefatti ai bordi, voce originale. "
                        "Scrivi nel riquadro il modello migliore e i difetti degli altri (vuoto = nessuna differenza che conti)."}


def source_items_for(case: str, review_dir: str) -> list:
    import review
    fixtures = os.path.join(review.ORCH, "fixtures")
    video = os.path.join(fixtures, "persona_parlante_11s.mp4" if "parlante" in case else "persona_intera_12s.mp4")
    portrait = os.path.join(fixtures, "ritratto_sfondo_neutro_pulito.png")
    items = []
    sources = [("video originale", video, True)] + ([("ritratto (personaggio nuovo)", portrait, False)] if case.startswith("q41") else [])
    for label, path, is_video in sources:
        if os.path.isfile(path):
            items.append({"role": "orig", "label": label, "path": path, "thumb": review._thumb(path, review_dir, f"{case}_src_{len(items)}"),
                          "big": review._thumb(path, review_dir, f"{case}_src_{len(items)}", True), "video": is_video})
    return items


def build_review(entries: list, review_dir: str) -> str:
    import review
    os.makedirs(review_dir, exist_ok=True)
    cards = [card_for_case(case, [e for e in entries if e["case"] == case], review_dir, source_items_for(case, review_dir)) for case in sorted({e["case"] for e in entries})]
    data = {"nome": "confronto di quantizzazione del DiT H3", "run": review_dir, "creato": time.strftime("%Y-%m-%d %H:%M"), "cards": cards}
    payload = json.dumps(data, ensure_ascii=False, default=str)
    with open(os.path.join(review_dir, "data.json"), "w", encoding="utf-8") as handle:
        handle.write(payload)
    index = os.path.join(review_dir, "index.html")
    with open(index, "w", encoding="utf-8") as handle:
        handle.write(review.PAGE.replace("/*DATA*/null", payload.replace("</", "<\\/")))
    return index


def read_entries(jsonl: str) -> list:
    return [json.loads(line) for line in open(jsonl, encoding="utf-8") if line.strip()] if os.path.isfile(jsonl) else []


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) >= 6 and args[0] == "add":
        entry = build_entry(args[2], args[3], args[4], args[5], args[6] if len(args) > 6 else "")
        with open(args[1], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("registrato:", entry["id"], entry["model"], entry["esito"], entry["tempo_s"])
        return 0
    if len(args) >= 3 and args[0] == "report":
        entries = read_entries(args[1])
        text = table(entries)
        with open(args[2], "a", encoding="utf-8") as handle:
            handle.write("\n" + text + "\n")
        review_dir = args[3] if len(args) > 3 else os.path.join(os.path.dirname(args[2]), "review_quant_" + time.strftime("%Y%m%d_%H%M"))
        print(text)
        print("pagina di revisione:", build_review(entries, review_dir))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
