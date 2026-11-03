"""Controlli AUTOMATICI sui risultati dei casi, PRIMA della pagina di revisione (decisione dell'utente 03/11).

Regola: l'utente non deve rigiudicare (a) risultati che non sono cambiati, (b) risultati che falliscono in modo misurabile.
Ogni controllo restituisce un dict {"nome", "esito": "ok"|"ROSSO", "motivo", "misure"}; run_suite scrive l'elenco in
r["autocheck"] e, se c'e un ROSSO, mette esito="rosso" (il caso non va alla pagina). `"baseline": true` nel caso = confronto col
comportamento di oggi: si misura e si mostra, ma non diventa rosso.

Controlli (soglie TARATE sui casi bocciati del 03/11, vedi `python tests/app_live/autocheck.py --calibra`):
  (a) unchanged   hash del risultato (pixel per le immagini, byte per i video) uguale a quello di un giro precedente dello stesso caso.
  (b) integra     «incollata, non rigenerata»: correlazione dei dettagli fini (passa-alto) fra uscita e composito d'ingresso dentro la sagoma (soglia 0.50)
                  + velo di colore (spostamento della media globale >= 40: «layer rosso semitrasparente» della maschera 837 px) + scala del volto (0.8-1.2).
                  Misura scelta perche la media della differenza NON basta: i motori Qwen cambiano solo le tonalita (differenza media 30-40/255,
                  quasi nessun pixel < 8) ma conservano ogni dettaglio (correlazione 0.86-0.95); Kontext/Krea rigenerano davvero (0.00-0.12).
                  + identita: somiglianza ArcFace fra la foto del soggetto e il volto piu simile nell'uscita (Kontext «diventa cinese» 0.02-0.20,
                  Krea «la ragazza manca» <= 0.0; Qwen/FLUX.2 0.83-0.93).
  (c) inpaint     l'oggetto richiesto nella maschera: coseno CLIP (DFN5B ViT-H/14, open_clip, CPU) fra il ritaglio attorno alla maschera e
                  «a photo of <oggetto>». Fill/Klein/Qwen 0.21-0.33; FLUX base «macchia/foglietti/cacca» 0.09-0.16.
  (d) video       identita: ArcFace sui fotogrammi (2 per secondo) contro la foto del personaggio; oggetti senza volto (astronauta,
                  orsacchiotto...) col coseno CLIP del testo sui fotogrammi. MSR 2.3/2.5 di oggi: volto <= 0.42, astronauta <= 0.14 (fixture 0.27).

Modelli (GIA sul disco, niente download): CLIP = apple/DFN5B-CLIP-ViT-H-14-384 nella cache HF dell'utente, via open_clip nel venv
MMAudio (torch CPU: non tocca la GPU); volti = antelopev2 (scrfd + glintr100) via insightface nell'inference_server_venv, provider CPU.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live"
HISTORY = os.path.join(ORCH, "autocheck_hash.json")
CLIP_PY = os.environ.get("AC_CLIP_PY", r"E:\ai\Tools\MMAudio\venv\Scripts\python.exe")
FACE_PY = os.environ.get("AC_FACE_PY", r"E:\ai\Tools\inference_server_venv\Scripts\python.exe")

THRESH = {
    "integra_hp_ncc": 0.50,      # >= : incollata (dettagli fini identici al composito). Tarata sul vuoto fra 0.12 (Kontext, rigenerato davvero)
                                 # e 0.72 (FLUX.2, incollato): nessun caso di oggi sta in mezzo
    "integra_velo": 40.0,        # >= : velo di colore su tutta l'immagine (spostamento della media di un canale, /255). Qwen con maschera 837 px: 83; gli altri <= 19
    "integra_scala": (0.80, 1.20),   # larghezza del volto in uscita / nel composito. Tarata su UN solo caso (Qwen «rigenera» 03: 1.23, «grande il doppio»)
    "integra_face_sim": 0.50,    # <  : identita persa o soggetto assente
    "inpaint_clip_cos": 0.18,    # <  : l'oggetto richiesto non c'e
    "video_face_sim": 0.50,      # max sui fotogrammi < : personaggio non riconosciuto
    "video_clip_cos": 0.20,      # max sui fotogrammi < : oggetto/personaggio senza volto assente
}
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")


def _f(ok: bool, nome: str, motivo: str = "", **misure) -> dict:
    return {"nome": nome, "esito": "ok" if ok else "ROSSO", "motivo": "" if ok else motivo, "misure": misure}


# ── (a) invariato ─────────────────────────────────────────────────────────────────────────────────────────────────────
def file_hash(path: str) -> str:
    """Immagini: md5 dei PIXEL (i metadati PNG cambiano a ogni salvataggio); video/altro: md5 dei byte."""
    if path.lower().endswith(VIDEO_EXT):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()
    from PIL import Image
    return hashlib.md5(Image.open(path).convert("RGB").tobytes()).hexdigest()


def load_history() -> dict:
    try:
        return json.load(open(HISTORY, encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return {}


def save_history(h: dict) -> None:
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    json.dump(h, open(HISTORY, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


def seed_history(orch: str = ORCH) -> dict:
    """Riempie la storia con tutti i giri passati (results.jsonl con esito ok): serve alla prima esecuzione."""
    h = load_history()
    for rj in sorted(glob.glob(os.path.join(orch, "*", "results.jsonl"))):
        run = os.path.basename(os.path.dirname(rj))
        for ln in open(rj, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except Exception:                                # noqa: BLE001
                continue
            if r.get("esito") not in ("ok", "rosso") or r.get("dry") or not r.get("uscite"):
                continue
            p = r["uscite"][0]
            if not os.path.isfile(p):
                continue
            try:
                hv = file_hash(p)
            except Exception:                                # noqa: BLE001
                continue
            lst = h.setdefault(str(r.get("id")), [])
            if not any(e["hash"] == hv and e["run"] == run for e in lst):
                lst.append({"hash": hv, "run": run})
    save_history(h)
    return h


def check_unchanged(case_id: str, out_path: str, run: str, hist: dict) -> dict:
    hv = file_hash(out_path)
    prev = [e for e in hist.get(case_id, []) if e["run"] != run]
    same = [e["run"] for e in prev if e["hash"] == hv]
    return _f(not same, "unchanged", f"identico a un giro precedente dello stesso caso ({same[0] if same else ''}): nulla e cambiato", hash=hv[:10], uguale_a=same[:3])


# ── helper pesanti in subprocess (venv diversi) ───────────────────────────────────────────────────────────────────────
def _run_worker(py: str, script: str, job: dict, timeout: int = 900) -> dict:
    if not os.path.isfile(py):
        raise RuntimeError(f"manca {py}")
    env = dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    p = subprocess.run([py, "-I", os.path.join(HERE, script)], input=json.dumps(job), capture_output=True, text=True, env=env,
                       timeout=timeout, encoding="utf-8", errors="replace")
    for ln in reversed(p.stdout.splitlines()):
        if ln.startswith("@@JSON@@"):
            return json.loads(ln[8:])
    raise RuntimeError(f"{script} senza risposta (rc={p.returncode}): {(p.stderr or p.stdout)[-400:]}")


def clip_scores(items: list) -> list:
    """items: [{"image": p, "mask": p|None, "pad": 0.5, "texts": [..]}] -> [[cos per testo], ...]"""
    return _run_worker(CLIP_PY, "_ac_clip.py", {"items": items})["scores"]


def face_sims(ref: str, images: list) -> list:
    """-> [{"n": volti, "max": somiglianza massima col volto della foto `ref`}] (max=-1 se nessun volto)."""
    return _run_worker(FACE_PY, "_ac_face.py", {"ref": ref, "images": images})["res"]


def video_frames(video: str, fps: float = 2.0, max_frames: int = 24) -> list:
    d = tempfile.mkdtemp(prefix="acv_")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"fps={fps}", os.path.join(d, "f_%03d.png")], capture_output=True)
    fr = sorted(glob.glob(os.path.join(d, "f_*.png")))
    if len(fr) > max_frames:
        step = len(fr) / max_frames
        fr = [fr[int(i * step)] for i in range(max_frames)]
    return fr


# ── (b) Integra ───────────────────────────────────────────────────────────────────────────────────────────────────────
def pasted_measures(out_path: str, comp_path: str, pos_mask_path: str) -> dict:
    import numpy as np
    import scipy.ndimage as ndi
    from PIL import Image
    c = Image.open(comp_path).convert("RGB")
    o = Image.open(out_path).convert("RGB")
    if o.size != c.size:
        o = o.resize(c.size, Image.LANCZOS)
    pm = np.asarray(Image.open(pos_mask_path).convert("L").resize(c.size)) > 127
    er = ndi.binary_erosion(pm, iterations=6)
    if er.sum() < 100:
        er = pm
    oo, cc = np.asarray(o).astype(np.float32), np.asarray(c).astype(np.float32)
    d = np.abs(oo - cc).max(2)

    def hp(a, s=3.0):
        g = a.mean(2)
        return g - ndi.gaussian_filter(g, s)
    a, b = hp(oo)[er], hp(cc)[er]
    ncc = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 0.0
    return {"hp_ncc": round(ncc, 3), "diff_media": round(float(d[er].mean()), 1), "quota_diff_lt8": round(float((d[er] < 8).mean()), 3)}


def check_integra(out_path: str, comp_path: str, pos_mask_path: str, ref_path: str, ctx_mask_path: str = "", scala: bool = True) -> list:
    """(b) Integra. `scala=False` per gli approcci senza maschera (Kontext a strisce): il volto puo spostarsi legittimamente."""
    import numpy as np
    from PIL import Image
    res = []
    m = pasted_measures(out_path, comp_path, pos_mask_path)
    res.append(_f(m["hp_ncc"] < THRESH["integra_hp_ncc"], "integra_incollata",
                  f"incollata, non rigenerata: i dettagli fini della persona coincidono col composito (correlazione {m['hp_ncc']} >= {THRESH['integra_hp_ncc']}; "
                  f"cambiano solo le tonalita: differenza media {m['diff_media']}/255)", **m))
    c = Image.open(comp_path).convert("RGB")
    o = Image.open(out_path).convert("RGB").resize(c.size)
    shift = float(np.abs(np.asarray(o).reshape(-1, 3).astype(np.float32).mean(0) - np.asarray(c).reshape(-1, 3).astype(np.float32).mean(0)).max())
    res.append(_f(shift < THRESH["integra_velo"], "integra_velo", f"velo di colore su tutta l'immagine (spostamento della media {shift:.0f}/255 >= {THRESH['integra_velo']})",
                  shift_globale=round(shift, 1)))
    if ref_path and os.path.isfile(ref_path):
        fs_out, fs_comp = face_sims(ref_path, [out_path, comp_path])
        res.append(_f(fs_out["max"] >= THRESH["integra_face_sim"], "integra_identita",
                      f"identita persa o soggetto assente: somiglianza volto {fs_out['max']} < {THRESH['integra_face_sim']} ({fs_out['n']} volti nell'uscita)",
                      face_sim=fs_out["max"], volti=fs_out["n"], ncc_volto_riferimento=fs_out.get("ncc_riferimento")))
        if scala and fs_out.get("larghezza") and fs_comp.get("larghezza") and fs_out["max"] >= THRESH["integra_face_sim"]:
            r = fs_out["larghezza"] / fs_comp["larghezza"]
            lo, hi = THRESH["integra_scala"]
            res.append(_f(lo <= r <= hi, "integra_scala", f"la persona ha cambiato scala: volto {r:.2f}x rispetto al composito (ammesso {lo}-{hi})", rapporto_volto=round(r, 2)))
    return res


# ── (c) inpaint ───────────────────────────────────────────────────────────────────────────────────────────────────────
def check_inpaint(out_path: str, mask_path: str, obj_text: str) -> list:
    s = clip_scores([{"image": out_path, "mask": mask_path, "pad": 0.5, "texts": ["a photo of " + obj_text]}])[0][0]
    return [_f(s >= THRESH["inpaint_clip_cos"], "inpaint_oggetto",
               f"l'oggetto richiesto ({obj_text}) non c'e nella maschera: punteggio testo-immagine {s:.3f} < {THRESH['inpaint_clip_cos']}",
               clip_cos=round(s, 3))]


# ── (d) video con riferimenti ─────────────────────────────────────────────────────────────────────────────────────────
def check_video_refs(video: str, subjects: list) -> list:
    """subjects: [{"nome": "...", "face_ref": path}  oppure  {"nome": "...", "text": "an astronaut in a spacesuit"}]"""
    fr = video_frames(video)
    res = []
    for sj in subjects:
        if sj.get("face_ref"):
            sims = face_sims(sj["face_ref"], fr)
            mx = max([x["max"] for x in sims], default=-1)
            res.append(_f(mx >= THRESH["video_face_sim"], "video_volto_" + sj["nome"],
                          f"{sj['nome']}: nessun fotogramma somiglia al riferimento (somiglianza massima {mx:.2f} < {THRESH['video_face_sim']})",
                          face_sim_max=round(mx, 3), fotogrammi=len(fr)))
        else:
            sc = clip_scores([{"image": f, "mask": None, "texts": ["a photo of " + sj["text"]]} for f in fr])
            mx = max([x[0] for x in sc], default=0.0)
            res.append(_f(mx >= THRESH["video_clip_cos"], "video_oggetto_" + sj["nome"],
                          f"{sj['nome']}: non compare in nessun fotogramma (punteggio testo-immagine massimo {mx:.3f} < {THRESH['video_clip_cos']})",
                          clip_cos_max=round(mx, 3), fotogrammi=len(fr)))
    return res


# ── orchestrazione per caso ───────────────────────────────────────────────────────────────────────────────────────────
def _find(folder: str, suffix: str) -> str:
    g = glob.glob(os.path.join(folder, "*" + suffix))
    return g[0] if g else ""


def evaluate(case: dict, r: dict, run: str, hist: dict) -> list:
    """Controlli applicabili al caso (da `case["autocheck"]`, o dedotti dai casi storici). Non lancia mai: un controllo che non riesce
    a girare diventa {"esito": "n/d"} e NON blocca (meglio un controllo mancante che un falso rosso)."""
    spec = dict(case.get("autocheck") or {})
    outs = [p for p in (r.get("uscite") or []) if isinstance(p, str) and os.path.isfile(p)]
    if not outs:
        return []
    out = outs[0]
    res = []
    try:
        res.append(check_unchanged(str(case.get("id")), out, run, hist))
    except Exception as e:                                   # noqa: BLE001
        res.append({"nome": "unchanged", "esito": "n/d", "motivo": repr(e)[:200], "misure": {}})
    kind = spec.get("kind") or ("integra" if case.get("kind") == "integrate" else "inpaint" if case.get("kind") == "inpaint"
                                else "video_refs" if case.get("refs") else "")
    try:
        folder = os.path.dirname(out)
        if kind == "integra":
            ms = r.get("maschere") or {}
            comp = ms.get("composito") or _find(folder, "integrate_input.png")
            pos = ms.get("posizione") or _find(folder, "integrate_position_mask.png") or _find(folder, "posizione*.png")
            ref = spec.get("ref") or case.get("layer") or ""
            if comp and pos:
                res += check_integra(out, comp, pos, ref, ms.get("contesto") or _find(folder, "contesto*.png"), scala=spec.get("scala", True))
            else:
                res.append({"nome": "integra", "esito": "n/d", "motivo": "composito o maschera di posizione non salvati", "misure": {}})
        elif kind == "inpaint":
            mk = _find(folder, "_frame1_mask.png") or _find(folder, "mask.png")
            obj = spec.get("oggetto") or spec.get("text")
            if mk and obj:
                res += check_inpaint(out, mk, obj)
            else:
                res.append({"nome": "inpaint", "esito": "n/d", "motivo": "manca la maschera salvata o `autocheck.oggetto` nel caso", "misure": {}})
        elif kind == "video_refs":
            subj = spec.get("subjects") or []
            if subj:
                res += check_video_refs(out, subj)
            else:
                res.append({"nome": "video", "esito": "n/d", "motivo": "manca `autocheck.subjects` nel caso", "misure": {}})
    except Exception as e:                                   # noqa: BLE001
        res.append({"nome": kind or "controllo", "esito": "n/d", "motivo": repr(e)[:300], "misure": {}})
    if case.get("baseline"):                                  # confronto col default: misurato e mostrato, mai rosso
        for x in res:
            if x["esito"] == "ROSSO":
                x["esito"] = "ROSSO(baseline)"
    return res


def apply(case: dict, r: dict, run: str, hist: dict) -> None:
    """Scrive r["autocheck"]; se c'e un ROSSO (non baseline) e il caso era ok -> esito="rosso" con il motivo; aggiorna la storia."""
    if r.get("dry") or r.get("esito") != "ok":
        return
    ck = evaluate(case, r, run, hist)
    r["autocheck"] = ck
    reds = [x for x in ck if x["esito"] == "ROSSO"]
    if reds:
        r["esito"] = "rosso"
        r["motivo"] = "[controllo automatico] " + " | ".join(f"{x['nome']}: {x['motivo']}" for x in reds)
    outs = [p for p in (r.get("uscite") or []) if isinstance(p, str) and os.path.isfile(p)]
    if outs:
        try:
            hist.setdefault(str(case.get("id")), []).append({"hash": file_hash(outs[0]), "run": run})
        except Exception:                                    # noqa: BLE001
            pass


# ── taratura sui casi di oggi ─────────────────────────────────────────────────────────────────────────────────────────
def calibra() -> int:
    """Stampa le misure sui casi del 03/11 (file veri in orch/app_live) e se i controlli li classificano come atteso."""
    import re
    FIX = os.path.join(ORCH, "fixtures")
    ref = os.path.join(FIX, "ritratto_960x1024.png")
    com = open(r"E:\ai\ltxstudio\docs\pubblicato_2026-11-03T18-47-37-019Z.md", encoding="utf-8").read()
    bocciati = set(re.findall(r"^## (\S+)", com, re.M))
    rows = []
    for run in ("run_t9a_1103_151127", "run_t9b_1103_151127", "run_t10c_1103_163004"):
        for ln in open(os.path.join(ORCH, run, "results.jsonl"), encoding="utf-8"):
            r = json.loads(ln)
            if r.get("esito") == "ok" and r.get("uscite"):
                rows.append((run, r))
    errori = 0
    comp_by_pos = {}
    for run, r in rows:                                       # composito di riserva per i casi «rigenera» (non lo salvavano): stesso mask di posizione
        f = os.path.dirname(r["uscite"][0])
        c, pm = _find(f, "integrate_input.png"), _find(f, "*position_mask.png") or _find(f, "posizione*.png")
        if c and pm:
            from PIL import Image
            import numpy as np
            comp_by_pos[hashlib.md5(np.asarray(Image.open(pm).convert("L")).tobytes()).hexdigest()] = c
    print(f"{'caso':62s} {'bocc.':5s} esito controlli")
    for run, r in rows:
        i = r["id"]
        f = os.path.dirname(r["uscite"][0])
        tags = []
        try:
            if "integra" in i:
                pm = _find(f, "posizione*.png") or _find(f, "*position_mask.png")
                from PIL import Image
                import numpy as np
                comp = _find(f, "integrate_input.png") or comp_by_pos.get(hashlib.md5(np.asarray(Image.open(pm).convert("L")).tobytes()).hexdigest(), "")
                for x in check_integra(r["uscite"][0], comp, pm, ref):
                    tags.append(f"{x['nome']}={x['esito']}{json.dumps(x['misure'])}")
            elif "inpaint" in i:
                mk = _find(f, "_frame1_mask.png")
                for x in check_inpaint(r["uscite"][0], mk, "brass compass"):
                    tags.append(f"{x['nome']}={x['esito']}{json.dumps(x['misure'])}")
            elif "video_msr" in i:
                for x in check_video_refs(r["uscite"][0], [{"nome": "ragazza", "face_ref": ref},
                                                           {"nome": "astronauta", "text": "an astronaut in a spacesuit"}]):
                    tags.append(f"{x['nome']}={x['esito']}{json.dumps(x['misure'])}")
        except Exception as e:                               # noqa: BLE001
            tags.append("ERRORE " + repr(e)[:150])
        if tags:
            rosso = any("=ROSSO" in t for t in tags)
            if i in bocciati and not rosso:
                errori += 1
            print(f"{i[:62]:62s} {'SI' if i in bocciati else 'no':5s} {'ROSSO' if rosso else 'verde':5s} {' '.join(tags)}")
    print(f"\nbocciati dall'utente NON rossi: {errori}")
    return 0


if __name__ == "__main__":
    if "--calibra" in sys.argv:
        raise SystemExit(calibra())
    if "--semina" in sys.argv:
        print({k: len(v) for k, v in seed_history().items()})
