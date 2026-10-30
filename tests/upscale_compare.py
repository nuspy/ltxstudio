"""Confronto upscaler video: SeedVR2 vs FlashVSR v1.1 vs FlashVSR+ (Turbo / Base) — task 20.

USA LA GPU: lanciare SOLO col «via» e UN test per volta sotto il guardiano, ad es.
    python tests\\live_guard.py --min-commit-gb 20 --max-proc-gb 70 --kill-servers-on-exit -- ^
        E:\\AI\\Tools\\inference_server_venv\\Scripts\\python.exe tests\\upscale_compare.py run --engine fvp_turbo ^
        --clip <lq.mp4> --name testo --scale 2

Sottocomandi:
  prep    --src VIDEO --name N [--start S] [--dur 10] [--height 360] [--outdir D]
          ritaglia una clip (>=10 s) e la riduce ad altezza LQ: e' l'ingresso comune dei motori.
  run     --engine {seedvr2,flashvsr11,fvp_turbo,fvp_base,fvp_turbo_tc2} --clip LQ --name N [--scale 2] [--results J]
          esegue UN motore su UNA clip e aggiunge la voce a J (schema campagna, gruppo «Upscaler a confronto»):
          tempo totale, s/frame, picco VRAM (nvidia-smi, meno la base), picco RAM (albero processi), nitidezza
          (varianza del Laplaciano su 8 frame), PSNR del downscale dell'uscita contro l'ingresso LQ.
  report  --results J --outdir D     fotogrammi affiancati (ritaglio 1:1 + intero) e miniature per la pagina.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GROUP = "Upscaler a confronto"
ENGINES = {
    "seedvr2": "SeedVR2 3B fp8 (nativo)",
    "flashvsr11": "FlashVSR v1.1",
    "fvp_turbo": "FlashVSR+ Turbo",
    "fvp_base": "FlashVSR+ Base",
    "fvp_turbo_tc2": "FlashVSR+ Turbo (movimento 2)",
}


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = (s["avg_frame_rate"].split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    dur = float(j["format"]["duration"])
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": fps, "dur": dur, "frames": int(s.get("nb_frames") or round(dur * fps))}


# ------------------------------------------------------------------ metriche ----
def sample_frames(path, n=8):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(total * 0.05, total * 0.95, n).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append((int(i), f))
    cap.release()
    return out


def metrics(lq_path, out_path):
    import cv2
    import numpy as np
    lq = dict(sample_frames(lq_path))
    out = sample_frames(out_path)
    sharp, psnr = [], []
    for i, f in out:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        ref = lq.get(i)
        if ref is None:                                   # frame vicino
            k = min(lq, key=lambda x: abs(x - i))
            ref = lq[k]
        small = cv2.resize(f, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((small.astype(np.float64) - ref.astype(np.float64)) ** 2))
        psnr.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    return {"nitidezza_laplaciano": round(float(np.mean(sharp)), 1), "psnr_downscale_db": round(float(np.mean(psnr)), 2),
            "frame_campionati": len(out)}


class Monitor(threading.Thread):
    """Picco VRAM (nvidia-smi: memoria usata dall'intera GPU, si sottrae la base) e RAM (albero processi)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.base = self._vram()
        self.vram_peak = self.base
        self.ram_peak = 0.0

    @staticmethod
    def _vram():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            return float(r.stdout.strip().splitlines()[0])
        except Exception:                                  # noqa: BLE001
            return 0.0

    def run(self):
        import psutil
        me = psutil.Process()
        while not self.stop:
            self.vram_peak = max(self.vram_peak, self._vram())
            try:
                tot = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                self.ram_peak = max(self.ram_peak, tot / 2 ** 20)
            except Exception:                              # noqa: BLE001
                pass
            time.sleep(0.5)


# ------------------------------------------------------------------ esecuzione ----
def run_engine(engine, clip, scale, out_path, log):
    info = probe(clip)
    if engine == "seedvr2":
        from core import seedvr2_service as sv
        res = int(round(info["h"] * scale / 2.0)) * 2
        return sv.upscale_video(clip, out_path, resolution=res, batch_size=5, chunk_size=120, temporal_overlap=3, on_log=log)
    if engine == "flashvsr11":
        from core import flashvsr_service as fv
        return fv.upscale_video(clip, out_path, target_scale=scale, on_log=log)
    from core import flashvsrplus_service as fvp
    mode = "base" if engine == "fvp_base" else "turbo"
    return fvp.upscale_video(clip, out_path, mode=mode, target_scale=scale,
                             temporal_compression=2 if engine.endswith("tc2") else 4, on_log=log)


def cmd_run(a):
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(a.clip)
    out = str(out_dir / f"DIAG_cmp_{a.name}_{a.engine}.mp4")
    lines: list[str] = []
    mon = Monitor()
    mon.start()
    t0 = time.time()
    err = None
    try:
        run_engine(a.engine, a.clip, a.scale, out, lambda l: (lines.append(l), print(l, flush=True)))
    except Exception as e:                                 # noqa: BLE001
        err = str(e)
    dt = time.time() - t0
    mon.stop = True
    time.sleep(0.6)
    entry = {
        "key": f"cmp:{a.name}:{a.engine}", "gruppo": GROUP, "arch": "upscaler", "id_test": a.name, "variante": a.engine,
        "nome": f"{ENGINES.get(a.engine, a.engine)} su «{a.name}»", "tipo": "Upscaler video", "file_modello": ENGINES.get(a.engine, a.engine),
        "trigger": [], "forza": None, "prompt": None, "fonte_prompt": None, "pagina": None, "seed": None,
        "risoluzione": f"{info['w']}x{info['h']} -> x{a.scale:g}", "durata_s": round(info["dur"], 1), "passi": None,
        "tempo_s": round(dt, 1), "s_per_frame": round(dt / max(1, info["frames"]), 3),
        "picco_vram_mb": int(mon.vram_peak - mon.base), "picco_ram_mb": int(mon.ram_peak),
        "esito": "OK" if err is None else "FALLITO", "esito_tecnico": "FILE" if err is None and os.path.exists(out) else "ERRORE",
        "controlli": {}, "lora_applicate": [], "motivo": err, "nota": "", "output": out if err is None else None,
        "miniatura": None, "ora": time.strftime("%Y-%m-%dT%H:%M:%S"), "nota_vista": "",
    }
    if err is None and os.path.exists(out):
        entry["controlli"] = {"file": out, "tipo": "video", **{k: v for k, v in probe(out).items()}}
        entry["metriche"] = metrics(a.clip, out)
        for l in lines:                                    # statistiche interne del runner FlashVSR+
            if "STATS " in l:
                try:
                    entry["stats_runner"] = json.loads(l.split("STATS ", 1)[1])
                except Exception:                          # noqa: BLE001
                    pass
    res = Path(a.results)
    data = json.loads(res.read_text("utf-8")) if res.exists() else []
    data = [e for e in data if e.get("key") != entry["key"]] + [entry]
    res.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: entry.get(k) for k in ("key", "esito", "tempo_s", "s_per_frame", "picco_vram_mb", "picco_ram_mb", "metriche")}, ensure_ascii=False))
    return 0 if err is None else 1


def cmd_prep(a):
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cmp_lq_{a.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(a.start), "-i", a.src, "-t", str(a.dur), "-an",
                    "-vf", f"scale=-2:{a.height}:flags=area", "-c:v", "libx264", "-crf", "12", "-preset", "slow",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    print(json.dumps({"clip": str(out), **probe(out)}))
    return 0


def cmd_report(a):
    import cv2
    import numpy as np
    data = [e for e in json.loads(Path(a.results).read_text("utf-8")) if e.get("gruppo") == GROUP and e.get("output")]
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {}
    for e in data:
        by.setdefault(e["id_test"], []).append(e)
    for name, es in by.items():
        es.sort(key=lambda e: e["variante"])
        caps = []
        cols = []
        for e in es:
            cap = cv2.VideoCapture(e["output"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
            ok, f = cap.read()
            cap.release()
            if not ok:
                continue
            h, w = f.shape[:2]
            cy, cx = h // 2, w // 2
            crop = f[max(0, cy - 150):cy + 150, max(0, cx - 200):cx + 200]            # ritaglio 1:1 (nessun ridimensionamento)
            thumb = cv2.resize(f, (480, int(480 * h / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"thumb_{name}_{e['variante']}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            e["miniatura"] = str(out_dir / f"thumb_{name}_{e['variante']}.jpg")
            cv2.putText(crop, ENGINES.get(e["variante"], e["variante"]), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cols.append(crop)
        if cols:
            hh = min(c.shape[0] for c in cols)
            strip = np.hstack([c[:hh] for c in cols])
            cv2.imwrite(str(out_dir / f"strip_{name}.png"), strip)
    allj = json.loads(Path(a.results).read_text("utf-8"))
    for e in allj:
        for d in data:
            if d["key"] == e.get("key"):
                e["miniatura"] = d.get("miniatura")
    Path(a.results).write_text(json.dumps(allj, ensure_ascii=False, indent=1), "utf-8")
    print("report:", out_dir)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--src", required=True); p.add_argument("--name", required=True)
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--dur", type=float, default=10.0)
    p.add_argument("--height", type=int, default=360); p.add_argument("--outdir", default=str(ROOT / "output"))
    r = sub.add_parser("run"); r.add_argument("--engine", required=True, choices=list(ENGINES)); r.add_argument("--clip", required=True)
    r.add_argument("--name", required=True); r.add_argument("--scale", type=float, default=2.0)
    r.add_argument("--results", default=r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\task20_risultati.json")
    r.add_argument("--outdir", default=str(ROOT / "output"))
    q = sub.add_parser("report"); q.add_argument("--results", default=r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\task20_risultati.json")
    q.add_argument("--outdir", default=str(ROOT / "output" / "cmp_report"))
    a = ap.parse_args()
    return {"prep": cmd_prep, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
