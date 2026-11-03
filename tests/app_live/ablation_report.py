"""Tabella e immagine di confronto dell'ABLAZIONE del doppione (turno 16a, 03/11).
Input: un file jsonl con una riga per caso {"id", "clipdir", "video"} (lo scrive misure_h3_lib.sh), le maschere e il video sorgente.
Per ogni variante: k mediana del ritardo temporale e quota di frame col fantasma (measure_lag), fedelta della testa e persona fuori sagoma 0-6 s (ghost_metric),
sfondo vecchio nella fascia attorno alla sagoma (measure_oldbg, riga TOTALE). Immagine: per ogni variante un fotogramma a 2,5 s e uno a 4,5 s.
Uso: python ablation_report.py <casi.jsonl> <cartella maschere> <video sorgente> <tabella.txt> <confronto.jpg>"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ghost_metric as G  # noqa: E402
import measure_lag as L  # noqa: E402


def winpath(p: str) -> str:
    """Percorsi di Git Bash (/e/ai/...) -> percorsi Windows (E:/ai/...), come li scrive misure_h3_lib.sh nel jsonl."""
    import re
    return (p[1].upper() + ":" + p[2:]) if re.match(r"^/[a-zA-Z]/", p or "") else p


def oldbg_total(video: str, source: str, masks: str) -> str:
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "measure_oldbg.py"), video, source, masks], capture_output=True, text=True, timeout=900).stdout
        line = [ln for ln in out.splitlines() if ln.strip().startswith("TOTALE")]
        return line[0].split(":", 1)[1].strip() if line else "n/d"
    except Exception as ex:                                  # noqa: BLE001
        return f"errore {ex}"


def build_rows(entries, masks, source):
    rows = []
    mlist = sorted(glob.glob(os.path.join(masks, "*.png")))
    for en in entries:
        row = {"id": en["id"], "video": en.get("video", "")}
        try:
            gen = np.load(os.path.join(en["clipdir"], "gen.npy"), mmap_mode="r")
            src = np.load(os.path.join(en["clipdir"], "clip_src.npy"), mmap_mode="r")
            sm = L.summary(L.lag_series(gen, src, mlist))
            row.update(k_mediana=sm["k_mediana"], quota_fantasma=(sm["n_fantasma"] / sm["n"]) if sm["n"] else None)
        except Exception as ex:                              # noqa: BLE001
            row["lag_errore"] = str(ex)
        try:
            gm = G.measure_video(en["video"], masks, ((0, 6),), 6, 10, None, 0, source).get("0-6s", {})
            row.update(fedelta_testa=gm.get("fedelta_testa"), fuori_sagoma=gm.get("quota_fuori_sagoma"), volto_fantasma=gm.get("frame_con_volto_fantasma"))
        except Exception as ex:                              # noqa: BLE001
            row["ghost_errore"] = str(ex)
        row["oldbg"] = oldbg_total(en["video"], source, masks) if en.get("video") else "n/d"
        rows.append(row)
    return rows


def fmt(v, f="%.3f"):
    return "n/d" if v is None else (f % v)


def table(rows) -> str:
    out = ["ABLAZIONE (solo clip 1): k = ritardo temporale mediano della persona ridisegnata (0 = sincrona), fantasma = quota di frame col fantasma;",
           "fedelta testa (piu bassa = meglio) e persona fuori sagoma sono 0-6 s; sfondo vecchio = TOTALE di measure_oldbg.", ""]
    out.append("%-34s %8s %10s %12s %12s %12s  %s" % ("variante", "k med", "fantasma", "fedelta", "fuori sag.", "volto fant.", "sfondo vecchio"))
    for r in rows:
        out.append("%-34s %8s %10s %12s %12s %12s  %s" % (r["id"][:34], fmt(r.get("k_mediana"), "%.0f"), fmt(r.get("quota_fantasma"), "%.2f"),
                                                         fmt(r.get("fedelta_testa")), fmt(r.get("fuori_sagoma")), fmt(r.get("volto_fantasma"), "%.2f"), r.get("oldbg", "n/d")))
    return "\n".join(out)


def comparison_image(rows, path, times=(2.5, 4.5), w=432):
    tiles = []
    for r in rows:
        cap = cv2.VideoCapture(r.get("video") or "")
        line = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 24))
            ok, f = cap.read()
            f = cv2.resize(f, (w, int(w * 9 / 16))) if ok else np.zeros((int(w * 9 / 16), w, 3), np.uint8)
            label = "%s  %.1fs" % (r["id"].replace("abl_", "")[:30], t)
            cv2.putText(f, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(f, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            line.append(f)
        tiles.append(np.hstack(line))
    if tiles:
        cv2.imwrite(path, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    entries = [json.loads(ln) for ln in open(a[0], encoding="utf-8") if ln.strip()]
    for en in entries:
        en["clipdir"], en["video"] = winpath(en.get("clipdir", "")), winpath(en.get("video", ""))
    rows = build_rows(entries, a[1], a[2])
    txt = table(rows)
    with open(a[3], "a", encoding="utf-8") as fh:
        fh.write("\n" + txt + "\n")
    print(txt)
    comparison_image(rows, a[4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
