"""Grafico a candele: ComfyUI vs Backend nativo (caricamento + generazione).

Legge output/bench_native.json e output/bench_comfy.json (prodotti da _bench.py / _bench_comfy.py)
e genera output/comfy_compare.png. Ogni candela: corpo = tempo di CARICAMENTO modello (da
"gen" a "load_gen"); il tappo inferiore = generazione pura (modello già caricato). I valori
sono annotati. Esegui con un Python che ha matplotlib (es. il python di sistema).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

OUT = r"E:\ai\ltxstudio\output"
COMFY_COLOR = "#e8893b"
NATIVE_COLOR = "#3b9ee8"


def _load(name: str) -> dict:
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def main() -> int:
    nat = _load("bench_native.json")
    com = _load("bench_comfy.json")
    models = list(dict.fromkeys(list(com.keys()) + list(nat.keys())))
    if not models:
        print("Nessun dato benchmark."); return 1
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.4), 6.2))
    w = 0.30

    def candle(x: float, d: dict):
        gen, lg = d.get("gen"), d.get("load_gen")
        col = COMFY_COLOR if d.get("_who") == "comfy" else NATIVE_COLOR
        if gen is None or lg is None:
            return
        ax.add_patch(Rectangle((x - w / 2, gen), w, max(lg - gen, 0.2),
                               facecolor=col, edgecolor="black", alpha=0.88))
        ax.plot([x - w / 2, x + w / 2], [gen, gen], color="black", lw=1.6)
        ax.annotate(f"{lg:.0f}s", (x, lg), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.annotate(f"gen {gen:.0f}s", (x, gen), ha="center", va="top", fontsize=7)

    xt, xl = [], []
    for i, m in enumerate(models):
        xc = i * 1.0
        if m in com:
            d = dict(com[m]); d["_who"] = "comfy"; candle(xc - 0.18, d)
        if m in nat:
            d = dict(nat[m]); d["_who"] = "native"; candle(xc + 0.18, d)
        xt.append(xc); xl.append(m)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("secondi  (più basso = più veloce)")
    ax.set_title("ComfyUI vs Backend nativo — caricamento modello + generazione\n"
                 "Corpo candela = caricamento · tappo inferiore = generazione pura (modello caricato)")
    ax.legend(handles=[Patch(facecolor=COMFY_COLOR, label="ComfyUI"),
                       Patch(facecolor=NATIVE_COLOR, label="Backend nativo")])
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(OUT, "comfy_compare.png")
    plt.savefig(out, dpi=130)
    print("SAVED", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
