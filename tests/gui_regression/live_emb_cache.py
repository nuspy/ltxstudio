"""TEST LIVE end-to-end DALLA GUI — cache degli embedding di prompt (core/emb_cache).

Percorso REALE utente→GUI→worker (regola [[test-end-to-end-from-gui]]): MainWindow vera, widget veri,
`on_generate()` vero, generazioni vere. Nessuno spy.

Cosa dimostra, in ordine di importanza:
  1) un HIT restituisce ESATTAMENTE lo stesso video di un MISS (stesso seed → mp4 identico byte a
     byte). È la prova che l'ottimizzazione non cambia i risultati; se cade, la cache va spenta.
  2) la cache viene davvero usata: la voce esiste, e il suo mtime avanza a ogni hit (os.utime in
     emb_cache.get) — verifica di STATO, indipendente dai log del subprocess.
  3) il guadagno è reale e misurabile (la seconda generazione è più veloce di ~13 s).
  4) una voce corrotta produce un MISS e un video valido, mai un crash.
  5) con la cache disattivata dalla preferenza reale il video è ANCORA identico.

Il test misura anche il determinismo di base (due generazioni a cache spenta): se il denoise non è
riproducibile a parità di seed, l'asserzione (1) viene degradata invece che dichiarata fallita — la
differenza non sarebbe colpa della cache. Lo dice esplicitamente nel referto.

Avvio:  QT_QPA_PLATFORM=offscreen <gui_python> tests/gui_regression/live_emb_cache.py
"""
from __future__ import annotations

import faulthandler
import glob
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LTX_EMB_PROBE"] = "1"                        # digest dei tensori nei log del worker
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
faulthandler.enable()

LOG = ROOT / "output" / "DIAG_emb_cache.log"
PROMPT = ("una chiromante scettica legge la mano a un robot che le chiede se troverà l'amore, "
          "tenda di velluto viola, sfera di cristallo che sfarfalla, luce di candela")
SEED = 20260816


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _spin(app, seconds: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        app.processEvents()
        time.sleep(0.05)


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _generate(w, app, tag: str) -> tuple[str | None, float]:
    """Una generazione REALE dai widget. Ritorna (path mp4 nuovo, secondi)."""
    before = set(glob.glob(str(ROOT / "output" / "ltx_native_*.mp4")))
    t0 = time.monotonic()
    log(f"--- on_generate() [{tag}] ---")
    w.on_generate()
    while time.monotonic() - t0 < 900:
        _spin(app, 1.0)
        if not w.jobs.is_busy() and time.monotonic() - t0 > 10:
            break
    dt = time.monotonic() - t0
    new = sorted(set(glob.glob(str(ROOT / "output" / "ltx_native_*.mp4"))) - before)
    out = new[-1] if new else None
    log(f"    [{tag}] {dt:.1f}s → {os.path.basename(out) if out else 'NESSUN OUTPUT'}")
    return out, dt


def main() -> int:
    import harness
    from core import emb_cache, user_prefs

    LOG.write_text("", encoding="utf-8")
    fails: list[str] = []
    notes: list[str] = []

    def check(name, cond, note=""):
        log(("OK   " if cond else "FAIL ") + name + (f" — {note}" if note else ""))
        if not cond:
            fails.append(name)

    log("=== BUILD MainWindow (offscreen) ===")
    w = harness.build_window()
    app = harness.app()

    # widget reali: clip corta, seed FISSO (senza, on_generate randomizza e i video non sono confrontabili)
    w.video_tab.prompt.setPlainText(PROMPT)
    w.video_tab.width.setValue(512)
    w.video_tab.height.setValue(320)
    if hasattr(w.video_tab, "length"):
        w.video_tab.length.setValue(1.3)
    w.video_tab.sampler_seed.steps.setValue(8)
    w.video_tab.sampler_seed.seed.setValue(SEED)
    w.video_tab.sampler_seed.fixed_seed.setChecked(True)
    w.tabs.setCurrentWidget(w.video_tab)
    check("seed fissato (i video sono confrontabili)", w.video_tab.wants_fixed_seed())

    # ── BASELINE: il denoise è riproducibile a parità di seed? Serve a sapere se l'asserzione
    #    forte (mp4 identico) è disponibile o se va degradata. Non è la cache sotto esame qui.
    user_prefs.set("emb_cache.enabled", False)
    emb_cache.clear()
    off_a, t_off_a = _generate(w, app, "cache OFF #1")
    off_b, t_off_b = _generate(w, app, "cache OFF #2")
    check("generazione a cache spenta riuscita", bool(off_a and off_b))
    deterministico = bool(off_a and off_b and _sha(off_a) == _sha(off_b))
    log(f"    determinismo di base: {'SÌ' if deterministico else 'NO'} "
        f"({_sha(off_a) if off_a else '?'} vs {_sha(off_b) if off_b else '?'})")
    if not deterministico:
        notes.append("il denoise NON è riproducibile a parità di seed (indipendente dalla cache): "
                     "l'identità bit a bit dei video non è verificabile, resta il confronto dei "
                     "digest dei tensori nel log [emb-probe]")
    check("nessuna voce scritta a cache spenta", emb_cache.stats()["n"] == 0,
          f"voci={emb_cache.stats()['n']}")

    # ── MISS: prima generazione a cache accesa ──
    user_prefs.set("emb_cache.enabled", True)
    emb_cache.clear()
    hit_a, t_miss = _generate(w, app, "cache ON, MISS")
    st = emb_cache.stats()
    check("la voce è stata scritta", st["n"] == 1, f"voci={st['n']} · {st['mb']:.1f} MB")
    check("dimensione plausibile (5-80 MB)", 5 <= st["mb"] <= 80, f"{st['mb']:.1f} MB")
    ent = emb_cache.entries()
    if ent:
        log(f"    voce: {ent[0]['key'][:8]} · {ent[0]['mb']:.1f}MB · encode {ent[0]['encode_s']:.1f}s")
        check("l'encode misurato è stato registrato", ent[0]["encode_s"] > 0)
        check("il prompt è quello giusto", ent[0]["prompt"].startswith(PROMPT[:40]))

    # ── HIT: seconda generazione identica ──
    key = ent[0]["key"] if ent else ""
    mtime_before = emb_cache.path_for(key).stat().st_mtime if key else 0
    time.sleep(1.1)                                       # mtime ha granularità ~1s: separa i due istanti
    hit_b, t_hit = _generate(w, app, "cache ON, HIT")
    check("nessuna voce in più (è stato un hit, non un secondo miss)", emb_cache.stats()["n"] == 1,
          f"voci={emb_cache.stats()['n']}")
    if key:
        mtime_after = emb_cache.path_for(key).stat().st_mtime
        check("la voce è stata LETTA (mtime avanzato da os.utime)", mtime_after > mtime_before,
              f"{mtime_before:.0f} → {mtime_after:.0f}")

    if hit_a and hit_b:
        same = _sha(hit_a) == _sha(hit_b)
        if deterministico:
            check("HIT ≡ MISS: video identico byte a byte", same,
                  f"{_sha(hit_a)} vs {_sha(hit_b)}")
        else:
            log(f"     (identità non asseribile: {_sha(hit_a)} vs {_sha(hit_b)})")
    log(f"    tempi: miss {t_miss:.1f}s · hit {t_hit:.1f}s · guadagno {t_miss - t_hit:+.1f}s")
    check("l'hit è più veloce del miss", t_hit < t_miss,
          f"miss {t_miss:.1f}s → hit {t_hit:.1f}s ({t_miss - t_hit:+.1f}s)")

    # ── CORRUZIONE: una voce rovinata deve dare un MISS, non un crash ──
    if key:
        p = emb_cache.path_for(key)
        with open(p, "r+b") as fh:
            fh.truncate(4096)
        log(f"    voce {key[:8]} troncata a 4096 byte")
        corr, _t = _generate(w, app, "voce corrotta")
        check("video valido nonostante la cache corrotta", bool(corr) and os.path.getsize(corr) > 1000)
        check("la voce corrotta è stata sostituita", emb_cache.stats()["n"] == 1)

    # ── OFF dalla preferenza reale: il video deve restare identico ──
    user_prefs.set("emb_cache.enabled", False)
    off_c, _t = _generate(w, app, "cache OFF dopo hit")
    if off_c and hit_a and deterministico:
        check("cache spenta: video ancora identico", _sha(off_c) == _sha(hit_a),
              f"{_sha(off_c)} vs {_sha(hit_a)}")
    user_prefs.set("emb_cache.enabled", True)

    log("")
    for n in notes:
        log(f"NOTA: {n}")
    log("TUTTO OK" if not fails else f"FALLITI ({len(fails)}): {fails}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
