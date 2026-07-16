#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ottimizza per il WEB gli asset della landing (da eseguire prima dell'integrazione/deploy).

Il worker LTX scrive mp4 a bitrate altissimo (vid-long-01 = 132MB per 60s!) e lascia in assets/ i
chunk intermedi del chunking (vid-long-01_chunkN.mp4): roba da lavorazione, non da pubblicare.

Cosa fa (idempotente — si può rilanciare):
  1. rimuove i *_chunkN.mp4 (file di lavoro del chunking autoregressivo)
  2. ricomprime ogni mp4 per il web: h264 CRF 28, preset slow, +faststart (play immediato in streaming),
     audio AAC 128k. L'hero resta muto. Tipico: 10-20× più leggero, invisibile in un player di pagina.
  3. salta i file già ottimizzati (marker _optimized.json con la dimensione attesa)

Uso:  python scripts/landing_optimize.py [--dry-run] [--crf 28]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "marketing" / "tadamoo_landing" / "assets"
WORK = ASSETS / "_work"
MARKER = ASSETS / "_optimized.json"
HERO = "hero-bg-01"                                   # sfondo: muto (autoplay) e già leggero


def log(m: str) -> None:
    print(m, flush=True)


def ffmpeg_exe() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                 # noqa: BLE001
        raise RuntimeError("ffmpeg non trovato")


def mb(p: Path) -> float:
    return p.stat().st_size / 1048576


def load_marker() -> dict:
    if MARKER.exists():
        try:
            return json.loads(MARKER.read_text(encoding="utf-8"))
        except Exception:                             # noqa: BLE001
            pass
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ottimizza gli asset video della landing per il web")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--crf", type=int, default=28, help="qualità h264 (più alto = più leggero)")
    args = ap.parse_args()

    if not ASSETS.exists():
        log(f"assets non trovati: {ASSETS}")
        return 2

    # 1) chunk intermedi del chunking → via da assets (non sono asset)
    chunks = sorted(ASSETS.glob("*_chunk*.mp4"))
    tot_chunk = sum(mb(c) for c in chunks)
    if chunks:
        log(f"CHUNK INTERMEDI da rimuovere: {len(chunks)} file, {tot_chunk:.1f} MB")
        for c in chunks:
            log(f"  - {c.name} ({mb(c):.1f} MB)")
            if not args.dry_run:
                WORK.mkdir(parents=True, exist_ok=True)
                shutil.move(str(c), str(WORK / c.name))   # non li cancello: li archivio in _work
        if not args.dry_run:
            log(f"  → spostati in {WORK.name}/")

    # 2) ricompressione web dei video pubblicabili
    marker = load_marker()
    vids = [v for v in sorted(ASSETS.glob("*.mp4"))
            if "_chunk" not in v.name and not v.name.startswith("placeholder")]
    log("")
    log(f"VIDEO da ottimizzare (CRF {args.crf}):")
    saved = 0.0
    for v in vids:
        before = mb(v)
        if v.stem == HERO:                            # già ricompresso in generazione (web=True): non toccarlo,
            log(f"  = {v.name:<24} {before:6.2f} MB  (hero: già web, non ricomprimo)")
            continue                                  # una seconda passata h264 degraderebbe e basta
        if marker.get(v.name) == round(before, 3):
            log(f"  = {v.name:<24} {before:6.2f} MB  (già ottimizzato)")
            continue
        if args.dry_run:
            log(f"  ▶ {v.name:<24} {before:6.2f} MB  → ricomprimo")
            continue
        tmp = v.with_suffix(".opt.mp4")
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(v),
               "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        cmd += ["-an"] if v.stem == HERO else ["-c:a", "aac", "-b:a", "128k"]
        cmd += [str(tmp)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:    # noqa: PERF203
            log(f"  ! {v.name}: ffmpeg fallito ({e}) — lascio l'originale")
            tmp.unlink(missing_ok=True)
            continue
        after = mb(tmp)
        if after >= before:                           # già compresso bene: tieni l'originale
            log(f"  = {v.name:<24} {before:6.2f} MB  (ricompressione inutile, tengo l'originale)")
            tmp.unlink(missing_ok=True)
            marker[v.name] = round(before, 3)
            continue
        tmp.replace(v)
        saved += before - after
        marker[v.name] = round(mb(v), 3)
        log(f"  ✔ {v.name:<24} {before:6.2f} → {after:5.2f} MB  ({before / max(after, .01):.1f}× più leggero)")

    if not args.dry_run:
        MARKER.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    # 3) riepilogo peso pubblicabile
    pub = [f for f in ASSETS.iterdir()
           if f.is_file() and not f.name.startswith(("_", "placeholder")) and "_chunk" not in f.name]
    tot = sum(mb(f) for f in pub)
    log("")
    log(f"RISPARMIO: {saved:.1f} MB (video) + {tot_chunk:.1f} MB (chunk archiviati)")
    log(f"PESO PUBBLICABILE: {tot:.1f} MB su {len(pub)} file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
