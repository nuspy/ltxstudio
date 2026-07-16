#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integra i media generati dentro marketing/tadamoo_landing/index.html (Fase B del README).

La pagina disegna ogni esempio con `devHTML(spec)` (uno "sviluppo" grafico in CSS). TUTTI i renderer
(single/slider/slider-sat/result-sats/yt/ytpair/audio) passano da lì: basta insegnare a devHTML a
mostrare il media VERO quando il pezzo ce l'ha, e ogni modalità si integra da sola.

Cosa fa (idempotente — si può rilanciare a ogni nuovo asset prodotto):
  1. CSS      → classe .dev-real (immagini/video che riempiono il frame come lo sviluppo)
  2. devHTML  → se il pezzo ha img:/video:, rende <img>/<video>; altrimenti lo sviluppo di prima
  3. ytFrame  → se il pezzo ha video:, player <video controls preload="none"> SELF-HOSTED (niente YouTube)
  4. EX       → inietta img:"assets/<id>.webp" / video:"assets/<id>.mp4" / audio:"assets/<id>.mp3"
                nei pezzi il cui file ESISTE davvero (i mancanti restano sviluppi grafici: la pagina
                resta bella e coerente anche a media incompleti)
  5. <audio>  → src dal data-asset-id; hero → data-video-src="assets/hero-bg-01.mp4"

Uso:  python scripts/landing_integrate.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANDING = REPO / "marketing" / "tadamoo_landing"
INDEX = LANDING / "index.html"
ASSETS = LANDING / "assets"

MEDIA_FIELD = {".webp": "img", ".mp4": "video", ".mp3": "audio"}
MIN_BYTES = 1024                                     # sotto = file rotto/vuoto, ignoralo

CSS_MARK = "/* --- media reali (Fase B) --- */"
CSS_BLOCK = """
/* --- media reali (Fase B) --- */
.dev-real{width:100%;height:100%;object-fit:cover;display:block;border-radius:inherit;background:#0a0f18}
.exd-frame video.dev-real,.half video.dev-real{cursor:pointer}
"""

DEV_MARK = "if(spec.img)"
DEV_NEW = """function devHTML(spec){
  /* Fase B: se il pezzo ha un media reale lo mostriamo al posto dello sviluppo grafico.
     Tutti i renderer passano di qui, quindi slider/satelliti/coppie si integrano da soli. */
  if(spec.img) return `<img class="dev-real" src="${spec.img}" alt="${spec.alt||spec.tag||''}" loading="lazy" decoding="async">`;
  if(spec.video) return `<video class="dev-real" src="${spec.video}" muted loop playsinline preload="metadata" aria-label="${spec.tag||''}"></video>`;
"""

YT_MARK = "if(p.video)"
YT_NEW = """function ytFrame(p,cls){
  /* Video SELF-HOSTED (niente YouTube): preload="none" → si scarica solo al play, non pesa sul first-paint. */
  if(p.video) return `<div class="${cls||'exd-frame'}" data-asset-id="${p.id}">
    <video class="dev-real" src="${p.video}" controls preload="none" playsinline aria-label="${p.tag}"></video>
    <span class="exd-tag">${p.tag}</span>
  </div>`;
"""


def log(m: str) -> None:
    print(m, flush=True)


def available_media() -> dict[str, tuple[str, str]]:
    """id-pezzo → (campo, path-relativo) per i file realmente presenti e non vuoti."""
    out: dict[str, tuple[str, str]] = {}
    for f in sorted(ASSETS.iterdir()):
        if not f.is_file() or f.name.startswith(("_", "placeholder")):
            continue
        field = MEDIA_FIELD.get(f.suffix.lower())
        if not field or f.stat().st_size < MIN_BYTES:
            continue
        out[f.stem] = (field, f"assets/{f.name}")
    return out


def patch_css(html: str) -> tuple[str, bool]:
    if CSS_MARK in html:
        return html, False
    i = html.find("</style>")
    if i < 0:
        raise RuntimeError("</style> non trovato in index.html")
    return html[:i] + CSS_BLOCK + html[i:], True


def patch_dev_html(html: str) -> tuple[str, bool]:
    if DEV_MARK in html:
        return html, False
    old = "function devHTML(spec){\n"
    if old not in html:
        raise RuntimeError("devHTML non trovato (firma cambiata?)")
    return html.replace(old, DEV_NEW, 1), True


def patch_yt_frame(html: str) -> tuple[str, bool]:
    if YT_MARK in html:
        return html, False
    old = "function ytFrame(p,cls){\n"
    if old not in html:
        raise RuntimeError("ytFrame non trovato (firma cambiata?)")
    return html.replace(old, YT_NEW, 1), True


def patch_ex_pieces(html: str, media: dict[str, tuple[str, str]]) -> tuple[str, int]:
    """Inietta img:/video:/audio: nei pezzi di EX che hanno un file vero. Idempotente."""
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        pid = m.group("pid")
        got = media.get(pid)
        if not got:
            return m.group(0)                        # nessun file → resta sviluppo grafico
        field, path = got
        if f'{field}:"' in m.group(0):               # già iniettato
            return m.group(0)
        n += 1
        return f'{{id:"{pid}",{field}:"{path}",'

    html = re.sub(r'\{id:"(?P<pid>[a-z0-9\-]+)",', sub, html)
    return html, n


def patch_cmp_stage(html: str, media: dict[str, tuple[str, str]]) -> tuple[str, int]:
    """Lo slider prima/dopo #cmp-stage vive nel MARKUP (non in EX): le due .side contengono la grafica
    (city+rainveil / city+sunbloom). Le sostituiamo con le due immagini vere, tenendo le .lab."""
    src, out = media.get("img-edit-01-src"), media.get("img-edit-01-out")
    if not (src and out) or "cmp-real" in html:
        return html, 0
    n = 0
    for side_cls, lab_cls, path, alt in (
            ("before", "l", src[1], "Prima della modifica"),
            ("after", "r", out[1], "Dopo la modifica")):
        pat = re.compile(r'(<div class="side %s">)(.*?)(<span class="lab %s">)' % (side_cls, lab_cls), re.S)
        m = pat.search(html)
        if not m:
            continue
        img = (f'\n          <img class="dev-real cmp-real" src="{path}" alt="{alt}" '
               f'loading="lazy" decoding="async">\n          ')
        html = html[:m.start(2)] + img + html[m.end(2):]
        n += 1
    if n:
        log(f"  slider #cmp-stage: {n} lati collegati alle immagini reali")
    return html, n


def patch_audio_and_hero(html: str, media: dict[str, tuple[str, str]]) -> tuple[str, int]:
    """<audio> → src reale; hero → data-video-src.

    Gli <audio> NON stanno nel markup: li genera mediaHTML come template literal, con l'id INTERPOLATO
    (data-asset-id="${p[0].id}") → una regex sull'id letterale non li trova mai. Patchiamo il TEMPLATE
    aggiungendo src dal campo audio: del pezzo (vuoto → nessuna sorgente, resta lo sviluppo grafico)."""
    n = 0
    for idx in (0, 1):                                   # case 'audio' usa p[0]; 'audiopair' usa p[0] e p[1]
        old = f'data-asset-id="${{p[{idx}].id}}"></audio>'
        new = f'data-asset-id="${{p[{idx}].id}}" src="${{p[{idx}].audio||\'\'}}"></audio>'
        if old in html:
            cnt = html.count(old)
            html = html.replace(old, new)
            n += cnt
    if n:
        log(f"  <audio>: {n} template collegati al campo audio: del pezzo")

    hero = media.get("hero-bg-01")
    if hero and 'data-video-src=""' in html:
        html = html.replace('data-video-src=""', f'data-video-src="{hero[1]}"', 1)
        n += 1
        log("  hero: data-video-src collegato")
    return html, n


def main() -> int:
    ap = argparse.ArgumentParser(description="Integra i media generati in index.html")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not INDEX.exists():
        log(f"index.html non trovato: {INDEX}")
        return 2
    media = available_media()
    if not media:
        log("nessun media in assets/ — niente da integrare")
        return 0

    log(f"MEDIA DISPONIBILI ({len(media)}):")
    for pid, (field, path) in media.items():
        log(f"  {field:<6} {pid:<24} → {path}")

    html = INDEX.read_text(encoding="utf-8")
    html, c1 = patch_css(html)
    html, c2 = patch_dev_html(html)
    html, c3 = patch_yt_frame(html)
    html, n_ex = patch_ex_pieces(html, media)
    html, n_cmp = patch_cmp_stage(html, media)
    html, n_ah = patch_audio_and_hero(html, media)

    log("")
    log(f"PATCH: css={'sì' if c1 else 'già'}  devHTML={'sì' if c2 else 'già'}  "
        f"ytFrame={'sì' if c3 else 'già'}  pezzi-EX={n_ex}  cmp-stage={n_cmp}  audio/hero={n_ah}")

    if args.dry_run:
        log("(dry-run: nessuna scrittura)")
        return 0

    bak = LANDING / "index.beforeB.html"
    if not bak.exists():
        shutil.copy2(INDEX, bak)                     # backup una-tantum del pre-integrazione
        log(f"backup: {bak.name}")
    INDEX.write_text(html, encoding="utf-8")
    log(f"scritto: {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
