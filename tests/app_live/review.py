"""Pagina di revisione del giro (task 25): miniatura cliccabile per caso, cosa è stato fatto / cosa verificare,
prompt, modello, LoRA, originale (e maschera) affiancati per gli edit, un textbox per caso (localStorage) e, in
fondo, «Pubblica» che crea un .md con i SOLI casi con textbox compilato (vuoto = superato).

    python -m tests.app_live.review build <run_dir> [nome]     # genera orch\\app_live\\review_<data>\\index.html
    python -m tests.app_live.review serve <review_dir> [porta]  # server locale: «Pubblica» scrive il .md nella cartella
Aperta da disco (file://) «Pubblica» scarica il file (ripiego).
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import time

ORCH = r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live"
IMG = (".png", ".jpg", ".jpeg", ".webp")
VID = (".mp4", ".mov", ".webm")


def _thumb(src: str, dst_dir: str, name: str, big: bool = False):
    """Ritorna il percorso relativo della miniatura (JPEG) di un'immagine o di un frame video; None se impossibile."""
    try:
        from PIL import Image
        os.makedirs(dst_dir, exist_ok=True)
        rel = f"img/{name}{'_big' if big else ''}.jpg"
        dst = os.path.join(dst_dir, rel)
        ext = os.path.splitext(src)[1].lower()
        if ext in VID:
            import cv2
            cap = cv2.VideoCapture(src)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
            ok, fr = cap.read()
            cap.release()
            if not ok:
                return None
            im = Image.fromarray(fr[:, :, ::-1])
        else:
            im = Image.open(src).convert("RGB")
        im.thumbnail((1600, 1600) if big else (480, 480))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst, quality=88 if big else 80)
        return rel
    except Exception:                                        # noqa: BLE001
        return None


def _overlay(img_path: str, mask_path: str, dst_dir: str, name: str):
    """Miniatura (piccola e grande) della foto con la MASCHERA sovrapposta in rosso (bianco/alpha della maschera = zona rigenerata)."""
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(img_path).convert("RGB")
        mk = Image.open(mask_path)
        a = np.asarray(mk.convert("RGBA"))
        m = (a[..., 3] > 127) & (a[..., :3].max(axis=2) > 127) if mk.mode in ("RGBA", "LA", "P") and a[..., 3].min() < 255 else (a[..., :3].max(axis=2) > 127)
        m = np.asarray(Image.fromarray((m * 255).astype("uint8")).resize(im.size, Image.NEAREST)) > 127
        arr = np.asarray(im).astype(np.float32)
        arr[m] = 0.55 * arr[m] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
        arr[edge] = (255, 255, 0)
        out = Image.fromarray(arr.clip(0, 255).astype("uint8"))
        res = {}
        for big in (False, True):
            o = out.copy()
            o.thumbnail((1600, 1600) if big else (480, 480))
            rel = f"img/{name}{'_big' if big else ''}.jpg"
            os.makedirs(os.path.join(dst_dir, "img"), exist_ok=True)
            o.save(os.path.join(dst_dir, rel), quality=88 if big else 80)
            res["big" if big else "thumb"] = rel
        return res
    except Exception:                                        # noqa: BLE001
        return None


def _what_to_check(c: dict, r: dict) -> str:
    if c.get("verify"):
        return c["verify"]
    fn = c.get("fn")
    bits = []
    if r.get("dry"):
        return "PROVA A SECCO: nessuna generazione. Controllare solo che modello, LoRA, forza e prompt qui sotto siano quelli impostati."
    if fn == "create_image":
        bits.append("L'immagine corrisponde al prompt e al formato richiesto, senza deformazioni o macchie.")
        if c.get("loras"):
            bits.append("L'effetto/stile delle LoRA elencate è visibile (e non è un'immagine identica a quella senza LoRA).")
    elif fn == "video":
        bits.append("Il video corrisponde al prompt, movimento fluido, nessun lampeggio, audio presente se richiesto.")
        if c.get("loras"):
            bits.append("L'effetto delle LoRA elencate è visibile.")
    else:
        bits.append("Il risultato corrisponde a quanto impostato.")
    return " ".join(bits)


def build(run_dir: str, nome: str = "giro") -> str:
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") if l.strip()]
    stamp = time.strftime("%Y%m%d_%H%M")
    rd = os.path.join(ORCH, f"review_{stamp}")
    os.makedirs(rd, exist_ok=True)
    cards = []
    for i, r in enumerate(rows):
        c = r.get("case", {})
        cid = str(c.get("id", r.get("id", i)))
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cid)
        calls = r.get("calls") or []
        q = next((x for x in calls if x["via"] == "queue"), None)
        p = (q or {}).get("params", {}) if q else {}
        ws = r.get("widget_state") or {}
        outs = r.get("uscite") or []
        media = []
        for k, o in enumerate(outs[:4]):
            t, b = _thumb(o, rd, f"{safe}_out{k}"), _thumb(o, rd, f"{safe}_out{k}", True)
            media.append({"path": o, "thumb": t, "big": b, "video": o.lower().endswith(VID)})
        origs = []
        iniz = None
        if c.get("fn") == "video":                                          # «Immagine iniziale: nessuna / miniatura» (bug 31/10)
            sp_ = p.get("image_path") or next((k.get("path") for k in (p.get("keyframes") or []) if isinstance(k, dict) and k.get("path")), None)
            iniz = {"path": sp_ or "", "thumb": _thumb(sp_, rd, f"{safe}_iniz") if sp_ and os.path.isfile(sp_) else None,
                    "big": _thumb(sp_, rd, f"{safe}_iniz", True) if sp_ and os.path.isfile(sp_) else None}
        for key in (("init_image", "mask", "mask_path", "control_image") if c.get("fn") == "video" else ("init_image", "image_path", "mask", "mask_path", "control_image")):
            v = p.get(key)
            if isinstance(v, str) and os.path.isfile(v):
                origs.append({"label": key, "path": v, "thumb": _thumb(v, rd, f"{safe}_{key}"), "big": _thumb(v, rd, f"{safe}_{key}", True)})
        for k, rr in enumerate(p.get("references") or []):                  # riferimenti multipli video (+ sfondo): originali affiancati
            v = rr.get("path") if isinstance(rr, dict) else rr
            if isinstance(v, str) and os.path.isfile(v):
                lab = "SFONDO" if isinstance(rr, dict) and rr.get("role") == "background" else f"riferimento {k + 1}"
                origs.append({"label": lab, "path": v, "thumb": _thumb(v, rd, f"{safe}_vref{k}"), "big": _thumb(v, rd, f"{safe}_vref{k}", True)})
        for key2 in ("video",):
            v = p.get(key2)
            if isinstance(v, str) and os.path.isfile(v):
                origs.append({"label": "video originale", "path": v, "thumb": _thumb(v, rd, f"{safe}_srcvid"), "big": _thumb(v, rd, f"{safe}_srcvid", True)})
        for k, v in enumerate(p.get("ref_images") or []):
            if isinstance(v, str) and os.path.isfile(v):
                origs.append({"label": f"riferimento {k + 1}", "path": v, "thumb": _thumb(v, rd, f"{safe}_ref{k}"), "big": _thumb(v, rd, f"{safe}_ref{k}", True)})
        _im = next((p.get(k) for k in ("init_image", "image_path") if isinstance(p.get(k), str) and os.path.isfile(p.get(k))), None)
        _mk = next((p.get(k) for k in ("mask", "mask_path") if isinstance(p.get(k), str) and os.path.isfile(p.get(k))), None)
        if _im and _mk and c.get("fn") != "video":                           # inpaint / Integra: MASCHERA SOVRAPPOSTA alla foto (richiesta 03/11)
            ov = _overlay(_im, _mk, rd, f"{safe}_overlay")
            if ov:
                origs.append({"label": "MASCHERA SOVRAPPOSTA (rosso = zona rigenerata)", "path": _mk, **ov})
        loras = [{"file": os.path.basename(str(L.get("file"))), "strength": L.get("strength"), "trigger": ", ".join(L.get("triggers") or [])}
                 for L in (ws.get("loras") or [])]
        cards.append({
            "id": cid, "fn": c.get("fn"), "esito": r.get("esito"), "motivo": r.get("motivo") or "", "dry": bool(r.get("dry")),
            "tempo": r.get("tempo_s"), "vram": r.get("vram_peak_mb"), "ram": r.get("ram_peak_mb"), "commit": r.get("commit_free_min_mb"),
            "handler": r.get("handler"), "arch": p.get("arch") or c.get("arch"), "modello": p.get("model") or (ws.get("model") or c.get("model")),
            "prompt": c.get("prompt") or p.get("prompt"), "negativo": c.get("negative") or p.get("negative"),
            "params": {k: v for k, v in p.items() if k not in ("prompt", "negative", "loras")}, "loras": loras,
            "media": media, "orig": origs, "verifica": _what_to_check(c, r), "descr": c.get("descr") or "",
            "dialoghi": [d.get("text", "")[:300] for d in (r.get("dialoghi") or [])], "iniz": iniz,
        })
    data = {"nome": nome, "run": run_dir, "creato": time.strftime("%Y-%m-%d %H:%M"), "cards": cards}
    open(os.path.join(rd, "data.json"), "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, default=str))
    open(os.path.join(rd, "index.html"), "w", encoding="utf-8").write(PAGE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")))
    shutil.copy2(os.path.join(run_dir, "results.jsonl"), os.path.join(rd, "results.jsonl"))
    return os.path.join(rd, "index.html")


def serve(rd: str, port: int = 8766):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, unquote, urlparse
    roots = [os.path.abspath(rd), r"E:\ai\ltxstudio", ORCH, r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch"]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/file":
                f = os.path.abspath(unquote(parse_qs(u.query).get("p", [""])[0]))
                if not any(f.lower().startswith(r.lower()) for r in roots) or not os.path.isfile(f):
                    return self._send(404, b"no")
                ctype = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".webm": "video/webm"}.get(os.path.splitext(f)[1].lower(), "application/octet-stream")
                return self._send(200, open(f, "rb").read(), ctype)
            rel = u.path.lstrip("/") or "index.html"
            f = os.path.abspath(os.path.join(rd, rel))
            if not f.startswith(os.path.abspath(rd)) or not os.path.isfile(f):
                return self._send(404, b"no")
            ctype = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".json": "application/json"}.get(os.path.splitext(f)[1].lower(), "application/octet-stream")
            self._send(200, open(f, "rb").read(), ctype)

        def do_POST(self):
            if self.path != "/publish":
                return self._send(404, b"no")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            name = f"pubblicato_{time.strftime('%Y%m%d_%H%M%S')}.md"
            path = os.path.join(rd, name)
            open(path, "wb").write(body)
            self._send(200, json.dumps({"ok": True, "path": path}).encode(), "application/json")
    print(f"revisione su http://127.0.0.1:{port}/  (Ctrl+C per chiudere); «Pubblica» scrive in {rd}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()


def open_in_browser(index_html: str, port: int = 8766) -> str:
    """Apre la pagina di revisione nel browser, servita da `serve` (cosi «Pubblica» scrive il .md nella cartella).
    Il server resta vivo in un processo staccato (si chiude con la finestra del terminale o dal task manager)."""
    import socket
    import webbrowser
    rd = os.path.dirname(index_html)
    for p in range(port, port + 20):                     # prima porta libera
        s = socket.socket()
        s.settimeout(0.5)
        busy = s.connect_ex(("127.0.0.1", p)) == 0
        s.close()
        if not busy:
            port = p
            break
    kw = {"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" else {}      # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", rd, str(port)], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **kw)
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open(url)
    return url


PAGE = r"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisione giro test</title>
<style>
:root{--bg:#f6f7f9;--fg:#15171c;--mut:#5b6270;--card:#fff;--bd:#d9dde4;--ok:#1b7a3d;--ko:#b3261e;--wa:#9a6700;--ac:#2456d6}
@media (prefers-color-scheme:dark){:root{--bg:#101216;--fg:#e8eaee;--mut:#9aa3b2;--card:#181b21;--bd:#2b303a;--ok:#5fcf86;--ko:#ff8a80;--wa:#e3b341;--ac:#8aa8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif;padding:0 16px 120px;max-width:1100px;margin-inline:auto;overflow-x:hidden;overflow-wrap:anywhere;word-break:break-word}.card,.row>div{min-width:0;max-width:100%}
h1{font-size:1.3rem;margin:16px 0 4px}.sub{color:var(--mut);font-size:.9rem}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--bd);display:flex;gap:8px;flex-wrap:wrap;align-items:center;z-index:5}
button,select{min-height:44px;padding:0 14px;border-radius:8px;border:1px solid var(--bd);background:var(--card);color:var(--fg);font:inherit;cursor:pointer}
button.pri{background:var(--ac);color:#fff;border-color:var(--ac)}button:focus-visible,textarea:focus-visible,a:focus-visible{outline:3px solid var(--ac);outline-offset:2px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;margin:14px 0;padding:12px}
.head{display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}.id{font-weight:700;overflow-wrap:anywhere}
.tag{font-size:.8rem;padding:2px 8px;border-radius:99px;border:1px solid currentColor}.ok{color:var(--ok)}.fallito,.errore,.crash{color:var(--ko)}.dry{color:var(--wa)}
.meta{color:var(--mut);font-size:.85rem;overflow-wrap:anywhere}.row{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.thumb{display:block;max-width:100%}.thumb img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--bd);display:block}
.cap{font-size:.78rem;color:var(--mut)}.box{background:var(--bg);border-radius:8px;padding:8px;font-size:.88rem;overflow-wrap:anywhere;white-space:pre-wrap}
.todo{border-left:4px solid var(--ac);padding:4px 10px;margin:8px 0}textarea{width:100%;min-height:76px;border-radius:8px;border:1px solid var(--bd);background:var(--bg);color:var(--fg);font:inherit;padding:8px}
details summary{cursor:pointer;min-height:44px;display:flex;align-items:center;color:var(--mut)}
.vb{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}.vbtn{min-width:96px}.vbtn.okb.on{background:var(--ok);color:#fff;border-color:var(--ok)}.vbtn.kob.on{background:var(--ko);color:#fff;border-color:var(--ko)}.err{color:var(--ko)}#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:50;padding:8px}#lb.on{display:flex}#lb img,#lb video{max-width:100%;max-height:100%}
.foot{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--bd);padding:10px 16px calc(10px + env(safe-area-inset-bottom));display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;z-index:6}
</style></head><body>
<h1 id="t"></h1><div class="sub" id="s"></div>
<div class="bar"><select id="flt" aria-label="Filtro"><option value="all">Tutti</option><option value="bad">Non riusciti dal driver</option><option value="note">Con mia nota</option></select><span class="sub" id="cnt"></span></div>
<div id="list"></div>
<div class="foot"><span id="fc" class="sub"></span><button class="pri" id="pub">Pubblica</button><span id="pm" class="sub" role="status"></span></div>
<div id="lb" role="dialog" aria-label="Anteprima"></div>
<script>
const D=/*DATA*/null;const KEY='rev:'+D.creato+':'+D.nome;
const H=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let notes={};try{notes=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let ver={};try{ver=JSON.parse(localStorage.getItem(KEY+':v')||'{}')}catch(e){}
const isBad=c=>c.esito!=='ok';
const OKRE=/^\s*(ok|va bene|superato|tutto ok)\W*$/i;
const verdict=c=>ver[c.id]||(OKRE.test(notes[c.id]||'')?'ok':'');
const fileUrl=p=>location.protocol==='file:'?('file:///'+p.replace(/\\/g,'/')):('/file?p='+encodeURIComponent(p));
document.getElementById('t').textContent='Revisione: '+D.nome;
document.getElementById('s').textContent=D.creato+' · '+D.cards.length+' casi · cartella '+D.run;
function card(c,i){const m=c.media.map(x=>`<div><a class="thumb" href="#" data-big="${H(x.video?fileUrl(x.path):x.big||'')}" data-video="${x.video?1:0}" aria-label="Apri ${H(c.id)}">${x.thumb?`<img loading="lazy" src="${x.thumb}" alt="risultato ${H(c.id)}">`:'<span class=meta>(nessuna miniatura)</span>'}</a><div class="cap">${x.video?'▶ video · ':''}${H(x.path)}</div></div>`).join('');
const o=c.orig.map(x=>`<div><a class="thumb" href="#" data-big="${H(x.big||'')}"><img loading="lazy" src="${x.thumb||''}" alt="${H(x.label)}"></a><div class="cap">ORIGINALE · ${H(x.label)}</div></div>`).join('');
const lo=c.loras.length?c.loras.map(l=>`${H(l.file)} @${H(l.strength)}${l.trigger?' · trigger: '+H(l.trigger):''}`).join('<br>'):'nessuna';
return `<section class="card" data-i="${i}"><div class="head"><span class="id">${H(c.id)}</span><span class="tag ${H(c.esito)}">${H(c.esito)}</span>${c.dry?'<span class="tag dry">prova a secco</span>':''}<span class="meta">${H(c.fn)} · ${c.tempo!=null?c.tempo+' s':''}${c.vram?' · VRAM '+c.vram+' MB':''}${c.ram?' · RAM '+c.ram+' MB':''}</span></div>
${c.descr?`<div class="meta">${H(c.descr)}</div>`:''}${c.motivo?`<div class="err">${H(c.motivo)}</div>`:''}${c.dialoghi.length?`<div class="err">Errore in GUI: ${H(c.dialoghi.join(' | '))}</div>`:''}
<div class="row">${o}${m||'<span class=meta>nessuna uscita</span>'}</div>
<div class="todo"><b>Da verificare:</b> ${H(c.verifica)}</div>
<div class="meta">arch <b>${H(c.arch)}</b> · modello <b>${H(c.modello)}</b> · handler ${H(c.handler)}<br>LoRA: ${lo}</div>
<details><summary>Prompt e parametri</summary><div class="box"><b>Prompt:</b> ${H(c.prompt)}\n<b>Negativo:</b> ${H(c.negativo)}\n<b>Parametri al backend:</b> ${H(JSON.stringify(c.params))}</div></details>
<div class="vb" role="group" aria-label="Giudizio su ${H(c.id)}"><button type="button" class="vbtn okb${verdict(c)==='ok'?' on':''}" data-v="ok" data-i="${i}" aria-pressed="${verdict(c)==='ok'}">✓ ok</button><button type="button" class="vbtn kob${verdict(c)==='ko'?' on':''}" data-v="ko" data-i="${i}" aria-pressed="${verdict(c)==='ko'}">✗ non ok</button>${isBad(c)?'<span class="err">il driver lo ha dato come NON riuscito: finisce comunque nel file pubblicato, con il suo errore</span>':''}</div><label class="cap" for="n${i}">Il tuo giudizio (facoltativo; un «ok» da solo = superato)</label><textarea id="n${i}" data-i="${i}" placeholder="Scrivi solo se c'è un problema…">${H(notes[c.id]||'')}</textarea></section>`}
function render(){const f=document.getElementById('flt').value;const l=D.cards.map((c,i)=>({c,i})).filter(({c})=>f==='all'||(f==='bad'&&c.esito!=='ok')||(f==='note'&&(notes[c.id]||'').trim()));
document.getElementById('list').innerHTML=l.map(({c,i})=>card(c,i)).join('');cnt()}
function cnt(){const n=Object.values(notes).filter(v=>v.trim()).length;const nk=D.cards.filter(c=>verdict(c)==='ko').length,nb=D.cards.filter(c=>isBad(c)).length;document.getElementById('fc').textContent=n+' con nota · '+nk+' non ok · '+nb+' falliti (inclusi di diritto) su '+D.cards.length;document.getElementById('cnt').textContent=D.cards.filter(c=>c.esito!=='ok').length+' non riusciti dal driver'}
document.addEventListener('input',e=>{if(e.target.matches('textarea[data-i]')){notes[D.cards[e.target.dataset.i].id]=e.target.value;try{localStorage.setItem(KEY,JSON.stringify(notes))}catch(x){}cnt()}});
document.addEventListener('click',e=>{const b=e.target.closest('button.vbtn');if(!b)return;const c=D.cards[b.dataset.i];ver[c.id]=(ver[c.id]===b.dataset.v)?'':b.dataset.v;try{localStorage.setItem(KEY+':v',JSON.stringify(ver))}catch(x){}b.closest('section').querySelectorAll('button.vbtn').forEach(x=>{const on=ver[c.id]===x.dataset.v;x.classList.toggle('on',on);x.setAttribute('aria-pressed',on)});cnt()});
document.getElementById('flt').onchange=render;
document.addEventListener('click',e=>{const a=e.target.closest('a.thumb');if(!a)return;e.preventDefault();const lb=document.getElementById('lb');const u=a.dataset.big;if(!u)return;lb.innerHTML=a.dataset.video==='1'?`<video src="${u}" controls autoplay></video>`:`<img src="${u}" alt="">`;lb.classList.add('on')});
document.getElementById('lb').onclick=()=>{const lb=document.getElementById('lb');lb.classList.remove('on');lb.innerHTML=''};
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.getElementById('lb').click()});
document.getElementById('pub').onclick=async()=>{const sel=D.cards.filter(c=>isBad(c)||verdict(c)==='ko'||((notes[c.id]||'').trim()&&verdict(c)!=='ok'));const pm=document.getElementById('pm');
if(!sel.length){pm.textContent='Nessun caso non ok e nessun fallito: tutti superati.';return}
let md='# Casi non superati — '+D.nome+' ('+D.creato+')\n\nCartella del giro: '+D.run+'\n\n';
sel.forEach(c=>{md+='## '+c.id+'\n- funzione: '+c.fn+'\n- arch: '+(c.arch||'')+'\n- modello: '+(c.modello||'')+'\n- LoRA: '+(c.loras.length?c.loras.map(l=>l.file+' @'+l.strength).join(', '):'nessuna')+'\n- file: '+(c.media.map(m=>m.path).join(', ')||'nessuno')+'\n- esito del driver: '+c.esito+(c.motivo?' ('+c.motivo+')':'')+(c.dialoghi.length?'\n- errore in GUI: '+c.dialoghi.join(' | '):'')+'\n- giudizio: '+(verdict(c)==='ko'?'NON OK':(isBad(c)?'FALLITO (driver)':'senza giudizio'))+'\n- **utente:** '+((notes[c.id]||'').trim().replace(/\n/g,'\n  ')||'(nessun commento)')+'\n\n'});
try{const r=await fetch('/publish',{method:'POST',body:md});if(r.ok){const j=await r.json();pm.textContent='Salvato: '+j.path;return}}catch(x){}
const b=new Blob([md],{type:'text/markdown'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='pubblicato_'+new Date().toISOString().replace(/[:.]/g,'-')+'.md';a.click();pm.textContent='File scaricato ('+sel.length+' casi)'};
render();
</script></body></html>"""

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        print(build(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "giro"))
    elif len(sys.argv) >= 3 and sys.argv[1] == "serve":
        serve(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8766)
    else:
        print(__doc__)
