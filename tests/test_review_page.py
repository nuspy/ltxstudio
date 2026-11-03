"""Pagina di revisione (tests/app_live/review.py): «Pubblica» include i falliti del driver con il loro errore e i «non ok»; un «ok» da solo = superato."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "app_live"))
import review  # noqa: E402


@pytest.mark.skipif(shutil.which("node") is None, reason="serve node per eseguire lo script della pagina")
def test_publish_includes_failed_and_ko_and_skips_ok(tmp_path):
    HJS = tmp_path / "rev_h.js"

    cards = []
    for cid, esito, mot in (("a_ok", "ok", ""), ("b_fail", "fallito", "OOM xyz"), ("c_noted", "ok", ""), ("d_oknote", "ok", ""), ("e_ko", "ok", ""), ("f_clean", "ok", "")):
        cards.append({"id": cid, "fn": "edit", "esito": esito, "motivo": mot, "dry": False, "tempo": 1, "vram": 1, "ram": 1, "commit": 1, "handler": "h", "arch": "x", "modello": "m",
                      "prompt": "p", "negativo": "", "params": {}, "loras": [], "media": [], "orig": [], "verifica": "v", "descr": "", "dialoghi": ["dlg"] if esito != "ok" else [], "iniz": None})
    data = {"nome": "t", "run": "r", "creato": "now", "cards": cards}
    js = re.search(r"<script>(.*)</script>", review.PAGE, re.S).group(1).replace("/*DATA*/null", json.dumps(data))
    harness = """
    const store={};global.localStorage={getItem:k=>store[k]||null,setItem:(k,v)=>{store[k]=v}};
    const els={};const mk=()=>({textContent:'',innerHTML:'',value:'all',classList:{remove(){},add(){},toggle(){}},addEventListener(){},click(){}});
    global.location={protocol:'http:'};
    global.document={getElementById:id=>els[id]||(els[id]=mk()),addEventListener(){},querySelectorAll(){return[]}};
    let body=null;global.fetch=async(u,o)=>{body=o.body;return {ok:true,json:async()=>({path:'x'})}};
    global.URL={createObjectURL(){return''}};
    """ + js + """
    notes['c_noted']='la luce e sbagliata';notes['d_oknote']='ok';ver['e_ko']='ko';
    document.getElementById('pub').onclick().then(()=>{console.log(body)});
    """
    open(str(HJS), "w", encoding="utf-8").write(harness)
    r = subprocess.run(["node", str(HJS)], capture_output=True, text=True)
    out = r.stdout
    assert "## b_fail" in out and "OOM xyz" in out and "FALLITO" in out
    assert "## c_noted" in out and "la luce e sbagliata" in out
    assert "## e_ko" in out and "NON OK" in out
    assert "## a_ok" not in out and "## d_oknote" not in out and "## f_clean" not in out


def _fake_run(tmp_path, kind, params, uscite, finiti_out, case_extra=None):
    import json
    from PIL import Image
    rd = tmp_path / "run"
    rd.mkdir(exist_ok=True)
    paths = []
    for n in uscite:
        p = rd / n
        Image.new("RGB", (32, 24), (80, 90, 100)).save(p)
        paths.append(str(p))
    for k in ("init_image", "image_path", "mask", "mask_path"):
        if k in params:
            params[k] = str(rd / params[k])
    case = {"id": "c1", "fn": "edit", "kind": kind, **(case_extra or {})}
    row = {"id": "c1", "case": case, "esito": "ok", "uscite": paths, "calls": [{"via": "queue", "params": params}],
           "finiti": [{"status": "success", "outputs": [str(rd / n) for n in finiti_out]}]}
    (rd / "results.jsonl").write_text(json.dumps(row), encoding="utf-8")
    return rd


def test_roles_come_from_driver_data_not_from_file_names(tmp_path, monkeypatch):
    """Integra Kontext: il job ha prodotto piu file; solo il primo e il RISULTATO da valutare, gli altri sono l'ingresso incollato e la maschera (ruoli dai parametri)."""
    import json
    monkeypatch.setattr(review, "ORCH", str(tmp_path))
    rd = _fake_run(tmp_path, "integrate", {"image_path": "out_1_integrate_input.png", "mask_path": "out_2_integrate_mask.png"},
                   ["out_0_native_inpaint_5048.png", "out_1_integrate_input.png", "out_2_integrate_mask.png", "out_3_boh.png"],
                   ["native_inpaint_5048.png", "integrate_input.png", "integrate_mask.png"])
    idx = review.build(str(rd), "t")
    data = json.load(open(Path(idx).with_name("data.json"), encoding="utf-8"))
    items = {Path(i["path"]).name: i for i in data["cards"][0]["items"]}
    assert items["out_0_native_inpaint_5048.png"]["role"] == "result"
    assert items["out_1_integrate_input.png"]["role"] == "model_input" and "NON VALUTARE" in items["out_1_integrate_input.png"]["label"]
    assert items["out_2_integrate_mask.png"]["role"] == "mask" and items["out_2_integrate_mask.png"]["label"].startswith("MASCHERA")
    assert items["out_3_boh.png"]["role"] == "service" and "non valutare" in items["out_3_boh.png"]["label"]      # file senza ruolo noto: mai un risultato
    assert data["cards"][0]["items"][0]["role"] == "result"                                                        # il risultato e il primo
    assert items["out_0_native_inpaint_5048.png"]["label"] == "RISULTATO – DA VALUTARE"


def test_inpaint_input_is_the_original_and_multiple_results_are_numbered(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(review, "ORCH", str(tmp_path))
    rd = _fake_run(tmp_path, "inpaint", {"image_path": "frame1.png", "mask_path": "frame1_mask.png"},
                   ["out_0_a.png", "out_1_b.png", "out_2_frame1.png", "out_3_frame1_mask.png"], ["a.png", "b.png"])
    idx = review.build(str(rd), "t")
    cards = json.load(open(Path(idx).with_name("data.json"), encoding="utf-8"))["cards"]
    labs = {Path(i["path"]).name: i["label"] for i in cards[0]["items"]}
    assert labs["out_0_a.png"] == "RISULTATO – DA VALUTARE – Risultato 1" and labs["out_1_b.png"].endswith("Risultato 2")
    assert labs["out_2_frame1.png"].startswith("ORIGINALE")                      # inpaint: l'ingresso e la foto originale
    assert cards[0]["verifica"]


def test_page_shows_big_labels_and_cosa_guardare_first(tmp_path):
    """La pagina: riga «Cosa guardare» in cima, il risultato per primo con etichetta grande, l'ingresso del modello etichettato «NON VALUTARE»."""
    import json
    import re
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("serve node")

    def item(role, lab, path):
        return {"role": role, "label": lab, "path": path, "thumb": "img/x.jpg", "big": "img/x.jpg", "video": False}
    card = {"id": "c", "fn": "edit", "esito": "ok", "motivo": "", "dry": False, "tempo": 1, "vram": 1, "ram": 1, "commit": 1, "handler": "h", "arch": "x", "modello": "m",
            "prompt": "p", "negativo": "", "params": {}, "loras": [], "media": [], "orig": [], "verifica": "Guarda se la persona e ridisegnata.", "descr": "", "dialoghi": [],
            "items": [item("model_input", "INPUT AL MODELLO – NON VALUTARE", "a.png"), item("result", "RISULTATO – DA VALUTARE", "b.png")]}
    data = {"nome": "t", "run": "r", "creato": "now", "cards": [card]}
    js = re.search(r"<script>(.*)</script>", review.PAGE, re.S).group(1).replace("/*DATA*/null", json.dumps(data))
    harness = ("const store={};global.localStorage={getItem:k=>store[k]||null,setItem:(k,v)=>{store[k]=v}};const els={};"
               "const mk=()=>({textContent:'',innerHTML:'',value:'all',classList:{remove(){},add(){},toggle(){}},addEventListener(){}});"
               "global.location={protocol:'http:'};global.document={getElementById:id=>els[id]||(els[id]=mk()),addEventListener(){},querySelectorAll(){return[]}};"
               "global.fetch=async()=>({ok:false});" + js + "\nconsole.log(card(D.cards[0],0));")
    f = tmp_path / "h.js"
    f.write_text(harness, encoding="utf-8")
    out = subprocess.run(["node", str(f)], capture_output=True, text=True, encoding="utf-8").stdout
    assert out.index("Cosa guardare") < out.index("RISULTATO – DA VALUTARE") < out.index("INPUT AL MODELLO – NON VALUTARE"), out[:600]
    assert 'class="lab result"' in out and "Guarda se la persona e ridisegnata." in out


def test_wide_mask_case_shows_two_masks_and_the_result_with_todays_mask_next_to_it(tmp_path, monkeypatch):
    """Casi `_maschera_larga`: due maschere sovrapposte (posizione rossa, contesto gialla) con etichetta chiara e, accanto al risultato da valutare, quello con la
    maschera di oggi (dallo stesso caso base di un giro precedente)."""
    import json
    from PIL import Image
    monkeypatch.setattr(review, "ORCH", str(tmp_path))
    base = tmp_path / "base"
    base.mkdir()
    Image.new("RGB", (32, 24), (200, 10, 10)).save(base / "out_0_base.png")
    (base / "results.jsonl").write_text(json.dumps({"id": "c0", "esito": "ok", "uscite": [str(base / "out_0_base.png")]}), encoding="utf-8")
    rd = _fake_run(tmp_path, "integrate", {"image_path": "out_1_integrate_input.png", "mask_path": "out_2_integrate_mask.png"},
                   ["out_0_native.png", "out_1_integrate_input.png", "out_2_integrate_mask.png"], ["native.png"],
                   case_extra={"id": "c0_maschera_larga_25", "baseline_id": "c0"})
    row = json.loads((rd / "results.jsonl").read_text(encoding="utf-8"))
    for n, (x0, x1) in (("pos", (8, 16)), ("ctx", (2, 28))):
        im = Image.new("L", (32, 24), 0)
        im.paste(255, (x0, 4, x1, 20))
        im.save(rd / f"{n}.png")
    row["maschere"] = {"posizione": str(rd / "pos.png"), "contesto": str(rd / "ctx.png"), "context_px": 120}
    (rd / "results.jsonl").write_text(json.dumps(row), encoding="utf-8")
    idx = review.build(str(rd), "t", baseline_dirs=[str(base)])
    card = json.load(open(Path(idx).with_name("data.json"), encoding="utf-8"))["cards"][0]
    roles = [i["role"] for i in card["items"]]
    assert roles[0] == "result" and roles[1] == "result_base"                    # il confronto sta accanto al risultato da valutare
    two = next(i for i in card["items"] if i["role"] == "mask_two")
    assert "ROSSO = POSIZIONE" in two["label"] and "GIALLO = CONTESTO" in two["label"] and "120 px" in two["label"]
    assert next(i for i in card["items"] if i["role"] == "result_base")["label"].startswith("RISULTATO CON LA MASCHERA DI OGGI")
