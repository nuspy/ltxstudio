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
