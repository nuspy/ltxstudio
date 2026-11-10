/* Traduzione automatica della pagina con il traduttore integrato nel browser (API Translator): il testo non lascia il computer.
   Menu proprio con bandiera e sigla. Scelta salvata; senza scelta si usa la lingua del browser. */
(function () {
  "use strict";
  // [codice, codice per il browser, bandiera (flag-icons)]
  var L = [["sq","sq","al"],["ar","ar","sa"],["hy","hy","am"],["az","az","az"],["be","be","by"],["bn","bn","bd"],["bs","bs","ba"],["bg","bg","bg"],["ca","ca","es-ct"],["zh-CN","zh","cn"],["zh-TW","zh-Hant","tw"],["ko","ko","kr"],["hr","hr","hr"],["da","da","dk"],["sk","sk","sk"],["sl","sl","si"],["es","es","es"],["et","et","ee"],["tl","tl","ph"],["fi","fi","fi"],["fr","fr","fr"],["ka","ka","ge"],["ja","ja","jp"],["el","el","gr"],["hi","hi","in"],["id","id","id"],["en","en","gb"],["is","is","is"],["it","it","it"],["he","he","il"],["kk","kk","kz"],["lv","lv","lv"],["lt","lt","lt"],["mk","mk","mk"],["ms","ms","my"],["mt","mt","mt"],["no","no","no"],["nl","nl","nl"],["fa","fa","ir"],["pl","pl","pl"],["pt","pt","pt"],["ro","ro","ro"],["ru","ru","ru"],["sr","sr","rs"],["sw","sw","ke"],["sv","sv","se"],["th","th","th"],["cs","cs","cz"],["tr","tr","tr"],["uk","uk","ua"],["ur","ur","pk"],["hu","hu","hu"],["uz","uz","uz"],["vi","vi","vn"],["de","de","de"]];
  var SRC = "it", KEY = "tdm-lang";
  var HAS = typeof self.Translator !== "undefined" && self.Translator && typeof self.Translator.create === "function";
  // nome di ogni lingua scritto nella lingua stessa: l'ordine non cambia con la lingua della pagina
  var ENDO = { sq:"Shqip", ar:"العربية", hy:"Հայերեն", az:"Azərbaycan", be:"Беларуская", bn:"বাংলা", bs:"Bosanski", bg:"Български", ca:"Català", "zh-CN":"简体中文", "zh-TW":"繁體中文", ko:"한국어", hr:"Hrvatski", da:"Dansk", sk:"Slovenčina", sl:"Slovenščina", es:"Español", et:"Eesti", tl:"Filipino", fi:"Suomi", fr:"Français", ka:"ქართული", ja:"日本語", el:"Ελληνικά", hi:"हिन्दी", id:"Indonesia", en:"English", is:"Íslenska", it:"Italiano", he:"עברית", kk:"Қазақ тілі", lv:"Latviešu", lt:"Lietuvių", mk:"Македонски", ms:"Melayu", mt:"Malti", no:"Norsk", nl:"Nederlands", fa:"فارسی", pl:"Polski", pt:"Português", ro:"Română", ru:"Русский", sr:"Српски", sw:"Kiswahili", sv:"Svenska", th:"ไทย", cs:"Čeština", tr:"Türkçe", uk:"Українська", ur:"اردو", hu:"Magyar", uz:"Oʻzbek", vi:"Tiếng Việt", de:"Deutsch" };
  function name(c) { return ENDO[c] || c; }
  function sigla(c) { return c.indexOf("zh") === 0 ? c.replace("-", "·").toUpperCase() : c.toUpperCase(); }
  var items = L.map(function (x) { return { c: x[0], b: x[1], f: x[2], n: name(x[0]) }; });
  var coll = new Intl.Collator("en", { sensitivity: "base" });
  items.sort(function (a, b) { return coll.compare(a.n, b.n); });
  function byCode(c) { return items.filter(function (i) { return i.c === c; })[0]; }

  function store(v) { try { if (v == null) return localStorage.getItem(KEY); localStorage.setItem(KEY, v); } catch (e) {} return null; }
  function browserLang() {
    var l = ((navigator.languages && navigator.languages[0]) || navigator.language || SRC).toLowerCase();
    if (/^zh-(tw|hk|hant)/.test(l)) return "zh-TW";
    if (l.indexOf("zh") === 0) return "zh-CN";
    if (l.indexOf("nb") === 0 || l.indexOf("nn") === 0) return "no";
    var b = l.split("-")[0]; return byCode(b) ? b : SRC;
  }

  [].forEach.call(document.querySelectorAll(".logo,[data-count],#ck,.tag,.fi"), function (n) { n.setAttribute("translate", "no"); });

  // interfaccia
  var host = document.createElement("div"); host.className = "lang"; host.id = "lang";
  host.innerHTML = "<button type='button' class='lbtn mono' translate='no' aria-haspopup='listbox' aria-expanded='false' aria-label='Lingua della pagina'><span class='fi'></span><b></b><i aria-hidden='true'></i></button>" +
    "<ul class='llist' translate='no' role='listbox' tabindex='-1'></ul><p class='lmsg' translate='no' role='status'></p>";
  var nav = document.getElementById("nav"), pill = nav && nav.querySelector(".pill");
  if (nav) nav.insertBefore(host, pill || null);
  else { // pagina senza barra: selettore fisso in alto a destra, con il proprio stile
    host.classList.add("float"); document.body.appendChild(host);
    var fl = document.createElement("link"); fl.rel = "stylesheet"; fl.href = "https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/css/flag-icons.min.css"; document.head.appendChild(fl);
    var st = document.createElement("style"); st.textContent =
      ".lang.float{position:fixed;top:14px;right:14px;z-index:2147483000;font-family:system-ui,sans-serif}" +
      ".lang .lbtn{display:flex;align-items:center;gap:8px;background:rgba(10,9,24,.72);backdrop-filter:blur(10px);border:1px solid rgba(243,234,214,.28);color:#f3ead6;padding:.55em .8em;font:600 12px/1 ui-monospace,monospace;letter-spacing:.12em;cursor:pointer;border-radius:8px;min-height:40px}" +
      ".lang .lbtn:hover,.lang.open .lbtn{border-color:#ffd166}.lang .lbtn .fi{width:1.25em;height:.95em;background-size:cover;border-radius:1px}" +
      ".lang .lbtn i{width:6px;height:6px;border-right:1px solid #ffd166;border-bottom:1px solid #ffd166;transform:rotate(45deg) translateY(-2px)}.lang.open .lbtn i{transform:rotate(-135deg)}" +
      ".lang .llist{position:absolute;right:0;top:calc(100% + 8px);width:min(300px,92vw);max-height:min(62vh,440px);overflow-y:auto;list-style:none;margin:0;background:rgba(10,9,24,.97);border:1px solid rgba(243,234,214,.22);border-radius:8px;padding:6px;display:none}" +
      ".lang.open .llist{display:block}.lang .llist li{display:grid;grid-template-columns:1.6em 3.2em 1fr;align-items:center;gap:10px;padding:.55em .7em;cursor:pointer;color:#e8e0cc;font-size:.9rem;min-height:40px}" +
      ".lang .llist li:hover,.lang .llist li:focus{background:rgba(255,209,102,.12);outline:none}.lang .llist li[aria-selected=true]{color:#ffd166}.lang .llist li b{font:600 .7rem ui-monospace,monospace;letter-spacing:.1em;color:#ffd166}" +
      ".lang .llist .fi{width:1.6em;height:1.2em;background-size:cover}.lang .llist li.off{opacity:.35;cursor:not-allowed}" +
      ".lang .lmsg{display:none;position:absolute;right:0;top:calc(100% + 8px);width:min(300px,92vw);margin:0;background:rgba(10,9,24,.97);border:1px solid rgba(243,234,214,.22);border-radius:8px;padding:.8rem 1rem;font-size:.85rem;color:#e8e0cc;line-height:1.45}" +
      ".lang.hint .lbtn::after{content:\"\";width:7px;height:7px;border-radius:50%;background:#ffd166}.lang.note .lmsg{display:block}.lang.open .lmsg{display:none}.lang.fail .lmsg{border-color:#ff5d73}";
    document.head.appendChild(st);
  }
  var btn = host.querySelector(".lbtn"), ul = host.querySelector(".llist"), msg = host.querySelector(".lmsg"), cur = SRC;
  ul.innerHTML = items.map(function (i) {
    return "<li role='option' data-c='" + i.c + "' tabindex='-1'><span class='fi fi-" + i.f + "'></span><b class='mono'>" + sigla(i.c) + "</b><span lang='" + (i.c === "zh-CN" ? "zh-Hans" : i.c === "zh-TW" ? "zh-Hant" : i.c) + "'>" + i.n + "</span></li>";
  }).join("");
  var sayT = null;
  function say(t, err) { clearTimeout(sayT); host.classList.remove("hint"); msg.textContent = t || ""; host.classList.toggle("fail", !!err); host.classList.toggle("note", !!t);
    if (t) sayT = setTimeout(function () { host.classList.remove("note"); }, err ? 12000 : 8000); }
  function hint(t) { host.classList.add("hint"); btn.title = t; btn.setAttribute("aria-label", "Lingua della pagina. " + t); }
  msg.addEventListener("click", function () { host.classList.remove("note"); });
  function paint(extra) {
    var it = byCode(cur) || byCode(SRC);
    btn.querySelector(".fi").className = "fi fi-" + it.f; btn.querySelector("b").textContent = sigla(it.c) + (extra || "");
    [].forEach.call(ul.children, function (li) { li.setAttribute("aria-selected", li.dataset.c === cur ? "true" : "false"); });
  }
  function open(v) { host.classList.toggle("open", v); btn.setAttribute("aria-expanded", v ? "true" : "false");
    if (v) { var s = ul.querySelector("[aria-selected=true]"); if (s) ul.scrollTop = s.offsetTop - 60; } }
  btn.addEventListener("click", function (e) { e.stopPropagation(); open(!host.classList.contains("open")); });
  document.addEventListener("click", function (e) { if (!host.contains(e.target)) open(false); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") open(false); });
  ul.addEventListener("click", function (e) { var li = e.target.closest("li"); if (!li || li.classList.contains("off")) return; open(false); choose(li.dataset.c, true); });
  ul.addEventListener("keydown", function (e) { if (e.key === "Enter" && e.target.dataset.c) { open(false); choose(e.target.dataset.c, true); } });

  // lingue che questo browser sa tradurre da italiano
  var avail = {};
  function probe() {
    if (!HAS) return;
    items.forEach(function (i) { if (i.c === SRC) return;
      self.Translator.availability({ sourceLanguage: SRC, targetLanguage: i.b }).then(function (a) {
        avail[i.c] = a; var li = ul.querySelector("[data-c='" + i.c + "']");
        if (li && a === "unavailable") { li.classList.add("off"); li.setAttribute("aria-disabled", "true"); li.title = "Questo browser non ha il pacchetto per questa lingua."; }
      }).catch(function () {});
    });
  }

  // motore: nodi di testo dell'intera pagina, anche quelli aggiunti dopo (finestre, grafici, scene)
  var orig = new Map(), done = new WeakMap(), tr = null, obs = null, busy = false, queue = [], token = 0, okN = 0, errN = 0, errMsg = "";
  var SKIP = { SCRIPT: 1, STYLE: 1, NOSCRIPT: 1, TEXTAREA: 1, CODE: 0 };
  function skipped(n) {
    for (var e = n.parentNode; e && e.nodeType === 1; e = e.parentNode) {
      if (SKIP[e.nodeName] === 1 || e.getAttribute("translate") === "no" || e.id === "lang") return true;
    } return false;
  }
  function textNodes(root) {
    var out = [], w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null), n;
    while ((n = w.nextNode())) { if (/\S/.test(n.nodeValue) && !skipped(n) && done.get(n) !== n.nodeValue) out.push(n); }
    return out;
  }
  function flush(my) {
    if (busy || !tr) return; busy = true;
    (function step() {
      if (my !== token) { busy = false; return; }
      if (!queue.length) { busy = false; if (!okN && errN) { cur = SRC; paint(); say("La traduzione non è partita (" + errMsg + ").", true); } return; }
      var batch = queue.splice(0, 24);
      Promise.all(batch.map(function (n) {
        var src = n.nodeValue; if (!src || done.get(n) === src) return null;
        var lead = src.match(/^\s*/)[0], trail = src.match(/\s*$/)[0];
        return tr.translate(src.trim()).then(function (t) {
          if (my !== token || !n.parentNode) return; if (!orig.has(n)) orig.set(n, src);
          var v = lead + t + trail; done.set(n, v); n.nodeValue = v; okN++;
        }).catch(function (e) { errN++; if (!errMsg) { errMsg = (e && (e.name + ": " + e.message)) || String(e); console.warn("translate", e); } });
      })).then(step, step);
    })();
  }
  function enqueue(nodes, my) { queue.push.apply(queue, nodes); flush(my); }
  function watch(my) {
    if (obs) obs.disconnect();
    obs = new MutationObserver(function (list) {
      var add = [];
      list.forEach(function (m) {
        if (m.type === "characterData") { var n = m.target; if (/\S/.test(n.nodeValue) && !skipped(n) && done.get(n) !== n.nodeValue) add.push(n); }
        else m.addedNodes.forEach(function (a) { if (a.nodeType === 3) { if (/\S/.test(a.nodeValue) && !skipped(a) && done.get(a) !== a.nodeValue) add.push(a); } else if (a.nodeType === 1) add.push.apply(add, textNodes(a)); });
      });
      if (add.length) enqueue(add, my);
    });
    obs.observe(document.body, { childList: true, subtree: true, characterData: true });
  }
  function restore() {
    token++; queue = []; if (obs) { obs.disconnect(); obs = null; }
    orig.forEach(function (v, n) { if (n.parentNode) n.nodeValue = v; }); orig.clear();
    if (tr && tr.destroy) { try { tr.destroy(); } catch (e) {} } tr = null; busy = false;
    document.documentElement.lang = SRC; document.documentElement.classList.remove("translated");
  }
  function apply(code, fromUser) {
    var it = byCode(code), my = ++token; restore(); token = my; okN = 0; errN = 0; errMsg = "";
    paint(" …"); say("");
    var opts = { sourceLanguage: SRC, targetLanguage: it.b };
    self.Translator.availability(opts).then(function (a) {
      if (a === "unavailable") { cur = SRC; paint(); say("Il tuo browser non può tradurre in questa lingua.", true); return; }
      if (a !== "available" && !fromUser) { cur = SRC; paint(); store(null); hint("Per tradurre in " + it.n + " serve scaricare il pacchetto della lingua: scegli la lingua dal menu."); return; }
      return self.Translator.create({ sourceLanguage: SRC, targetLanguage: it.b, monitor: function (m) {
        m.addEventListener("downloadprogress", function (e) { paint(" " + Math.round((e.loaded || 0) * 100) + "%"); say("Scarico il pacchetto della lingua nel browser (una volta sola)."); });
      } }).then(function (t) {
        if (my !== token) { try { t.destroy(); } catch (e) {} return; }
        tr = t; paint(); say("");
        document.documentElement.lang = it.c; document.documentElement.classList.add("translated");
        watch(my); enqueue(textNodes(document.body), my);
      });
    }).catch(function (e) { cur = SRC; paint(); console.warn("translate", e); say(e && e.name === "NotAllowedError" ? "Il browser chiede un gesto dell'utente: scegli di nuovo la lingua dal menu." : "Traduzione non riuscita: " + (e && e.message ? e.message : e), true); });
  }
  function choose(code, user) {
    cur = code; store(code);
    if (code === SRC) { restore(); paint(); say(""); return; }
    if (!HAS) { cur = SRC; paint(); if (!user) return; say("Su questo dispositivo la traduzione dal menu non è disponibile. Usa quella del browser: ⋮ → Traduci (Chrome), o l'icona di traduzione nella barra degli indirizzi.", true); return; }
    apply(code, user);
  }

  var saved = store(), first = saved || browserLang();
  cur = byCode(first) ? first : SRC; paint(); probe();
  if (cur !== SRC) choose(cur, false);
  window.TDMTranslate = { set: function (c) { choose(c, true); }, supported: HAS };
})();
