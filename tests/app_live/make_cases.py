"""Genera i casi del giro «tutte le architetture x modelli» DALL'INVENTARIO REALE dell'app (non da liste a mano).

    python tests/app_live/run_suite.py --inventory-out inventario.json      # avvia l'app e salva i menu veri
    python tests/app_live/make_cases.py inventario.json casi.json [--tier quick|full] [--fixtures DIR]

Per ogni funzione (Crea immagine, Modifica: inpaint/outpaint/Integra, Video) e per ogni arch/modello presente nei menu:
un caso «liscio» e, per il primo modello di ogni arch (o tutti con --tier full), un caso con UNA LoRA di quell'arch
(stesso seed del caso liscio). Le voci non scaricate (⬇) e i segnaposto sono elencati in `skip` del file, non testati.
Prompt creativi (mai gattini/cani/ballerine), SFW; i menu restano in modalita NSFW solo per vedere tutti i modelli.
"""
import argparse
import json
import os
import re

SKIP_TEXT = ("⬇", "scegli un modello", "Auto (")
NO_TEST = ("q8_0",)          # GGUF Q8 LTX-2.5: non si testa piu (decisione utente 30/10)

PROMPTS = {
    "img": [
        "a lighthouse keeper repairing a huge brass lamp during a storm, rain on the glass, cinematic lighting",
        "an abandoned greenhouse overgrown with giant ferns, soft morning light, photorealistic",
        "a street market stall selling clockwork birds at dusk, warm lanterns, shallow depth of field",
        "a glass pyramid in a salt flat reflecting a pink sky, ultra wide, calm",
        "an astronaut brewing tea in a rusty diner on Mars, neon sign, film grain",
    ],
    "video": [
        "a paper boat drifting down a rain-filled gutter, close-up, shallow depth of field, ambient rain",
        "a lantern festival on a river at night, floating lights slowly drifting, camera gently rising",
        "a beekeeper walking through a field of tall sunflowers at golden hour, slow tracking shot",
    ],
}
SIZES = {"Krea 2": (960, 1024), "FLUX": (1024, 1024), "Qwen-Image 2.1": (1024, 1024), "FLUX.3 Klein": (1024, 1024),
         "Z-Image": (1024, 1024), "Ernie": (1024, 1024)}


def ok_text(t):
    return not any(x.lower() in t.lower() for x in SKIP_TEXT)


def pick_lora(loras):
    sfw = [l for l in loras if not re.search(r"nsfw|sex|pussy|penis|breast|fellat|bukkake|anal|cum|nude|erotic|porn|deepthroat", (l["file"] or "") + " " + (l["name"] or ""), re.I)]
    return (sfw or loras or [None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory")
    ap.add_argument("out")
    ap.add_argument("--tier", default="full", choices=["quick", "full"])
    ap.add_argument("--fixtures", default=r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\fixtures")
    a = ap.parse_args()
    inv = json.load(open(a.inventory, encoding="utf-8"))
    cases, skip, n = [], [], 0
    scena = os.path.join(a.fixtures, "scena_1152x832.png")

    # ── Crea immagine ──
    for arch, v in inv["create"].items():
        models = [m for m in v["models"] if ok_text(m["text"])]
        skip += [{"funzione": "crea", "arch": arch, "voce": m["text"]} for m in v["models"] if not ok_text(m["text"])]
        lo = pick_lora(v["loras"])
        for k, m in enumerate(models):
            n += 1
            base = {"fn": "create_image", "nsfw": True, "arch": arch, "model": m["text"], "w": SIZES.get(arch, (1024, 1024))[0],
                    "h": SIZES.get(arch, (1024, 1024))[1], "seed": 3000 + n, "prompt": PROMPTS["img"][n % len(PROMPTS["img"])],
                    "timeout": 900}
            cases.append({**base, "id": f"crea_{re.sub(r'[^a-z0-9]+', '-', arch.lower())}_{k:02d}_liscio", "descr": f"Crea immagine · {arch} · {m['text'][:60]} · senza LoRA"})
            if lo and (k == 0 or a.tier == "full" and False):
                trig = (lo.get("triggers") or [""])[0]
                cases.append({**base, "id": f"crea_{re.sub(r'[^a-z0-9]+', '-', arch.lower())}_{k:02d}_lora", "loras": [{"piece": lo["file"], "strength": 0.8}],
                              "prompt": (trig + ", " if trig else "") + base["prompt"], "descr": f"Crea immagine · {arch} · {m['text'][:50]} · LoRA {lo['file']} @0.8 (stesso seed)"})

    # ── Modifica: inpaint / outpaint (menu per arch) ──
    for key, v in inv["edit"].items():
        if not key.startswith("inpaint:") or key == "inpaint:":
            continue
        arch = key.split(":", 1)[1]
        models = [m for m in v["models"] if ok_text(m["text"]) and m["data"] != "__placeholder__"]
        for k, m in enumerate(models):
            n += 1
            for kind in ("inpaint", "outpaint") if k == 0 else ("inpaint",):
                cases.append({"id": f"modifica_{kind}_{arch}_{k:02d}", "fn": "edit", "kind": kind, "arch": arch, "model": m["text"], "source": scena,
                              "seed": 4000 + n, "timeout": 900,
                              "prompt": "replace the masked area with a glowing brass compass on the table" if kind == "inpaint" else "extend the diner scene to the sides with more rusty booths and warm window light",
                              "descr": f"Modifica · {kind} · arch {arch} · {m['text'][:60]}"})
    # ── Modifica: Integra / edit per istruzione (motore x modello) ──
    for eng, v in inv["edit"].get("integrate_engines", {}).items():
        for k, m in enumerate(v["models"]):
            if not ok_text(m["text"]):
                skip.append({"funzione": "integra", "arch": eng, "voce": m["text"]})
                continue
            n += 1
            cases.append({"id": f"modifica_integra_{re.sub(r'[^a-z0-9]+', '-', eng.lower())}_{k:02d}", "fn": "edit", "kind": "integrate", "engine": eng, "model": m["text"],
                          "source": scena, "layer": os.path.join(a.fixtures, "ritratto_960x1024.png"), "seed": 5000 + n, "timeout": 900,
                          "prompt": "blend the pasted woman into the diner as a customer sitting at the counter, matching the warm light",
                          "descr": f"Modifica · Integra (edit per istruzione) · {eng} · {m['text'][:55]}"})

    # ── Video ──
    for arch, v in inv["video"].items():
        models = [m for m in v["models"] if not (isinstance(m["data"], dict) and m["data"].get("kind") in ("placeholder", "auto"))]
        lo = pick_lora(v["loras"])
        if arch == "lingbot":                          # finding: il checkbox rewriter e ignorato e il rewriter carica ~79 GB (commit -> 3 GB)
            skip += [{"funzione": "video", "arch": arch, "voce": m["text"], "motivo": "Prompt Rewriter sempre attivo (checkbox ignorato) e carica ~79 GB: porta il commit a ~3 GB. Da correggere in Fase 3, poi si testa"} for m in models]
            continue
        for k, m in enumerate(models):
            low = (m["text"] + json.dumps(m["data"])).lower()
            if any(x in low for x in NO_TEST):
                skip.append({"funzione": "video", "arch": arch, "voce": m["text"], "motivo": "GGUF Q8 LTX-2.5: non si testa piu"})
                continue
            if m["text"].startswith("⬇"):
                skip.append({"funzione": "video", "arch": arch, "voce": m["text"], "motivo": "pesi non scaricati"})
                continue
            n += 1
            h3 = arch == "minimax_h3"
            base = {"fn": "video", "nsfw": True, "arch": arch, "model": m["text"], "w": 704 if not h3 else 544, "h": 448 if not h3 else 960,
                    "secs": 4 if not h3 else 5, "seed": 6000 + n, "prompt": PROMPTS["video"][n % 3], "timeout": 1800 if not h3 else 2400}
            if arch == "lingbot":                      # il rewriter di default (Qwen3.6-27B) porta il commit a ~3 GB: finding, nel giro e spento
                base["widgets"] = {"video_tab.lingbot_rewriter": False}
            cases.append({**base, "id": f"video_{arch}_{k:02d}_liscio", "descr": f"Video · {v.get('label', arch)} · {m['text'][:60]}"})
            if lo and k == 1 and arch == "ltx":
                trig = (lo.get("triggers") or [""])[0]
                cases.append({**base, "id": f"video_{arch}_{k:02d}_lora", "loras": [{"piece": lo["file"], "strength": 0.8}],
                              "prompt": (trig + ", " if trig else "") + base["prompt"], "descr": f"Video · {m['text'][:45]} · LoRA {lo['file']} @0.8 (stesso seed)"})
    if a.tier == "quick":
        seen, q = set(), []
        for c in cases:
            key = (c["fn"], c.get("arch"), c.get("kind"), c.get("engine"))
            if key not in seen or c["id"].endswith("_lora"):
                seen.add(key)
                q.append(c)
        cases = q
    json.dump({"nome": f"Giro tutte le architetture x modelli ({a.tier})", "skip": skip, "cases": cases}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    by = {}
    for c in cases:
        by[c["fn"]] = by.get(c["fn"], 0) + 1
    print(len(cases), "casi", by, "| saltati:", len(skip))


if __name__ == "__main__":
    main()
