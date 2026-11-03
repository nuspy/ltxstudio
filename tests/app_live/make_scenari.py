"""Scenari di prova per Integra, inpaint e video MSR (decisione dell'utente 03/11 sera): «crea scene diverse, e veramente monotono».

Sei scenari SFW, adulti inventati, colorati, con persone DIVERSE in ciascuno. Per ogni scenario servono:
  scena    immagine con un POSTO LIBERO in primo piano (Integra inserisce una persona, l'inpaint fa comparire un oggetto)
  sfondo   la stessa ambientazione SENZA persone (sfondo dei video MSR)
  soggetti A, B, C  foto di riferimento su fondo neutro (Integra: A; MSR: 2-3 personaggi)
  oggetto  cosa deve comparire nell'inpaint (diverso per scenario) + posizione della maschera sulla scena

Le immagini si generano dall'app (Genera, FLUX Dev) nel primo turno GPU. Questo script scrive i file dei casi:

    python tests/app_live/make_scenari.py fixture     -> orch/app_live/casi_fixture_scenari_20261103.json   (30 immagini, un turno)
    python tests/app_live/make_scenari.py esperimenti -> orch/app_live/casi_exp_integra_20261103.json  (12 job)
                                                         orch/app_live/casi_exp_inpaint_20261103.json  (12 job)
                                                         orch/app_live/casi_exp_msr_20261103.json      (10 job)

I casi di prova RUOTANO fra gli scenari: nello stesso motore nessuno scenario si ripete e due casi consecutivi non usano mai la stessa scena
(verificato da `verifica_rotazione`; 6 scenari per 10-12 casi per giro = al massimo 2 volte lo stesso scenario in tutto il giro, con motori diversi).
"""
from __future__ import annotations

import json
import os
import sys

ORCH = r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live"
FIX = os.path.join(ORCH, "fixtures", "scenari")

NEUTRO = "standing on a plain light grey seamless studio backdrop, soft even studio lighting, sharp focus, full figure visible from head to knees, photorealistic photo"
ADULTI = "all people are adults, fully and tastefully dressed, family friendly"

SCENARI = {
    "spiaggia": {
        "titolo": "Spiaggia tropicale, amiche a beach volley",
        "scena": "Wide photorealistic photo of a sunny tropical beach with turquoise water and palm trees, a group of cheerful adult women friends in colourful swimsuits and "
                 "sarongs playing beach volleyball on the left and laughing together under a striped umbrella on the right, golden hour light, an empty patch of clean "
                 "sand in the centre foreground, vivid colours, " + ADULTI,
        "sfondo": "Wide photorealistic photo of a sunny tropical beach with turquoise water, palm trees, a striped umbrella and a volleyball net, golden hour light, "
                  "nobody in the picture, vivid colours",
        "soggetti": {
            "A": "a cheerful adult woman about 30 with a wide straw sun hat, wavy dark-blonde hair, a coral-coloured swimsuit and a light sarong, laughing",
            "B": "an adult woman about 26, athletic, short black hair, a turquoise swimsuit with a white shirt tied at the waist, smiling",
            "C": "an adult man about 35 with a short beard, flowered swim shorts and an open white linen shirt, relaxed smile",
        },
        "oggetto": "a colourful inflatable beach ball", "ellisse": (0.55, 0.88, 0.10, 0.07),
        "msr": "{A} laughs and waves at the camera while {B} bounces a beach ball, both on the tropical beach shown as background, warm sunny light",
    },
    "palazzo": {
        "titolo": "Palazzo orientale in stile avventura (harem da film)",
        "scena": "Cinematic photo of a luxurious Arabian-nights palace hall in the style of a Hollywood adventure film: silk curtains, carved arches, mosaic floor, a fountain "
                 "with sparkling water, heaps of velvet cushions, hanging lanterns, a few adult dancers in elegant embroidered veil costumes dancing in the background "
                 "(fully dressed, no nudity), an empty richly woven carpet in the centre foreground, " + ADULTI,
        "sfondo": "Cinematic photo of a luxurious Arabian-nights palace hall: silk curtains, carved arches, mosaic floor, a fountain with sparkling water, velvet cushions, "
                  "hanging lanterns, nobody in the picture",
        "soggetti": {
            "A": "an adult woman dancer about 28 in an elegant emerald and gold embroidered costume with a long sheer veil on her shoulders and gold jewellery, graceful pose",
            "B": "an adult man about 45, a dignified vizier with a dark pointed beard, a wine-red turban and a long embroidered coat, arms crossed",
            "C": "an adult woman about 35 with a golden headdress and a deep red silk gown with wide sleeves, regal bearing",
        },
        "oggetto": "a large fan of white and turquoise feathers", "ellisse": (0.50, 0.89, 0.11, 0.07),
        "msr": "{A} dances gracefully and {B} watches with a smile, both in the palace hall shown as background, flickering lantern light",
    },
    "sfilata": {
        "titolo": "Passerella di una sfilata di moda",
        "scena": "Photo of a high fashion runway show: a long glossy catwalk with dramatic spotlights, bursts of camera flashes, an audience on both sides, two adult models in "
                 "sculptural avant-garde gowns walking in the distance, an empty stretch of runway in the centre foreground, rich colours, " + ADULTI,
        "sfondo": "Photo of a high fashion runway show venue: a long glossy catwalk with dramatic spotlights and a dark audience on both sides, nobody on the runway",
        "soggetti": {
            "A": "an adult female fashion model about 27 in a sculptural metallic silver avant-garde dress with a dramatic high collar, confident runway pose",
            "B": "an adult female fashion model about 24 with a voluminous floral tulle gown in pink and orange and bold red lipstick, mid-step pose",
            "C": "an adult male fashion designer about 40 in a bold electric-blue suit with a gold brooch, hands on hips",
        },
        "oggetto": "a small teddy bear dressed as an astronaut with a tiny white helmet", "ellisse": (0.52, 0.88, 0.10, 0.07),
        "msr": "{A} walks down the runway and {B} follows her, camera flashes sparkling, both on the fashion show runway shown as background",
    },
    "battaglia": {
        "titolo": "Battaglia epica fantasy",
        "scena": "Epic cinematic fantasy battlefield photo: warriors, men and women in plate armour and leather, clashing swords on a dusty plain, tall banners flying, clouds of dust, "
                 "dramatic stormy sunset sky, a patch of trampled empty ground in the centre foreground, " + ADULTI,
        "sfondo": "Epic cinematic fantasy battlefield photo: a dusty plain under a dramatic stormy sunset, tall banners flying, smoke in the distance, nobody in the picture",
        "soggetti": {
            "A": "an adult woman knight about 30 in polished silver plate armour with a red cape, holding a long sword point-down, determined look",
            "B": "an adult bearded warrior about 40 in a fur-trimmed leather armour holding a round wooden shield and a battle axe",
            "C": "an adult elf archer woman about 28 with braided hair, a green hooded cloak and a longbow, calm focused look",
        },
        "oggetto": "a golden winged helmet", "ellisse": (0.50, 0.88, 0.11, 0.08),
        "msr": "{A} raises her sword and {B} roars beside her, both on the battlefield shown as background, dust swirling, stormy light",
    },
    "galeone": {
        "titolo": "Ciurma pirata sul ponte di un galeone",
        "scena": "Photo of the deck of a pirate galleon at sunrise: a crew of adult pirates, men and women in tricorn hats and leather coats, hauling ropes and laughing, tall masts, "
                 "sails and rigging, sparkling sea, a clear empty stretch of wooden deck in the centre foreground, " + ADULTI,
        "sfondo": "Photo of the deck of a pirate galleon at sunrise: tall masts, sails and rigging, sparkling sea, wooden deck, nobody in the picture",
        "soggetti": {
            "A": "an adult woman pirate captain about 35 in a long red coat, a black tricorn hat with a feather, a sword at the belt, confident grin",
            "B": "an adult young sailor man about 22 with a striped shirt, a blue bandana and rolled-up trousers, cheerful smile",
            "C": "an adult woman ship's cook about 45 with a white apron over a patched dress and a red headscarf, warm smile",
        },
        "oggetto": "an open wooden treasure chest full of gold coins", "ellisse": (0.50, 0.89, 0.12, 0.08),
        "msr": "{A} points to the horizon and {B} cheers, both on the galleon deck shown as background, morning sea breeze",
    },
    "carnevale": {
        "titolo": "Carnevale in maschera",
        "scena": "Photo of a Venetian carnival masquerade street at dusk with strings of warm lights, adult revellers in ornate masks and feathered costumes dancing, falling confetti, "
                 "colourful decorations, an empty stretch of cobblestones in the centre foreground, " + ADULTI,
        "sfondo": "Photo of a Venetian carnival street at dusk with strings of warm lights, colourful decorations and falling confetti, nobody in the picture",
        "soggetti": {
            "A": "an adult woman about 32 in a baroque ball gown of gold and burgundy, holding a feathered mask on a stick in her hand with her face fully visible, smiling",
            "B": "an adult man about 38 in a colourful harlequin costume with a diamond pattern and a tall hat, face fully visible, mischievous smile",
            "C": "an adult woman about 26 in a costume of peacock feathers in blue and green, face fully visible, laughing",
        },
        "oggetto": "a colourful parrot", "ellisse": (0.50, 0.88, 0.10, 0.08),
        "msr": "{A} twirls her mask and {B} dances beside her, both in the carnival street shown as background, confetti in the air",
    },
}
ORDINE = list(SCENARI)
SEED0 = 6100

# modelli (stringhe esatte dei menu dell'app, dall'inventario del 02/11)
M_QWEN_TURBO = "Qwen-Image 2.1 Turbo NVFP4 (8 passi, RTX 50xx)  · ~5.3GB"
M_KLEIN = "Flux\\realDream_klein9BV2.safetensors"
M_KONTEXT = "Flux\\FLUX_Kontext_Dev.safetensors"
M_LTX25 = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors  · diffusion"
M_LTX23 = "LTXVideo\\v2\\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors  · diffusion"


def fx(scen, nome):
    return os.path.join(FIX, scen, nome)


def casi_fixture() -> dict:
    cases, seed = [], SEED0
    for sc in ORDINE:
        d = SCENARI[sc]
        items = [("scena", d["scena"], 1152, 832), ("sfondo", d["sfondo"], 1152, 832)]
        items += [(f"soggetto_{k}", f"Full-figure studio photo of {v}, {NEUTRO}, {ADULTI}", 896, 1152) for k, v in d["soggetti"].items()]
        for nome, prompt, w, h in items:
            seed += 1
            cases.append({"id": f"fixscen_{sc}_{nome}", "fn": "create_image", "nsfw": True, "arch": "FLUX", "model": "Flux\\FLUX_Dev.safetensors", "w": w, "h": h,
                          "seed": seed, "timeout": 900, "prompt": prompt, "fixture_dest": fx(sc, nome + ".png"),
                          "descr": f"Fixture scenario «{d['titolo']}» · {nome} (FLUX Dev)",
                          "verify": ("La scena e quella dello scenario, colorata e fotorealistica, con un POSTO LIBERO in primo piano al centro?" if nome == "scena" else
                                     "Nessuna persona, stessa ambientazione?" if nome == "sfondo" else
                                     "Una sola persona adulta, figura intera su fondo grigio neutro, volto ben visibile, vestita?")})
    return {"nome": "Fixture scenari 03/11 (6 scenari x 5 immagini)", "skip": [], "cases": cases}


def _base_integra(sc, k, **kw):
    d = SCENARI[sc]
    c = {"fn": "edit", "kind": "integrate", "source": fx(sc, "scena.png"), "layer": fx(sc, "soggetto_A.png"), "seed": 5200 + k, "timeout": 900,
         "prompt": f"blend the person into the scene as part of it, matching the light, colours and perspective of {d['titolo'].lower()}",
         "scenario": sc, "autocheck": {"kind": "integra", "ref": fx(sc, "soggetto_A.png")}}
    c.update(kw)
    return c


def _base_inpaint(sc, k, model, arch="Flux", **kw):
    d = SCENARI[sc]
    c = {"fn": "edit", "kind": "inpaint", "arch": arch, "model": model, "source": fx(sc, "scena.png"), "seed": 4200 + k, "timeout": 900,
         "mask_ellipse": list(d["ellisse"]),
         "prompt": f"{d['oggetto']}, photorealistic, lit by the scene light, soft contact shadows",
         "scenario": sc, "autocheck": {"kind": "inpaint", "oggetto": d["oggetto"]}}
    c.update(kw)
    return c


def _base_msr(sc, k, model, release, subj=("A", "B"), prompt_style="breve", **kw):
    d = SCENARI[sc]
    toks = {s: f"@image_{i + 1}" for i, s in enumerate(subj)}
    pr = d["msr"].format(**toks)
    if prompt_style == "dettagliato":
        pr = d["msr"].format(**{s: f"{toks[s]} ({d['soggetti'][s]})" for s in subj})
    c = {"fn": "video", "nsfw": True, "arch": "ltx", "model": model, "w": 704, "h": 960, "secs": 4, "seed": 8600 + k, "timeout": 2400,
         "refs": [fx(sc, f"soggetto_{s}.png") for s in subj], "background": fx(sc, "sfondo.png"), "prompt": pr, "scenario": sc,
         "autocheck": {"kind": "video_refs", "subjects": [{"nome": s, "face_ref": fx(sc, f"soggetto_{s}.png")} for s in subj]}}
    c.update(kw)
    return c


def esperimenti_integra() -> dict:
    S = lambda i: ORDINE[i % 6]          # noqa: E731  rotazione
    spec = [  # (id, scenario, parametri)
        # ── Qwen-Image 2.1 Turbo NVFP4: 5 scenari diversi ──
        ("int_qwen_base", S(0), dict(engine="Qwen-Image 2.1", model=M_QWEN_TURBO, baseline=True)),
        ("int_qwen_armonizza45", S(1), dict(engine="Qwen-Image 2.1", model=M_QWEN_TURBO, approach="armonizza", approach_strength=0.45,
                                            expect_params={"mask_ref": False})),
        ("int_qwen_armonizza65", S(2), dict(engine="Qwen-Image 2.1", model=M_QWEN_TURBO, approach="armonizza", approach_strength=0.65,
                                            expect_params={"mask_ref": False})),
        ("int_qwen_soggetto_rif", S(3), dict(engine="Qwen-Image 2.1", model=M_QWEN_TURBO, regenera=True)),
        ("int_qwen_armonizza55_ctx212", S(4), dict(engine="Qwen-Image 2.1", model=M_QWEN_TURBO, approach="armonizza", approach_strength=0.55, ctx_px=212,
                                                   expect_params={"mask_ref": False})),
        # ── FLUX.2 Klein (realDream 9B): 4 scenari diversi ──
        ("int_klein_base", S(5), dict(engine="FLUX.2 (sperimentale)", model=M_KLEIN, baseline=True, expect_params={"arch": "flux2_inpaint"})),
        ("int_klein_rif_inpaint100", S(0), dict(engine="FLUX.2 (sperimentale)", model=M_KLEIN, approach="klein_ref_inpaint", approach_strength=1.0,
                                                expect_params={"arch": "flux2_inpaint", "strength": 1.0})),
        ("int_klein_rif_inpaint85", S(1), dict(engine="FLUX.2 (sperimentale)", model=M_KLEIN, approach="klein_ref_inpaint", approach_strength=0.85,
                                               expect_params={"arch": "flux2_inpaint", "strength": 0.85})),
        ("int_klein_armonizza50", S(2), dict(engine="FLUX.2 (sperimentale)", model=M_KLEIN, approach="klein_armonizza", approach_strength=0.5,
                                             expect_params={"arch": "flux2_inpaint", "strength": 0.5})),
        # ── Kontext FLUX.1: 3 scenari diversi ──
        ("int_kontext_base", S(3), dict(engine="Kontext (FLUX.1)", model=M_KONTEXT, baseline=True, expect_params={"arch": "inpaint"})),
        ("int_kontext_stitch_a", S(4), dict(engine="Kontext (FLUX.1)", model=M_KONTEXT, approach="kontext_stitch", expect_params={"arch": "kontext"})),
        ("int_kontext_stitch_b", S(5), dict(engine="Kontext (FLUX.1)", model=M_KONTEXT, approach="kontext_stitch", expect_params={"arch": "kontext"})),
    ]
    cases = []
    for k, (cid, sc, kw) in enumerate(spec):
        c = _base_integra(sc, k, **kw)
        c["id"] = cid
        c["descr"] = f"Integra · {kw.get('engine')} · {kw.get('approach') or ('soggetto come riferimento' if kw.get('regenera') else 'comportamento di oggi (confronto)')} · scenario {sc}"
        cases.append(c)
    return {"nome": "Esperimenti Integra 03/11 (approcci alternativi, scenari nuovi)", "skip": [], "cases": cases}


def esperimenti_inpaint() -> dict:
    S = lambda i: ORDINE[i % 6]          # noqa: E731
    DEV, KREA, GETPHAT, FUX = ("FLUX_Dev  · FLUX base: inpaint, outpaint, sostituzione di oggetti", "FLUX_Krea_Dev  (base, foto)",
                               "getphat Flux Reality v30 Hardcore FP8  (base, NSFW)",
                               "fuxCapacityNSFWPornFlux_20_(SwarmUI)  · FLUX base: inpaint, outpaint, sostituzione di oggetti")
    spec = [
        ("inp_dev_base", S(0), DEV, {"baseline": True}),
        ("inp_dev_crop128", S(1), DEV, {"inject": {"padding_mask_crop": 128}, "expect_params": {"padding_mask_crop": 128}}),
        ("inp_dev_crop64", S(2), DEV, {"inject": {"padding_mask_crop": 64}, "expect_params": {"padding_mask_crop": 64}}),
        ("inp_dev_maschera_grande", S(3), DEV, {"mask_scale": 1.5}),
        ("inp_dev_maschera_grande_crop128", S(4), DEV, {"mask_scale": 1.5, "inject": {"padding_mask_crop": 128}, "expect_params": {"padding_mask_crop": 128}}),
        ("inp_dev_crop128_guid5", S(5), DEV, {"inject": {"padding_mask_crop": 128}, "widgets": {"frame1_tab.inp_guidance": 5.0}, "expect_params": {"padding_mask_crop": 128}}),
        ("inp_krea_crop128", S(0), KREA, {"inject": {"padding_mask_crop": 128}, "expect_params": {"padding_mask_crop": 128}}),
        ("inp_getphat_crop128", S(1), GETPHAT, {"inject": {"padding_mask_crop": 128}, "expect_params": {"padding_mask_crop": 128}}),
        ("inp_fuxcapacity_crop128", S(2), FUX, {"inject": {"padding_mask_crop": 128}, "expect_params": {"padding_mask_crop": 128}}),
        ("inp_fill_controllo", S(3), "FLUX_DEV_Fill  (Fill inpaint/outpaint, 22GB)", {"baseline": True}),
        ("inp_klein_controllo", S(4), "miraclein_flux2klein_nsfw_v2  (FLUX.2 Klein, edit)", {"arch": "Flux2", "baseline": True}),
        ("inp_solo_oggetto_crop128", S(0), DEV, {"inject": {"padding_mask_crop": 128}, "expect_params": {"padding_mask_crop": 128}, "solo_oggetto": True}),
    ]
    cases = []
    for k, (cid, sc, model, kw) in enumerate(spec):
        kw = dict(kw)
        arch = kw.pop("arch", "Flux")
        sc_ell = list(SCENARI[sc]["ellisse"])
        ms = kw.pop("mask_scale", None)
        solo = kw.pop("solo_oggetto", False)
        c = _base_inpaint(sc, k, model, arch=arch, **kw)
        if ms:
            c["mask_ellipse"] = [sc_ell[0], sc_ell[1] - 0.01, sc_ell[2] * ms, sc_ell[3] * ms]
        if solo:
            c["prompt"] = SCENARI[sc]["oggetto"]
        c["id"] = cid
        c["descr"] = f"Inpaint · {model.split('  ')[0]} · {', '.join(f'{a}={b}' for a, b in (kw.get('inject') or {}).items()) or 'comportamento di oggi'} · scenario {sc}"
        cases.append(c)
    return {"nome": "Esperimenti inpaint FLUX base 03/11 (ritaglio attorno alla maschera, scenari nuovi)", "skip": [], "cases": cases}


def esperimenti_msr() -> dict:
    S = lambda i: ORDINE[i % 6]          # noqa: E731
    spec = [
        # LTX 2.5 (slot embedding): la LoRA era applicata? base = comportamento di oggi (senza pesi della LoRA)
        ("msr25_base_senza_lora", S(0), M_LTX25, "2.5", {"baseline": True, "inject": {"msr_apply_lora": False}}),
        ("msr25_lora", S(1), M_LTX25, "2.5", {}),
        ("msr25_lora_dettagliato", S(2), M_LTX25, "2.5", {"prompt_style": "dettagliato"}),
        ("msr25_lora_forza07", S(3), M_LTX25, "2.5", {"inject": {"msr_strength": 0.7}}),
        ("msr25_lora_3soggetti", S(4), M_LTX25, "2.5", {"subj": ("A", "B", "C")}),
        # LTX 2.3 (pseudo-video, IC-LoRA)
        ("msr23_base", S(5), M_LTX23, "2.3", {"baseline": True}),
        ("msr23_dettagliato", S(0), M_LTX23, "2.3", {"prompt_style": "dettagliato"}),
        ("msr23_forza_guida08", S(1), M_LTX23, "2.3", {"prompt_style": "dettagliato", "inject": {"control_strength": 0.8}}),
        ("msr23_lora_forza12", S(2), M_LTX23, "2.3", {"prompt_style": "dettagliato", "inject": {"msr_strength": 1.2}}),
        ("msr23_3soggetti", S(3), M_LTX23, "2.3", {"prompt_style": "dettagliato", "subj": ("A", "B", "C")}),
    ]
    cases = []
    for k, (cid, sc, model, rel, kw) in enumerate(spec):
        c = _base_msr(sc, k, model, rel, **kw)
        c["id"] = cid
        c["descr"] = f"Video MSR LTX {rel} · {', '.join(f'{a}={b}' for a, b in (kw.get('inject') or {}).items()) or kw.get('prompt_style', 'comportamento di oggi')} · scenario {sc}"
        cases.append(c)
    return {"nome": "Esperimenti video MSR 03/11 (riferimenti, scenari nuovi)", "skip": [], "cases": cases}


def verifica_rotazione(spec: dict) -> list:
    """Rotazione: nello stesso sotto-gruppo (stesso motore) nessuno scenario si ripete, e due casi CONSECUTIVI non usano mai lo stesso scenario."""
    errori = []
    for sg in ("int_qwen", "int_klein", "int_kontext", "inp_dev", "msr25", "msr23"):
        sc = [c["scenario"] for c in spec["cases"] if c["id"].startswith(sg)]
        if len(sc) != len(set(sc)):
            errori.append(f"{sg}: scenari ripetuti {sc}")
    seq = [c.get("scenario") for c in spec["cases"]]
    for i in range(1, len(seq)):
        if seq[i] and seq[i] == seq[i - 1]:
            errori.append(f"casi consecutivi con lo stesso scenario: {spec['cases'][i - 1]['id']} / {spec['cases'][i]['id']}")
    return errori


def raccogli(run_dir: str) -> int:
    """Dopo il turno delle fixture: copia ogni uscita in fixtures/scenari/<scenario>/<nome>.png e fa i controlli automatici possibili
    (soggetto: UN solo volto e fondo neutro; scena/sfondo: punteggio CLIP con l'ambientazione). Nessun blocco: il giudizio e dell'utente, nella pagina del giro."""
    import shutil
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import autocheck as ac
    import numpy as np
    from PIL import Image
    n_ok = 0
    for ln in open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8"):
        r = json.loads(ln)
        dest = (r.get("case") or {}).get("fixture_dest")
        if not dest or r.get("esito") != "ok" or not r.get("uscite"):
            print(f"fixture {r.get('id')}: NON generata ({r.get('esito')}) {str(r.get('motivo'))[:120]}")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(r["uscite"][0], dest)
        n_ok += 1
        nome = os.path.basename(dest)
        note = []
        try:
            if nome.startswith("soggetto"):
                fs = ac.face_sims(dest, [dest])[0]
                note.append(f"volti={fs['n']}" + ("" if fs["n"] == 1 else "  <-- ATTENZIONE: serve UN solo volto"))
                a = np.asarray(Image.open(dest).convert("RGB")).astype(np.float32)
                bordo = np.concatenate([a[:12].reshape(-1, 3), a[-12:].reshape(-1, 3), a[:, :12].reshape(-1, 3), a[:, -12:].reshape(-1, 3)])
                dev = float(np.abs(bordo - np.median(bordo, 0)).mean())
                note.append(f"fondo_neutro_dev={dev:.1f}" + ("" if dev < 25 else "  <-- fondo NON uniforme"))
            else:
                sc = os.path.basename(os.path.dirname(dest))
                txt = SCENARI[sc]["titolo"]
                cos = ac.clip_scores([{"image": dest, "mask": None, "texts": ["a photo of " + txt]}])[0][0]
                note.append(f"clip_ambientazione={cos:.3f}")
        except Exception as e:                               # noqa: BLE001
            note.append("controllo n/d: " + repr(e)[:100])
        print(f"fixture {r['id']} -> {dest} | " + " ".join(note))
    print(f"fixture copiate: {n_ok}")
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(ORCH, exist_ok=True)
    if mode == "fixture":
        out = {os.path.join(ORCH, "casi_fixture_scenari_20261103.json"): casi_fixture()}
    elif mode == "esperimenti":
        out = {os.path.join(ORCH, "casi_exp_integra_20261103.json"): esperimenti_integra(),
               os.path.join(ORCH, "casi_exp_inpaint_20261103.json"): esperimenti_inpaint(),
               os.path.join(ORCH, "casi_exp_msr_20261103.json"): esperimenti_msr()}
    elif mode == "raccogli":
        return raccogli(sys.argv[2])
    else:
        print(__doc__)
        return 2
    for p, spec in out.items():
        err = verifica_rotazione(spec) if mode == "esperimenti" else []
        json.dump(spec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(p, len(spec["cases"]), "casi", ("ERRORI: " + str(err)) if err else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
