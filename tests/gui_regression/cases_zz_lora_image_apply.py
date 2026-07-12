"""Casi APPLICAZIONE LoRA IMMAGINE: una generazione FLUX reale per OGNI LoRA della libreria.

Speculare a cases_lora_tab (video): scopre "questa LoRA immagine rompe/degrada la generazione"
e in run_real lascia in output/ un PNG per LoRA da ispezionare a occhio. Il selettore usato è
il LoraTab EMBEDDED del tab Immagini (frame1_tab.gen_lora_tab, arch Flux).

Prompt CREATIVI e ruotati (mai uguali, mai gattini/cani) così la galleria è anche varia.
"""
from __future__ import annotations

import harness as H


def _enum_flux_loras():
    try:
        from core.lora_catalog import LoraCatalog
        cat = LoraCatalog(compatible_arch="Flux")
        return [(e.name, e.comfy_name, e.nsfw) for e in cat.loras(allow_nsfw=True)]
    except Exception:                                            # noqa: BLE001
        return []


_LORAS = _enum_flux_loras()

# prompt ruotati per varietà della galleria (regola: creativi, sempre diversi)
_PROMPTS = [
    "a jazz trio of skeletons in tuxedos performing on a neon rooftop at midnight",
    "an elegant witch brewing aurora borealis in a giant teacup, cozy cabin kitchen",
    "a retro-futuristic diner on the rings of Saturn, waitress on roller skates",
    "a majestic paper-origami dragon soaring over a candle-lit medieval city",
    "an underwater library with jellyfish lamps and a mermaid librarian reading",
    "a tiny knight riding a giant snail through a mushroom forest at dawn",
    "a 1920s detective office where the detective is a sentient chess piece",
    "an opera singer on a cliff conducting a storm with her voice, dramatic light",
    "a street food market on the moon, vendors in vintage space suits, lanterns",
    "a glassblower shaping a living phoenix from molten glass, sparks flying",
]


def _row_by_name(t, name):
    return t._rows.get(name)


def _make_case(idx, name, comfy, nsfw):
    def _case(w, spy):
        f = w.frame1_tab
        try:                                          # NSFW visibili così TUTTE le LoRA sono in lista
            f.img_nsfw.setCurrentIndex(1)
        except Exception:                             # noqa: BLE001
            pass
        f.gen_lora_tab.refresh()
        row = _row_by_name(f.gen_lora_tab, name)
        assert row is not None, f"LoRA immagine '{name}' assente dalla lista (filtro arch/NSFW?)"
        row["cb"].setChecked(True)
        row["sp"].setValue(0.8)
        sel = {s["name"] for s in f.gen_lora_tab.selected_loras()}
        assert comfy in sel, f"LoRA '{name}' non in selected_loras(): {sel}"
        # modello base esplicito e stabile per la galleria (niente auto-select)
        for i in range(f.gen_model.count()):
            if "FLUX_Dev.safetensors" in str(f.gen_model.itemData(i) or f.gen_model.itemText(i)):
                f.gen_model.setCurrentIndex(i)
                break
        f.prompt.setPlainText(_PROMPTS[idx % len(_PROMPTS)])
        w.generate_image()
        assert spy.calls, f"generate_image non ha inviato nulla con la LoRA '{name}'"
        p = spy.calls[-1]["params"]
        sent = [d.get("name") for d in (p.get("loras") or [])]
        assert comfy in sent, f"LoRA immagine '{name}' NON passata al backend: {sent}"
    return _case


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


CASES = []
for _i, (_name, _comfy, _nsfw) in enumerate(_LORAS):
    CASES.append((
        f"lora_image_apply_{_safe(_name)}",
        f"LoRA immagine '{_name}'{' [NSFW]' if _nsfw else ''}: selezione + generazione FLUX reale con la LoRA applicata.",
        _make_case(_i, _name, _comfy, _nsfw),
    ))
