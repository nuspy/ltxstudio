"""Rewriter proprietario LingBot-Video — prosa → caption JSON + auto-negative (venv LingBot).

Eseguito nel venv `lingbot_venv` (transformers 5.x + peft). Due stadi come da README ufficiale:
  1) VLM base (Qwen3.6-27B) SENZA LoRA → espande il prompt utente
  2) stesso VLM base CON la rewriter-LoRA → converte in caption JSON strutturata
poi un passo Auto-Negative genera il negative specifico per quella caption.

Uso:  python -m workers.lingbot_rewriter --prompt "..." --mode t2v --out result.json
      [--first-frame path.png]
Env:  LINGBOT_REWRITER_BASE (cartella VLM base), LINGBOT_REWRITER_LORA (cartella rewriter-LoRA)

Scrive result.json = {"prompt_json": "<json string>", "negative": "..."}.

⚠️ LIVE-TEST PENDENTE: la ricetta esatta (template chat, nome adapter, formato caption) va confermata
contro il pacchetto ufficiale. Struttura completa; punti da validare marcati con # VALIDATE. In caso di
errore scrive comunque un result.json col prompt originale così il chiamante non si blocca mai.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EXPAND_SYS = ("You are a prompt engineer for a text-to-video model. Expand the user's prompt into a "
               "vivid, concrete description of a single continuous shot: subject, action, setting, "
               "camera, lighting, mood. Keep it physically plausible. Output prose only.")
_JSON_SYS = ("Convert the description into a STRICT JSON caption for the LingBot-Video DiT. Output ONLY "
             "the JSON object.")
_NEG_SYS = ("Given a video caption, output a short negative prompt (comma-separated) of artefacts to avoid "
            "for THIS specific caption. Output only the negative text.")


def _load():
    """Carica il VLM base + la rewriter-LoRA (peft). Ritorna (model, tok, has_lora)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = os.environ["LINGBOT_REWRITER_BASE"]
    lora = os.environ.get("LINGBOT_REWRITER_LORA", "")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    has_lora = False
    if lora and os.path.isdir(lora):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora, adapter_name="rewriter")
            has_lora = True
        except Exception as ex:                              # noqa: BLE001
            print(f"[rewriter] LoRA non caricata ({ex})", flush=True)
    return model, tok, has_lora


def _chat(model, tok, system: str, user: str, *, use_lora: bool, has_lora: bool, max_new: int = 400) -> str:
    import torch
    # VALIDATE: attiva/disattiva l'adapter per-stadio (stadio 1 = base, stadio 2 = base+LoRA)
    if has_lora:
        try:
            if use_lora:
                model.set_adapter("rewriter")
            else:
                model.disable_adapter().__enter__()          # VALIDATE: gestione adapter off
        except Exception:                                    # noqa: BLE001
            pass
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    except Exception:                                        # noqa: BLE001
        inputs = tok(system + "\n\n" + user, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mode", default="t2v")
    ap.add_argument("--first-frame", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result = {"prompt_json": a.prompt, "negative": ""}
    try:
        model, tok, has_lora = _load()
        expanded = _chat(model, tok, _EXPAND_SYS, a.prompt, use_lora=False, has_lora=has_lora)      # stadio 1
        caption = _chat(model, tok, _JSON_SYS, expanded, use_lora=True, has_lora=has_lora)          # stadio 2
        negative = _chat(model, tok, _NEG_SYS, caption, use_lora=False, has_lora=has_lora, max_new=120)
        # valida che la caption sia JSON; altrimenti impacchetta come {"caption": ...}
        try:
            json.loads(caption)
            prompt_json = caption
        except Exception:                                    # noqa: BLE001
            prompt_json = json.dumps({"caption": expanded or a.prompt}, ensure_ascii=False)
        result = {"prompt_json": prompt_json, "negative": negative}
    except Exception as ex:                                  # noqa: BLE001
        print(f"[rewriter] errore ({ex}) → prompt originale", flush=True)

    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
