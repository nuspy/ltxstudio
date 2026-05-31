"""Generazione del frame iniziale da prompt (#3).

Usa un golden txt2img catturato (Z-Image/Flux) come template, parametrizzato con prompt e
dimensioni. Coerente con l'architettura golden-prompt del resto dell'app: se il template non
è ancora stato catturato, è disponibile() ritorna False e l'UI lo segnala.

Convenzione file template (in templates/): txt2img_<model>.api.json
Binding minimi cercati per euristica (text/prompt, width/height) sui nodi comuni.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class FrameGenerator:
    def __init__(self, model_key: str = "zimage"):
        self.model_key = model_key
        self.template_path = TEMPLATES_DIR / f"txt2img_{model_key}.api.json"
        self._template = None
        if self.template_path.exists():
            try:
                self._template = json.loads(self.template_path.read_text(encoding="utf-8"))
            except Exception:
                self._template = None

    def available(self) -> bool:
        return self._template is not None

    @staticmethod
    def list_available() -> list[str]:
        return [p.stem.replace("txt2img_", "") for p in TEMPLATES_DIR.glob("txt2img_*.api.json")]

    def build_prompt(self, prompt: str, width: int, height: int, seed: int) -> dict:
        if not self._template:
            raise RuntimeError(
                f"Template txt2img mancante: {self.template_path.name}. "
                "Cattura il golden del workflow txt2img (vedi README_capture).")
        p = copy.deepcopy(self._template)
        for nid, node in p.items():
            ct = node.get("class_type", "")
            ins = node.get("inputs", {})
            # prompt positivo: CLIPTextEncode / PrimitiveStringMultiline con testo letterale
            if ct in ("CLIPTextEncode",) and isinstance(ins.get("text"), str):
                # euristica: il primo CLIPTextEncode non-vuoto e' il positivo
                if "negative" not in node.get("_meta", {}).get("title", "").lower():
                    ins["text"] = prompt
            if ct in ("PrimitiveStringMultiline",) and isinstance(ins.get("value"), str):
                ins["value"] = prompt
            # dimensioni
            for k in ("width", "height"):
                if isinstance(ins.get(k), int):
                    ins[k] = width if k == "width" else height
            # seed
            for k in ("seed", "noise_seed"):
                if isinstance(ins.get(k), int):
                    ins[k] = seed
        return p
