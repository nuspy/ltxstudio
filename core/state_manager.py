"""Stato centrale dell'app + valutazione conflitti.

Single source of truth: i tab leggono/scrivono qui. Ogni set() rivaluta le regole.
Gli Implies si auto-applicano (preset); gli altri conflitti vengono restituiti alla UI
che mostra il dialog con i fix.
"""
from __future__ import annotations

from typing import Any

from core.conflict_rules import Conflict, Rule, default_rules


class StateManager:
    def __init__(self, rules: list[Rule] | None = None):
        self.state: dict[str, Any] = {
            "audio_native": True, "audio_clone": False, "audio_user": False,
            "audio_mode": "native",
            "model_standard": True, "model_gguf": False, "model_eros": False,
            "nsfw_mode": "sfw",
            "controlnet_enabled": False,
        }
        self.rules = rules or default_rules()

    def set(self, key: str, value: Any, auto_apply_implies: bool = True) -> list[Conflict]:
        self.state[key] = value
        return self.evaluate(auto_apply_implies)

    def update(self, values: dict, auto_apply_implies: bool = True) -> list[Conflict]:
        self.state.update(values)
        return self.evaluate(auto_apply_implies)

    def evaluate(self, auto_apply_implies: bool = True, max_iter: int = 10) -> list[Conflict]:
        """Valuta le regole. Auto-applica gli Implies (con guard anti-ciclo). Restituisce
        i conflitti residui (non auto-risolti)."""
        from core.conflict_rules import Implies
        for _ in range(max_iter):
            changed = False
            for r in self.rules:
                if auto_apply_implies and isinstance(r, Implies):
                    c = r.check(self.state)
                    if c and c.fixes:
                        c.fixes[0].apply(self.state)
                        changed = True
            if not changed:
                break
        conflicts = []
        for r in self.rules:
            from core.conflict_rules import Implies as _Imp
            if isinstance(r, _Imp):
                continue
            c = r.check(self.state)
            if c:
                conflicts.append(c)
        return conflicts

    def apply_fix(self, conflict: Conflict, index: int = 0) -> list[Conflict]:
        if conflict.fixes:
            conflict.fixes[index].apply(self.state)
        return self.evaluate()

    def blocking(self) -> list[Conflict]:
        return [c for c in self.evaluate(auto_apply_implies=False) if c.severity == "blocking"]


if __name__ == "__main__":
    sm = StateManager()
    print("Stato iniziale conflitti:", [(c.rule_id, c.severity) for c in sm.evaluate()])

    print("\n-- Attivo ControlNet senza video --")
    cs = sm.set("controlnet_enabled", True)
    for c in cs:
        print(f"  [{c.severity}] {c.rule_id}: {c.message}")

    print("\n-- Passo a NSFW (deve forzare 10EROS via Implies) --")
    sm.set("nsfw_mode", "nsfw")
    print("  model_eros =", sm.state["model_eros"], "| model_standard =", sm.state["model_standard"])

    print("\n-- Attivo due modelli (ExclusiveGroup) --")
    sm.state["model_standard"] = True; sm.state["model_eros"] = True
    cs = sm.evaluate(auto_apply_implies=False)
    for c in cs:
        if c.rule_id == "model_one":
            print(f"  conflitto: {c.message}; applico fix '{c.fixes[0].label}'")
            sm.apply_fix(c)
            print("  dopo fix: std=", sm.state["model_standard"], "eros=", sm.state["model_eros"])

    print("\n-- Audio: nessuna sorgente attiva --")
    sm.update({"audio_native": False, "audio_clone": False, "audio_user": False})
    cs = sm.evaluate(auto_apply_implies=False)
    for c in cs:
        if c.rule_id == "audio_one":
            print(f"  {c.message} -> fix '{c.fixes[0].label}'")
            sm.apply_fix(c)
            print("  dopo fix: native=", sm.state["audio_native"])
