"""Regole di compatibilità dichiarative per LTX Studio.

Le restrizioni `ExclusiveGroup` (max one) / `ExactlyOne` (always one) replicano i
toggleRestriction dei Fast Groups Bypasser del workflow Pluto (5753 modello, 5660 audio).
Ogni regola valuta lo stato (dict) e produce Conflict con eventuali fix (auto-correzioni).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Conflict:
    rule_id: str
    severity: str           # "blocking" | "warning"
    message: str
    fixes: list["Fix"] = field(default_factory=list)


@dataclass
class Fix:
    label: str
    apply: Callable[[dict], None]   # muta lo stato


class Rule:
    def check(self, state: dict) -> Conflict | None:
        raise NotImplementedError


@dataclass
class Requires(Rule):
    """Se `when` è vero, allora `needs` deve essere valorizzato."""
    id: str
    when: Callable[[dict], bool]
    needs: str
    message: str
    severity: str = "blocking"

    def check(self, state):
        if self.when(state) and not state.get(self.needs):
            return Conflict(self.id, self.severity, self.message)
        return None


@dataclass
class ExactlyOne(Rule):
    """Esattamente una delle chiavi booleane deve essere True (always one)."""
    id: str
    keys: list[str]
    default: str
    message: str
    severity: str = "warning"

    def check(self, state):
        on = [k for k in self.keys if state.get(k)]
        if len(on) == 1:
            return None
        def fix_state(s, chosen=self.default, keys=self.keys):
            for k in keys:
                s[k] = (k == chosen)
        msg = self.message + f" (attive: {on or 'nessuna'})"
        return Conflict(self.id, self.severity, msg,
                        [Fix(f"Tieni solo '{self.default}'", fix_state)])


@dataclass
class ExclusiveGroup(Rule):
    """Al massimo una delle chiavi booleane può essere True (max one)."""
    id: str
    keys: list[str]
    message: str
    severity: str = "warning"

    def check(self, state):
        on = [k for k in self.keys if state.get(k)]
        if len(on) <= 1:
            return None
        keep = on[0]
        def fix_state(s, keep=keep, keys=self.keys):
            for k in keys:
                if k != keep:
                    s[k] = False
        return Conflict(self.id, self.severity, self.message + f" (attive: {on})",
                        [Fix(f"Tieni solo '{keep}'", fix_state)])


@dataclass
class Implies(Rule):
    """Se `when` è vero, imposta automaticamente dei valori (preset propagation)."""
    id: str
    when: Callable[[dict], bool]
    set_values: dict
    message: str = ""
    severity: str = "warning"

    def check(self, state):
        if self.when(state):
            need = {k: v for k, v in self.set_values.items() if state.get(k) != v}
            if need:
                def fix_state(s, vals=self.set_values):
                    s.update(vals)
                return Conflict(self.id, self.severity,
                                self.message or f"Auto-impostazione: {self.set_values}",
                                [Fix("Applica automaticamente", fix_state)])
        return None


def default_rules() -> list[Rule]:
    return [
        Requires("cnet_needs_video", lambda s: s.get("controlnet_enabled"),
                 "controlnet_ref_video", "Il ControlNet richiede un video di riferimento."),
        Requires("clone_needs_sample", lambda s: s.get("audio_mode") == "clone",
                 "voice_sample", "La clonazione voce richiede un campione audio (file o registrazione)."),
        ExactlyOne("audio_one", ["audio_mute", "audio_clone", "audio_native", "audio_user"],
                   default="audio_mute", message="Serve esattamente una sorgente audio (o Muto)."),
        ExclusiveGroup("model_one", ["model_standard", "model_gguf", "model_eros"],
                       message="Solo un modello LTX-2.3 attivo per volta."),
        Implies("nsfw_preset", lambda s: s.get("nsfw_mode") == "nsfw",
                {"model_eros": True, "model_standard": False, "model_gguf": False},
                message="NSFW → uso il modello 10EROS."),
    ]
