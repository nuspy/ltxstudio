"""Cache dello STATO DEL TESTO di un clip H3 (embedding del testo, tag, riferimenti normalizzati) dentro la cartella del clip.

Quando un clip va in OOM e il worker lo rifa col ripiego (piu blocchi in RAM), la codifica del testo non cambia: a 736p costa ~360 s (turno 40 del 04/11). Il primo
tentativo salva lo stato prodotto dalla fase del testo, il secondo lo ripristina senza caricare il text encoder. Valido solo con la stessa chiave del clip (`want_key.txt`
scritta da run_edit: sorgente, maschere, parametri, prompt). Un file illeggibile non blocca: si riscrive il testo da zero e il motivo e nel log."""
from __future__ import annotations

import os
import pickle

import torch

MISSING = object()


class OnCuda:
    """Segna un tensore che stava sulla GPU: nel file e su CPU, al ripristino torna sul dispositivo."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


def to_storage(value):
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu()
        return OnCuda(cpu) if value.is_cuda else cpu
    if isinstance(value, (list, tuple)):
        return type(value)(to_storage(item) for item in value)
    if isinstance(value, dict):
        return {name: to_storage(item) for name, item in value.items()}
    return value


def from_storage(value, device):
    if isinstance(value, OnCuda):
        return value.tensor.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(from_storage(item, device) for item in value)
    if isinstance(value, dict):
        return {name: from_storage(item, device) for name, item in value.items()}
    return value


class TextStateCache:
    def __init__(self, path: str, key: str, log):
        self.path, self.key, self.log = path, key, log

    @classmethod
    def for_request(cls, params: dict, log):
        """La cache di una richiesta: solo negli edit a clip (params['_h3_edit']['text_cache'] = {path, key}); None altrimenti."""
        info = (params.get("_h3_edit") or {}).get("text_cache")
        return cls(info["path"], info["key"], log) if info and info.get("key") else None

    def save(self, state_values: dict, input_values: dict, kwargs_mapping: dict) -> None:
        produced = {name: to_storage(value) for name, value in state_values.items() if input_values.get(name, MISSING) is not value}
        payload = {"key": self.key, "inputs": [name for name in state_values if name not in produced], "produced": produced,
                   "kwargs_mapping": {kind: list(names) for kind, names in kwargs_mapping.items()}}
        temporary = self.path + ".tmp"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, self.path)
        except OSError as error:
            self.log(f"cache del testo: salvataggio non riuscito ({type(error).__name__}: {error}); un riavvio ricodificherebbe il testo")
            return
        self.log(f"cache del testo salvata ({os.path.getsize(self.path) / 2**20:.0f} MB): un riavvio dopo OOM non ricodifica il testo")

    def load(self):
        """Il contenuto salvato se esiste ed e della stessa chiave; None se manca o non vale (con il motivo nel log quando il file esiste ma non e usabile)."""
        if not os.path.isfile(self.path):
            return None
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, ValueError, AttributeError, ImportError, pickle.UnpicklingError) as error:
            self.log(f"cache del testo illeggibile ({type(error).__name__}: {error}): ricodifico il testo")
            return None
        if payload.get("key") != self.key:
            self.log("cache del testo di un altro clip/parametri: ricodifico il testo")
            return None
        return payload

    @staticmethod
    def restore(payload: dict, inputs: dict, device, new_state):
        """Riempie `new_state` (PipelineState vuoto) con gli ingressi della richiesta e i valori salvati; ritorna lo stato."""
        for name in payload["inputs"]:
            new_state.set(name, inputs[name])
        for name, value in payload["produced"].items():
            new_state.set(name, from_storage(value, device))
        new_state.kwargs_mapping = {kind: list(names) for kind, names in payload["kwargs_mapping"].items()}
        return new_state
