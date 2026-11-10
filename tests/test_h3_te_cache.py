"""Cache dello stato del testo di un clip H3 (riavvio dopo OOM senza ricodificare il testo, 04/11)."""
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import h3_te_cache as cache_module  # noqa: E402

VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"


@dataclass
class FakeState:
    values: dict = field(default_factory=dict)
    kwargs_mapping: dict = field(default_factory=dict)

    def set(self, key, value, kwargs_type=None):
        self.values[key] = value


class Reference:                                                 # un riferimento normalizzato (oggetto Python semplice, come i MiniMaxH3*Reference)
    def __init__(self, frames):
        self.frames = frames


def make_inputs():
    return {"prompt": "a refuge", "num_frames": 141, "references": ["ref"]}


def make_state_values(inputs):
    return {**inputs, "prompt_embeds": torch.randn(1, 6, 8), "text_token_tags": torch.arange(6), "normalized_references": [Reference(torch.ones(2, 2))]}


def test_roundtrip_restores_inputs_produced_values_and_kwargs_mapping(tmp_path):
    logs = []
    cache = cache_module.TextStateCache(str(tmp_path / "te_state.pt"), "chiave-1", logs.append)
    inputs = make_inputs()
    values = make_state_values(inputs)
    cache.save(values, inputs, {"denoiser_input_fields": ["prompt_embeds"]})
    assert (tmp_path / "te_state.pt").is_file() and any("cache del testo salvata" in m for m in logs)
    payload = cache.load()
    assert payload is not None and sorted(payload["inputs"]) == ["num_frames", "prompt", "references"] and set(payload["produced"]) == {"prompt_embeds", "text_token_tags", "normalized_references"}
    state = cache_module.TextStateCache.restore(payload, inputs, "cpu", FakeState())
    assert state.values["prompt"] is inputs["prompt"] and state.values["references"] is inputs["references"]          # gli ingressi sono quelli della richiesta
    assert torch.equal(state.values["prompt_embeds"], values["prompt_embeds"]) and torch.equal(state.values["text_token_tags"], values["text_token_tags"])
    assert torch.equal(state.values["normalized_references"][0].frames, torch.ones(2, 2))
    assert state.kwargs_mapping == {"denoiser_input_fields": ["prompt_embeds"]}


def test_cuda_tensors_return_to_the_device_and_nested_containers_survive():
    marker = cache_module.OnCuda(torch.ones(3))
    restored = cache_module.from_storage({"a": [marker, (torch.zeros(1),)]}, "cpu")
    assert torch.equal(restored["a"][0], torch.ones(3)) and isinstance(restored["a"][1], tuple)
    stored = cache_module.to_storage({"x": [torch.ones(2), 5, "testo"]})
    assert torch.equal(stored["x"][0], torch.ones(2)) and stored["x"][1:] == [5, "testo"]


def test_other_key_or_unreadable_file_means_no_cache_with_a_log_line(tmp_path):
    logs = []
    path = str(tmp_path / "te_state.pt")
    cache_module.TextStateCache(path, "chiave-1", logs.append).save(make_state_values(make_inputs()), make_inputs(), {})
    assert cache_module.TextStateCache(path, "chiave-2", logs.append).load() is None and any("altro clip" in m for m in logs)
    Path(path).write_bytes(b"non e un file torch")
    assert cache_module.TextStateCache(path, "chiave-1", logs.append).load() is None and any("illeggibile" in m for m in logs)
    assert cache_module.TextStateCache(str(tmp_path / "assente.pt"), "k", logs.append).load() is None


def test_for_request_only_with_a_key():
    assert cache_module.TextStateCache.for_request({}, print) is None
    assert cache_module.TextStateCache.for_request({"_h3_edit": {"text_cache": {"path": "x", "key": ""}}}, print) is None
    request = {"_h3_edit": {"text_cache": {"path": "x.pt", "key": "abc"}}}
    assert cache_module.TextStateCache.for_request(request, print).key == "abc"


def test_wiring_in_worker_and_clip_runner():
    worker = (ROOT / "workers" / "minimax_h3_infer.py").read_text(encoding="utf-8")
    run_source = worker.split("def run(p: dict)", 1)[1]
    assert "state = _text_state(W, p, mode, kw, edit, blocks_text, g_text)" in run_source and "load_text_encoder(W)" not in run_source      # il testo e in una funzione a parte
    assert "te_cache = " not in worker and "cache.save(state.values, kw, state.kwargs_mapping)" in worker
    edit = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert '"text_cache": {"path": str(jd / "te_state.pt"), "key": _want_key(jd)}' in edit and '(jd / "te_state.pt").unlink(missing_ok=True)' in edit


@pytest.mark.skipif(not os.path.isfile(VENV_PY), reason="serve il venv di H3")
def test_restore_into_a_real_pipeline_state():
    script = (
        "import sys, torch; sys.path.insert(0, r'%s')\n"
        "from diffusers.modular_pipelines import PipelineState\n"
        "import h3_te_cache as c\n"
        "state = PipelineState(); state.set('prompt', 'p'); state.set('prompt_embeds', torch.ones(2, 3), 'denoiser_input_fields')\n"
        "cache = c.TextStateCache('x.pt', 'k', print)\n"
        "import tempfile, os\n"
        "cache.path = os.path.join(tempfile.mkdtemp(), 'te.pt')\n"
        "cache.save(state.values, {'prompt': state.values['prompt']}, state.kwargs_mapping)\n"
        "new = c.TextStateCache.restore(cache.load(), {'prompt': 'p'}, 'cpu', PipelineState())\n"
        "assert new.get('prompt') == 'p' and torch.equal(new.get('prompt_embeds'), torch.ones(2, 3)) and new.kwargs_mapping == state.kwargs_mapping\n"
        "assert list(new.get_by_kwargs('denoiser_input_fields')) == ['prompt_embeds']\n"
        "print('TE CACHE OK')\n" % (ROOT / "workers"))
    result = subprocess.run([VENV_PY, "-c", script], capture_output=True, text=True, timeout=300)
    assert "TE CACHE OK" in result.stdout, result.stdout[-600:] + result.stderr[-900:]
