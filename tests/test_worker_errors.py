"""core.worker_errors.explain — la causa di un worker fallito arriva intera (02/11)."""
from core.worker_errors import explain

_TB = ("Traceback (most recent call last):\n"
       '  File "E:\\AI\\Tools\\LTX-2\\packages\\ltx-core\\src\\ltx_core\\utils.py", line 61, in find_matching_file\n'
       "    raise FileNotFoundError(...)\n"
       "FileNotFoundError: No files matching pattern 'tokenizer.model' found under "
       "E:\\AI\\Models\\text_encoders\\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors\n")


def test_cause_is_whole_and_first():
    out = explain("[stage] Caricamento modello\n", _TB)
    first = out.splitlines()[0]
    assert first.startswith("FileNotFoundError: No files matching pattern 'tokenizer.model'")
    assert first.endswith("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    assert "find_matching_file" in out            # la coda del traceback c'e'


def test_not_cut_at_500_chars():
    big = "Traceback (most recent call last):\n" + "".join(f'  File "f{i}.py", line {i}, in g\n    x\n' for i in range(80)) + "ValueError: boom " + "x" * 600
    out = explain("", big, limit=4000)
    assert out.splitlines()[0].endswith("x" * 600)           # causa intera anche se lunga
    assert len(out) > 1000


def test_no_traceback_uses_last_line():
    assert explain("", "").startswith("errore sconosciuto") or explain("", "", return_code=3).startswith("il worker")
    assert explain("ok\nfallito: manca il file X\n", "").splitlines()[0] == "fallito: manca il file X"


def test_stdout_traceback_when_stderr_empty():
    out = explain(_TB, "")
    assert out.splitlines()[0].startswith("FileNotFoundError")


def test_tail_limit_applies_to_tail_only():
    tb = "Traceback (most recent call last):\n" + "  File x\n" * 2000 + "RuntimeError: la causa\n"
    out = explain("", tb, limit=300)
    assert out.splitlines()[0] == "RuntimeError: la causa"
    assert len(out) < 700
