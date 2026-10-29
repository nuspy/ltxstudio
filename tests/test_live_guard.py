"""Test di tests/live_guard.py (guardiano di memoria per i test dal vivo). Nessuna GPU; processi veri ma innocui
(sleep). ATTENZIONE: nei test col kill vero il pattern dei server è DISATTIVATO (`(?!)` non combacia mai): il test non
deve poter uccidere server di altri agenti che girano in parallelo."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psutil  # noqa: E402
import pytest  # noqa: E402

import live_guard as lg  # noqa: E402

NEVER = r"(?!)"
SLEEP = [sys.executable, "-c", "import time; time.sleep(120)"]


def _alive(pid):
    return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


class _Rec:
    """popen che ricorda il figlio, per verificarne la morte."""
    child = None

    def __call__(self, cmd, **kw):
        self.child = subprocess.Popen(cmd, **kw)
        return self.child


def test_breach_kills_the_test_tree_and_writes_the_reason(tmp_path):
    rec = _Rec()
    seq = iter([40000.0, 38000.0, 9000.0] + [9000.0] * 100)             # il commit libero crolla al 3° campione
    g = lg.Guard(SLEEP, min_commit_mb=20 * 1024, interval=0.05, out_dir=str(tmp_path), pattern=NEVER,
                 read_commit=lambda: next(seq), read_ram=lambda: 80000.0, popen=rec, say=lambda m: None)
    t0 = time.time()
    rc = g.run()
    assert rc == lg.EXIT_BREACH == 99
    assert time.time() - t0 < 30
    rec.child.wait(timeout=10)
    assert rec.child.poll() is not None                                  # il test è MORTO
    txt = open(g.reason_file, encoding="utf-8").read()
    assert "TEST UCCISO" in txt and "commit libero 8.8 GB < soglia 20.0 GB" in txt and str(rec.child.pid) in txt
    assert any(f.endswith(".csv") for f in os.listdir(tmp_path))         # campioni su CSV


def test_breach_also_kills_grandchildren(tmp_path):
    code = ("import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
            "print(p.pid,flush=True);time.sleep(120)")
    pids = {}

    def popen(cmd, **kw):
        c = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, **kw)
        pids["child"] = c
        pids["grand"] = int(c.stdout.readline())                         # PID del nipote
        return c
    seq = iter([40000.0, 1000.0] + [1000.0] * 50)
    g = lg.Guard([sys.executable, "-c", code], min_commit_mb=20 * 1024, interval=0.05, out_dir=str(tmp_path),
                 pattern=NEVER, read_commit=lambda: next(seq), read_ram=lambda: 80000.0, popen=popen, say=lambda m: None)
    assert g.run() == 99
    pids["child"].wait(timeout=10)
    deadline = time.time() + 10
    while time.time() < deadline and _alive(pids["grand"]):
        time.sleep(0.1)
    assert not _alive(pids["grand"])                                     # morto anche il discendente


def test_healthy_run_returns_the_test_exit_code_and_kills_nothing(tmp_path):
    g = lg.Guard([sys.executable, "-c", "import sys; sys.exit(3)"], min_commit_mb=20 * 1024, interval=0.05,
                 out_dir=str(tmp_path), pattern=NEVER, read_commit=lambda: 90000.0, read_ram=lambda: 80000.0,
                 say=lambda m: None)
    assert g.run() == 3 and g.reason_file is None


def test_command_that_does_not_start(tmp_path):
    g = lg.Guard(["__comando_inesistente__"], out_dir=str(tmp_path), pattern=NEVER, say=lambda m: None)
    assert g.run() == lg.EXIT_NO_START


def _p(pid, ppid, cmd, mb=100.0, name="python.exe"):
    return {"pid": pid, "ppid": ppid, "name": name, "cmd": cmd, "private_mb": mb}


def test_pick_victims_test_tree_and_new_servers_only():
    procs = [_p(1, 0, "explorer"), _p(499, 1, "terminal"), _p(500, 499, "python live_guard.py"),
             _p(100, 500, "python live_t12.py"), _p(101, 100, "python -m inference_server.server"),
             _p(200, 1, "python -m inference_server.persistent_runner"),            # NUOVO server, staccato dal test
             _p(300, 1, "python -m inference_server.persistent_runner"),            # server GIÀ vivo prima del test
             _p(400, 1, "python altro.py"), _p(601, 499, "python workers.ltx2_server")]   # nuovo ma non nell'albero: pattern
    v = {p["pid"] for p in lg.pick_victims(procs, 100, {300}, lg.SERVER_PATTERN, 500)}
    assert v == {100, 101, 200, 601}
    assert 300 not in v and 400 not in v and 500 not in v and 499 not in v     # snapshot, estranei, se stesso e antenati: MAI


def test_thresholds_commit_ram_and_single_process():
    g = lg.Guard(["x"], min_commit_mb=20480, min_ram_mb=8192, max_proc_mb=60 * 1024, pattern=NEVER)
    ok = [_p(1, 0, "a", 1000)]
    assert g._check(30000, 30000, ok) is None
    assert "commit libero" in g._check(19000, 30000, ok)
    assert "RAM libera" in g._check(30000, 4000, ok)
    assert "tetto" in g._check(30000, 30000, [_p(7, 0, "a", 90 * 1024)])          # il caso del 17/10: ~90 GB in un processo
    off = lg.Guard(["x"], min_commit_mb=20480, pattern=NEVER)
    assert off._check(30000, 100, [_p(7, 0, "a", 90 * 1024)]) is None            # RAM e tetto per-processo spenti di default


def test_default_readers_return_numbers():
    assert lg.read_commit_mb() > 0 and lg.read_ram_mb() > 0
    assert any(p["pid"] == os.getpid() for p in lg.list_procs())
