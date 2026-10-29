"""live_guard — CINTURA DI SICUREZZA per ogni test dal vivo (task 12, dopo il crash di sistema del 17/10/2026).

Lancia un comando di test e, ogni N secondi, misura il COMMIT libero (RAM + pagefile: il limite VERO su Windows) e la
RAM fisica. Se il commit libero scende sotto la soglia (default 20 GB) — o la RAM sotto la sua, o un singolo processo
supera il tetto — UCCIDE l'albero dei processi del test e i server che il test ha avviato, e scrive il MOTIVO
(stdout/stderr + file DIAG_live_guard_<ora>.txt in output\\ + CSV dei campioni).

Uso (dalla radice del repo):
    python tests\\live_guard.py [--min-commit-gb 20] [--min-ram-gb 0] [--max-proc-gb 0] [--interval 2]
                               [--out-dir output] [--kill-servers-on-exit] -- <comando del test ...>

  Es.:  python tests\\live_guard.py --min-commit-gb 20 -- python orch03\\live_run\\live_t12.py L1new

Codice d'uscita: quello del test; 99 se il guardiano ha ucciso il test per soglia; 98 se il comando non parte.

Cosa uccide (e cosa MAI):
  - il processo del test e tutti i suoi discendenti (psutil, anche i detached finché il genitore è vivo);
  - ogni altro processo il cui comando combacia con i pattern dei SERVER GPU (inference_server, persistent_runner,
    job_runner, workers.ltx2_server/krea2_server/krea_rt_server/ltx2_infer, measure_worker…) e che NON esisteva prima
    dell'avvio del test (snapshot iniziale): i server dell'utente già vivi non si toccano mai.
  - mai se stesso né i propri antenati.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# processi che il test può aver lanciato e che tengono memoria GPU/commit anche dopo la sua morte
SERVER_PATTERN = (r"inference_server[\\/. ]|inference_server\.server|persistent_runner|job_runner|measure_worker|"
                  r"workers[\\/.](ltx2_server|ltx2_infer|krea2_server|krea2_infer|krea_rt_server|qwen_image21_infer|flux3_infer|"
                  r"wan_multitalk|minimax_h3_infer)")
EXIT_BREACH = 99
EXIT_NO_START = 98


def read_commit_mb() -> float:
    """Commit disponibile (MB), ullAvailPageFile. 1e9 se non rilevabile."""
    sys.path.insert(0, ROOT)
    try:
        from core import st_direct
        return float(st_direct.commit_avail_mb())
    except Exception:                                       # noqa: BLE001
        return 1e9


def read_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2**20
    except Exception:                                       # noqa: BLE001
        return 1e9


def list_procs() -> list[dict]:
    """[{pid, ppid, name, cmd, private_mb}] di tutti i processi visibili."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            mi = p.memory_info()
            priv = getattr(mi, "private", None) or mi.vms
            out.append({"pid": p.info["pid"], "ppid": p.info["ppid"], "name": p.info["name"] or "",
                        "cmd": " ".join(p.info["cmdline"] or []), "private_mb": priv / 2**20})
        except Exception:                                   # noqa: BLE001 — processo sparito / accesso negato
            continue
    return out


def kill_pid(pid: int) -> None:
    """Uccide `pid` e il suo albero (taskkill /T su Windows; psutil altrove)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return
    try:
        import psutil
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except Exception:                               # noqa: BLE001
                pass
        p.kill()
    except Exception:                                       # noqa: BLE001
        pass


def descendants(procs: list[dict], root_pid: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for p in procs:
        kids.setdefault(p["ppid"], []).append(p["pid"])
    seen, stack = set(), [root_pid]
    while stack:
        cur = stack.pop()
        for c in kids.get(cur, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def ancestors(procs: list[dict], pid: int) -> set[int]:
    by = {p["pid"]: p["ppid"] for p in procs}
    out, cur = set(), pid
    while cur in by and by[cur] not in out and by[cur] != cur:
        cur = by[cur]
        out.add(cur)
    return out


def pick_victims(procs: list[dict], root_pid: int, snapshot: set[int], pattern: str, self_pid: int) -> list[dict]:
    """Chi uccidere: il test + discendenti + server nuovi (non presenti nello snapshot iniziale). Mai se stesso/antenati."""
    rx = re.compile(pattern, re.I)
    protect = {self_pid} | ancestors(procs, self_pid)
    tree = {root_pid} | descendants(procs, root_pid)
    out = []
    for p in procs:
        if p["pid"] in protect:
            continue
        if p["pid"] in tree or (p["pid"] not in snapshot and rx.search(p["cmd"] or "")):
            out.append(p)
    return out


class Guard:
    def __init__(self, cmd, *, min_commit_mb=20480.0, min_ram_mb=0.0, max_proc_mb=0.0, interval=2.0,
                 out_dir=None, pattern=SERVER_PATTERN, kill_servers_on_exit=False,
                 read_commit=read_commit_mb, read_ram=read_ram_mb, procs=list_procs, kill=kill_pid,
                 popen=subprocess.Popen, sleep=time.sleep, clock=time.time, say=None):
        self.cmd = list(cmd)
        self.min_commit_mb, self.min_ram_mb, self.max_proc_mb = float(min_commit_mb), float(min_ram_mb), float(max_proc_mb)
        self.interval = float(interval)
        self.out_dir = out_dir or os.path.join(ROOT, "output")
        self.pattern = pattern
        self.kill_servers_on_exit = kill_servers_on_exit
        self.read_commit, self.read_ram, self.procs, self.kill = read_commit, read_ram, procs, kill
        self.popen, self.sleep, self.clock = popen, sleep, clock
        self.say = say or (lambda m: print(m, file=sys.stderr, flush=True))
        self.reason_file = None
        self.samples = []

    # ── ciclo ────────────────────────────────────────────────────────────────
    def _check(self, commit, ram, procs):
        """Ritorna il motivo di violazione, oppure None."""
        if commit < self.min_commit_mb:
            return (f"commit libero {commit / 1024:.1f} GB < soglia {self.min_commit_mb / 1024:.1f} GB")
        if self.min_ram_mb and ram < self.min_ram_mb:
            return f"RAM libera {ram / 1024:.1f} GB < soglia {self.min_ram_mb / 1024:.1f} GB"
        if self.max_proc_mb:
            big = max(procs, key=lambda p: p["private_mb"], default=None)
            if big and big["private_mb"] > self.max_proc_mb:
                return (f"processo {big['pid']} ({big['name']}) usa {big['private_mb'] / 1024:.1f} GB "
                        f"> tetto {self.max_proc_mb / 1024:.1f} GB")
        return None

    def run(self) -> int:
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.out_dir, f"DIAG_live_guard_{stamp}.csv")
        snapshot = {p["pid"] for p in self.procs() if re.search(self.pattern, p["cmd"] or "", re.I)}
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            child = self.popen(self.cmd, **kw)
        except Exception as e:                              # noqa: BLE001
            self.say(f"LIVE_GUARD: comando non avviabile: {e}")
            return EXIT_NO_START
        self.say(f"LIVE_GUARD: test avviato (PID {child.pid}); soglie: commit libero ≥ {self.min_commit_mb / 1024:.0f} GB"
                 + (f", RAM libera ≥ {self.min_ram_mb / 1024:.0f} GB" if self.min_ram_mb else "")
                 + (f", processo ≤ {self.max_proc_mb / 1024:.0f} GB" if self.max_proc_mb else "")
                 + f"; campione ogni {self.interval:g}s → {csv_path}")
        t0 = self.clock()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "commit_free_mb", "ram_free_mb", "max_proc_mb"])
            while True:
                rc = child.poll()
                commit, ram = self.read_commit(), self.read_ram()
                procs = self.procs()
                big = max((p["private_mb"] for p in procs), default=0.0)
                w.writerow([f"{self.clock() - t0:.1f}", f"{commit:.0f}", f"{ram:.0f}", f"{big:.0f}"])
                fh.flush()
                self.samples.append((commit, ram, big))
                reason = self._check(commit, ram, procs)
                if reason:
                    return self._breach(reason, child, procs, snapshot, commit, ram)
                if rc is not None:
                    if self.kill_servers_on_exit:
                        for p in pick_victims(procs, child.pid, snapshot, self.pattern, os.getpid()):
                            self.kill(p["pid"])
                    return rc
                self.sleep(self.interval)

    def _breach(self, reason, child, procs, snapshot, commit, ram) -> int:
        victims = pick_victims(procs, child.pid, snapshot, self.pattern, os.getpid())
        for p in sorted(victims, key=lambda p: -p["pid"]):   # prima i figli (PID più alti), poi la radice
            self.kill(p["pid"])
        try:
            child.kill()
        except Exception:                                    # noqa: BLE001
            pass
        lines = [f"LIVE_GUARD: TEST UCCISO — {reason}",
                 f"ora: {time.strftime('%Y-%m-%d %H:%M:%S')}  commit libero {commit / 1024:.1f} GB  RAM libera {ram / 1024:.1f} GB",
                 f"comando: {' '.join(self.cmd)}",
                 f"uccisi {len(victims)} processi:"]
        lines += [f"  PID {p['pid']:>6}  {p['private_mb'] / 1024:6.1f} GB  {p['name']}  {p['cmd'][:140]}" for p in victims]
        text = "\n".join(lines)
        self.say(text)
        self.reason_file = os.path.join(self.out_dir, f"DIAG_live_guard_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(self.reason_file, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            pass
        return EXIT_BREACH


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Guardiano di memoria per test dal vivo (uccide il test se il commit scende).")
    ap.add_argument("--min-commit-gb", type=float, default=20.0, help="commit libero minimo (GB), default 20")
    ap.add_argument("--min-ram-gb", type=float, default=0.0, help="RAM fisica libera minima (GB), 0 = non controllare")
    ap.add_argument("--max-proc-gb", type=float, default=0.0, help="tetto di memoria per singolo processo (GB), 0 = off")
    ap.add_argument("--interval", type=float, default=2.0, help="secondi fra i campioni, default 2")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--kill-servers-on-exit", action="store_true", help="a fine test uccide anche i server nuovi rimasti")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- comando del test")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("manca il comando del test (dopo --)")
    g = Guard(cmd, min_commit_mb=a.min_commit_gb * 1024, min_ram_mb=a.min_ram_gb * 1024,
              max_proc_mb=a.max_proc_gb * 1024, interval=a.interval, out_dir=a.out_dir,
              kill_servers_on_exit=a.kill_servers_on_exit)
    return g.run()


if __name__ == "__main__":
    raise SystemExit(main())
